# -*- coding: utf-8 -*-
"""
04 - Zones & Shot Detection Sandbox
=====================================
Define goal regions and robot zones, then test shot detection logic.

CONCEPT:
    A "shot" is a ball that:
    1. Was near a robot zone (attribution)
    2. Started moving upward/fast (launch detection)
    3. Either enters a goal region (scored) or doesn't (missed)

    Since these are lob shots into a 6ft bucket, we're looking for:
    - Upward velocity (negative vy in image coords, since y=0 is top)
    - Sufficient speed
    - Proximity to a robot at launch time
    - Eventual entry (or not) into goal region

WORKFLOW:
    1. Run 01 and 03 first to get detection + tracking working
    2. Set VIDEO_PATH below
    3. Define goal regions interactively
    4. Define robot starting zones (or just watch and label later)
    5. Run shot detection and review results

CONTROLS:
    g      - Define goal regions (rectangles) interactively
    G      - Define goal regions (polygons) interactively
    z      - Define robot zones interactively
    SPACE  - Play/Pause
    n      - Step forward one frame
    b      - Step backward one frame
    s      - Save zones + config
    q/ESC  - Quit

Author: Clay / Claude sandbox
"""

import sys
import os
import time
import csv
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, save_config, open_video, apply_roi,
    BallDetector, CentroidTracker, draw_hud, draw_zones,
    point_in_rect, point_in_polygon, draw_polygon,
    RobotDetector, draw_robots, create_ball_tracker
)


VIDEO_PATH = ""  # <-- SET THIS
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]


# ============================================================================
# SHOT DETECTOR
# ============================================================================

class ShotDetector:
    """
    Detects "shot events" from tracked ball data.

    A shot event is created when a tracked ball:
    1. Gains significant upward velocity (vy < threshold, since y increases downward)
    2. Is moving fast enough to be a shot, not just rolling
    3. Is near a robot zone at the time of launch

    The shot is then tracked to see if the ball enters a goal region.

    OCCLUSION-SAFE ATTRIBUTION:
        Shot attribution is locked at launch time and never re-evaluated.
        This means if a ball passes through another robot's zone after launch,
        it remains attributed to the original launching robot.
    """

    def __init__(self, config, robot_detector=None, fps=30.0):
        """
        Initialize shot detector.

        Args:
            config: Configuration dict
            robot_detector: Optional RobotDetector for dynamic robot tracking.
                           If None, falls back to static robot zones from config.
            fps: Video frame rate (used to scale velocity thresholds)
        """
        self.config = config
        self.robot_detector = robot_detector
        self.fps = fps

        shot_cfg = config.get("shot_detection", {})

        # Config thresholds are tuned for 30fps - scale for actual fps
        # At higher fps, per-frame velocity is lower (same real-world speed)
        fps_scale = 30.0 / fps  # e.g., 0.5 for 60fps

        self.min_upward_vy = shot_cfg.get("min_upward_velocity", -3.0) * fps_scale
        self.min_downward_vy = shot_cfg.get("min_downward_velocity", 2.0) * fps_scale
        self.require_bbox_exit = shot_cfg.get("require_bbox_exit", False)
        self.min_speed = shot_cfg.get("min_speed", 5.0) * fps_scale
        self.min_shot_speed = shot_cfg.get("min_shot_speed", 8.0) * fps_scale
        self.proximity = shot_cfg.get("proximity_to_robot", 120)

        # Goal proximity thresholds (for trajectory-based classification)
        self.goal_proximity_x = shot_cfg.get("goal_proximity_x", 150)
        self.goal_proximity_y = shot_cfg.get("goal_proximity_y", 100)
        self.classify_field_passes = shot_cfg.get("classify_field_passes", True)

        # Scale flight frames for fps (more frames at higher fps for same duration)
        base_flight_frames = shot_cfg.get("min_flight_frames", 4)
        self.min_flight_frames = max(1, int(base_flight_frames / fps_scale))

        # Velocity smoothing: use averaged velocity to reduce jitter at high fps
        # At high fps, per-frame velocity is noisy due to small centroid errors
        self.use_smoothed_velocity = fps > 45  # Auto-enable for high fps
        self.velocity_smoothing_window = max(2, int(3 / fps_scale))  # ~3 frames at 30fps

        # Print scaled thresholds if fps differs from reference
        if abs(fps - 30.0) > 1.0:
            smooth_note = f", smoothing={self.velocity_smoothing_window}f" if self.use_smoothed_velocity else ""
            print(f"  [SHOTS] FPS-adjusted thresholds for {fps:.0f}fps: "
                  f"vy<{self.min_upward_vy:.1f}, speed>{self.min_speed:.1f}, "
                  f"flight>={self.min_flight_frames} frames{smooth_note}")

        self.goal_regions = config.get("goal_regions", {}).get("regions", [])
        self.robot_zones = config.get("robot_zones", {}).get("robots", [])

        # Human player zones (for shot attribution)
        hp_cfg = config.get("human_player_zones", {})
        self.hp_zones_enabled = hp_cfg.get("enabled", False)
        self.hp_zones = hp_cfg.get("zones", [])

        # Active shot tracking
        self.shots = []           # list of ShotEvent
        self.active_shots = {}    # obj_id -> ShotEvent (in-flight)
        self.ball_launch_candidates = {}  # obj_id -> frame count of upward motion
        self.ball_launch_origins = {}  # obj_id -> (x, y, frame) when upward flight started
        self.ball_prev_positions = {}  # obj_id -> (prev_x, prev_y) for bbox exit detection
        self.scored_ball_ids = set()  # obj_ids that have entered a goal (prevent bounce-out double-counting)

        # Trajectory fitting for robust shot detection
        traj_cfg = shot_cfg
        self.trajectory_fit_window = traj_cfg.get("trajectory_fit_window", 8)
        self.min_trajectory_r_squared = traj_cfg.get("min_trajectory_r_squared", 0.85)
        self.enable_launch_extrapolation = traj_cfg.get("enable_launch_extrapolation", True)
        self.enable_goal_prediction = traj_cfg.get("enable_goal_prediction", True)

        # Callbacks for real-time streaming (Phase 2)
        self._on_shot_launched = None
        self._on_shot_resolved = None

    def set_callbacks(self, on_shot_launched=None, on_shot_resolved=None):
        """
        Register callbacks for real-time shot event notifications.

        Args:
            on_shot_launched: Callback(shot_event, frame_num) when a shot is detected
            on_shot_resolved: Callback(shot_event, frame_num) when a shot is resolved (scored/missed)

        For streaming pipeline, this allows immediate CSV logging without waiting
        for the full video to process.
        """
        self._on_shot_launched = on_shot_launched
        self._on_shot_resolved = on_shot_resolved

    def update(self, tracked_objects, frame_num):
        """
        Check all tracked objects for shot events.
        Call once per frame after tracker.update().

        Returns:
            list: Object IDs that should be removed from tracking (e.g., balls
                  that entered goals). Caller should pass these to tracker.remove_objects().
        """
        ids_to_remove = []

        for obj_id, obj in tracked_objects.items():
            # Skip balls that have already scored (bounce-out prevention)
            if obj_id in self.scored_ball_ids:
                continue

            if obj.disappeared > 0:
                # If this ball was being tracked as a shot and disappeared,
                # it might have landed
                if obj_id in self.active_shots:
                    shot = self.active_shots[obj_id]
                    if not shot.resolved:
                        shot.resolve("missed", frame_num)
                        if self._on_shot_resolved:
                            self._on_shot_resolved(shot, frame_num)
                    del self.active_shots[obj_id]
                # Clean up tracking state for disappeared ball
                if obj_id in self.ball_prev_positions:
                    del self.ball_prev_positions[obj_id]
                if obj_id in self.ball_launch_candidates:
                    del self.ball_launch_candidates[obj_id]
                if obj_id in self.ball_launch_origins:
                    del self.ball_launch_origins[obj_id]
                continue

            # Get previous position for bbox exit detection
            prev_pos = self.ball_prev_positions.get(obj_id, (obj.cx, obj.cy))

            # Check if ball is already being tracked as a shot
            if obj_id in self.active_shots:
                shot = self.active_shots[obj_id]
                shot.update_position(obj.cx, obj.cy, frame_num)

                # Check if ball entered goal region
                for goal in self.goal_regions:
                    if self._point_in_goal(obj.cx, obj.cy, goal):
                        # Rigorous check: only count as scored if ball is moving DOWNWARD
                        # This prevents rim bounces (where ball enters goal region sideways/upward)
                        # from being counted as scored
                        if obj.vy >= self.min_downward_vy:
                            shot.resolve("scored", frame_num, goal["name"])
                            if self._on_shot_resolved:
                                self._on_shot_resolved(shot, frame_num)
                            self.scored_ball_ids.add(obj_id)
                            ids_to_remove.append(obj_id)
                            del self.active_shots[obj_id]
                            break
                        # Ball in goal region but not moving downward - might be bouncing off rim
                        # Don't resolve yet, let it continue tracking

                # Check if shot has been in flight too long (timeout)
                if (obj_id in self.active_shots and
                    frame_num - shot.launch_frame > 90):  # ~3 sec at 30fps
                    shot.resolve("missed", frame_num)
                    if self._on_shot_resolved:
                        self._on_shot_resolved(shot, frame_num)
                    del self.active_shots[obj_id]

                # Update previous position
                self.ball_prev_positions[obj_id] = (obj.cx, obj.cy)
                continue

            # ---- New shot detection ----
            # Get velocity (smoothed for high fps, instantaneous otherwise)
            if self.use_smoothed_velocity and hasattr(obj, 'get_smoothed_velocity'):
                _, vy, speed = obj.get_smoothed_velocity(self.velocity_smoothing_window)
            else:
                vy, speed = obj.vy, obj.speed

            # Is this ball moving upward and fast?
            if vy < self.min_upward_vy and speed > self.min_speed:
                # Count consecutive upward frames
                if obj_id not in self.ball_launch_candidates:
                    self.ball_launch_candidates[obj_id] = 1
                    # Store the origin position when upward flight FIRST starts
                    # This is where the ball was when it began moving upward
                    self.ball_launch_origins[obj_id] = (obj.cx, obj.cy, frame_num)
                else:
                    self.ball_launch_candidates[obj_id] += 1

                # Enough consecutive upward frames = confirmed shot candidate
                if self.ball_launch_candidates[obj_id] >= self.min_flight_frames:
                    # Get the origin position where upward flight started
                    origin_x, origin_y, origin_frame = self.ball_launch_origins.get(
                        obj_id, (obj.cx, obj.cy, frame_num)
                    )

                    # Trajectory fitting for robust shot validation
                    traj_fit = None
                    if hasattr(obj, 'trail') and len(obj.trail) >= 4:
                        traj_fit = self._fit_trajectory(obj.trail)

                    # Launch point extrapolation for better robot attribution
                    extrap_point = None
                    if traj_fit and self.enable_launch_extrapolation:
                        extrap_point = self._extrapolate_launch_point(obj.trail, traj_fit)

                    # Use extrapolated launch point if available and robot detector exists
                    attr_x = extrap_point[0] if extrap_point else origin_x
                    attr_y = extrap_point[1] if extrap_point else origin_y

                    # Determine robot attribution based on ORIGIN position (not current)
                    robot_name = None
                    should_create_shot = True
                    attribution_method = "unknown"
                    robot_candidates = []

                    if self.require_bbox_exit and self.robot_detector is not None:
                        # STRICT MODE: Attribution based on where the ball STARTED its flight
                        # Try extrapolated point first, then original origin
                        inside, origin_robot = self.robot_detector.point_in_robot_bbox(
                            attr_x, attr_y
                        )
                        if not inside and extrap_point:
                            # Try original origin if extrapolation didn't help
                            inside, origin_robot = self.robot_detector.point_in_robot_bbox(
                                origin_x, origin_y
                            )
                        if inside:
                            robot_name = origin_robot
                            attribution_method = "bbox_exit"
                        else:
                            robot_name = self._check_hp_zones(attr_x, attr_y)
                            if robot_name is not None:
                                attribution_method = "hp_zone"
                            else:
                                robot_name = "unknown"
                                attribution_method = "unattributed"
                    else:
                        # RELAXED MODE: Use nearest robot fallback
                        robot_name = self._find_nearest_robot(obj)
                        attribution_method = "nearest_robot"

                    if should_create_shot:
                        # Classify this ball movement (shot vs field_pass vs ignored)
                        classification = self._classify_shot(
                            origin_x, origin_y, obj.vx, obj.vy, speed
                        )

                        # Skip if classified as ignored (slow bounces, ejector dribbles)
                        if classification == "ignored":
                            del self.ball_launch_candidates[obj_id]
                            if obj_id in self.ball_launch_origins:
                                del self.ball_launch_origins[obj_id]
                            continue

                        # Get track confidence if available
                        track_confidence = getattr(obj, 'confidence', None)

                        # Build audit trail
                        audit = {
                            "attribution_method": attribution_method,
                            "track_confidence": track_confidence,
                            "extrapolated_launch": extrap_point is not None,
                        }

                        # Use origin position for the shot record
                        shot = ShotEvent(
                            shot_id=len(self.shots),
                            obj_id=obj_id,
                            launch_x=origin_x,
                            launch_y=origin_y,
                            launch_frame=origin_frame,
                            robot_name=robot_name,
                            classification=classification,
                            track_confidence=track_confidence,
                            trajectory_fit=traj_fit,
                            audit=audit,
                        )
                        self.shots.append(shot)
                        self.active_shots[obj_id] = shot

                        # Fire callback for real-time streaming
                        if self._on_shot_launched:
                            self._on_shot_launched(shot, frame_num)

                        # Tag the tracked object
                        obj.shot_id = shot.shot_id
                        obj.robot_id = robot_name
                        obj.classification = classification

                        # Clean up candidate tracking
                        del self.ball_launch_candidates[obj_id]
                        if obj_id in self.ball_launch_origins:
                            del self.ball_launch_origins[obj_id]
            else:
                # Reset candidate counter if ball stops going up
                if obj_id in self.ball_launch_candidates:
                    del self.ball_launch_candidates[obj_id]
                if obj_id in self.ball_launch_origins:
                    del self.ball_launch_origins[obj_id]

            # Update previous position
            self.ball_prev_positions[obj_id] = (obj.cx, obj.cy)

        return ids_to_remove

    def _point_in_goal(self, px, py, goal):
        """
        Check if point is inside goal region.
        Supports both polygon and rectangle formats for backwards compatibility.
        """
        # Check for polygon format first
        if "polygon" in goal and goal["polygon"]:
            return point_in_polygon(px, py, goal["polygon"])
        # Fall back to rectangle format
        return point_in_rect(px, py, goal)

    def _find_nearest_robot(self, obj):
        """
        Find which robot/human player the ball is closest to at launch time.

        Priority order:
        1. Human player zones (if enabled and ball is inside)
        2. Dynamic RobotDetector (if available)
        3. Static robot zones from config

        Note: Attribution is locked at launch time (occlusion-safe).
        """
        # Check human player zones first (highest priority)
        if self.hp_zones_enabled and self.hp_zones:
            for zone in self.hp_zones:
                if self._point_in_hp_zone(obj.cx, obj.cy, zone):
                    return zone.get("name", "human_player")

        # Try dynamic robot detection
        if self.robot_detector is not None:
            robot_id = self.robot_detector.get_robot_at_position(
                obj.cx, obj.cy, max_distance=self.proximity
            )
            if robot_id != "unknown":
                return robot_id

        # Fall back to static robot zones
        if not self.robot_zones:
            return "unknown"

        min_dist = float("inf")
        nearest = "unknown"

        for robot in self.robot_zones:
            # Center of robot zone
            rx = robot["x"] + robot["w"] / 2
            ry = robot["y"] + robot["h"] / 2
            dist = ((obj.cx - rx)**2 + (obj.cy - ry)**2)**0.5

            if dist < min_dist and dist < self.proximity:
                min_dist = dist
                nearest = robot.get("name", "unknown")

        return nearest

    def _point_in_hp_zone(self, px, py, zone):
        """
        Check if point is inside a human player zone.
        Supports both polygon and rectangle formats.
        """
        # Check for polygon format first
        if "polygon" in zone and zone["polygon"]:
            return point_in_polygon(px, py, zone["polygon"])
        # Fall back to rectangle format
        if "x" in zone:
            return point_in_rect(px, py, zone)
        return False

    def _check_hp_zones(self, x, y):
        """
        Check if a point is inside any human player zone.

        Args:
            x, y: Point to check

        Returns:
            str or None: Zone name if inside an HP zone, None otherwise
        """
        if not self.hp_zones_enabled or not self.hp_zones:
            return None

        for zone in self.hp_zones:
            if self._point_in_hp_zone(x, y, zone):
                return zone.get("name", "human_player")

        return None

    def _fit_trajectory(self, trail):
        """
        Fit a parabolic trajectory to a sequence of (x, y) positions.

        Fits y = a*t² + b*t + c and x = d*t + e using least squares.
        A valid shot trajectory has:
        - Negative initial vertical velocity (b < 0, upward)
        - Positive curvature (a > 0, parabolic arc — gravity pulls down)
        - Good R² fit (actually following a parabolic path, not random noise)

        Args:
            trail: deque or list of (x, y) tuples (most recent last)

        Returns:
            dict with trajectory fit data, or None if insufficient data:
            {
                "a": float,              # quadratic coefficient (curvature)
                "b": float,              # linear coefficient (initial velocity)
                "c": float,              # offset
                "r_squared": float,      # goodness of fit (0-1)
                "launch_vy": float,      # estimated launch vertical velocity
                "launch_vx": float,      # estimated launch horizontal velocity
                "launch_speed": float,   # speed at launch
                "launch_angle": float,   # angle in degrees from horizontal
                "predicted_apex_y": float,  # predicted highest point
            }
        """
        positions = list(trail)
        n = min(len(positions), self.trajectory_fit_window)
        if n < 4:  # Need at least 4 points for a meaningful parabolic fit
            return None

        recent = positions[-n:]
        t = np.arange(n, dtype=np.float64)
        y_vals = np.array([p[1] for p in recent], dtype=np.float64)
        x_vals = np.array([p[0] for p in recent], dtype=np.float64)

        # Fit y = a*t² + b*t + c (parabolic trajectory in image coords)
        try:
            coeffs = np.polyfit(t, y_vals, 2)
            a, b, c = coeffs

            # R² for y-axis fit
            y_pred = np.polyval(coeffs, t)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Linear fit for x-axis: x = d*t + e
            x_coeffs = np.polyfit(t, x_vals, 1)
            launch_vx = float(x_coeffs[0])

            launch_vy = float(b)  # dy/dt at t=0
            launch_speed = math.sqrt(launch_vx ** 2 + launch_vy ** 2)
            launch_angle = math.degrees(math.atan2(-launch_vy, abs(launch_vx)))  # Negative vy = upward

            # Predicted apex (where dy/dt = 0): t_apex = -b / (2a)
            predicted_apex_y = float(c)
            if a > 0:  # Parabola opens downward in real world (up in image)
                t_apex = -b / (2 * a)
                if t_apex > 0:
                    predicted_apex_y = float(a * t_apex ** 2 + b * t_apex + c)

            return {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "r_squared": float(r_squared),
                "launch_vy": launch_vy,
                "launch_vx": launch_vx,
                "launch_speed": launch_speed,
                "launch_angle": launch_angle,
                "predicted_apex_y": predicted_apex_y,
            }
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _extrapolate_launch_point(self, trail, traj_fit):
        """
        Extrapolate backward along fitted trajectory to estimate launch point.

        This helps when the ball's first detected position was already in flight,
        outside the robot's bounding box. By extrapolating backward, we can
        estimate where the ball was when it was actually launched.

        Args:
            trail: Position history
            traj_fit: Dict from _fit_trajectory()

        Returns:
            (x, y) estimated launch point, or None
        """
        if traj_fit is None or not self.enable_launch_extrapolation:
            return None

        positions = list(trail)
        if len(positions) < 2:
            return None

        # Extrapolate backward 2-3 frames from the start of the fit window
        n = min(len(positions), self.trajectory_fit_window)
        start_x = positions[-n][0]
        start_y = positions[-n][1]

        # Use fitted velocities to go backward
        extrap_frames = 3
        launch_x = start_x - traj_fit["launch_vx"] * extrap_frames
        launch_y = start_y - traj_fit["launch_vy"] * extrap_frames

        return (launch_x, launch_y)

    def _classify_shot(self, launch_x, launch_y, vx, vy, speed):
        """
        Classify a detected ball movement into shot, field_pass, or ignored.

        Three-tier classification:
        - Below min_shot_speed: IGNORED (slow bounces, ejector dribbles)
        - Above min_shot_speed + trajectory near goal: SHOT
        - Above min_shot_speed + NOT near goal: FIELD_PASS

        Args:
            launch_x, launch_y: Launch position
            vx, vy: Velocity components
            speed: Ball speed

        Returns:
            str: "shot", "field_pass", or "ignored"
        """
        # Below shot threshold = ignore (bounces, ejector dribbles)
        if speed < self.min_shot_speed:
            return "ignored"

        # If classification is disabled, everything above threshold is a shot
        if not self.classify_field_passes:
            return "shot"

        # Check if trajectory passes near any goal
        if self._trajectory_near_goal(launch_x, launch_y, vx, vy):
            return "shot"

        return "field_pass"

    def _trajectory_near_goal(self, launch_x, launch_y, vx, vy):
        """
        Check if ball trajectory passes near any goal region.

        Projects the ball's trajectory as a ray from launch point and checks
        if it intersects an expanded rectangle around each goal region.

        Args:
            launch_x, launch_y: Launch position
            vx, vy: Velocity components

        Returns:
            bool: True if trajectory passes near a goal
        """
        if not self.goal_regions:
            return False

        for goal in self.goal_regions:
            # Get goal bounds (handle polygon or rect format)
            gx1, gy1, gx2, gy2 = self._get_goal_bounds(goal)

            # Expand bounds by proximity thresholds
            gx1 -= self.goal_proximity_x
            gx2 += self.goal_proximity_x
            gy1 -= self.goal_proximity_y
            gy2 += self.goal_proximity_y

            # Project trajectory - check if it intersects expanded region
            if self._trajectory_intersects_rect(launch_x, launch_y, vx, vy,
                                                 gx1, gy1, gx2, gy2):
                return True

        return False

    def _get_goal_bounds(self, goal):
        """
        Get bounding box for a goal region.

        Handles both polygon and rectangle formats.

        Args:
            goal: Goal dict with either 'polygon' or 'x,y,w,h' keys

        Returns:
            tuple: (x1, y1, x2, y2) bounding box
        """
        if "polygon" in goal and goal["polygon"]:
            # Get bounding box of polygon
            points = goal["polygon"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))
        else:
            # Rectangle format
            return (goal["x"], goal["y"],
                    goal["x"] + goal["w"], goal["y"] + goal["h"])

    def _trajectory_intersects_rect(self, x, y, vx, vy, rx1, ry1, rx2, ry2):
        """
        Check if a ray from (x,y) with direction (vx,vy) intersects a rectangle.

        Uses parametric ray intersection with axis-aligned bounding box.
        Only checks forward direction (positive t values).

        Args:
            x, y: Ray origin
            vx, vy: Ray direction (velocity)
            rx1, ry1, rx2, ry2: Rectangle bounds

        Returns:
            bool: True if ray intersects rectangle
        """
        # Handle near-zero velocities to avoid division issues
        epsilon = 1e-6

        # Check if already inside the rectangle
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            return True

        # For each axis, compute t values at which ray intersects the slab
        if abs(vx) < epsilon:
            # Ray is parallel to Y axis
            if not (rx1 <= x <= rx2):
                return False
            tx_min, tx_max = float('-inf'), float('inf')
        else:
            tx1 = (rx1 - x) / vx
            tx2 = (rx2 - x) / vx
            tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)

        if abs(vy) < epsilon:
            # Ray is parallel to X axis
            if not (ry1 <= y <= ry2):
                return False
            ty_min, ty_max = float('-inf'), float('inf')
        else:
            ty1 = (ry1 - y) / vy
            ty2 = (ry2 - y) / vy
            ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)

        # Find overlap of t ranges
        t_enter = max(tx_min, ty_min)
        t_exit = min(tx_max, ty_max)

        # Check if there's a valid intersection in the forward direction
        # t_enter <= t_exit means ranges overlap
        # t_exit >= 0 means intersection is in forward direction
        return t_enter <= t_exit and t_exit >= 0

    def get_stats(self):
        """Return summary statistics."""
        # Separate shots vs field passes
        shots_only = [s for s in self.shots if s.classification == "shot"]
        field_passes = [s for s in self.shots if s.classification == "field_pass"]

        total = len(shots_only)
        scored = sum(1 for s in shots_only if s.result == "scored")
        missed = sum(1 for s in shots_only if s.result == "missed")
        in_flight = sum(1 for s in self.active_shots.values() if s.classification == "shot")
        unresolved = total - scored - missed

        # By-robot breakdown (shots only, not field passes)
        by_robot = defaultdict(lambda: {"scored": 0, "missed": 0, "total": 0})
        for shot in shots_only:
            r = shot.robot_name
            by_robot[r]["total"] += 1
            if shot.result == "scored":
                by_robot[r]["scored"] += 1
            elif shot.result == "missed":
                by_robot[r]["missed"] += 1

        return {
            "shots_total": total,
            "shots_scored": scored,
            "shots_missed": missed,
            "shots_in_flight": in_flight,
            "shots_unresolved": unresolved,
            "field_passes": len(field_passes),
            "by_robot": dict(by_robot),
        }

    def export_csv(self, path="shot_log.csv"):
        """Export shot log to CSV with enhanced trajectory and audit data."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "shot_id", "classification", "robot", "launch_frame", "launch_x",
                "launch_y", "result", "result_frame", "goal_name",
                "launch_speed", "launch_angle", "trajectory_r2",
                "track_confidence", "attribution_method",
            ])
            for shot in self.shots:
                writer.writerow([
                    shot.shot_id, shot.classification, shot.robot_name, shot.launch_frame,
                    f"{shot.launch_x:.0f}", f"{shot.launch_y:.0f}",
                    shot.result or "in_flight", shot.result_frame or "",
                    shot.goal_name or "",
                    f"{shot.launch_speed:.1f}" if shot.launch_speed else "",
                    f"{shot.launch_angle:.1f}" if shot.launch_angle else "",
                    f"{shot.trajectory_r_squared:.3f}" if shot.trajectory_r_squared else "",
                    f"{shot.track_confidence:.2f}" if shot.track_confidence else "",
                    shot.audit.get("attribution_method", "") if shot.audit else "",
                ])
        print(f"[CSV] Exported {len(self.shots)} events to {path} "
              f"({sum(1 for s in self.shots if s.classification == 'shot')} shots, "
              f"{sum(1 for s in self.shots if s.classification == 'field_pass')} field passes)")


class ShotEvent:
    """
    A single shot event.

    OCCLUSION-SAFE ATTRIBUTION:
        The robot_name is captured at shot launch time and is never re-evaluated.
        This ensures that if the ball passes through another robot's zone
        after being launched, it remains correctly attributed to the robot
        that actually made the shot.

    CLASSIFICATION:
        - "shot": Ball aimed at a goal region (green tracer)
        - "field_pass": Ball thrown fast but NOT aimed at goal (cyan tracer)
        - "ignored": Slow bounces, ejector dribbles (no tracer)
    """

    def __init__(self, shot_id, obj_id, launch_x, launch_y, launch_frame,
                 robot_name="unknown", classification="shot",
                 track_confidence=None, trajectory_fit=None, audit=None):
        self.shot_id = shot_id
        self.obj_id = obj_id
        self.launch_x = launch_x
        self.launch_y = launch_y
        self.launch_frame = launch_frame
        self.robot_name = robot_name
        self.classification = classification  # "shot", "field_pass", or "ignored"

        self.positions = [(launch_x, launch_y, launch_frame)]
        self.result = None       # 'scored' or 'missed'
        self.result_frame = None
        self.goal_name = None
        self.resolved = False

        # Enhanced trajectory data
        self.track_confidence = track_confidence
        self.trajectory_fit = trajectory_fit  # dict from _fit_trajectory()
        self.launch_speed = trajectory_fit["launch_speed"] if trajectory_fit else None
        self.launch_angle = trajectory_fit["launch_angle"] if trajectory_fit else None
        self.trajectory_r_squared = trajectory_fit["r_squared"] if trajectory_fit else None

        # Attribution audit trail
        self.audit = audit or {}

    def update_position(self, x, y, frame_num):
        self.positions.append((x, y, frame_num))

    def resolve(self, result, frame_num, goal_name=None):
        self.result = result
        self.result_frame = frame_num
        self.goal_name = goal_name
        self.resolved = True


# ============================================================================
# INTERACTIVE ZONE SELECTION
# ============================================================================

def select_zones_on_frame(frame, zone_type="goal", existing=None):
    """Select rectangular zones on a frame. Returns list of zone dicts."""
    zones = list(existing or [])
    win_name = f"Select {zone_type} zones (ENTER=add, q=done)"

    while True:
        display = frame.copy()
        for i, z in enumerate(zones):
            color = (0, 255, 0) if zone_type == "goal" else (255, 165, 0)
            cv2.rectangle(display, (z["x"], z["y"]),
                          (z["x"]+z["w"], z["y"]+z["h"]), color, 2)
            cv2.putText(display, z.get("name", f"{zone_type}_{i}"),
                        (z["x"]+5, z["y"]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        r = cv2.selectROI(win_name, display, fromCenter=False)

        if r[2] == 0 or r[3] == 0:
            break

        name = input(f"  Name for this {zone_type} zone "
                     f"(or ENTER for '{zone_type}_{len(zones)}'): ").strip()
        if not name:
            name = f"{zone_type}_{len(zones)}"

        zone = {
            "name": name,
            "x": int(r[0]), "y": int(r[1]),
            "w": int(r[2]), "h": int(r[3]),
        }
        if zone_type == "robot":
            zone["color"] = [255, 165, 0]  # Orange default

        zones.append(zone)
        print(f"  Added: {zone}")

    cv2.destroyWindow(win_name)
    return zones


def select_polygon_zone(frame, zone_type="goal", existing=None):
    """
    Interactive polygon selection for goal regions.
    Click points to define vertices, press ENTER to finish, ESC to cancel.

    Args:
        frame: Frame to draw on
        zone_type: Type of zone (for labeling)
        existing: Existing polygon zones to display

    Returns:
        List of zone dicts with 'name' and 'polygon' keys
    """
    zones = list(existing or [])
    win_name = f"Select {zone_type} polygon (click vertices, ENTER=finish, ESC=cancel, 'q'=done)"

    while True:
        points = []
        display = frame.copy()

        # Draw existing zones
        for i, z in enumerate(zones):
            if "polygon" in z and z["polygon"]:
                display = draw_polygon(display, z["polygon"], (0, 255, 0), 2,
                                        label=z.get("name", f"{zone_type}_{i}"))
            elif "x" in z:  # Legacy rectangle format
                cv2.rectangle(display, (z["x"], z["y"]),
                              (z["x"]+z["w"], z["y"]+z["h"]), (0, 255, 0), 2)
                cv2.putText(display, z.get("name", f"{zone_type}_{i}"),
                            (z["x"]+5, z["y"]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        base_display = display.copy()

        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                display = base_display.copy()

                # Draw current polygon progress
                for i, pt in enumerate(points):
                    cv2.circle(display, tuple(pt), 5, (0, 255, 255), -1)
                    if i > 0:
                        cv2.line(display, tuple(points[i-1]), tuple(pt), (0, 255, 255), 2)

                # Show closing line if we have enough points
                if len(points) > 2:
                    cv2.line(display, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 1)

            elif event == cv2.EVENT_MOUSEMOVE and len(points) > 0:
                # Preview line to cursor
                preview = display.copy()
                cv2.line(preview, tuple(points[-1]), (x, y), (128, 128, 255), 1)
                if len(points) > 1:
                    cv2.line(preview, (x, y), tuple(points[0]), (128, 128, 255), 1)
                cv2.imshow(win_name, preview)
                return

        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback)

        print(f"\n  Click points to define {zone_type} polygon.")
        print("  ENTER = finish current polygon, ESC = cancel, 'q' = done adding zones")

        while True:
            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(points) >= 3:  # ENTER - finish polygon
                name = input(f"  Name for this {zone_type} zone "
                             f"(or ENTER for '{zone_type}_{len(zones)}'): ").strip()
                if not name:
                    name = f"{zone_type}_{len(zones)}"

                zone = {
                    "name": name,
                    "polygon": points,
                }
                zones.append(zone)
                print(f"  Added polygon: {name} with {len(points)} vertices")
                break

            elif key == 27:  # ESC - cancel current polygon
                print("  Cancelled current polygon")
                break

            elif key == ord('q'):  # Done adding zones
                cv2.destroyWindow(win_name)
                return zones

    cv2.destroyWindow(win_name)
    return zones


# ============================================================================
# DRAWING
# ============================================================================

def draw_shots(frame, shot_detector, tracked_objects, config):
    """Draw shot trails with color coding and zone overlays."""
    annotated = draw_zones(frame, config)

    for obj_id, obj in tracked_objects.items():
        if obj.disappeared > 0:
            continue

        cx, cy = int(obj.cx), int(obj.cy)

        # Check if this ball is part of an active or resolved shot
        shot = None
        for s in shot_detector.shots:
            if s.obj_id == obj_id:
                shot = s
                break
        if obj_id in shot_detector.active_shots:
            shot = shot_detector.active_shots[obj_id]

        # Color based on shot status and classification
        is_field_pass = shot and shot.classification == "field_pass"

        if is_field_pass:
            color = (255, 255, 0)     # Cyan = field pass
        elif shot and shot.result == "scored":
            color = (0, 255, 0)       # Green = scored
        elif shot and shot.result == "missed":
            color = (0, 0, 255)       # Red = missed
        elif shot and not shot.resolved:
            color = (0, 165, 255)     # Orange = in flight
        elif obj.is_moving:
            color = (0, 255, 255)     # Yellow = moving, not a shot
        else:
            color = (80, 80, 80)      # Gray = stationary

        # Trail
        if len(obj.trail) > 1:
            pts = list(obj.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                tc = tuple(int(c * alpha) for c in color)
                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(annotated, pt1, pt2, tc, 2)

        # Ball
        cv2.circle(annotated, (cx, cy), int(obj.radius), color, 2)

        # Draw origin marker (star) for "unknown" attributed shots
        if shot and shot.robot_name == "unknown" and not is_field_pass:
            origin_x, origin_y = int(shot.launch_x), int(shot.launch_y)
            cv2.drawMarker(annotated, (origin_x, origin_y),
                           (0, 255, 255),  # Yellow marker
                           cv2.MARKER_STAR, 15, 2)

        # Label
        if shot:
            if is_field_pass:
                label = f"P{shot.shot_id}"  # P for pass
            else:
                label = f"S{shot.shot_id}"
            if shot.robot_name != "unknown":
                label += f":{shot.robot_name}"
            if shot.result and not is_field_pass:
                label += f" [{shot.result.upper()}]"
            cv2.putText(annotated, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return annotated


# ============================================================================
# MAIN
# ============================================================================

def main():
    global VIDEO_PATH

    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    config = load_config()
    cap, vid_info = open_video(VIDEO_PATH)
    detector = BallDetector(config)

    tracker = create_ball_tracker(config)

    # Optional: dynamic robot tracking
    robot_detector = None
    robot_cfg = config.get("robot_detection", {})
    if robot_cfg.get("enabled", False):
        robot_detector = RobotDetector(config)
        print("[ROBOTS] Dynamic robot tracking enabled")

    shot_detector = ShotDetector(config, robot_detector=robot_detector)

    # --- Initial zone setup ---
    print("\n[ZONES] Set up goal regions and robot zones.")
    print("  Press 'g' during playback to define goal regions.")
    print("  Press 'z' during playback to define robot zones.")
    print("  Or define them now? (y/n): ", end="")

    if input().strip().lower() == 'y':
        ret, first_frame = cap.read()
        if ret:
            roi_frame = apply_roi(first_frame, config["roi"])

            print("\n  --- Goal Regions ---")
            goals = select_zones_on_frame(
                roi_frame, "goal",
                config.get("goal_regions", {}).get("regions", [])
            )
            config["goal_regions"]["regions"] = goals

            print("\n  --- Robot Zones ---")
            robots = select_zones_on_frame(
                roi_frame, "robot",
                config.get("robot_zones", {}).get("robots", [])
            )
            config["robot_zones"]["robots"] = robots

            save_config(config)
            shot_detector = ShotDetector(config, robot_detector=robot_detector)  # Re-init with new zones

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Main loop ---
    win_name = "Shot Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    playing = True
    playback_delay = int(1000 / vid_info["fps"])
    frame_num = 0

    print("\nControls: SPACE=play/pause  g=rect goals  G=poly goals  z=robots")
    print("          h=rect HP zone  H=poly HP zone  s=save  e=export CSV  q=quit\n")

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
        else:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                playing = True
                continue
            elif key == ord('n'):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
            elif key == ord('b'):
                frame_num = max(0, frame_num - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
            elif key == ord('g'):
                roi_frame = apply_roi(frame, config["roi"])
                goals = select_zones_on_frame(
                    roi_frame, "goal",
                    config.get("goal_regions", {}).get("regions", [])
                )
                config["goal_regions"]["regions"] = goals
                shot_detector = ShotDetector(config, robot_detector=robot_detector)
                continue
            elif key == ord('G'):
                roi_frame = apply_roi(frame, config["roi"])
                goals = select_polygon_zone(
                    roi_frame, "goal",
                    config.get("goal_regions", {}).get("regions", [])
                )
                config["goal_regions"]["regions"] = goals
                shot_detector = ShotDetector(config, robot_detector=robot_detector)
                continue
            elif key == ord('z'):
                roi_frame = apply_roi(frame, config["roi"])
                robots = select_zones_on_frame(
                    roi_frame, "robot",
                    config.get("robot_zones", {}).get("robots", [])
                )
                config["robot_zones"]["robots"] = robots
                shot_detector = ShotDetector(config, robot_detector=robot_detector)
                continue
            elif key == ord('h'):
                # Rectangle human player zone
                roi_frame = apply_roi(frame, config["roi"])
                hp_zones = select_zones_on_frame(
                    roi_frame, "human_player",
                    config.get("human_player_zones", {}).get("zones", [])
                )
                config.setdefault("human_player_zones", {})["zones"] = hp_zones
                config["human_player_zones"]["enabled"] = True
                shot_detector = ShotDetector(config, robot_detector=robot_detector)
                continue
            elif key == ord('H'):
                # Polygon human player zone
                roi_frame = apply_roi(frame, config["roi"])
                hp_zones = select_polygon_zone(
                    roi_frame, "human_player",
                    config.get("human_player_zones", {}).get("zones", [])
                )
                config.setdefault("human_player_zones", {})["zones"] = hp_zones
                config["human_player_zones"]["enabled"] = True
                shot_detector = ShotDetector(config, robot_detector=robot_detector)
                continue
            elif key == ord('s'):
                save_config(config)
                continue
            elif key == ord('e'):
                shot_detector.export_csv()
                continue
            else:
                continue

        # Process
        roi_frame = apply_roi(frame, config["roi"])
        detections = detector.detect(roi_frame)
        objects = tracker.update(detections)

        # Update robot tracking if enabled
        if robot_detector is not None:
            robot_detector.detect_and_track(roi_frame)

        ids_to_remove = shot_detector.update(objects, frame_num)

        # Remove balls that entered goals (prevents bounce-out double-counting)
        if ids_to_remove:
            tracker.remove_objects(ids_to_remove)

        # Draw
        display = draw_shots(roi_frame, shot_detector, objects, config)

        # Draw detected robots if enabled
        if robot_detector is not None:
            display = draw_robots(display, robot_detector.get_all_robots())

        # HUD with shot stats
        shot_stats = shot_detector.get_stats()
        stats = {
            "balls_detected": len(detections),
            "balls_tracked": sum(1 for o in objects.values() if o.disappeared == 0),
            "balls_moving": sum(1 for o in objects.values()
                                if o.disappeared == 0 and o.is_moving),
            **shot_stats,
        }
        display = draw_hud(display, stats, frame_num, vid_info["frame_count"])

        cv2.imshow(win_name, display)

        key = cv2.waitKey(playback_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = False
        elif key == ord('s'):
            save_config(config)
        elif key == ord('e'):
            shot_detector.export_csv()
        elif key == ord('+') or key == ord('='):
            playback_delay = max(1, playback_delay - 5)
        elif key == ord('-'):
            playback_delay = min(200, playback_delay + 5)

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    final = shot_detector.get_stats()
    print("\n" + "=" * 60)
    print("  SHOT DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total shots on goal:  {final['shots_total']}")
    print(f"  Scored:               {final['shots_scored']}")
    print(f"  Missed:               {final['shots_missed']}")
    print(f"  Unresolved:           {final['shots_unresolved']}")
    print(f"  Field passes:         {final.get('field_passes', 0)}")
    if final["by_robot"]:
        print(f"\n  By Robot (shots only):")
        for robot, robot_stats in final["by_robot"].items():
            pct = (robot_stats['scored'] / robot_stats['total'] * 100) if robot_stats['total'] > 0 else 0
            print(f"    {robot}: {robot_stats['scored']}/{robot_stats['total']} ({pct:.0f}%)")
    print("=" * 60)

    # Auto-export
    shot_detector.export_csv()


if __name__ == "__main__":
    main()
