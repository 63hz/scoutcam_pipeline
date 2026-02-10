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
    g      - Define goal regions interactively
    z      - Define robot zones interactively
    SPACE  - Play/Pause
    n/p    - Step frames
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
    point_in_rect
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
    """

    def __init__(self, config):
        self.config = config
        shot_cfg = config.get("shot_detection", {})
        self.min_upward_vy = shot_cfg.get("min_upward_velocity", -3.0)
        self.min_speed = shot_cfg.get("min_speed", 5.0)
        self.proximity = shot_cfg.get("proximity_to_robot", 120)
        self.min_flight_frames = shot_cfg.get("min_flight_frames", 4)

        self.goal_regions = config.get("goal_regions", {}).get("regions", [])
        self.robot_zones = config.get("robot_zones", {}).get("robots", [])

        # Active shot tracking
        self.shots = []           # list of ShotEvent
        self.active_shots = {}    # obj_id -> ShotEvent (in-flight)
        self.ball_launch_candidates = {}  # obj_id -> frame count of upward motion

    def update(self, tracked_objects, frame_num):
        """
        Check all tracked objects for shot events.
        Call once per frame after tracker.update().
        """
        for obj_id, obj in tracked_objects.items():
            if obj.disappeared > 0:
                # If this ball was being tracked as a shot and disappeared,
                # it might have landed
                if obj_id in self.active_shots:
                    shot = self.active_shots[obj_id]
                    if not shot.resolved:
                        shot.resolve("missed", frame_num)
                    del self.active_shots[obj_id]
                continue

            # Check if ball is already being tracked as a shot
            if obj_id in self.active_shots:
                shot = self.active_shots[obj_id]
                shot.update_position(obj.cx, obj.cy, frame_num)

                # Check if ball entered goal region
                for goal in self.goal_regions:
                    if point_in_rect(obj.cx, obj.cy, goal):
                        shot.resolve("scored", frame_num, goal["name"])
                        del self.active_shots[obj_id]
                        break

                # Check if shot has been in flight too long (timeout)
                if (obj_id in self.active_shots and
                    frame_num - shot.launch_frame > 90):  # ~3 sec at 30fps
                    shot.resolve("missed", frame_num)
                    del self.active_shots[obj_id]

                continue

            # ---- New shot detection ----
            # Is this ball moving upward and fast?
            if obj.vy < self.min_upward_vy and obj.speed > self.min_speed:
                # Count consecutive upward frames
                if obj_id not in self.ball_launch_candidates:
                    self.ball_launch_candidates[obj_id] = 1
                else:
                    self.ball_launch_candidates[obj_id] += 1

                # Enough consecutive upward frames = confirmed shot
                if self.ball_launch_candidates[obj_id] >= self.min_flight_frames:
                    # Find nearest robot zone for attribution
                    robot_name = self._find_nearest_robot(obj)

                    shot = ShotEvent(
                        shot_id=len(self.shots),
                        obj_id=obj_id,
                        launch_x=obj.cx,
                        launch_y=obj.cy,
                        launch_frame=frame_num,
                        robot_name=robot_name,
                    )
                    self.shots.append(shot)
                    self.active_shots[obj_id] = shot

                    # Tag the tracked object
                    obj.shot_id = shot.shot_id
                    obj.robot_id = robot_name

                    del self.ball_launch_candidates[obj_id]
            else:
                # Reset candidate counter if ball stops going up
                if obj_id in self.ball_launch_candidates:
                    del self.ball_launch_candidates[obj_id]

    def _find_nearest_robot(self, obj):
        """Find which robot zone the ball is closest to."""
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

    def get_stats(self):
        """Return summary statistics."""
        total = len(self.shots)
        scored = sum(1 for s in self.shots if s.result == "scored")
        missed = sum(1 for s in self.shots if s.result == "missed")
        in_flight = len(self.active_shots)
        unresolved = total - scored - missed

        by_robot = defaultdict(lambda: {"scored": 0, "missed": 0, "total": 0})
        for shot in self.shots:
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
            "by_robot": dict(by_robot),
        }

    def export_csv(self, path="shot_log.csv"):
        """Export shot log to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shot_id", "robot", "launch_frame", "launch_x",
                             "launch_y", "result", "result_frame", "goal_name"])
            for shot in self.shots:
                writer.writerow([
                    shot.shot_id, shot.robot_name, shot.launch_frame,
                    f"{shot.launch_x:.0f}", f"{shot.launch_y:.0f}",
                    shot.result or "in_flight", shot.result_frame or "",
                    shot.goal_name or "",
                ])
        print(f"[CSV] Exported {len(self.shots)} shots to {path}")


class ShotEvent:
    """A single shot event."""

    def __init__(self, shot_id, obj_id, launch_x, launch_y, launch_frame,
                 robot_name="unknown"):
        self.shot_id = shot_id
        self.obj_id = obj_id
        self.launch_x = launch_x
        self.launch_y = launch_y
        self.launch_frame = launch_frame
        self.robot_name = robot_name

        self.positions = [(launch_x, launch_y, launch_frame)]
        self.result = None       # 'scored' or 'missed'
        self.result_frame = None
        self.goal_name = None
        self.resolved = False

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

        # Color based on shot status
        if shot and shot.result == "scored":
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

        # Label
        if shot:
            label = f"S{shot.shot_id}"
            if shot.robot_name != "unknown":
                label += f":{shot.robot_name}"
            if shot.result:
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

    track_cfg = config["tracking"]
    tracker = CentroidTracker(
        max_disappeared=track_cfg["max_frames_missing"],
        max_distance=track_cfg["max_distance"],
        trail_length=track_cfg["trail_length"],
    )

    shot_detector = ShotDetector(config)

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
            shot_detector = ShotDetector(config)  # Re-init with new zones

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Main loop ---
    win_name = "Shot Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    playing = True
    playback_delay = int(1000 / vid_info["fps"])
    frame_num = 0

    print("\nControls: SPACE=play/pause  g=goals  z=robots  s=save  "
          "e=export CSV  q=quit\n")

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
            elif key == ord('p'):
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
                shot_detector = ShotDetector(config)
                continue
            elif key == ord('z'):
                roi_frame = apply_roi(frame, config["roi"])
                robots = select_zones_on_frame(
                    roi_frame, "robot",
                    config.get("robot_zones", {}).get("robots", [])
                )
                config["robot_zones"]["robots"] = robots
                shot_detector = ShotDetector(config)
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
        shot_detector.update(objects, frame_num)

        # Draw
        display = draw_shots(roi_frame, shot_detector, objects, config)

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
    print(f"  Total shots detected: {final['shots_total']}")
    print(f"  Scored:               {final['shots_scored']}")
    print(f"  Missed:               {final['shots_missed']}")
    print(f"  Unresolved:           {final['shots_unresolved']}")
    if final["by_robot"]:
        print(f"\n  By Robot:")
        for robot, stats in final["by_robot"].items():
            print(f"    {robot}: {stats['total']} shots "
                  f"({stats['scored']} scored, {stats['missed']} missed)")
    print("=" * 60)

    # Auto-export
    shot_detector.export_csv()


if __name__ == "__main__":
    main()
