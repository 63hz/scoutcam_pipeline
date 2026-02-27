# -*- coding: utf-8 -*-
"""
05 - Full Pipeline (Annotated Output Video)
=============================================
Process the entire video through the detection → tracking → shot detection
pipeline and produce an annotated output video.

KEY FEATURE: Two-pass processing for retroactive shot coloring.
    Pass 1: Process all frames, detect all shots, determine outcomes
    Pass 2: Re-render with full knowledge of shot outcomes
            (so the trail color is green/red from the START of the shot,
             not just after it resolves)

This makes it visually obvious "this shot will go in" or "this shot will miss"
from the moment the ball leaves the robot.

OUTPUT:
    - Annotated video file (MP4)
    - Shot log CSV
    - Summary stats to console

USAGE:
    1. Tune everything with scripts 01-04 first
    2. Set VIDEO_PATH below
    3. Run. Go get coffee. Come back to a finished annotated video.

Author: Clay / Claude sandbox
"""

import sys
import os
import time
import json

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, open_video, apply_roi,
    BallDetector, CentroidTracker, draw_hud, draw_zones,
    RobotDetector, YOLORobotDetector, draw_robots,
    create_ball_tracker, TrackingDiagnostics
)

# Import ShotDetector from script 04
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "zones_and_shots",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_zones_and_shots.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ShotDetector = _mod.ShotDetector


VIDEO_PATH = ""  # <-- SET THIS
OUTPUT_PATH = ""  # Leave empty for auto-naming
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]


# ============================================================================
# TWO-PASS PIPELINE
# ============================================================================

def run_pipeline(video_path, output_path=None):
    """
    Two-pass pipeline:
        Pass 1: Detection + Tracking + Shot Detection → build shot database
        Pass 2: Re-render video with retroactive shot coloring
    """
    config = load_config()
    cap, vid_info = open_video(video_path)

    if not output_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path),
                                    f"{base}_annotated.mp4")

    detector = BallDetector(config)

    # ==== PASS 1: Analyze ====
    print("\n" + "=" * 60)
    print("  PASS 1: Analyzing video...")
    print("=" * 60)

    tracker = create_ball_tracker(config)

    # Optional: dynamic robot tracking (YOLO or HSV)
    robot_detector = None
    yolo_cfg = config.get("yolo_robot_detection", {})
    hsv_cfg = config.get("robot_detection", {})

    if yolo_cfg.get("enabled", False):
        try:
            robot_detector = YOLORobotDetector(config)
            print("  [ROBOTS] YOLO-based robot tracking enabled")
        except (ImportError, FileNotFoundError) as e:
            print(f"  [ROBOTS] YOLO failed: {e}")
            print("  [ROBOTS] Falling back to HSV detection...")
            if hsv_cfg.get("enabled", False):
                robot_detector = RobotDetector(config)
                print("  [ROBOTS] HSV robot tracking enabled")
    elif hsv_cfg.get("enabled", False):
        robot_detector = RobotDetector(config)
        print("  [ROBOTS] HSV robot tracking enabled")

    shot_detector = ShotDetector(config, robot_detector=robot_detector, fps=vid_info["fps"])
    diagnostics = TrackingDiagnostics()

    # Store per-frame data for pass 2
    frame_data = []  # list of {detections, object_states}
    total_frames = vid_info["frame_count"]
    t_start = time.time()

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = apply_roi(frame, config["roi"])
        detections = detector.detect(roi_frame)
        objects = tracker.update(detections)

        # Update robot tracking if enabled
        if robot_detector is not None:
            robot_detector.detect_and_track(roi_frame)

        ids_to_remove = shot_detector.update(objects, frame_num)
        diagnostics.update(objects, frame_num)

        # Remove balls that entered goals (prevents bounce-out double-counting)
        if ids_to_remove:
            tracker.remove_objects(ids_to_remove)

        # Save lightweight state for pass 2
        obj_states = {}
        for oid, obj in objects.items():
            if obj.disappeared == 0:
                obj_states[oid] = {
                    "cx": obj.cx, "cy": obj.cy,
                    "radius": obj.radius,
                    "trail": list(obj.trail),
                    "is_moving": obj.is_moving,
                    "vx": obj.vx, "vy": obj.vy,
                    "speed": obj.speed,
                }

        # Save robot states if tracking enabled
        robot_states = {}
        if robot_detector is not None:
            for rid, robot in robot_detector.robots.items():
                if robot.disappeared == 0:
                    robot_states[rid] = {
                        "cx": robot.cx, "cy": robot.cy,
                        "bbox": robot.bbox,
                        "robot_bbox": robot.robot_bbox,  # Expanded bbox for attribution
                        "alliance": robot.alliance,
                        "identity": robot.identity,
                    }

        frame_data.append({
            "detections_count": len(detections),
            "object_states": obj_states,
            "robot_states": robot_states,
        })

        frame_num += 1
        if frame_num % 100 == 0:
            elapsed = time.time() - t_start
            fps = frame_num / elapsed
            eta = (total_frames - frame_num) / fps if fps > 0 else 0
            print(f"  Pass 1: {frame_num}/{total_frames} "
                  f"({fps:.0f} fps, ETA {eta:.0f}s)")

    t_pass1 = time.time() - t_start
    print(f"  Pass 1 complete: {frame_num} frames in {t_pass1:.1f}s "
          f"({frame_num/t_pass1:.0f} fps)")

    # Print tracking quality summary
    diag_summary = diagnostics.get_summary()
    print(f"  Tracking: {diag_summary.get('total_tracks', 0)} tracks, "
          f"avg lifespan {diag_summary.get('avg_lifespan_frames', 0):.0f} frames, "
          f"avg hit rate {diag_summary.get('avg_hit_rate', 0):.1%}")

    # Export track lifecycle log
    track_log_path = os.path.splitext(output_video)[0] + "_track_log.csv"
    diagnostics.export(track_log_path)

    # Print tracker stats if available
    if hasattr(tracker, 'get_stats'):
        tstats = tracker.get_stats()
        print(f"  Tracker stats: {tstats.get('matches', 0)} matches, "
              f"{tstats.get('reidentifications', 0)} re-IDs, "
              f"{tstats.get('dead_tracks', 0)} dead tracks")

    # Build shot lookup: obj_id -> shot result (with launch_frame for proper tracer coloring)
    # Also build a list for computing live stats during Pass 2
    shot_results = {}
    shot_events = []  # List of all shots with timing info for live stats
    for shot in shot_detector.shots:
        shot_info = {
            "shot_id": shot.shot_id,
            "result": shot.result or "unresolved",
            "robot": shot.robot_name,
            "launch_frame": shot.launch_frame,  # Track when shot was detected
            "result_frame": shot.result_frame,  # When shot was resolved (scored/missed)
            "launch_x": shot.launch_x,  # For origin marker
            "launch_y": shot.launch_y,
            "classification": shot.classification,  # "shot" or "field_pass"
        }
        shot_results[shot.obj_id] = shot_info
        shot_events.append(shot_info)

    def compute_live_stats(current_frame):
        """Compute shot stats as of a specific frame (for live HUD updates)."""
        from collections import defaultdict

        # Only count shots that have been launched by current_frame
        active_shots = [s for s in shot_events
                        if s["launch_frame"] <= current_frame and s["classification"] == "shot"]
        active_passes = [s for s in shot_events
                         if s["launch_frame"] <= current_frame and s["classification"] == "field_pass"]

        # Count results only if they've been resolved by current_frame
        scored = sum(1 for s in active_shots
                     if s["result"] == "scored" and s["result_frame"] and s["result_frame"] <= current_frame)
        missed = sum(1 for s in active_shots
                     if s["result"] == "missed" and s["result_frame"] and s["result_frame"] <= current_frame)

        # In-flight = launched but not yet resolved
        in_flight = len(active_shots) - scored - missed

        # By-robot breakdown (shots only)
        by_robot = defaultdict(lambda: {"scored": 0, "missed": 0, "total": 0})
        for shot in active_shots:
            r = shot["robot"]
            # Only count if launched by current frame
            by_robot[r]["total"] += 1
            # Only count result if resolved by current frame
            if shot["result_frame"] and shot["result_frame"] <= current_frame:
                if shot["result"] == "scored":
                    by_robot[r]["scored"] += 1
                elif shot["result"] == "missed":
                    by_robot[r]["missed"] += 1

        return {
            "shots_total": len(active_shots),
            "shots_scored": scored,
            "shots_missed": missed,
            "shots_in_flight": in_flight,
            "field_passes": len(active_passes),
            "by_robot": dict(by_robot),
        }

    # Print stats
    final = shot_detector.get_stats()
    print(f"\n  Shots: {final['shots_total']} total, "
          f"{final['shots_scored']} scored, {final['shots_missed']} missed, "
          f"{final.get('field_passes', 0)} field passes")

    # ==== PASS 2: Render annotated video ====
    print("\n" + "=" * 60)
    print("  PASS 2: Rendering annotated video...")
    print("=" * 60)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get ROI dimensions for output
    ret, test_frame = cap.read()
    roi_test = apply_roi(test_frame, config["roi"])
    out_h, out_w = roi_test.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*config["output"].get("codec", "mp4v"))
    out_fps = vid_info["fps"]
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}")
        return

    t_start = time.time()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= len(frame_data):
            break

        roi_frame = apply_roi(frame, config["roi"])
        fd = frame_data[frame_num]

        # Draw zones
        annotated = draw_zones(roi_frame, config)

        # Draw robots if tracking enabled
        robot_states = fd.get("robot_states", {})
        show_attribution_zone = config.get("output", {}).get("show_attribution_zone", False)
        if robot_states:
            for rid, rs in robot_states.items():
                x, y, w, h = rs["bbox"]
                alliance = rs["alliance"]
                identity = rs["identity"]
                robot_bbox = rs.get("robot_bbox")

                # Color based on alliance
                if alliance == "red":
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                # Draw expanded robot bbox (attribution zone) if enabled
                if show_attribution_zone and robot_bbox is not None:
                    x1, y1, x2, y2 = robot_bbox
                    # Dashed rectangle for attribution zone (lighter color)
                    for i in range(x1, x2, 16):
                        cv2.line(annotated, (i, y1), (min(i+8, x2), y1), color, 1)
                        cv2.line(annotated, (i, y2), (min(i+8, x2), y2), color, 1)
                    for i in range(y1, y2, 16):
                        cv2.line(annotated, (x1, i), (x1, min(i+8, y2)), color, 1)
                        cv2.line(annotated, (x2, i), (x2, min(i+8, y2)), color, 1)

                # Draw bounding box (bumper)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                # Draw label
                label = identity
                cv2.putText(annotated, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Get tracer mode from config (all, shots_only, none)
        tracer_mode = config.get("output", {}).get("tracer_mode", "all")

        # Draw tracked objects with retroactive shot coloring
        for oid, state in fd["object_states"].items():
            cx, cy = int(state["cx"]), int(state["cy"])
            radius = int(state["radius"])

            # Check if this is a shot/field_pass and get launch frame
            is_tracked_event = oid in shot_results
            if is_tracked_event:
                sr = shot_results[oid]
                launch_frame = sr["launch_frame"]
                classification = sr.get("classification", "shot")
                is_shot = classification == "shot"
                is_field_pass = classification == "field_pass"
            else:
                launch_frame = None
                is_shot = False
                is_field_pass = False

            # Apply tracer_mode filtering
            # - "none": skip all tracers
            # - "shots_only": skip non-shot balls (but include field_passes if enabled)
            # - "all": draw everything (default)
            if tracer_mode == "none":
                continue
            if tracer_mode == "shots_only" and not is_tracked_event:
                continue

            # Determine color from shot result (retroactive!)
            # Only apply shot color AFTER launch_frame to prevent pre-launch coloring
            if is_tracked_event and (launch_frame is None or frame_num >= launch_frame):
                result = sr["result"]
                if is_field_pass:
                    # Cyan for field passes
                    color = (255, 255, 0)     # Cyan (BGR)
                elif result == "scored":
                    color = (0, 255, 0)       # Green
                elif result == "missed":
                    color = (0, 0, 255)       # Red
                else:
                    color = (0, 165, 255)     # Orange (unresolved)
            elif state["is_moving"]:
                color = (0, 255, 255)         # Yellow (moving, not shot)
            else:
                color = (60, 60, 60)          # Dim gray (stationary)

            # Trail - only draw from launch_frame onward if this is a shot
            trail = state["trail"]
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    tc = tuple(int(c * alpha) for c in color)
                    pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    cv2.line(annotated, pt1, pt2, tc, 2)

            # Ball
            cv2.circle(annotated, (cx, cy), radius, color, 2)

            # Draw origin marker (star) for "unknown" attributed shots
            if is_tracked_event and sr["robot"] == "unknown" and is_shot:
                origin_x, origin_y = int(sr["launch_x"]), int(sr["launch_y"])
                # Yellow star marker at launch position
                cv2.drawMarker(annotated, (origin_x, origin_y),
                               (0, 255, 255),  # Yellow marker
                               cv2.MARKER_STAR, 15, 2)

            # Label for shots
            if is_tracked_event:
                if is_field_pass:
                    label = f"P{sr['shot_id']}"  # P for pass
                else:
                    label = f"S{sr['shot_id']}"
                if sr["robot"] != "unknown":
                    label += f":{sr['robot']}"
                if not is_field_pass:
                    label += f" [{sr['result'].upper()}]"
                cv2.putText(annotated, label, (cx + 8, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # HUD with LIVE stats (computed as of current frame)
        live_shot_stats = compute_live_stats(frame_num)
        stats = {
            "balls_detected": fd["detections_count"],
            "balls_tracked": len(fd["object_states"]),
            "balls_moving": sum(1 for s in fd["object_states"].values()
                                if s["is_moving"]),
            **live_shot_stats,
        }
        annotated = draw_hud(annotated, stats, frame_num, total_frames)

        writer.write(annotated)
        frame_num += 1

        if frame_num % 100 == 0:
            elapsed = time.time() - t_start
            fps = frame_num / elapsed
            eta = (total_frames - frame_num) / fps if fps > 0 else 0
            print(f"  Pass 2: {frame_num}/{total_frames} "
                  f"({fps:.0f} fps, ETA {eta:.0f}s)")

    writer.release()
    cap.release()

    t_pass2 = time.time() - t_start
    print(f"  Pass 2 complete: {frame_num} frames in {t_pass2:.1f}s")
    print(f"\n  Output video: {output_path}")

    # Export CSV
    csv_path = os.path.splitext(output_path)[0] + "_shots.csv"
    shot_detector.export_csv(csv_path)

    # Final summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input:   {video_path}")
    print(f"  Output:  {output_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Frames:  {frame_num}")
    print(f"  Time:    Pass1={t_pass1:.1f}s  Pass2={t_pass2:.1f}s  "
          f"Total={t_pass1+t_pass2:.1f}s")
    print(f"\n  Shot Summary:")
    print(f"    Shots:      {final['shots_total']}")
    print(f"    Scored:     {final['shots_scored']}")
    print(f"    Missed:     {final['shots_missed']}")
    print(f"    Passes:     {final.get('field_passes', 0)}")
    if final["by_robot"]:
        print(f"\n  By Robot (shots only):")
        for robot, rs in final["by_robot"].items():
            pct = (rs["scored"] / rs["total"] * 100) if rs["total"] > 0 else 0
            print(f"    {robot}: {rs['scored']}/{rs['total']} "
                  f"({pct:.0f}% accuracy)")
    print("=" * 60)

    return shot_detector


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    if not OUTPUT_PATH:
        base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(VIDEO_PATH)),
                                    f"{base}_annotated.mp4")

    run_pipeline(VIDEO_PATH, OUTPUT_PATH)
