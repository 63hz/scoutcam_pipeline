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
    RobotDetector, draw_robots
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


VIDEO_PATH = "C:/Users/Clay/scoutcam_pipeline/kettering1.mkv"  # <-- SET THIS
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

    track_cfg = config["tracking"]
    tracker = CentroidTracker(
        max_disappeared=track_cfg["max_frames_missing"],
        max_distance=track_cfg["max_distance"],
        trail_length=track_cfg["trail_length"],
    )

    # Optional: dynamic robot tracking
    robot_detector = None
    robot_cfg = config.get("robot_detection", {})
    if robot_cfg.get("enabled", False):
        robot_detector = RobotDetector(config)
        print("  [ROBOTS] Dynamic robot tracking enabled")

    shot_detector = ShotDetector(config, robot_detector=robot_detector)

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

        frame_data.append({
            "detections_count": len(detections),
            "object_states": obj_states,
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

    # Build shot lookup: obj_id -> shot result
    shot_results = {}
    for shot in shot_detector.shots:
        shot_results[shot.obj_id] = {
            "shot_id": shot.shot_id,
            "result": shot.result or "unresolved",
            "robot": shot.robot_name,
        }

    # Print stats
    final = shot_detector.get_stats()
    print(f"\n  Shots: {final['shots_total']} total, "
          f"{final['shots_scored']} scored, {final['shots_missed']} missed")

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

        # Draw tracked objects with retroactive shot coloring
        for oid, state in fd["object_states"].items():
            cx, cy = int(state["cx"]), int(state["cy"])
            radius = int(state["radius"])

            # Determine color from shot result (retroactive!)
            if oid in shot_results:
                result = shot_results[oid]["result"]
                if result == "scored":
                    color = (0, 255, 0)       # Green
                elif result == "missed":
                    color = (0, 0, 255)       # Red
                else:
                    color = (0, 165, 255)     # Orange (unresolved)
            elif state["is_moving"]:
                color = (0, 255, 255)         # Yellow (moving, not shot)
            else:
                color = (60, 60, 60)          # Dim gray (stationary)

            # Trail
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

            # Label for shots
            if oid in shot_results:
                sr = shot_results[oid]
                label = f"S{sr['shot_id']}"
                if sr["robot"] != "unknown":
                    label += f":{sr['robot']}"
                label += f" [{sr['result'].upper()}]"
                cv2.putText(annotated, label, (cx + 8, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # HUD
        stats = {
            "balls_detected": fd["detections_count"],
            "balls_tracked": len(fd["object_states"]),
            "balls_moving": sum(1 for s in fd["object_states"].values()
                                if s["is_moving"]),
            **final,
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
    print(f"    Total:      {final['shots_total']}")
    print(f"    Scored:     {final['shots_scored']}")
    print(f"    Missed:     {final['shots_missed']}")
    if final["by_robot"]:
        print(f"\n  By Robot:")
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
    VIDEO_PATH, OUTPUT_PATH

    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    if not OUTPUT_PATH:
        base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(VIDEO_PATH)),
                                    f"{base}_annotated.mp4")

    run_pipeline(VIDEO_PATH, OUTPUT_PATH)
