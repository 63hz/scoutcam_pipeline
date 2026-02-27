# -*- coding: utf-8 -*-
"""
03 - Tracking Sandbox
======================
Multi-ball tracking with centroid association, trails, and velocity vectors.

This is where we go from "detecting balls in each frame" to "following balls
across frames" — the foundation for shot detection and attribution.

WORKFLOW:
    1. Tune detection with 01_hsv_tuning.py first
    2. Set VIDEO_PATH below
    3. Run to see tracking in action
    4. Adjust tracking params (max_distance, max_frames_missing) as needed

WHAT TO LOOK FOR:
    - Are ball IDs stable? (same ball keeps same ID across frames)
    - Are fast-moving balls being tracked? (increase max_distance if not)
    - Are stationary balls cluttering things? (try motion filtering)
    - Are IDs being re-assigned too quickly? (increase max_frames_missing)
    - Do trails look reasonable for ball trajectories?

CONTROLS:
    SPACE  - Play/Pause
    n/p    - Next/Previous frame
    m      - Toggle motion-only filter (only show moving balls)
    t      - Toggle trail drawing
    v      - Toggle velocity vectors
    i      - Toggle ID labels
    s      - Save current tracking config
    d      - Toggle debug view
    q/ESC  - Quit

Author: Clay / Claude sandbox
"""

import sys
import os
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, save_config, open_video, apply_roi,
    BallDetector, CentroidTracker, draw_hud, create_ball_tracker
)

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_PATH = ""  # <-- SET THIS
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_tracking_frame(frame, tracker_objects, show_trails=True,
                        show_velocity=False, show_ids=True,
                        motion_only=False):
    """
    Custom drawing for tracking sandbox with toggle-able layers.
    """
    annotated = frame.copy()

    for obj_id, obj in tracker_objects.items():
        if obj.disappeared > 0:
            continue

        if motion_only and not obj.is_moving:
            continue

        cx, cy = int(obj.cx), int(obj.cy)

        # Color: moving = bright yellow, stationary = dim gray
        if obj.is_moving:
            color = (0, 255, 255)
        else:
            color = (80, 80, 80)

        # Trail
        if show_trails and len(obj.trail) > 1:
            pts = list(obj.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                tc = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                thickness = max(1, int(2 * alpha))
                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(annotated, pt1, pt2, tc, thickness)

        # Ball circle
        cv2.circle(annotated, (cx, cy), int(obj.radius), color, 2)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)  # center dot

        # Velocity vector
        if show_velocity and obj.is_moving:
            # Scale velocity for visibility
            vx_draw = int(obj.vx * 5)
            vy_draw = int(obj.vy * 5)
            end_pt = (cx + vx_draw, cy + vy_draw)
            cv2.arrowedLine(annotated, (cx, cy), end_pt, (255, 0, 255), 2,
                            tipLength=0.3)

        # ID label
        if show_ids:
            label = f"#{obj.id}"
            if obj.is_moving:
                label += f" v={obj.speed:.0f}"
            cv2.putText(annotated, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

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

    # Display toggles
    show_trails = True
    show_velocity = False
    show_ids = True
    motion_only = False
    debug_mode = False
    playing = True
    playback_delay = int(1000 / vid_info["fps"])
    frame_num = 0

    # Stats
    max_simultaneous = 0
    total_ids_created = 0

    win_name = "Ball Tracking"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Tracking config trackbars
    win_ctrl = "Tracking Controls"
    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_ctrl, 400, 150)
    cv2.createTrackbar("Max Dist", win_ctrl, track_cfg["max_distance"], 200,
                       lambda x: None)
    cv2.createTrackbar("Max Missing", win_ctrl, track_cfg["max_frames_missing"], 30,
                       lambda x: None)
    cv2.createTrackbar("Trail Len", win_ctrl, track_cfg["trail_length"], 100,
                       lambda x: None)

    print("\nControls: SPACE=play/pause  m=motion-only  t=trails  v=velocity  "
          "i=ids  s=save  q=quit\n")

    while True:
        # Read tracking params from trackbars
        new_max_dist = cv2.getTrackbarPos("Max Dist", win_ctrl)
        new_max_missing = cv2.getTrackbarPos("Max Missing", win_ctrl)
        new_trail_len = cv2.getTrackbarPos("Trail Len", win_ctrl)

        if new_max_dist != tracker.max_distance:
            tracker.max_distance = max(1, new_max_dist)
        if new_max_missing != tracker.max_disappeared:
            tracker.max_disappeared = max(1, new_max_missing)
        # trail_length can't be changed on existing deques easily,
        # but new objects will use the new length
        tracker.trail_length = max(2, new_trail_len)

        # Frame reading
        if playing:
            ret, frame = cap.read()
            if not ret:
                print(f"\n[DONE] Processed {frame_num} frames")
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
            elif key == ord('m'):
                motion_only = not motion_only
                print(f"[FILTER] Motion only: {motion_only}")
                continue
            elif key == ord('t'):
                show_trails = not show_trails
                continue
            elif key == ord('v'):
                show_velocity = not show_velocity
                continue
            elif key == ord('i'):
                show_ids = not show_ids
                continue
            elif key == ord('s'):
                config["tracking"]["max_distance"] = tracker.max_distance
                config["tracking"]["max_frames_missing"] = tracker.max_disappeared
                config["tracking"]["trail_length"] = tracker.trail_length
                save_config(config)
                continue
            elif key == ord('d'):
                debug_mode = not debug_mode
                continue
            else:
                continue

        # Process
        t_start = time.time()
        roi_frame = apply_roi(frame, config["roi"])
        detections = detector.detect(roi_frame)
        objects = tracker.update(detections)
        t_elapsed = time.time() - t_start

        # Stats
        active_count = sum(1 for o in objects.values() if o.disappeared == 0)
        moving_count = sum(1 for o in objects.values()
                          if o.disappeared == 0 and o.is_moving)
        max_simultaneous = max(max_simultaneous, active_count)
        total_ids_created = tracker.next_id

        # Draw
        if debug_mode:
            mask = detector.get_mask(roi_frame)
            from frc_tracker_utils import create_debug_view
            display = create_debug_view(roi_frame, mask, detections)
        else:
            display = draw_tracking_frame(
                roi_frame, objects,
                show_trails=show_trails,
                show_velocity=show_velocity,
                show_ids=show_ids,
                motion_only=motion_only,
            )

        # HUD
        stats = {
            "balls_detected": len(detections),
            "balls_tracked": active_count,
            "balls_moving": moving_count,
        }
        display = draw_hud(display, stats, frame_num, vid_info["frame_count"])

        # Extra stats line
        perf_text = (f"{t_elapsed*1000:.1f}ms | "
                     f"IDs created: {total_ids_created} | "
                     f"Max simultaneous: {max_simultaneous}")
        cv2.putText(display, perf_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(win_name, display)

        # Key handling during playback
        key = cv2.waitKey(playback_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = False
        elif key == ord('m'):
            motion_only = not motion_only
            print(f"[FILTER] Motion only: {motion_only}")
        elif key == ord('t'):
            show_trails = not show_trails
        elif key == ord('v'):
            show_velocity = not show_velocity
        elif key == ord('i'):
            show_ids = not show_ids
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('s'):
            config["tracking"]["max_distance"] = tracker.max_distance
            config["tracking"]["max_frames_missing"] = tracker.max_disappeared
            config["tracking"]["trail_length"] = tracker.trail_length
            save_config(config)
        elif key == ord('+') or key == ord('='):
            playback_delay = max(1, playback_delay - 5)
        elif key == ord('-'):
            playback_delay = min(200, playback_delay + 5)

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 50)
    print("  TRACKING SUMMARY")
    print("=" * 50)
    print(f"  Total unique IDs assigned: {total_ids_created}")
    print(f"  Max simultaneous tracked:  {max_simultaneous}")
    print(f"  Tracker settings:")
    print(f"    max_distance:      {tracker.max_distance}")
    print(f"    max_disappeared:   {tracker.max_disappeared}")
    print(f"    trail_length:      {tracker.trail_length}")
    print("=" * 50)


if __name__ == "__main__":
    main()
