# -*- coding: utf-8 -*-
"""
02 - Detection Validation
==========================
Run ball detection across the video with saved config and visualize results.
Useful for validating tuning before moving to tracking.

WORKFLOW:
    1. Run 01_hsv_tuning.py first to tune and save config
    2. Set VIDEO_PATH below
    3. Run this script to see detection across the video
    4. Review stats and identify problem areas

CONTROLS:
    SPACE  - Play/Pause
    n/p    - Next/Previous frame
    +/-    - Speed up/slow down playback
    d      - Toggle debug quad view
    h      - Toggle detection histogram (balls per frame)
    q/ESC  - Quit

SPYDER CELL MODE:
    This script also works with Spyder cells (# %%) for interactive
    exploration. Run cells individually to inspect specific frames.

Author: Clay / Claude sandbox
"""

import sys
import os
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, open_video, apply_roi,
    BallDetector, draw_detections, create_debug_view, draw_hud
)

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_PATH = ""  # <-- SET THIS
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

# ============================================================================
# %% CELL: Load and validate
# ============================================================================

def run_validation():
    global VIDEO_PATH

    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    config = load_config()
    cap, vid_info = open_video(VIDEO_PATH)
    detector = BallDetector(config)

    # Stats collection
    detection_counts = []
    frame_times = []

    win_name = "Detection Validation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    playing = True
    debug_mode = False
    playback_delay = int(1000 / vid_info["fps"])  # ms per frame at 1x
    frame_num = 0

    print("\nControls: SPACE=play/pause, n/p=step, d=debug, +/-=speed, q=quit\n")

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                print(f"\n[STATS] Processed {frame_num} frames")
                break
            frame_num += 1
        else:
            # In pause mode, just re-display
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
            elif key == ord('d'):
                debug_mode = not debug_mode
                continue
            else:
                continue

        # Process frame
        t_start = time.time()
        roi_frame = apply_roi(frame, config["roi"])
        detections = detector.detect(roi_frame)
        t_elapsed = time.time() - t_start

        detection_counts.append(len(detections))
        frame_times.append(t_elapsed)

        # Build display
        if debug_mode:
            mask = detector.get_mask(roi_frame)
            display = create_debug_view(roi_frame, mask, detections)
        else:
            display = draw_detections(roi_frame, detections)

        # HUD
        stats = {
            "balls_detected": len(detections),
            "balls_tracked": 0,
            "balls_moving": 0,
        }
        display = draw_hud(display, stats, frame_num, vid_info["frame_count"])

        # Performance info
        fps_actual = 1.0 / t_elapsed if t_elapsed > 0 else 0
        perf_text = f"Detection: {t_elapsed*1000:.1f}ms ({fps_actual:.0f} fps potential)"
        cv2.putText(display, perf_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(win_name, display)

        # Key handling during playback
        key = cv2.waitKey(playback_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = False
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('+') or key == ord('='):
            playback_delay = max(1, playback_delay - 5)
            print(f"[SPEED] Delay: {playback_delay}ms")
        elif key == ord('-'):
            playback_delay = min(200, playback_delay + 5)
            print(f"[SPEED] Delay: {playback_delay}ms")

    cap.release()
    cv2.destroyAllWindows()

    # ---- Print Summary Stats ----
    if detection_counts:
        counts = np.array(detection_counts)
        times = np.array(frame_times)
        print("\n" + "=" * 50)
        print("  DETECTION SUMMARY")
        print("=" * 50)
        print(f"  Frames processed:  {len(counts)}")
        print(f"  Balls per frame:   mean={counts.mean():.1f}, "
              f"min={counts.min()}, max={counts.max()}, "
              f"std={counts.std():.1f}")
        print(f"  Processing time:   mean={times.mean()*1000:.1f}ms, "
              f"max={times.max()*1000:.1f}ms")
        print(f"  Potential FPS:     {1.0/times.mean():.0f}")
        print("=" * 50)

    return detection_counts, frame_times


# ============================================================================
# %% CELL: Run it
# ============================================================================

if __name__ == "__main__":
    detection_counts, frame_times = run_validation()

    # In Spyder, you can now inspect detection_counts and frame_times
    # in the Variable Explorer, plot them, etc.
    #
    # Quick plot (uncomment if you want):
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    # ax1.plot(detection_counts)
    # ax1.set_ylabel("Balls Detected")
    # ax1.set_xlabel("Frame")
    # ax1.set_title("Detection Count Over Time")
    # ax2.plot(np.array(frame_times) * 1000)
    # ax2.set_ylabel("Processing Time (ms)")
    # ax2.set_xlabel("Frame")
    # ax2.set_title("Per-Frame Processing Time")
    # plt.tight_layout()
    # plt.show()
