# -*- coding: utf-8 -*-
"""
01 - HSV Tuning Sandbox
========================
Interactive parameter tuning for yellow ball detection.

WORKFLOW:
    1. Set VIDEO_PATH below
    2. Run the script (in external terminal from Spyder, or standalone)
    3. Select ROI to crop out chirons
    4. Use trackbars to tune HSV range, morphology, and contour filters
    5. Step through frames with 'n' (next) and 'p' (prev) to test robustness
    6. Press 's' to save config, 'q' to quit

CONTROLS:
    SPACE  - Play/Pause video
    n      - Next frame (when paused)
    p      - Previous frame (when paused)
    d      - Toggle debug view (2x2 quad view)
    s      - Save current parameters to config
    r      - Re-select ROI
    q/ESC  - Quit

SPYDER NOTES:
    This script uses cv2.imshow + trackbars which require a real window.
    In Spyder: Run > Configuration per file > Execute in an external system terminal
    OR just run from terminal: python 01_hsv_tuning.py

Author: Clay / Claude sandbox
"""

import sys
import os

import cv2
import numpy as np

# Add parent dir to path if running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, save_config, open_video, apply_roi,
    select_roi_interactive, BallDetector, draw_detections,
    create_debug_view, print_gpu_status, get_gpu_status
)

# ============================================================================
# CONFIGURATION - EDIT THIS
# ============================================================================

VIDEO_PATH = ""  # <-- SET THIS to your video file path

# If empty, will prompt for file via dialog or argument
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

# ============================================================================
# TRACKBAR CALLBACK (no-op, we read values in the loop)
# ============================================================================

def nothing(x):
    pass

# ============================================================================
# MAIN
# ============================================================================

def main():
    global VIDEO_PATH

    print("=" * 60)
    print("  FRC Ball Tracker - HSV Tuning")
    print("=" * 60)

    # Print GPU diagnostics at startup
    print_gpu_status()

    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] File not found: {VIDEO_PATH}")
        return

    # Load config
    config = load_config()
    config["video"]["default_path"] = VIDEO_PATH

    # Open video
    cap, vid_info = open_video(VIDEO_PATH)

    # ---- ROI Selection ----
    print("\n[STEP 1] ROI Selection")
    print("Draw a rectangle to crop out chirons/overlays.")
    print("Press ENTER to confirm, or ESC to use saved ROI.\n")

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read video")
        return

    try:
        r = cv2.selectROI("Select ROI - ENTER to confirm, ESC for saved",
                           first_frame, fromCenter=False)
        cv2.destroyWindow("Select ROI - ENTER to confirm, ESC for saved")
        if r[2] > 0 and r[3] > 0:
            config["roi"] = {"x": int(r[0]), "y": int(r[1]),
                             "w": int(r[2]), "h": int(r[3])}
            print(f"[ROI] Set to: {config['roi']}")
        else:
            print(f"[ROI] Using saved: {config['roi']}")
    except Exception:
        print(f"[ROI] Using saved: {config['roi']}")

    # Reset to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Create Tuning Windows ----
    win_main = "Ball Detection Tuning"
    win_ctrl = "HSV + Filter Controls"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)

    # HSV trackbars
    hsv = config["hsv_yellow"]
    cv2.createTrackbar("H Low",  win_ctrl, hsv["h_low"],  179, nothing)
    cv2.createTrackbar("H High", win_ctrl, hsv["h_high"], 179, nothing)
    cv2.createTrackbar("S Low",  win_ctrl, hsv["s_low"],  255, nothing)
    cv2.createTrackbar("S High", win_ctrl, hsv["s_high"], 255, nothing)
    cv2.createTrackbar("V Low",  win_ctrl, hsv["v_low"],  255, nothing)
    cv2.createTrackbar("V High", win_ctrl, hsv["v_high"], 255, nothing)

    # Morphology trackbars
    morph = config["morphology"]
    cv2.createTrackbar("Open K",   win_ctrl, morph["open_kernel"],  21, nothing)
    cv2.createTrackbar("Close K",  win_ctrl, morph["close_kernel"], 21, nothing)
    cv2.createTrackbar("Dilate K", win_ctrl, morph["dilate_kernel"], 15, nothing)
    cv2.createTrackbar("Dilate N", win_ctrl, morph["dilate_iterations"], 5, nothing)

    # Contour filter trackbars (single ball size limits)
    cf = config["contour_filter"]
    cv2.createTrackbar("Min Area", win_ctrl, cf["min_area"], 500, nothing)
    cv2.createTrackbar("Max Area", win_ctrl, cf["max_area"], 2000, nothing)
    cv2.createTrackbar("Min Circ%", win_ctrl, int(cf["min_circularity"] * 100), 100, nothing)

    # NMS Cluster Splitting trackbars (preferred over watershed)
    nms_cfg = config.get("cluster_splitting", {"enabled": True, "min_ball_radius": 12,
                                                 "peak_threshold": 0.7, "area_multiplier": 1.5,
                                                 "max_cluster_area": 2000})
    cv2.createTrackbar("NMS Split", win_ctrl, 1 if nms_cfg.get("enabled", True) else 0, 1, nothing)
    cv2.createTrackbar("NMS Radius", win_ctrl, nms_cfg.get("min_ball_radius", 12), 50, nothing)
    cv2.createTrackbar("NMS Peak%", win_ctrl, int(nms_cfg.get("peak_threshold", 0.7) * 100), 100, nothing)
    cv2.createTrackbar("NMS AreaMul%", win_ctrl, int(nms_cfg.get("area_multiplier", 1.5) * 100), 400, nothing)
    cv2.createTrackbar("Max Cluster", win_ctrl, min(nms_cfg.get("max_cluster_area", 2000), 10000), 10000, nothing)

    # Watershed trackbars (legacy - only used if NMS is disabled)
    ws = config.get("watershed", {"enabled": False, "peak_ratio": 0.45,
                                   "min_peak_distance": 8, "area_multiplier": 1.8})
    cv2.createTrackbar("Watershed", win_ctrl, 1 if ws.get("enabled", False) else 0, 1, nothing)
    cv2.createTrackbar("WS Peak%", win_ctrl, int(ws.get("peak_ratio", 0.45) * 100), 80, nothing)
    cv2.createTrackbar("WS MinDist", win_ctrl, ws.get("min_peak_distance", 8), 40, nothing)
    cv2.createTrackbar("WS AreaMul%", win_ctrl, int(ws.get("area_multiplier", 1.8) * 100), 500, nothing)

    # GPU toggle
    gpu_status = get_gpu_status()
    gpu_available = gpu_status["available"] and config.get("gpu", {}).get("enabled", False)
    cv2.createTrackbar("GPU", win_ctrl, 1 if gpu_available else 0, 1, nothing)

    # Frame control
    cv2.createTrackbar("Frame", win_main, 0, vid_info["frame_count"] - 1, nothing)

    # State
    debug_mode = False
    playing = False
    current_frame_num = 0

    print("\n[STEP 2] Tune HSV and filters")
    print("Controls: SPACE=play/pause, n/p=step, d=debug view, s=save, q=quit\n")

    detector = BallDetector(config)

    while True:
        # Read trackbar values and update config
        config["hsv_yellow"]["h_low"]  = cv2.getTrackbarPos("H Low",  win_ctrl)
        config["hsv_yellow"]["h_high"] = cv2.getTrackbarPos("H High", win_ctrl)
        config["hsv_yellow"]["s_low"]  = cv2.getTrackbarPos("S Low",  win_ctrl)
        config["hsv_yellow"]["s_high"] = cv2.getTrackbarPos("S High", win_ctrl)
        config["hsv_yellow"]["v_low"]  = cv2.getTrackbarPos("V Low",  win_ctrl)
        config["hsv_yellow"]["v_high"] = cv2.getTrackbarPos("V High", win_ctrl)

        # Ensure kernel sizes are odd and >= 1
        ok = max(1, cv2.getTrackbarPos("Open K",   win_ctrl)) | 1
        ck = max(1, cv2.getTrackbarPos("Close K",  win_ctrl)) | 1
        dk = max(1, cv2.getTrackbarPos("Dilate K", win_ctrl)) | 1
        config["morphology"]["open_kernel"]  = ok
        config["morphology"]["close_kernel"] = ck
        config["morphology"]["dilate_kernel"] = dk
        config["morphology"]["dilate_iterations"] = cv2.getTrackbarPos("Dilate N", win_ctrl)

        config["contour_filter"]["min_area"] = cv2.getTrackbarPos("Min Area", win_ctrl)
        config["contour_filter"]["max_area"] = max(
            config["contour_filter"]["min_area"] + 1,
            cv2.getTrackbarPos("Max Area", win_ctrl)
        )
        config["contour_filter"]["min_circularity"] = cv2.getTrackbarPos("Min Circ%", win_ctrl) / 100.0

        # NMS Cluster Splitting params (preferred)
        if "cluster_splitting" not in config:
            config["cluster_splitting"] = {"enabled": True, "method": "nms",
                                            "min_ball_radius": 12, "peak_threshold": 0.7,
                                            "area_multiplier": 1.5, "max_cluster_area": 2000}
        config["cluster_splitting"]["enabled"] = bool(cv2.getTrackbarPos("NMS Split", win_ctrl))
        config["cluster_splitting"]["min_ball_radius"] = max(5, cv2.getTrackbarPos("NMS Radius", win_ctrl))
        config["cluster_splitting"]["peak_threshold"] = max(0.3, cv2.getTrackbarPos("NMS Peak%", win_ctrl) / 100.0)
        config["cluster_splitting"]["area_multiplier"] = max(1.1, cv2.getTrackbarPos("NMS AreaMul%", win_ctrl) / 100.0)
        config["cluster_splitting"]["max_cluster_area"] = max(100, cv2.getTrackbarPos("Max Cluster", win_ctrl))

        # Watershed params (legacy - disable if NMS is enabled)
        if "watershed" not in config:
            config["watershed"] = {"enabled": False, "peak_ratio": 0.45,
                                    "min_peak_distance": 8, "area_multiplier": 1.8}
        # Only enable watershed if NMS is disabled
        ws_requested = bool(cv2.getTrackbarPos("Watershed", win_ctrl))
        config["watershed"]["enabled"] = ws_requested and not config["cluster_splitting"]["enabled"]
        config["watershed"]["peak_ratio"] = max(0.1, cv2.getTrackbarPos("WS Peak%", win_ctrl) / 100.0)
        config["watershed"]["min_peak_distance"] = max(3, cv2.getTrackbarPos("WS MinDist", win_ctrl))
        config["watershed"]["area_multiplier"] = max(1.1, cv2.getTrackbarPos("WS AreaMul%", win_ctrl) / 100.0)

        # GPU
        if "gpu" not in config:
            config["gpu"] = {"enabled": False}
        config["gpu"]["enabled"] = bool(cv2.getTrackbarPos("GPU", win_ctrl))

        # Update detector with new config
        detector.update_config(config)

        # Handle frame position
        if playing:
            current_frame_num += 1
            if current_frame_num >= vid_info["frame_count"]:
                current_frame_num = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            # Check if user moved the frame trackbar
            tb_frame = cv2.getTrackbarPos("Frame", win_main)
            if tb_frame != current_frame_num:
                current_frame_num = tb_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame_num = 0
            continue

        # Apply ROI
        roi_frame = apply_roi(frame, config["roi"])

        # Detect
        detections = detector.detect(roi_frame)
        mask = detector.get_mask(roi_frame)

        # Display
        if debug_mode:
            display = create_debug_view(roi_frame, mask, detections, detector)
        else:
            display = draw_detections(roi_frame, detections)

        # HUD text
        nms_enabled = config.get("cluster_splitting", {}).get("enabled", False)
        ws_enabled = config.get("watershed", {}).get("enabled", False) and not nms_enabled
        if nms_enabled:
            split_status = "NMS:ON"
        elif ws_enabled:
            split_status = "WS:ON"
        else:
            split_status = "Split:OFF"

        gpu_text = "GPU:ON" if detector.use_gpu else "GPU:OFF"
        split_info = f"  splits:{detector._debug_split_count}" if (nms_enabled or ws_enabled) else ""

        # Timing info
        timing = detector.get_timing()
        timing_text = f"{timing['total_ms']:.1f}ms"

        info_text = (f"Frame {current_frame_num}/{vid_info['frame_count']}  |  "
                     f"Detected: {len(detections)}{split_info}  |  "
                     f"{split_status}  {gpu_text}  {timing_text}  |  "
                     f"{'PLAYING' if playing else 'PAUSED'}  |  "
                     f"d=debug s=save q=quit")
        cv2.putText(display, info_text, (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win_main, display)

        # Update frame trackbar position
        if playing:
            cv2.setTrackbarPos("Frame", win_main, current_frame_num)

        # Key handling
        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):  # Space: toggle play
            playing = not playing
        elif key == ord('n'):  # Next frame
            playing = False
            current_frame_num += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        elif key == ord('p'):  # Previous frame
            playing = False
            current_frame_num = max(0, current_frame_num - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        elif key == ord('d'):  # Toggle debug view
            debug_mode = not debug_mode
            print(f"[DEBUG] {'ON' if debug_mode else 'OFF'}")
        elif key == ord('s'):  # Save config
            save_config(config)
            print(f"[SAVED] Detection count at save: {len(detections)}")
        elif key == ord('r'):  # Re-select ROI
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, reselect_frame = cap.read()
            if ret:
                r = cv2.selectROI("Re-select ROI", reselect_frame, fromCenter=False)
                cv2.destroyWindow("Re-select ROI")
                if r[2] > 0 and r[3] > 0:
                    config["roi"] = {"x": int(r[0]), "y": int(r[1]),
                                     "w": int(r[2]), "h": int(r[3])}
                    print(f"[ROI] Updated: {config['roi']}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] HSV tuning session complete.")


if __name__ == "__main__":
    main()
