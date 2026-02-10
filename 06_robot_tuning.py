# -*- coding: utf-8 -*-
"""
06 - Robot Bumper Tuning
========================
Interactive HSV tuning for robot bumper detection.

PURPOSE:
    This script helps you tune the HSV color ranges for detecting
    red and blue FRC robot bumpers. Once tuned, robots can be tracked
    across frames and used for shot attribution in the main pipeline.

    The goal is to get clean bumper detection WITHOUT false positives
    (random colored objects being detected as robots).

TRACKER:
    By default, this uses OC-SORT tracker (Kalman + Hungarian algorithm)
    which maintains robot IDs through brief occlusions. To switch back
    to the legacy centroid tracker, set "tracker": "centroid" in the
    robot_tracking config section.

WHAT YOU'RE TUNING:
    1. HSV Color Ranges - Adjust until bumpers are cleanly detected
       - Red bumpers use TWO hue ranges (red wraps around 0/180 in HSV)
       - Blue bumpers use one hue range
       - S (saturation) and V (value) help reject washed-out or dark colors

    2. Contour Filters - Adjust to reject non-bumper detections
       - Min/Max Area: Size in pixels (bumpers are typically 500-15000 px)
       - Aspect Ratio: Bumpers are wide rectangles (aspect 1.5-8.0)

STEP-BY-STEP WORKFLOW:
    1. Load a match video with visible robot bumpers
    2. Pause on a frame with clear bumper visibility (press SPACE)
    3. Press 'd' to see debug view (shows color masks)
    4. Adjust RED sliders until red bumpers show white in "Red Mask"
    5. Adjust BLUE sliders until blue bumpers show white in "Blue Mask"
    6. Switch back to normal view (press 'd') to see detections
    7. Adjust Min/Max Area and Aspect Ratio to filter false positives
    8. Press 's' to save the HSV values to config

UNDERSTANDING THE DEBUG VIEW:
    Top-Left:     Detected robots with bounding boxes
    Top-Right:    Red mask (white = detected as red)
    Bottom-Left:  Blue mask (white = detected as blue)
    Bottom-Right: Color overlay showing what's being detected

TUNING TIPS:
    - Start with Saturation: Raise S_Low to reject washed-out colors
    - Adjust Hue carefully: Small changes have big effects
    - Use Value (V) to handle lighting: Lower V_Low for dark venues
    - If detecting too much: Raise S_Low and/or narrow the Hue range
    - If missing bumpers: Widen the Hue range and/or lower S_Low

TEAM NUMBER ASSIGNMENT:
    - Press 'c' to enter correction mode, then click on a robot
    - Team numbers are SESSION-ONLY (not saved between runs)
    - This is because tracking IDs are ephemeral and can't persist

CONTROLS:
    SPACE  - Play/Pause
    n      - Step forward one frame
    b      - Step backward one frame
    r      - Toggle red bumper detection ON/OFF
    u      - Toggle blue bumper detection ON/OFF
    d      - Toggle debug view (shows color masks)
    c      - Click-to-correct mode (click robot, enter team number)
    s      - Save HSV config values
    q/ESC  - Quit

Author: Clay / Claude sandbox
"""

import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, save_config, open_video, apply_roi,
    RobotDetector, draw_robots, print_gpu_status
)


VIDEO_PATH = ""  # <-- SET THIS or pass as command line argument
if not VIDEO_PATH and len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]


def create_trackbar_window(config):
    """Create separate trackbar window with HSV controls."""
    win_name = "Robot HSV Controls"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 400, 700)

    robot_cfg = config.get("robot_detection", {})

    # Red HSV (two ranges because red wraps around 0/180)
    hsv_red = robot_cfg.get("hsv_red", {})
    cv2.createTrackbar("R H1 Low", win_name, hsv_red.get("h_low1", 0), 20, lambda x: None)
    cv2.createTrackbar("R H1 High", win_name, hsv_red.get("h_high1", 10), 20, lambda x: None)
    cv2.createTrackbar("R H2 Low", win_name, hsv_red.get("h_low2", 170), 180, lambda x: None)
    cv2.createTrackbar("R H2 High", win_name, hsv_red.get("h_high2", 180), 180, lambda x: None)
    cv2.createTrackbar("R S Low", win_name, hsv_red.get("s_low", 100), 255, lambda x: None)
    cv2.createTrackbar("R S High", win_name, hsv_red.get("s_high", 255), 255, lambda x: None)
    cv2.createTrackbar("R V Low", win_name, hsv_red.get("v_low", 80), 255, lambda x: None)
    cv2.createTrackbar("R V High", win_name, hsv_red.get("v_high", 255), 255, lambda x: None)

    # Blue HSV
    hsv_blue = robot_cfg.get("hsv_blue", {})
    cv2.createTrackbar("B H Low", win_name, hsv_blue.get("h_low", 100), 180, lambda x: None)
    cv2.createTrackbar("B H High", win_name, hsv_blue.get("h_high", 130), 180, lambda x: None)
    cv2.createTrackbar("B S Low", win_name, hsv_blue.get("s_low", 100), 255, lambda x: None)
    cv2.createTrackbar("B S High", win_name, hsv_blue.get("s_high", 255), 255, lambda x: None)
    cv2.createTrackbar("B V Low", win_name, hsv_blue.get("v_low", 80), 255, lambda x: None)
    cv2.createTrackbar("B V High", win_name, hsv_blue.get("v_high", 255), 255, lambda x: None)

    # Contour filters
    cv2.createTrackbar("Min Area", win_name, robot_cfg.get("min_bumper_area", 500), 5000, lambda x: None)
    cv2.createTrackbar("Max Area", win_name, robot_cfg.get("max_bumper_area", 15000), 30000, lambda x: None)
    cv2.createTrackbar("Min Aspect*10", win_name, int(robot_cfg.get("min_aspect_ratio", 1.5) * 10), 50, lambda x: None)
    cv2.createTrackbar("Max Aspect*10", win_name, int(robot_cfg.get("max_aspect_ratio", 8.0) * 10), 100, lambda x: None)

    return win_name


def read_trackbars(win_name, config):
    """Read trackbar values and update config."""
    robot_cfg = config.setdefault("robot_detection", {})

    # Red HSV
    hsv_red = robot_cfg.setdefault("hsv_red", {})
    hsv_red["h_low1"] = cv2.getTrackbarPos("R H1 Low", win_name)
    hsv_red["h_high1"] = cv2.getTrackbarPos("R H1 High", win_name)
    hsv_red["h_low2"] = cv2.getTrackbarPos("R H2 Low", win_name)
    hsv_red["h_high2"] = cv2.getTrackbarPos("R H2 High", win_name)
    hsv_red["s_low"] = cv2.getTrackbarPos("R S Low", win_name)
    hsv_red["s_high"] = cv2.getTrackbarPos("R S High", win_name)
    hsv_red["v_low"] = cv2.getTrackbarPos("R V Low", win_name)
    hsv_red["v_high"] = cv2.getTrackbarPos("R V High", win_name)

    # Blue HSV
    hsv_blue = robot_cfg.setdefault("hsv_blue", {})
    hsv_blue["h_low"] = cv2.getTrackbarPos("B H Low", win_name)
    hsv_blue["h_high"] = cv2.getTrackbarPos("B H High", win_name)
    hsv_blue["s_low"] = cv2.getTrackbarPos("B S Low", win_name)
    hsv_blue["s_high"] = cv2.getTrackbarPos("B S High", win_name)
    hsv_blue["v_low"] = cv2.getTrackbarPos("B V Low", win_name)
    hsv_blue["v_high"] = cv2.getTrackbarPos("B V High", win_name)

    # Contour filters
    robot_cfg["min_bumper_area"] = cv2.getTrackbarPos("Min Area", win_name)
    robot_cfg["max_bumper_area"] = cv2.getTrackbarPos("Max Area", win_name)
    robot_cfg["min_aspect_ratio"] = cv2.getTrackbarPos("Min Aspect*10", win_name) / 10.0
    robot_cfg["max_aspect_ratio"] = cv2.getTrackbarPos("Max Aspect*10", win_name) / 10.0

    return config


def create_debug_view(frame, detector, robots):
    """Create 2x2 debug view showing masks and detections."""
    h, w = frame.shape[:2]
    qw, qh = w // 2, h // 2

    # Top left: annotated frame with detections
    annotated = draw_robots(frame, robots)
    tl = cv2.resize(annotated, (qw, qh))
    cv2.putText(tl, "Detections", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Top right: red mask
    red_mask = detector._debug_red_mask
    if red_mask is not None:
        tr = cv2.resize(cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), (qw, qh))
        cv2.putText(tr, "Red Mask (white=detected)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        tr = np.zeros((qh, qw, 3), dtype=np.uint8)
        cv2.putText(tr, "Red: OFF (press 'r' to enable)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Bottom left: blue mask
    blue_mask = detector._debug_blue_mask
    if blue_mask is not None:
        bl = cv2.resize(cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR), (qw, qh))
        cv2.putText(bl, "Blue Mask (white=detected)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        bl = np.zeros((qh, qw, 3), dtype=np.uint8)
        cv2.putText(bl, "Blue: OFF (press 'u' to enable)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Bottom right: color overlay
    combined = np.zeros_like(frame)
    if red_mask is not None:
        combined[red_mask > 0] = [0, 0, 255]  # Red
    if blue_mask is not None:
        combined[blue_mask > 0] = [255, 0, 0]  # Blue
    # Blend with original
    overlay = cv2.addWeighted(frame, 0.5, combined, 0.5, 0)
    br = cv2.resize(overlay, (qw, qh))
    cv2.putText(br, "Detected Colors", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])


def draw_hud(frame, robots, frame_num, total_frames, show_red, show_blue, correction_mode, tracker_stats=None):
    """Draw HUD with robot stats and status."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Background panel - taller to fit tracker stats
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (380, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Stats
    visible = [r for r in robots.values() if r.disappeared == 0]
    red_count = sum(1 for r in visible if r.alliance == "red")
    blue_count = sum(1 for r in visible if r.alliance == "blue")
    identified = sum(1 for r in visible if r.team_number is not None)

    y = 20
    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"Robots: {len(visible)} (Red: {red_count}, Blue: {blue_count})",
        f"Identified: {identified}",
        f"Tracking: {'RED ' if show_red else ''}{'BLUE' if show_blue else ''}{'NONE' if not show_red and not show_blue else ''}",
    ]

    # Add tracker stats
    if tracker_stats:
        tracker_type = tracker_stats.get("tracker_type", "unknown").upper()
        active = tracker_stats.get("active_tracks", 0)
        confirmed = tracker_stats.get("confirmed_tracks", 0)
        if tracker_type == "OCSORT":
            matches = tracker_stats.get("matches", 0)
            lines.append(f"Tracker: {tracker_type} (active:{active} conf:{confirmed})")
            lines.append(f"Matches: {matches}")
        else:
            lines.append(f"Tracker: {tracker_type} (active:{active})")

    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22

    # Progress bar
    if total_frames > 0:
        bar_y = h - 8
        cv2.rectangle(out, (0, bar_y), (int(w * frame_num / total_frames), h), (0, 200, 255), -1)

    # Mode indicators
    if correction_mode:
        cv2.putText(out, "CORRECTION MODE - Click a robot, then type team # in console",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return out


def print_help():
    """Print detailed help to console."""
    print("\n" + "=" * 70)
    print("  ROBOT BUMPER TUNING - QUICK GUIDE")
    print("=" * 70)
    print("""
  GOAL: Tune HSV values so bumpers are cleanly detected (no false positives)

  STEP 1: Press 'd' to enter debug view (see the color masks)
  STEP 2: Adjust RED sliders until red bumpers show WHITE in "Red Mask"
  STEP 3: Adjust BLUE sliders until blue bumpers show WHITE in "Blue Mask"
  STEP 4: Press 'd' again to see actual detections with bounding boxes
  STEP 5: If false positives, adjust Min/Max Area or Aspect Ratio
  STEP 6: Press 's' to save your HSV values

  CONTROLS:
    SPACE  - Play/Pause video
    n / b  - Step forward / backward one frame
    d      - Toggle debug view (shows color masks)
    r / u  - Toggle red / blue detection
    c      - Correction mode (click robot, type team # in console)
    s      - Save config
    q      - Quit

  TIPS:
    - Pause on a frame with clear bumpers visible
    - Start by adjusting S_Low (saturation) to reject washed-out colors
    - Hue is sensitive - small changes have big effects
    - V_Low helps with dark venues (lower = more lenient)
""")
    print("=" * 70 + "\n")


def main():
    global VIDEO_PATH

    print_gpu_status()
    print_help()

    if not VIDEO_PATH:
        VIDEO_PATH = input("Enter video file path: ").strip().strip('"').strip("'")

    config = load_config()

    # Enable robot detection for tuning
    config.setdefault("robot_detection", {})["enabled"] = True

    cap, vid_info = open_video(VIDEO_PATH)
    detector = RobotDetector(config)

    # --- Setup windows ---
    video_win = "Robot Detection"
    cv2.namedWindow(video_win, cv2.WINDOW_NORMAL)

    trackbar_win = create_trackbar_window(config)

    # --- State ---
    playing = False
    show_debug = False
    show_red = True
    show_blue = True
    correction_mode = False
    playback_delay = int(1000 / vid_info["fps"])
    frame_num = 0
    current_frame = None

    # Mouse callback for correction mode
    def mouse_callback(event, x, y, flags, param):
        nonlocal correction_mode
        if event == cv2.EVENT_LBUTTONDOWN and correction_mode:
            # Scale coordinates if in debug view (each quadrant is half size)
            if show_debug:
                # Only respond to clicks in top-left quadrant (detections)
                roi_frame = apply_roi(current_frame, config["roi"])
                h, w = roi_frame.shape[:2]
                qw, qh = w // 2, h // 2
                if x > qw or y > qh:
                    print("  (Click in the top-left quadrant to select a robot)")
                    return
                # Scale up the coordinates
                x = x * 2
                y = y * 2

            # Find nearest robot
            robots = detector.get_all_robots()
            min_dist = float("inf")
            nearest = None

            for rid, robot in robots.items():
                dist = ((x - robot.cx)**2 + (y - robot.cy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = robot

            if nearest and min_dist < 100:
                print(f"\n  Selected: {nearest.identity} (tracking ID: {nearest.id})")
                team = input(f"  Enter team number (or ENTER to cancel): ").strip()
                if team and team.isdigit():
                    detector.apply_correction(nearest.id, team)
                    print(f"  -> Set to {nearest.alliance}_{team}")
                else:
                    print("  -> Cancelled")
                correction_mode = False
            else:
                print("  (No robot found near click location)")

    cv2.setMouseCallback(video_win, mouse_callback)

    # Read first frame
    ret, current_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return
    frame_num = 1

    print("Ready! Press 'd' to toggle debug view, SPACE to play/pause.\n")

    while True:
        # Read trackbars and update detector
        config = read_trackbars(trackbar_win, config)

        # Update alliance tracking based on toggles
        if show_red and show_blue:
            config["robot_detection"]["track_alliance"] = "both"
        elif show_red:
            config["robot_detection"]["track_alliance"] = "red"
        elif show_blue:
            config["robot_detection"]["track_alliance"] = "blue"
        else:
            config["robot_detection"]["track_alliance"] = "none"

        detector.update_config(config)

        # Apply ROI
        roi_frame = apply_roi(current_frame, config["roi"])

        # Detect and track robots
        robots = detector.detect_and_track(roi_frame)

        # Get tracker stats for display
        tracker_stats = detector.get_tracker_stats()

        # Draw
        if show_debug:
            display = create_debug_view(roi_frame, detector, robots)
        else:
            display = draw_robots(roi_frame, robots)
            display = draw_hud(display, robots, frame_num, vid_info["frame_count"],
                               show_red, show_blue, correction_mode, tracker_stats)

        cv2.imshow(video_win, display)

        # Handle input
        if playing:
            key = cv2.waitKey(playback_delay) & 0xFF
        else:
            key = cv2.waitKey(30) & 0xFF  # Still need to process trackbar changes

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
            print(f"  {'Playing...' if playing else 'Paused'}")
        elif key == ord('n'):
            ret, current_frame = cap.read()
            if not ret:
                print("  End of video")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, current_frame = cap.read()
                frame_num = 0
            frame_num += 1
            playing = False
        elif key == ord('b'):
            frame_num = max(0, frame_num - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, current_frame = cap.read()
            if not ret:
                break
            frame_num += 1
            playing = False
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"  Debug view: {'ON' if show_debug else 'OFF'}")
        elif key == ord('r'):
            show_red = not show_red
            print(f"  Red detection: {'ON' if show_red else 'OFF'}")
        elif key == ord('u'):
            show_blue = not show_blue
            print(f"  Blue detection: {'ON' if show_blue else 'OFF'}")
        elif key == ord('c'):
            correction_mode = True
            print("\n  CORRECTION MODE: Click on a robot in the video window...")
        elif key == ord('s'):
            save_config(config)
            print("  Config saved!")
        elif key == ord('h') or key == ord('?'):
            print_help()

        # Advance frame if playing
        if playing:
            ret, current_frame = cap.read()
            if not ret:
                playing = False
                print("  End of video (paused)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, current_frame = cap.read()
                frame_num = 0
            else:
                frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    robots = detector.get_all_robots()
    if robots:
        print("\n" + "=" * 50)
        print("  DETECTED ROBOTS (end of session)")
        print("=" * 50)
        for rid, robot in robots.items():
            marker = "[M]" if robot.manual_correction else ""
            print(f"  {robot.identity} {marker}")
        print("=" * 50)
        print("\n  Note: Team numbers are session-only and must be")
        print("  re-assigned if you run the script again.\n")


if __name__ == "__main__":
    main()
