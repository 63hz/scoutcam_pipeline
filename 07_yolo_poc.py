# -*- coding: utf-8 -*-
"""
07 - YOLO Robot Detection Proof of Concept
============================================
Standalone test script to evaluate YOLO-based robot detection as an alternative
to HSV bumper tracking. This does NOT modify existing code - it's purely for
evaluation purposes.

WHAT THIS TESTS:
    1. Can YOLO detect robots in FRC footage?
    2. What classes does the pretrained COCO model see? (likely "person")
    3. What FPS can we achieve on the 3090?
    4. How well does the built-in tracker handle occlusions?

REQUIREMENTS:
    pip install ultralytics

USAGE:
    python 07_yolo_poc.py [video_path]

CONTROLS:
    Space   - Pause/Resume
    Q/ESC   - Quit
    D       - Toggle debug info overlay
    T       - Cycle tracker types (None, ByteTrack, BoT-SORT)
    C       - Toggle confidence threshold (0.25, 0.5, 0.75)
    +/-     - Adjust playback speed

Author: Clay / Claude sandbox
"""

import sys
import os
import time
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import load_config, open_video, apply_roi

# Check for ultralytics and CUDA
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"\n  CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n" + "=" * 60)
        print("  ERROR: CUDA NOT AVAILABLE")
        print("=" * 60)
        print("  This POC requires GPU acceleration for meaningful benchmarks.")
        print("  PyTorch was installed without CUDA support.")
        print("")
        print("  To fix, reinstall PyTorch with CUDA:")
        print("    pip uninstall torch torchvision")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60 + "\n")
        sys.exit(1)
except ImportError:
    YOLO_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("\n" + "=" * 60)
    print("  ULTRALYTICS NOT INSTALLED")
    print("=" * 60)
    print("  To run this POC, install ultralytics:")
    print("    pip install ultralytics")
    print("  Then reinstall PyTorch with CUDA:")
    print("    pip uninstall torch torchvision")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60 + "\n")
    sys.exit(1)


# Classes from COCO that might be robots
ROBOT_CLASSES = {
    0: "person",      # Robots often detected as people
    # Other potentially useful classes:
    # 2: "car", 7: "truck" - unlikely but possible
}


class YOLODetector:
    """YOLO-based detector for proof of concept."""

    def __init__(self, model_name="yolov8n.pt", device="cuda"):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLO model to use. Options:
                - yolov8n.pt (nano - fastest, ~6MB)
                - yolov8s.pt (small - good balance, ~22MB)
                - yolov8m.pt (medium - more accurate, ~52MB)
                - yolov8l.pt (large - high accuracy, ~87MB)
                - yolov8x.pt (extra large - highest accuracy, ~137MB)
            device: "cuda" for GPU (required for benchmarking)
        """
        print(f"\n  Loading YOLO model: {model_name}")
        print(f"  Device: {device}")

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = 0.25
        self.tracker_type = None  # None, "bytetrack", "botsort"

        # Warmup
        print("  Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=device, verbose=False)
        print("  Model ready!\n")

    def detect(self, frame, use_tracker=False):
        """
        Run detection on a frame.

        Returns list of detections: [{
            "bbox": (x1, y1, x2, y2),
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "track_id": int or None (if tracking enabled)
        }]
        """
        if use_tracker and self.tracker_type:
            results = self.model.track(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                tracker=f"{self.tracker_type}.yaml",
                persist=True,
                verbose=False
            )
        else:
            results = self.model.predict(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False
            )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()

                det = {
                    "bbox": tuple(map(int, bbox)),
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": conf,
                    "track_id": None
                }

                # Add track ID if available
                if boxes.id is not None:
                    det["track_id"] = int(boxes.id[i])

                detections.append(det)

        return detections

    def set_tracker(self, tracker_type):
        """Set tracker type: None, 'bytetrack', or 'botsort'."""
        self.tracker_type = tracker_type
        # Reset tracking state when changing trackers
        if hasattr(self.model, 'predictor') and self.model.predictor:
            self.model.predictor.trackers = []


def classify_alliance_by_hsv(frame, bbox, config):
    """
    Use HSV analysis to determine if a detected object is red or blue alliance.
    This combines YOLO detection with HSV color classification.

    Returns: "red", "blue", or "unknown"
    """
    x1, y1, x2, y2 = bbox

    # Focus on lower portion of bbox (where bumpers are)
    bumper_height = int((y2 - y1) * 0.3)  # Bottom 30%
    roi = frame[y2 - bumper_height:y2, x1:x2]

    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Get HSV ranges from config
    robot_cfg = config.get("robot_detection", {})
    hsv_red = robot_cfg.get("hsv_red", {})
    hsv_blue = robot_cfg.get("hsv_blue", {})

    # Red mask (wraps around hue)
    red_lower1 = np.array([hsv_red.get("h_low1", 0), hsv_red.get("s_low", 100), hsv_red.get("v_low", 100)])
    red_upper1 = np.array([hsv_red.get("h_high1", 10), hsv_red.get("s_high", 255), hsv_red.get("v_high", 255)])
    red_lower2 = np.array([hsv_red.get("h_low2", 160), hsv_red.get("s_low", 100), hsv_red.get("v_low", 100)])
    red_upper2 = np.array([hsv_red.get("h_high2", 180), hsv_red.get("s_high", 255), hsv_red.get("v_high", 255)])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Blue mask
    blue_lower = np.array([hsv_blue.get("h_low", 100), hsv_blue.get("s_low", 100), hsv_blue.get("v_low", 100)])
    blue_upper = np.array([hsv_blue.get("h_high", 130), hsv_blue.get("s_high", 255), hsv_blue.get("v_high", 255)])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    # Need significant color presence
    min_pixels = roi.shape[0] * roi.shape[1] * 0.05  # 5% of ROI

    if red_pixels > blue_pixels and red_pixels > min_pixels:
        return "red"
    elif blue_pixels > red_pixels and blue_pixels > min_pixels:
        return "blue"
    return "unknown"


def draw_detections(frame, detections, config, show_debug=True):
    """Draw detection boxes and labels on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        track_id = det["track_id"]

        # Determine alliance color
        alliance = classify_alliance_by_hsv(frame, det["bbox"], config)

        # Color based on alliance
        if alliance == "red":
            color = (0, 0, 255)  # BGR red
        elif alliance == "blue":
            color = (255, 0, 0)  # BGR blue
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label
        label_parts = [class_name]
        if track_id is not None:
            label_parts.insert(0, f"ID:{track_id}")
        if alliance != "unknown":
            label_parts.append(alliance.upper())
        if show_debug:
            label_parts.append(f"{conf:.2f}")
        label = " | ".join(label_parts)

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def draw_stats(frame, stats):
    """Draw performance stats overlay."""
    lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Detections: {stats.get('detections', 0)}",
        f"Tracker: {stats.get('tracker', 'None')}",
        f"Conf: {stats.get('confidence', 0.25):.2f}",
        f"Frame: {stats.get('frame_num', 0)}/{stats.get('total_frames', 0)}",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25

    # Instructions at bottom
    h = frame.shape[0]
    instructions = "SPACE:Pause  Q:Quit  D:Debug  T:Tracker  C:Confidence  +/-:Speed"
    cv2.putText(frame, instructions, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def run_poc(video_path):
    """Run the YOLO proof of concept."""
    if not YOLO_AVAILABLE:
        return

    config = load_config()
    cap, vid_info = open_video(video_path)

    print("\n" + "=" * 60)
    print("  YOLO Robot Detection POC")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Resolution: {vid_info['width']}x{vid_info['height']}")
    print(f"  FPS: {vid_info['fps']}")
    print(f"  Frames: {vid_info['frame_count']}")
    print("=" * 60)

    # Try different model sizes - start with small for balance
    detector = YOLODetector(model_name="yolov8s.pt", device="cuda")

    cv2.namedWindow("YOLO POC", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO POC", 1280, 720)

    paused = False
    show_debug = True
    playback_speed = 1.0
    tracker_options = [None, "bytetrack", "botsort"]
    tracker_idx = 0
    conf_options = [0.25, 0.5, 0.75]
    conf_idx = 0

    frame_num = 0
    total_frames = vid_info["frame_count"]
    fps_history = []

    print("\n  Press SPACE to start...")
    paused = True

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n  End of video reached.")
                break

            # Apply ROI
            roi_frame = apply_roi(frame, config["roi"])

            # Run detection
            t_start = time.time()
            use_tracker = tracker_options[tracker_idx] is not None
            detections = detector.detect(roi_frame, use_tracker=use_tracker)
            t_detect = time.time() - t_start

            # Calculate FPS
            fps = 1.0 / t_detect if t_detect > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Filter to robot-like detections (person class often)
            # For POC, show all detections to see what YOLO finds

            # Draw
            annotated = draw_detections(roi_frame, detections, config, show_debug)

            stats = {
                "fps": avg_fps,
                "detections": len(detections),
                "tracker": tracker_options[tracker_idx] or "None",
                "confidence": detector.conf_threshold,
                "frame_num": frame_num,
                "total_frames": total_frames,
            }
            annotated = draw_stats(annotated, stats)

            frame_num += 1

            # Print periodic stats
            if frame_num % 100 == 0:
                print(f"  Frame {frame_num}/{total_frames} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Detections: {len(detections)}")
        else:
            # When paused, just redraw last frame
            if 'annotated' not in locals():
                ret, frame = cap.read()
                if ret:
                    roi_frame = apply_roi(frame, config["roi"])
                    annotated = roi_frame.copy()
                    cv2.putText(annotated, "PAUSED - Press SPACE to start",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("YOLO POC", annotated)

        # Handle playback speed
        wait_time = max(1, int((1000 / vid_info["fps"]) / playback_speed))
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # Space
            paused = not paused
            if paused:
                print("  [PAUSED]")
            else:
                print("  [PLAYING]")
        elif key == ord('d'):  # Debug toggle
            show_debug = not show_debug
            print(f"  Debug overlay: {'ON' if show_debug else 'OFF'}")
        elif key == ord('t'):  # Tracker toggle
            tracker_idx = (tracker_idx + 1) % len(tracker_options)
            detector.set_tracker(tracker_options[tracker_idx])
            print(f"  Tracker: {tracker_options[tracker_idx] or 'None'}")
        elif key == ord('c'):  # Confidence toggle
            conf_idx = (conf_idx + 1) % len(conf_options)
            detector.conf_threshold = conf_options[conf_idx]
            print(f"  Confidence threshold: {detector.conf_threshold}")
        elif key == ord('+') or key == ord('='):
            playback_speed = min(4.0, playback_speed * 1.5)
            print(f"  Playback speed: {playback_speed:.1f}x")
        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed / 1.5)
            print(f"  Playback speed: {playback_speed:.1f}x")

    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    print("\n" + "=" * 60)
    print("  POC COMPLETE")
    print("=" * 60)
    print(f"  Frames processed: {frame_num}")
    if fps_history:
        print(f"  Average FPS: {sum(fps_history) / len(fps_history):.1f}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    video_path = "C:/Users/Clay/scoutcam_pipeline/kettering1.mkv"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    run_poc(video_path)
