# -*- coding: utf-8 -*-
"""
07 - YOLO Robot Detection Proof of Concept
============================================
Test script to evaluate YOLO-based robot detection. Supports both pretrained
COCO models and custom-trained bumper detection models.

WHAT THIS TESTS:
    1. Can YOLO detect robots in FRC footage?
    2. With custom model: direct red/blue bumper detection
    3. With COCO model: HSV post-processing for alliance color
    4. What FPS can we achieve on the 3090?
    5. How well does the built-in tracker handle occlusions?

REQUIREMENTS:
    pip install ultralytics

USAGE:
    python 07_yolo_poc.py [options] [video_path]

OPTIONS:
    --model PATH    Model to use (default: models/bumper_detector.pt if exists)
    --expand N      Bbox expansion factor for robot body (default: 1.5)

CONTROLS:
    Space   - Pause/Resume
    Q/ESC   - Quit
    D       - Toggle debug info overlay
    T       - Cycle tracker types (None, ByteTrack, BoT-SORT)
    C       - Toggle confidence threshold (0.25, 0.5, 0.75)
    E       - Toggle bbox expansion visualization
    +/-     - Adjust playback speed

Author: Clay / Claude sandbox
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

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


# Paths
SCRIPT_DIR = Path(__file__).parent
DEFAULT_CUSTOM_MODEL = SCRIPT_DIR / "models" / "bumper_detector.pt"
DEFAULT_COCO_MODEL = "yolov8s.pt"


def get_default_model():
    """Return custom model if exists, otherwise fall back to COCO model."""
    if DEFAULT_CUSTOM_MODEL.exists():
        return str(DEFAULT_CUSTOM_MODEL)
    return DEFAULT_COCO_MODEL


class YOLODetector:
    """YOLO-based detector for proof of concept."""

    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model. If None, uses custom model if available.
                Options for pretrained:
                - yolov8n.pt (nano - fastest, ~6MB)
                - yolov8s.pt (small - good balance, ~22MB)
                - yolov8m.pt (medium - more accurate, ~52MB)
                Or custom trained:
                - models/bumper_detector.pt (custom FRC bumper detection)
            device: "cuda" for GPU (required for benchmarking)
        """
        if model_path is None:
            model_path = get_default_model()

        print(f"\n  Loading YOLO model: {model_path}")
        print(f"  Device: {device}")

        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = 0.25
        self.tracker_type = None  # None, "bytetrack", "botsort"
        self.model_path = model_path

        # Check if this is a bumper detection model (has red/blue bumper classes)
        self.is_bumper_model = self._check_bumper_model()
        if self.is_bumper_model:
            print(f"  Model type: Custom bumper detector")
            print(f"  Classes: {list(self.model.names.values())}")
        else:
            print(f"  Model type: COCO pretrained (will use HSV for alliance)")

        # Warmup
        print("  Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=device, verbose=False)
        print("  Model ready!\n")

    def _check_bumper_model(self):
        """Check if model has bumper-specific classes."""
        class_names = [name.lower() for name in self.model.names.values()]
        return any("bumper" in name for name in class_names)

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

    def get_alliance_from_class(self, class_name):
        """
        Determine alliance from class name (for bumper models).

        Returns: "red", "blue", or "unknown"
        """
        name_lower = class_name.lower()
        if "red" in name_lower:
            return "red"
        elif "blue" in name_lower:
            return "blue"
        return "unknown"


def expand_bbox_upward(bbox, expansion_factor=1.5, frame_height=None):
    """
    Expand bumper bbox upward to approximate full robot body.

    Bumpers are at the base of the robot. Shot attribution needs to consider
    the full robot body above the bumpers. This expands the bbox upward.

    Args:
        bbox: (x1, y1, x2, y2) bounding box
        expansion_factor: How much to expand upward (1.5 = add 150% of height above)
        frame_height: Optional frame height to clamp y1 >= 0

    Returns:
        Expanded (x1, y1, x2, y2) bounding box
    """
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    expanded_y1 = y1 - int(height * expansion_factor)

    if frame_height is not None:
        expanded_y1 = max(0, expanded_y1)
    else:
        expanded_y1 = max(0, expanded_y1)

    return (x1, expanded_y1, x2, y2)


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


def draw_detections(frame, detections, config, detector, show_debug=True,
                    show_expanded=False, expansion_factor=1.5):
    """Draw detection boxes and labels on frame."""
    annotated = frame.copy()
    frame_height = frame.shape[0]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        track_id = det["track_id"]

        # Determine alliance color - use class name for bumper models, HSV otherwise
        if detector.is_bumper_model:
            alliance = detector.get_alliance_from_class(class_name)
        else:
            alliance = classify_alliance_by_hsv(frame, det["bbox"], config)

        # Color based on alliance
        if alliance == "red":
            color = (0, 0, 255)  # BGR red
        elif alliance == "blue":
            color = (255, 0, 0)  # BGR blue
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw expanded bbox if enabled (dashed)
        if show_expanded:
            ex1, ey1, ex2, ey2 = expand_bbox_upward(
                det["bbox"], expansion_factor, frame_height
            )
            # Draw dashed rectangle by drawing segments
            for i in range(ex1, ex2, 10):
                cv2.line(annotated, (i, ey1), (min(i + 5, ex2), ey1), color, 1)
                cv2.line(annotated, (i, ey2), (min(i + 5, ex2), ey2), color, 1)
            for i in range(ey1, ey2, 10):
                cv2.line(annotated, (ex1, i), (ex1, min(i + 5, ey2)), color, 1)
                cv2.line(annotated, (ex2, i), (ex2, min(i + 5, ey2)), color, 1)

        # Draw bumper bounding box (solid)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label
        label_parts = [class_name]
        if track_id is not None:
            label_parts.insert(0, f"ID:{track_id}")
        if alliance != "unknown" and not detector.is_bumper_model:
            # Only show alliance in label for COCO models (bumper models have it in class name)
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
        f"Model: {stats.get('model_type', 'unknown')}",
        f"Detections: {stats.get('detections', 0)}",
        f"Tracker: {stats.get('tracker', 'None')}",
        f"Conf: {stats.get('confidence', 0.25):.2f}",
        f"Expand: {stats.get('expand', 'OFF')}",
        f"Frame: {stats.get('frame_num', 0)}/{stats.get('total_frames', 0)}",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25

    # Instructions at bottom
    h = frame.shape[0]
    instructions = "SPACE:Pause  Q:Quit  D:Debug  T:Tracker  C:Conf  E:Expand  +/-:Speed"
    cv2.putText(frame, instructions, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def run_poc(video_path, model_path=None, expansion_factor=1.5):
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

    # Load specified model or default
    detector = YOLODetector(model_path=model_path, device="cuda")

    cv2.namedWindow("YOLO POC", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO POC", 1280, 720)

    paused = False
    show_debug = True
    show_expanded = False
    playback_speed = 1.0
    tracker_options = [None, "bytetrack", "botsort"]
    tracker_idx = 0
    conf_options = [0.25, 0.5, 0.75]
    conf_idx = 0

    frame_num = 0
    total_frames = vid_info["frame_count"]
    fps_history = []

    # Model type for display
    model_type = "bumper" if detector.is_bumper_model else "COCO"

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
            annotated = draw_detections(
                roi_frame, detections, config, detector,
                show_debug=show_debug,
                show_expanded=show_expanded,
                expansion_factor=expansion_factor
            )

            stats = {
                "fps": avg_fps,
                "model_type": model_type,
                "detections": len(detections),
                "tracker": tracker_options[tracker_idx] or "None",
                "confidence": detector.conf_threshold,
                "expand": f"{expansion_factor:.1f}x" if show_expanded else "OFF",
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
        elif key == ord('e'):  # Expansion toggle
            show_expanded = not show_expanded
            print(f"  Bbox expansion: {'ON' if show_expanded else 'OFF'}")
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
    parser = argparse.ArgumentParser(
        description="YOLO Robot Detection Proof of Concept"
    )
    parser.add_argument(
        "video", nargs="?",
        default="C:/Users/Clay/scoutcam_pipeline/kettering1.mkv",
        help="Path to video file"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to YOLO model (default: models/bumper_detector.pt if exists)"
    )
    parser.add_argument(
        "--expand", type=float, default=1.5,
        help="Bbox expansion factor for robot body (default: 1.5)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    run_poc(args.video, model_path=args.model, expansion_factor=args.expand)
