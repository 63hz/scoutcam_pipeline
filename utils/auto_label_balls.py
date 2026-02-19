# -*- coding: utf-8 -*-
"""
Auto-Label Balls from Video using HSV Detection
=================================================
Extracts frames from video and generates YOLO-format labels using the
existing HSV BallDetector. Creates a ready-to-train dataset structure.

OUTPUT STRUCTURE:
    balls-dataset/
    ├── data.yaml           # Dataset configuration
    ├── train/
    │   ├── images/        # Training images
    │   └── labels/        # YOLO format labels (.txt)
    └── valid/
        ├── images/        # Validation images
        └── labels/        # YOLO format labels (.txt)

LABEL FORMAT (YOLO):
    Each .txt file contains one line per detection:
    <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0,1] relative to image dimensions.

USAGE:
    python utils/auto_label_balls.py video.mkv --output balls-dataset
    python utils/auto_label_balls.py video1.mkv video2.mkv --output balls-dataset
    python utils/auto_label_balls.py video.mkv --interval 15 --output balls-dataset

OPTIONS:
    --interval N    Extract every Nth frame (default: 30)
    --output PATH   Output dataset folder (default: balls-dataset)
    --split RATIO   Train/valid split ratio (default: 0.85)
    --review        Open interactive review window for each frame
    --min-balls N   Minimum balls to include a frame (default: 1)
    --config PATH   Config file path (default: frc_tracker_config.json)

WORKFLOW:
    1. Run this script to generate initial labels
    2. Review/correct labels in Roboflow or labelImg
    3. Train model with: python 09_train_ball_model.py

Author: Clay / Claude sandbox
"""

import sys
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from frc_tracker_utils import load_config, BallDetector, apply_roi


def extract_frames_with_detections(
    video_paths: List[str],
    config: dict,
    interval: int = 30,
    min_balls: int = 1,
    review: bool = False,
) -> List[Tuple[np.ndarray, List[dict]]]:
    """
    Extract frames from video(s) and detect balls using HSV detector.

    Args:
        video_paths: List of video file paths
        config: Config dict for BallDetector
        interval: Extract every Nth frame
        min_balls: Minimum balls required to include frame
        review: If True, show each frame for manual review

    Returns:
        List of (frame, detections) tuples
    """
    detector = BallDetector(config)
    roi_cfg = config.get("roi", {"x": 0, "y": 0, "w": 1920, "h": 1080})

    results = []

    for video_path in video_paths:
        print(f"\n[VIDEO] Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  ERROR: Cannot open video: {video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {frame_count} frames @ {fps:.1f} fps")
        print(f"  Extracting every {interval} frames...")

        frame_num = 0
        extracted = 0
        skipped_empty = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % interval == 0:
                # Apply ROI
                roi_frame = apply_roi(frame, roi_cfg)

                # Detect balls
                detections = detector.detect(roi_frame)

                # Skip frames with too few balls
                if len(detections) < min_balls:
                    skipped_empty += 1
                    frame_num += 1
                    continue

                if review:
                    # Show frame for manual review
                    display = roi_frame.copy()
                    for det in detections:
                        cx, cy = int(det["cx"]), int(det["cy"])
                        r = int(det.get("radius", 10))
                        cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)

                    cv2.putText(display, f"Frame {frame_num}: {len(detections)} balls",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display, "SPACE=keep, S=skip, Q=quit",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("Review", display)
                    key = cv2.waitKey(0) & 0xFF

                    if key == ord('q'):
                        print("  Review cancelled")
                        break
                    elif key == ord('s'):
                        frame_num += 1
                        continue

                results.append((roi_frame.copy(), detections))
                extracted += 1

                if extracted % 50 == 0:
                    print(f"  Extracted {extracted} frames...")

            frame_num += 1

        cap.release()
        print(f"  Extracted {extracted} frames ({skipped_empty} skipped with <{min_balls} balls)")

    if review:
        cv2.destroyAllWindows()

    return results


def convert_to_yolo_format(
    detections: List[dict],
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> List[str]:
    """
    Convert detections to YOLO format labels.

    Args:
        detections: List of detection dicts with 'cx', 'cy', 'radius' or 'bbox'
        img_width: Image width for normalization
        img_height: Image height for normalization
        class_id: Class ID (0 for 'ball')

    Returns:
        List of YOLO format label strings
    """
    labels = []

    for det in detections:
        cx = det["cx"]
        cy = det["cy"]

        # Get bounding box dimensions
        if "bbox" in det:
            x, y, w, h = det["bbox"]
        elif "radius" in det:
            r = det["radius"]
            w = h = r * 2
        else:
            w = h = 20  # Default size

        # Normalize to [0,1]
        x_center = cx / img_width
        y_center = cy / img_height
        width = w / img_width
        height = h / img_height

        # Clamp to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0.001, min(1, width))
        height = max(0.001, min(1, height))

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return labels


def create_dataset(
    frames_with_detections: List[Tuple[np.ndarray, List[dict]]],
    output_dir: Path,
    train_split: float = 0.85,
):
    """
    Create YOLO dataset structure from extracted frames.

    Args:
        frames_with_detections: List of (frame, detections) tuples
        output_dir: Output directory
        train_split: Fraction for training (rest goes to validation)
    """
    print(f"\n[DATASET] Creating dataset at: {output_dir}")

    # Create directory structure
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(frames_with_detections)
    split_idx = int(len(frames_with_detections) * train_split)

    train_data = frames_with_detections[:split_idx]
    valid_data = frames_with_detections[split_idx:]

    print(f"  Train: {len(train_data)} frames")
    print(f"  Valid: {len(valid_data)} frames")

    total_detections = 0

    # Save training data
    for i, (frame, detections) in enumerate(train_data):
        img_path = output_dir / "train" / "images" / f"frame_{i:05d}.jpg"
        label_path = output_dir / "train" / "labels" / f"frame_{i:05d}.txt"

        cv2.imwrite(str(img_path), frame)

        h, w = frame.shape[:2]
        labels = convert_to_yolo_format(detections, w, h)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        total_detections += len(detections)

    # Save validation data
    for i, (frame, detections) in enumerate(valid_data):
        img_path = output_dir / "valid" / "images" / f"frame_{i:05d}.jpg"
        label_path = output_dir / "valid" / "labels" / f"frame_{i:05d}.txt"

        cv2.imwrite(str(img_path), frame)

        h, w = frame.shape[:2]
        labels = convert_to_yolo_format(detections, w, h)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        total_detections += len(detections)

    # Create data.yaml
    data_yaml = f"""# FRC Ball Detection Dataset
# Auto-generated from HSV detections

path: {output_dir.absolute()}
train: train/images
val: valid/images

# Classes
nc: 1
names:
  0: ball
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(data_yaml)

    print(f"  Total detections: {total_detections}")
    print(f"  Average per frame: {total_detections / len(frames_with_detections):.1f}")
    print(f"  Created: {output_dir / 'data.yaml'}")


def preview_labels(dataset_dir: Path, num_samples: int = 5):
    """Preview random labeled samples from the dataset."""
    print(f"\n[PREVIEW] Showing {num_samples} random samples...")

    train_images = list((dataset_dir / "train" / "images").glob("*.jpg"))
    if not train_images:
        print("  No images found!")
        return

    samples = random.sample(train_images, min(num_samples, len(train_images)))

    for img_path in samples:
        label_path = dataset_dir / "train" / "labels" / (img_path.stem + ".txt")

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = map(float, parts)
                        # Convert back to pixels
                        cx = int(xc * w)
                        cy = int(yc * h)
                        box_w = int(bw * w)
                        box_h = int(bh * h)

                        # Draw
                        x1, y1 = cx - box_w // 2, cy - box_h // 2
                        x2, y2 = cx + box_w // 2, cy + box_h // 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        cv2.putText(frame, img_path.name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Preview", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label balls from video using HSV detection"
    )
    parser.add_argument(
        "videos", nargs="+",
        help="Video file(s) to process"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="balls-dataset",
        help="Output dataset folder (default: balls-dataset)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=30,
        help="Extract every Nth frame (default: 30)"
    )
    parser.add_argument(
        "--split", type=float, default=0.85,
        help="Train/valid split ratio (default: 0.85)"
    )
    parser.add_argument(
        "--min-balls", type=int, default=1,
        help="Minimum balls required to include frame (default: 1)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config file path (default: frc_tracker_config.json)"
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Interactive review of each frame"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview existing dataset labels"
    )

    args = parser.parse_args()

    # Resolve output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / args.output

    if args.preview:
        preview_labels(output_dir)
        return

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    print("=" * 60)
    print("  FRC BALL AUTO-LABELER")
    print("=" * 60)
    print(f"  Videos: {len(args.videos)}")
    print(f"  Interval: Every {args.interval} frames")
    print(f"  Min balls: {args.min_balls}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Extract frames
    frames_with_detections = extract_frames_with_detections(
        args.videos,
        config,
        interval=args.interval,
        min_balls=args.min_balls,
        review=args.review,
    )

    if not frames_with_detections:
        print("\n  ERROR: No frames extracted!")
        return

    print(f"\n  Total frames extracted: {len(frames_with_detections)}")

    # Create dataset
    create_dataset(frames_with_detections, output_dir, train_split=args.split)

    # Offer preview
    print("\n  Run preview? (y/n): ", end="")
    if input().strip().lower() == 'y':
        preview_labels(output_dir)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"\n  Next steps:")
    print(f"  1. Review/correct labels in Roboflow or labelImg")
    print(f"  2. Train model: python 09_train_ball_model.py --data {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
