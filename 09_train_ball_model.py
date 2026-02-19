# -*- coding: utf-8 -*-
"""
09 - Train YOLO Model for FRC Ball Detection
=============================================
Training script to create a custom YOLO model that detects yellow FRC balls
(fuel). Can use auto-labeled data from HSV detection or manually labeled data.

REQUIREMENTS:
    pip install ultralytics
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

DATASET:
    balls-dataset/ folder containing:
    - data.yaml (class definitions and paths)
    - train/images/ and train/labels/
    - valid/images/ and valid/labels/
    - test/images/ and test/labels/ (optional)

    Create dataset with: python utils/auto_label_balls.py video.mkv

USAGE:
    python 09_train_ball_model.py [options]

OPTIONS:
    --data PATH     Dataset folder (default: balls-dataset)
    --epochs N      Number of training epochs (default: 100)
    --batch N       Batch size (default: 16, adjust for VRAM)
    --imgsz N       Image size (default: 960 for better small ball detection)
    --model NAME    Base model (default: yolov8m.pt)
    --resume        Resume from last checkpoint
    --validate      Run validation only on existing model

OUTPUT:
    models/ball_detector.pt  - Trained model weights
    runs/train/ball_v*/      - Training logs and metrics

Author: Clay / Claude sandbox
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# Check for required packages
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("\n" + "=" * 60)
    print("  MISSING DEPENDENCIES")
    print("=" * 60)
    print("  Install required packages:")
    print("    pip install ultralytics")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60 + "\n")
    sys.exit(1)


# Project paths
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASET = "balls-dataset"
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_MODEL = MODELS_DIR / "ball_detector.pt"


def check_cuda():
    """Check CUDA availability and print GPU info."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("  WARNING: CUDA NOT AVAILABLE")
        print("=" * 60)
        print("  Training will use CPU and be VERY slow.")
        print("  For fast training, install PyTorch with CUDA:")
        print("    pip uninstall torch torchvision")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60 + "\n")
        return "cpu"
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {device_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
        return "cuda"


def check_dataset(dataset_dir):
    """Verify the dataset exists and is properly formatted."""
    data_yaml = dataset_dir / "data.yaml"

    if not dataset_dir.exists():
        print(f"\n  ERROR: Dataset not found at {dataset_dir}")
        print(f"\n  Create a dataset with:")
        print(f"    python utils/auto_label_balls.py video.mkv --output {dataset_dir}")
        return False

    if not data_yaml.exists():
        print(f"\n  ERROR: data.yaml not found at {data_yaml}")
        return False

    # Check for train/valid folders
    train_dir = dataset_dir / "train" / "images"
    valid_dir = dataset_dir / "valid" / "images"

    if not train_dir.exists():
        print(f"\n  ERROR: Training images not found at {train_dir}")
        return False

    if not valid_dir.exists():
        print(f"\n  ERROR: Validation images not found at {valid_dir}")
        return False

    # Count images
    train_count = len(list(train_dir.glob("*")))
    valid_count = len(list(valid_dir.glob("*")))

    # Check for test folder (optional)
    test_dir = dataset_dir / "test" / "images"
    test_count = len(list(test_dir.glob("*"))) if test_dir.exists() else 0

    print(f"\n  Dataset: {dataset_dir.name}")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {valid_count}")
    if test_count > 0:
        print(f"  Test images: {test_count}")

    # Read and display class info
    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    if "names" in data_cfg:
        print(f"  Classes: {data_cfg['names']}")

    return True


def train_model(args, dataset_dir):
    """Train the YOLO model on the ball dataset."""
    device = check_cuda()
    data_yaml = dataset_dir / "data.yaml"

    if not check_dataset(dataset_dir):
        return False

    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60)
    print(f"  Dataset: {dataset_dir.name}")
    print(f"  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {device}")
    print("=" * 60 + "\n")

    # Load base model
    model = YOLO(args.model)

    # Train with settings optimized for small object detection
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(SCRIPT_DIR / "runs" / "train"),
        name="ball",
        patience=20,           # Early stopping patience
        save=True,             # Save checkpoints
        plots=True,            # Generate training plots
        verbose=True,
        exist_ok=True,         # Allow overwriting existing run
        # Small object detection optimizations
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # Mixup augmentation
        copy_paste=0.1,        # Copy-paste augmentation for small objects
        scale=0.5,             # Scale augmentation (smaller to preserve ball size)
        degrees=10.0,          # Rotation augmentation
        translate=0.1,         # Translation augmentation
    )

    # Find the best model from training
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"

    if best_model_path.exists():
        # Copy best model to models directory
        shutil.copy(best_model_path, OUTPUT_MODEL)
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best model saved to: {OUTPUT_MODEL}")
        print(f"  Training logs: {results.save_dir}")
        print("=" * 60 + "\n")
        return True
    else:
        print("\n  ERROR: Training completed but best.pt not found")
        return False


def validate_model(model_path, dataset_dir):
    """Run validation on an existing model."""
    data_yaml = dataset_dir / "data.yaml"

    if not Path(model_path).exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        return False

    if not check_dataset(dataset_dir):
        return False

    print("\n" + "=" * 60)
    print("  RUNNING VALIDATION")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset_dir.name}")

    model = YOLO(model_path)
    metrics = model.val(
        data=str(data_yaml),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("  Per-class AP50:")
    for i, name in enumerate(model.names.values()):
        print(f"    {name}: {metrics.box.ap50[i]:.4f}")
    print("=" * 60 + "\n")

    return True


def benchmark_model(model_path):
    """Run inference benchmark on the model."""
    import numpy as np
    import time

    if not Path(model_path).exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        return False

    print("\n" + "=" * 60)
    print("  INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"  Model: {model_path}")

    model = YOLO(model_path)

    # Warmup
    print("  Warming up...")
    dummy = np.zeros((960, 960, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, verbose=False)

    # Benchmark
    print("  Running 100 iterations...")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"\n  Inference time: {times.mean()*1000:.1f}ms (avg)")
    print(f"  FPS: {1/times.mean():.1f}")
    print(f"  P50: {np.percentile(times, 50)*1000:.1f}ms")
    print(f"  P95: {np.percentile(times, 95)*1000:.1f}ms")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for FRC ball detection"
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATASET,
        help=f"Dataset folder name (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, reduce if OOM)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=960,
        help="Image size (default: 960, larger for small object detection)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="Base model (default: yolov8m.pt - medium size for balance)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation only on existing model"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference speed benchmark"
    )

    args = parser.parse_args()

    # Resolve dataset directory
    dataset_dir = SCRIPT_DIR / args.data

    if args.validate:
        model_path = args.model if args.model != "yolov8m.pt" else str(OUTPUT_MODEL)
        validate_model(model_path, dataset_dir)
    elif args.benchmark:
        model_path = args.model if args.model != "yolov8m.pt" else str(OUTPUT_MODEL)
        benchmark_model(model_path)
    elif args.resume:
        # Find last checkpoint
        last_pt = SCRIPT_DIR / "runs" / "train" / "ball" / "weights" / "last.pt"
        if last_pt.exists():
            print(f"\n  Resuming from: {last_pt}")
            model = YOLO(str(last_pt))
            model.train(resume=True)
        else:
            print(f"\n  ERROR: No checkpoint found at {last_pt}")
            print("  Run training first without --resume")
    else:
        train_model(args, dataset_dir)


if __name__ == "__main__":
    main()
