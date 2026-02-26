#!/usr/bin/env python3
"""
FRC Ball Tracker - Installation Verification Script

Run this after install_frc_tracker.bat to verify all dependencies are working.
Usage: python verify_install.py
"""

import sys
import subprocess
from pathlib import Path

def check_import(module_name, display_name=None, required=True):
    """Try to import a module and report status."""
    display_name = display_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  [OK] {display_name}: {version}")
        return True
    except ImportError as e:
        status = "MISSING (required)" if required else "not installed (optional)"
        print(f"  [{'!!' if required else '--'}] {display_name}: {status}")
        return False

def check_cuda():
    """Check PyTorch CUDA availability - detects CPU-only builds."""
    try:
        import torch
        version = torch.__version__
        is_cuda_build = '+cu' in version

        if not is_cuda_build:
            print(f"  [!!] PyTorch {version} is a CPU-ONLY build!")
            print(f"       This is the #1 cause of slow YOLO performance.")
            print(f"       Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
            return False

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  [OK] CUDA build: {version}")
            print(f"  [OK] CUDA runtime: {cuda_version}")
            print(f"  [OK] GPU device: {device_name}")
            # Actually test GPU compute
            try:
                t = torch.randn(100, 100, device='cuda')
                _ = t @ t
                torch.cuda.synchronize()
                print(f"  [OK] GPU compute test: PASS")
            except Exception as e:
                print(f"  [!!] GPU compute test FAILED: {e}")
                return False
            return True
        else:
            print(f"  [!!] PyTorch {version} has CUDA support but CUDA runtime not available")
            print(f"       Check NVIDIA driver installation")
            return False
    except ImportError:
        print("  [!!] PyTorch not installed")
        return False

def check_cupy():
    """Check CuPy GPU acceleration."""
    try:
        import cupy as cp
        # Try a simple GPU operation
        arr = cp.array([1, 2, 3])
        _ = arr.sum()
        print(f"  [OK] CuPy GPU acceleration: {cp.__version__}")
        return True
    except ImportError:
        print("  [--] CuPy not installed (optional - CPU fallback will be used)")
        return False
    except Exception as e:
        print(f"  [--] CuPy installed but GPU error: {e}")
        return False

def check_ffmpeg():
    """Check FFmpeg installation and NVENC support."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            print(f"  [OK] FFmpeg: {version_line}")

            # Check for NVENC
            encoders = subprocess.run(
                ['ffmpeg', '-encoders'],
                capture_output=True, text=True, timeout=10
            )
            if 'h264_nvenc' in encoders.stdout:
                print("  [OK] NVENC encoder available")
                return True
            else:
                print("  [--] NVENC encoder not found (will use CPU encoding)")
                return True  # FFmpeg works, just no NVENC
        else:
            print("  [!!] FFmpeg found but returned error")
            return False
    except FileNotFoundError:
        print("  [!!] FFmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("  [!!] FFmpeg check timed out")
        return False

def check_nvidia_smi():
    """Check NVIDIA driver installation."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"  [OK] NVIDIA GPU: {gpu_info}")
            return True
        else:
            print("  [!!] nvidia-smi returned error")
            return False
    except FileNotFoundError:
        print("  [!!] nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except subprocess.TimeoutExpired:
        print("  [!!] nvidia-smi check timed out")
        return False

def check_model_files():
    """Check for trained model files."""
    script_dir = Path(__file__).parent
    models_dir = script_dir / 'models'

    models = [
        ('bumper_detector.pt', 'Robot bumper detection'),
        ('ball_detector.pt', 'Ball detection (YOLO)'),
    ]

    found_any = False
    for model_file, description in models:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {model_file}: {size_mb:.1f} MB ({description})")
            found_any = True
        else:
            print(f"  [--] {model_file}: not found ({description})")

    if not found_any:
        print("       No trained models found. Train with 08_train_bumper_model.py")

    return found_any

def main():
    print("=" * 60)
    print("FRC Ball Tracker - Installation Verification")
    print("=" * 60)
    print()

    all_ok = True

    # Python version
    print("Python Version:")
    py_version = sys.version.split()[0]
    major, minor = map(int, py_version.split('.')[:2])
    if major >= 3 and minor >= 9:
        print(f"  [OK] Python {py_version}")
    else:
        print(f"  [!!] Python {py_version} - Python 3.9+ recommended")
        all_ok = False
    print()

    # System tools
    print("System Tools:")
    check_nvidia_smi()
    check_ffmpeg()
    print()

    # Core packages
    print("Core Python Packages:")
    all_ok &= check_import('numpy')
    all_ok &= check_import('cv2', 'opencv-python')
    all_ok &= check_import('scipy')
    print()

    # YOLO / ML packages
    print("Machine Learning:")
    all_ok &= check_import('torch', 'PyTorch')
    check_cuda()
    all_ok &= check_import('ultralytics', 'Ultralytics YOLO')
    print()

    # GPU acceleration
    print("GPU Acceleration:")
    check_cupy()
    print()

    # Optional packages
    print("Optional Packages:")
    check_import('easyocr', required=False)
    check_import('pandas', required=False)
    check_import('matplotlib', required=False)
    print()

    # Model files
    print("Trained Models:")
    check_model_files()
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("All required dependencies are installed!")
        print()
        print("Quick start:")
        print("  python 01_hsv_tuning.py path/to/video.mkv")
        print()
        print("Press 'h' in any script for help.")
    else:
        print("Some required dependencies are missing.")
        print("Run install_frc_tracker.bat or install manually.")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
