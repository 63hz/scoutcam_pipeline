@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo FRC Ball Tracker - Windows Setup Script
echo ============================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Not running as administrator.
    echo Some installations may require admin rights.
    echo.
)

REM ========================================
REM 1. Check Python
REM ========================================
echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo.

REM ========================================
REM 2. Check NVIDIA GPU
REM ========================================
echo [2/6] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: nvidia-smi not found
    echo Make sure NVIDIA drivers are installed
    echo Download from: https://www.nvidia.com/Download/index.aspx
) else (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)
echo.

REM ========================================
REM 3. Check/Install FFmpeg
REM ========================================
echo [3/6] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorLevel% neq 0 (
    echo FFmpeg not found. Attempting install via winget...
    winget install FFmpeg.FFmpeg --accept-source-agreements --accept-package-agreements
    if %errorLevel% neq 0 (
        echo.
        echo FFmpeg install failed. Manual installation required:
        echo 1. Download from https://www.gyan.dev/ffmpeg/builds/
        echo 2. Extract to C:\ffmpeg
        echo 3. Add C:\ffmpeg\bin to system PATH
        echo.
    )
) else (
    echo FFmpeg found:
    ffmpeg -version 2>&1 | findstr "ffmpeg version"
    echo.
    echo Checking NVENC support...
    ffmpeg -encoders 2>&1 | findstr h264_nvenc
    if %errorLevel% neq 0 (
        echo WARNING: h264_nvenc encoder not found
        echo NVENC may not be available - will fall back to CPU encoding
    )
)
echo.

REM ========================================
REM 4. Upgrade pip
REM ========================================
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM ========================================
REM 5. Install Python packages
REM ========================================
echo [5/6] Installing Python packages...
echo.

echo Installing core packages (numpy, opencv, scipy)...
pip install numpy opencv-python scipy

echo.
echo Installing YOLO (ultralytics)...
pip install ultralytics

echo.
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing GPU acceleration (CuPy for CUDA 12.x)...
pip install cupy-cuda12x

echo.
echo Installing optional packages (easyocr, pandas, matplotlib)...
pip install easyocr pandas matplotlib

echo.

REM ========================================
REM 6. Verify installation
REM ========================================
echo [6/6] Verifying installation...
echo.

echo Testing core imports...
python -c "import numpy; import cv2; import scipy; print('Core packages: OK')"

echo.
echo Testing YOLO...
python -c "from ultralytics import YOLO; print('Ultralytics YOLO: OK')"

echo.
echo Testing PyTorch CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Testing CuPy...
python -c "import cupy; print(f'CuPy: {cupy.__version__}')" 2>nul
if %errorLevel% neq 0 (
    echo CuPy: Not available (GPU morphology will use CPU fallback)
)

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Next steps:
echo 1. cd to your project directory
echo 2. Run: python 01_hsv_tuning.py your_video.mkv
echo 3. Press 'h' for help in any script
echo.
pause
