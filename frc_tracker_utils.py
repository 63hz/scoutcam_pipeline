# -*- coding: utf-8 -*-
"""
FRC Ball Tracker - Shared Utilities
====================================
Core components shared across all sandbox scripts.

Includes:
    - Config management (load/save JSON)
    - GPU acceleration (CuPy/CUDA when available)
    - BallDetector: HSV-based yellow ball detection with filtering
      and per-blob watershed splitting for clustered balls
    - CentroidTracker: Multi-object tracking via centroid association
    - Drawing helpers for annotated output
    - Video utilities

Author: Clay / Claude sandbox
"""

import json
import math
import os
import time
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np

# Optional scipy for Hungarian algorithm (OC-SORT tracker)
_SCIPY_AVAILABLE = False
try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# GPU DETECTION (runs at import time)
# ============================================================================

_GPU_AVAILABLE = False
_GPU_BACKEND = None
cp = None            # CuPy module, set if available
cp_ndimage = None    # CuPy scipy.ndimage

try:
    import cupy as _cp
    from cupyx.scipy import ndimage as _cp_ndimage
    if _cp.cuda.runtime.getDeviceCount() > 0:
        cp = _cp
        cp_ndimage = _cp_ndimage
        _GPU_AVAILABLE = True
        _GPU_BACKEND = "cupy"
        _dev = cp.cuda.Device(0)
        _name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        _vram = _dev.mem_info[1] / 1e9
        print(f"[GPU] CuPy: {_name}, {_vram:.1f} GB VRAM")
except ImportError:
    pass
except Exception as e:
    print(f"[GPU] CuPy detected but error: {e}")

if not _GPU_AVAILABLE:
    print("[GPU] No GPU backend. Install CuPy for 3090 acceleration:")
    print("[GPU]   pip install cupy-cuda12x    (for CUDA 12)")
    print("[GPU]   pip install cupy-cuda11x    (for CUDA 11)")


def get_gpu_status():
    """
    Return detailed GPU status for diagnostics.

    Returns dict with:
        - available: bool - whether GPU acceleration is available
        - backend: str - GPU backend name ("cupy" or None)
        - cuda_version: str - CUDA runtime version
        - driver_version: str - CUDA driver version
        - device_name: str - GPU device name
        - vram_total_gb: float - total VRAM in GB
        - vram_free_gb: float - free VRAM in GB
        - kernel_compiled: bool - whether custom CUDA kernel is available
    """
    status = {
        "available": _GPU_AVAILABLE,
        "backend": _GPU_BACKEND,
        "cuda_version": None,
        "driver_version": None,
        "device_name": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "kernel_compiled": _hsv_inrange_kernel is not None,
        "cupy_version": None,
    }

    if _GPU_AVAILABLE and cp is not None:
        try:
            status["cupy_version"] = cp.__version__

            # CUDA versions
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            status["cuda_version"] = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"

            driver_ver = cp.cuda.runtime.driverGetVersion()
            status["driver_version"] = f"{driver_ver // 1000}.{(driver_ver % 1000) // 10}"

            # Device info
            dev = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            status["device_name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

            # VRAM
            mem_free, mem_total = dev.mem_info
            status["vram_total_gb"] = round(mem_total / 1e9, 2)
            status["vram_free_gb"] = round(mem_free / 1e9, 2)

        except Exception as e:
            status["error"] = str(e)

    return status


def print_gpu_status():
    """Print formatted GPU status to console."""
    status = get_gpu_status()

    print("\n" + "=" * 60)
    print("  GPU DIAGNOSTICS")
    print("=" * 60)

    if not status["available"]:
        print("  Status: NOT AVAILABLE")
        print("  Reason: CuPy not installed or no CUDA device found")
        print("  Install: pip install cupy-cuda12x")
    else:
        print(f"  Status: AVAILABLE ({status['backend']})")
        print(f"  Device: {status['device_name']}")
        print(f"  CUDA Runtime: {status['cuda_version']}")
        print(f"  CUDA Driver: {status['driver_version']}")
        print(f"  CuPy Version: {status['cupy_version']}")
        print(f"  VRAM: {status['vram_free_gb']:.1f} / {status['vram_total_gb']:.1f} GB free")
        print(f"  Custom Kernel: {'YES (fused HSV)' if status['kernel_compiled'] else 'NO (array ops fallback)'}")

        if status.get("error"):
            print(f"  Warning: {status['error']}")

    print("=" * 60 + "\n")


# Pre-compiled CUDA kernel: fuses BGR->HSV + inRange into one pass.
# Touches each pixel exactly once instead of three separate operations,
# which is a huge win on memory-bandwidth-bound GPU workloads.
_HSV_INRANGE_KERNEL_SRC = r"""
extern "C" __global__
void bgr_hsv_inrange(
    const unsigned char* __restrict__ bgr,
    unsigned char* __restrict__ mask,
    const int total,
    const int h_lo, const int h_hi,
    const int s_lo, const int s_hi,
    const int v_lo, const int v_hi
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int base = idx * 3;
    float b = (float)bgr[base];
    float g = (float)bgr[base + 1];
    float r = (float)bgr[base + 2];

    float mx = fmaxf(r, fmaxf(g, b));
    float mn = fminf(r, fminf(g, b));
    float d  = mx - mn;

    // V channel (0-255)
    int v = (int)mx;

    // S channel (0-255)
    int s = (mx > 0.0f) ? (int)(d * 255.0f / mx) : 0;

    // H channel (0-179, OpenCV convention)
    int h = 0;
    if (d > 0.0f) {
        if      (mx == r) h = (int)(30.0f * (g - b) / d);
        else if (mx == g) h = (int)(30.0f * (b - r) / d) + 60;
        else              h = (int)(30.0f * (r - g) / d) + 120;
        if (h < 0) h += 180;
    }

    mask[idx] = (h >= h_lo && h <= h_hi &&
                 s >= s_lo && s <= s_hi &&
                 v >= v_lo && v <= v_hi) ? 255 : 0;
}
"""

_hsv_inrange_kernel = None
if _GPU_BACKEND == "cupy":
    try:
        _hsv_inrange_kernel = cp.RawKernel(_HSV_INRANGE_KERNEL_SRC,
                                            "bgr_hsv_inrange")
        print("[GPU] Fused CUDA kernel compiled OK")
    except Exception:
        # NVRTC not available (common on Windows/conda) — array-ops fallback
        # is still GPU-accelerated, just ~30% slower than the fused kernel
        print("[GPU] Custom kernel unavailable (NVRTC), using CuPy array ops")


# ============================================================================
# CONFIG MANAGEMENT
# ============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "frc_tracker_config.json"


def load_config(path=None):
    """Load configuration from JSON file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_config(config, path=None):
    """Save configuration to JSON file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[CONFIG] Saved to {path}")


# ============================================================================
# VIDEO UTILITIES
# ============================================================================

def open_video(path):
    """Open video file and print info. Returns (cap, info_dict)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    info = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0

    print(f"[VIDEO] {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    print(f"[VIDEO] {info['frame_count']} frames, {info['duration_sec']:.1f} sec")
    return cap, info


def apply_roi(frame, roi_cfg):
    """Crop frame to ROI defined in config."""
    x, y, w, h = roi_cfg["x"], roi_cfg["y"], roi_cfg["w"], roi_cfg["h"]
    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw))
    y = max(0, min(y, fh))
    w = min(w, fw - x)
    h = min(h, fh - y)
    return frame[y:y+h, x:x+w]


def select_roi_interactive(video_path):
    """Open first frame and let user draw ROI rectangle."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Cannot read first frame")

    print("[ROI] Draw rectangle on frame, then press ENTER or SPACE.")
    r = cv2.selectROI("Select ROI - ENTER to confirm", frame, fromCenter=False)
    cv2.destroyWindow("Select ROI - ENTER to confirm")

    if r[2] == 0 or r[3] == 0:
        print("[ROI] No ROI selected, using full frame.")
        h, w = frame.shape[:2]
        return {"x": 0, "y": 0, "w": w, "h": h}

    roi = {"x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])}
    print(f"[ROI] Selected: {roi}")
    return roi


# ============================================================================
# BALL DETECTOR
# ============================================================================

class BallDetector:
    """
    HSV-based yellow ball detector.

    Features:
        - HSV thresholding + morphological cleanup
        - Contour filtering (area, circularity, aspect ratio)
        - Per-blob watershed splitting for clustered/touching balls
        - GPU acceleration via CuPy when available

    Usage:
        detector = BallDetector(config)
        detections = detector.detect(frame)
    """

    def __init__(self, config):
        self.use_gpu = False
        self._debug_dist = None
        self._debug_markers = None
        self._debug_split_count = 0
        self._debug_nms_candidates = []  # NMS candidate peaks before suppression
        self._debug_nms_accepted = []     # NMS accepted peaks after suppression

        # Timing diagnostics
        self._timing_enabled = True
        self._timing_interval = 100  # Print summary every N frames
        self._frame_count = 0
        self._timing_mask_total = 0.0
        self._timing_detect_total = 0.0
        self._timing_split_total = 0.0
        self._last_mask_time = 0.0
        self._last_detect_time = 0.0
        self._last_split_time = 0.0

        self.update_config(config)

    def update_config(self, config):
        """Update all detector parameters from config dict."""
        # HSV range
        hsv = config["hsv_yellow"]
        self.hsv_lower = np.array([hsv["h_low"], hsv["s_low"], hsv["v_low"]])
        self.hsv_upper = np.array([hsv["h_high"], hsv["s_high"], hsv["v_high"]])

        # Morphology kernels (force odd sizes >= 3)
        morph = config["morphology"]
        ok = max(3, morph["open_kernel"]) | 1
        ck = max(3, morph["close_kernel"]) | 1
        dk = max(3, morph["dilate_kernel"]) | 1
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        self.dilate_iters = morph["dilate_iterations"]
        self._open_k = ok
        self._close_k = ck
        self._dilate_k = dk

        # Contour filters
        cf = config["contour_filter"]
        self.min_area = cf["min_area"]
        self.max_area = cf["max_area"]
        self.min_circularity = cf["min_circularity"]
        self.min_aspect = cf["min_aspect_ratio"]
        self.max_aspect = cf["max_aspect_ratio"]

        # Watershed (per-blob splitting) - legacy, kept for fallback
        ws = config.get("watershed", {})
        self.use_watershed = ws.get("enabled", False)
        self.ws_peak_ratio = ws.get("peak_ratio", 0.45)
        self.ws_min_peak_dist = ws.get("min_peak_distance", 8)
        self.ws_area_mult = ws.get("area_multiplier", 1.8)

        # New cluster splitting config (NMS-based, preferred over watershed)
        split_cfg = config.get("cluster_splitting", {})
        self._split_config = {
            "enabled": split_cfg.get("enabled", False),
            "method": split_cfg.get("method", "nms"),
            "min_ball_radius": split_cfg.get("min_ball_radius", 12),
            "peak_threshold": split_cfg.get("peak_threshold", 0.7),
            "area_multiplier": split_cfg.get("area_multiplier", 1.5),
            "max_cluster_area": split_cfg.get("max_cluster_area", 2000),
        }

        # GPU
        gpu_cfg = config.get("gpu", {})
        new_gpu = gpu_cfg.get("enabled", False) and _GPU_AVAILABLE
        if hasattr(self, 'use_gpu') and new_gpu != self.use_gpu:
            # Reset diagnostic flags so next get_mask prints the new state
            for attr in ('_gpu_confirmed', '_cpu_confirmed'):
                if hasattr(self, attr):
                    delattr(self, attr)
        self.use_gpu = new_gpu

    # ------------------------------------------------------------------ mask
    def get_mask(self, frame):
        """Generate binary mask via HSV threshold + morphology."""
        t0 = time.perf_counter()

        if self.use_gpu and _GPU_BACKEND == "cupy":
            if not hasattr(self, '_gpu_confirmed'):
                print(f"[GPU] GPU path ACTIVE — processing on GPU")
                self._gpu_confirmed = True
            mask = self._mask_gpu(frame)
        else:
            if not hasattr(self, '_cpu_confirmed'):
                print(f"[GPU] CPU path active (use_gpu={self.use_gpu}, "
                      f"backend={_GPU_BACKEND}, available={_GPU_AVAILABLE})")
                self._cpu_confirmed = True
            mask = self._mask_cpu(frame)

        self._last_mask_time = (time.perf_counter() - t0) * 1000  # ms
        self._timing_mask_total += self._last_mask_time
        return mask

    def _mask_cpu(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        if self.dilate_iters > 0:
            mask = cv2.dilate(mask, self.dilate_kernel,
                              iterations=self.dilate_iters)
        return mask

    def _mask_gpu(self, frame):
        """GPU path: CuPy array operations + morphology."""
        h, w = frame.shape[:2]
        n = h * w

        gpu_bgr = cp.asarray(frame)

        # Try fused kernel first, fall back to array ops if it fails
        gpu_mask_2d = None
        if _hsv_inrange_kernel is not None:
            try:
                gpu_mask = cp.zeros(n, dtype=cp.uint8)
                block = 256
                grid = (n + block - 1) // block
                _hsv_inrange_kernel(
                    (grid,), (block,),
                    (gpu_bgr.ravel(), gpu_mask,
                     np.int32(n),
                     np.int32(self.hsv_lower[0]), np.int32(self.hsv_upper[0]),
                     np.int32(self.hsv_lower[1]), np.int32(self.hsv_upper[1]),
                     np.int32(self.hsv_lower[2]), np.int32(self.hsv_upper[2]))
                )
                gpu_mask_2d = gpu_mask.reshape((h, w))
            except Exception:
                # NVRTC or kernel launch failed — fall back silently
                gpu_mask_2d = None

        if gpu_mask_2d is None:
            gpu_mask_2d = self._mask_gpu_fallback(gpu_bgr)

        # --- Step 2: transfer to CPU and do morphology with OpenCV ---
        # OpenCV morphology is highly optimized and avoids CuPy kernel compilation
        # issues with newer CUDA versions (12.9+)
        mask = cp.asnumpy(gpu_mask_2d)

        # Morphological operations on CPU (fast with OpenCV)
        se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (self._open_k, self._open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_open)

        se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (self._close_k, self._close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_close)

        if self.dilate_iters > 0:
            se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (self._dilate_k, self._dilate_k))
            mask = cv2.dilate(mask, se_dilate, iterations=self.dilate_iters)

        return mask

    def _mask_gpu_fallback(self, gpu_bgr):
        """CuPy array-op HSV+threshold (no custom kernel)."""
        f = gpu_bgr.astype(cp.float32)
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        mx = cp.maximum(r, cp.maximum(g, b))
        mn = cp.minimum(r, cp.minimum(g, b))
        d = mx - mn

        v = mx
        s = cp.where(mx > 0, d * 255.0 / mx, 0.0)

        h_ch = cp.zeros_like(mx)
        pos = d > 0
        mr = pos & (mx == r)
        mg = pos & (mx == g) & ~mr
        mb = pos & ~mr & ~mg
        h_ch[mr] = 30.0 * (g[mr] - b[mr]) / d[mr]
        h_ch[mg] = 30.0 * (b[mg] - r[mg]) / d[mg] + 60.0
        h_ch[mb] = 30.0 * (r[mb] - g[mb]) / d[mb] + 120.0
        h_ch = h_ch % 180

        ok = ((h_ch >= self.hsv_lower[0]) & (h_ch <= self.hsv_upper[0]) &
              (s >= self.hsv_lower[1]) & (s <= self.hsv_upper[1]) &
              (v >= self.hsv_lower[2]) & (v <= self.hsv_upper[2]))
        return ok.astype(cp.uint8) * 255

    # -------------------------------------------------------------- detect
    def detect(self, frame):
        """
        Detect yellow balls. Returns list of detection dicts.

        When cluster splitting is enabled, blobs larger than expected single-ball
        area are split using NMS-based peak detection on distance transform.
        Falls back to watershed if configured.
        """
        t0 = time.perf_counter()

        mask = self.get_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Clear debug state
        self._debug_nms_candidates = []
        self._debug_nms_accepted = []

        # Determine splitting method
        split_cfg = getattr(self, '_split_config', None)
        use_splitting = split_cfg and split_cfg.get("enabled", False)
        split_method = split_cfg.get("method", "nms") if split_cfg else "nms"

        if not use_splitting and not self.use_watershed:
            self._debug_split_count = 0
            detections = [d for d in (self._eval(c) for c in contours) if d]
        elif use_splitting and split_method == "nms":
            detections = self._detect_with_nms_splitting(mask, contours)
        else:
            # Legacy watershed path
            detections = self._detect_watershed(mask, contours)

        # Timing
        self._last_detect_time = (time.perf_counter() - t0) * 1000  # ms
        self._timing_detect_total += self._last_detect_time
        self._frame_count += 1

        # Print summary every N frames
        if self._timing_enabled and self._frame_count % self._timing_interval == 0:
            self._print_timing_summary()

        return detections

    def _print_timing_summary(self):
        """Print timing summary every N frames."""
        n = self._timing_interval
        avg_mask = self._timing_mask_total / n
        avg_detect = self._timing_detect_total / n
        avg_split = self._timing_split_total / n if self._timing_split_total > 0 else 0

        path = "GPU" if self.use_gpu else "CPU"
        split_mode = "NMS" if getattr(self, '_split_config', {}).get("enabled") else (
            "Watershed" if self.use_watershed else "None"
        )

        print(f"[TIMING] {n} frames: mask={avg_mask:.1f}ms, detect={avg_detect:.1f}ms, "
              f"split={avg_split:.1f}ms | path={path}, splitting={split_mode}")

        # Reset accumulators
        self._timing_mask_total = 0.0
        self._timing_detect_total = 0.0
        self._timing_split_total = 0.0

    def get_timing(self):
        """Get last frame timing in milliseconds."""
        return {
            "mask_ms": self._last_mask_time,
            "detect_ms": self._last_detect_time,
            "split_ms": self._last_split_time,
            "total_ms": self._last_detect_time,  # detect includes mask time
            "path": "GPU" if self.use_gpu else "CPU",
        }

    def _detect_watershed(self, mask, contours):
        """
        Per-blob watershed: only applies to oversized merged blobs.

        1. Compute median "normal" ball area from current frame
        2. Blobs <= threshold: evaluate normally
        3. Blobs > threshold: distance transform -> find local maxima
           -> seed watershed -> extract split contours
        """
        # Estimate single-ball area from median of normal-sized detections
        areas = [cv2.contourArea(c) for c in contours]
        normal = [a for a in areas if self.min_area <= a <= self.max_area]
        median_a = float(np.median(normal)) if normal else (self.min_area + self.max_area) / 2
        split_thresh = median_a * self.ws_area_mult

        # Estimated ball radius for peak spacing
        est_r = max(self.ws_min_peak_dist, int(math.sqrt(median_a / math.pi)))

        # Debug storage
        self._debug_dist = np.zeros(mask.shape, dtype=np.float32)
        self._debug_markers = np.zeros(mask.shape, dtype=np.int32)
        split_count = 0

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_area:
                continue

            # Normal-sized or impossibly huge -> standard path
            if area <= split_thresh or area > self.max_area * 10:
                d = self._eval(cnt)
                if d:
                    detections.append(d)
                continue

            # ---- Oversized blob: watershed split ----
            # Isolate this blob
            blob = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)

            # Distance transform (CPU -- only on small blob region)
            dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
            self._debug_dist = np.maximum(self._debug_dist, dist)

            if dist.max() == 0:
                continue

            # Local maxima via dilation comparison:
            # pixel is local max if it equals the max in its neighborhood
            pk = max(3, est_r) | 1
            dilated = cv2.dilate(
                dist,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
            )
            local_max = ((dist == dilated)
                         & (dist >= dist.max() * self.ws_peak_ratio)
                         & (blob > 0))

            n_seeds, seed_labels = cv2.connectedComponents(
                local_max.astype(np.uint8), connectivity=8
            )

            if n_seeds <= 2:
                # 0 or 1 peaks: can't split; try relaxed circularity
                d = self._eval(cnt, min_circ=0.15)
                if d:
                    detections.append(d)
                continue

            # Build watershed markers
            markers = np.zeros(mask.shape, dtype=np.int32)
            markers[blob == 0] = 1          # background
            expand_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for i in range(1, n_seeds):
                seed = (seed_labels == i).astype(np.uint8)
                seed = cv2.dilate(seed, expand_k, iterations=1)
                markers[seed > 0] = i + 1   # labels 2, 3, ...

            # Watershed
            cv2.watershed(cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR), markers)
            self._debug_markers = np.where(
                markers != 0, markers, self._debug_markers
            )

            # Extract each split region
            for i in range(2, n_seeds + 1):
                region = (markers == i).astype(np.uint8) * 255
                region = cv2.morphologyEx(
                    region, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                )
                rc, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                for c in rc:
                    d = self._eval(c)
                    if d:
                        d["was_split"] = True
                        detections.append(d)

            split_count += 1

        self._debug_split_count = split_count
        return detections

    # -------------------------------------------------------- NMS splitting
    def _detect_with_nms_splitting(self, mask, contours):
        """
        NMS-based cluster splitting: more robust than watershed for 3+ ball clusters.

        Algorithm:
        1. Compute median ball area to determine split threshold
        2. Normal blobs: evaluate directly
        3. Oversized blobs: distance transform + NMS peak finding
        """
        t0 = time.perf_counter()

        cfg = self._split_config
        min_ball_radius = cfg["min_ball_radius"]
        peak_thresh = cfg["peak_threshold"]
        area_mult = cfg["area_multiplier"]
        max_cluster_area = cfg["max_cluster_area"]

        # Estimate single-ball area from median of normal-sized detections
        areas = [cv2.contourArea(c) for c in contours]
        normal = [a for a in areas if self.min_area <= a <= self.max_area]
        median_a = float(np.median(normal)) if normal else (self.min_area + self.max_area) / 2
        split_thresh = median_a * area_mult

        # Estimated ball radius
        est_r = max(min_ball_radius, int(math.sqrt(median_a / math.pi)))

        # Debug storage
        self._debug_dist = np.zeros(mask.shape, dtype=np.float32)
        split_count = 0

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_area:
                continue

            # Normal-sized -> standard single-ball evaluation
            if area <= split_thresh:
                d = self._eval(cnt)
                if d:
                    detections.append(d)
                continue

            # Too large for splitting -> skip (likely noise or non-ball object)
            if area > max_cluster_area:
                continue

            # ---- Oversized blob: NMS-based splitting ----
            # Get bounding box for efficiency (only process blob ROI)
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Create blob mask (cropped to bounding box)
            blob_roi = np.zeros((bh, bw), dtype=np.uint8)
            shifted_cnt = cnt - np.array([bx, by])
            cv2.drawContours(blob_roi, [shifted_cnt], -1, 255, -1)

            # Distance transform on cropped blob
            dist_roi = cv2.distanceTransform(blob_roi, cv2.DIST_L2, 5)

            # Copy to full debug image
            self._debug_dist[by:by+bh, bx:bx+bw] = np.maximum(
                self._debug_dist[by:by+bh, bx:bx+bw], dist_roi
            )

            if dist_roi.max() == 0:
                continue

            # NMS peak finding
            centers = self._split_cluster_nms(
                dist_roi, blob_roi,
                min_radius=est_r,
                peak_thresh=peak_thresh
            )

            if len(centers) <= 1:
                # No split possible, try relaxed circularity
                d = self._eval(cnt, min_circ=0.15)
                if d:
                    detections.append(d)
                continue

            # Store candidates/accepted for debug visualization (offset to full image coords)
            for cx, cy, r, accepted in getattr(self, '_nms_debug_points', []):
                if accepted:
                    self._debug_nms_accepted.append((cx + bx, cy + by, r))
                else:
                    self._debug_nms_candidates.append((cx + bx, cy + by, r))

            # Create detections from accepted centers
            for cx, cy, r in centers:
                # Offset back to full image coordinates
                full_cx = cx + bx
                full_cy = cy + by

                # Create synthetic detection
                det = {
                    "cx": float(full_cx),
                    "cy": float(full_cy),
                    "radius": float(r),
                    "area": float(math.pi * r * r),
                    "circularity": 1.0,  # Assumed circular
                    "bbox": (int(full_cx - r), int(full_cy - r), int(2*r), int(2*r)),
                    "contour": cnt,  # Keep original contour for reference
                    "was_split": True,
                }
                detections.append(det)

            split_count += 1

        self._debug_split_count = split_count
        self._last_split_time = (time.perf_counter() - t0) * 1000
        self._timing_split_total += self._last_split_time

        return detections

    def _split_cluster_nms(self, dist, blob_mask, min_radius=12, peak_thresh=0.7):
        """
        NMS-based cluster splitting on distance transform.

        Algorithm:
        1. Find candidate peaks: pixels where dist > min_radius * peak_thresh
        2. Sort candidates by distance value (descending)
        3. NMS loop: accept peak if no accepted center within min_radius

        Args:
            dist: Distance transform of blob (cropped)
            blob_mask: Binary mask of blob (cropped)
            min_radius: Minimum expected ball radius
            peak_thresh: Fraction of min_radius to qualify as candidate

        Returns:
            List of (cx, cy, radius) tuples for detected ball centers
        """
        # Threshold for candidate peaks
        threshold = min_radius * peak_thresh

        # Find all candidate pixels (local maxima above threshold)
        # Use dilation to find local maxima
        kernel_size = max(3, min_radius // 2) | 1
        dilated = cv2.dilate(dist, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ))

        # Local max: pixel equals its dilated value and is above threshold
        local_max = (dist == dilated) & (dist >= threshold) & (blob_mask > 0)

        # Get candidate coordinates and their distance values
        candidates_y, candidates_x = np.where(local_max)
        if len(candidates_x) == 0:
            return []

        distances = dist[candidates_y, candidates_x]

        # Sort by distance value (descending) - stronger peaks first
        sorted_indices = np.argsort(-distances)
        candidates_x = candidates_x[sorted_indices]
        candidates_y = candidates_y[sorted_indices]
        distances = distances[sorted_indices]

        # NMS loop
        accepted = []
        suppression_radius = min_radius
        self._nms_debug_points = []

        for i in range(len(candidates_x)):
            cx, cy, d = int(candidates_x[i]), int(candidates_y[i]), float(distances[i])

            # Check if too close to any accepted center
            suppressed = False
            for ax, ay, ar in accepted:
                dist_to_accepted = math.sqrt((cx - ax)**2 + (cy - ay)**2)
                if dist_to_accepted < suppression_radius:
                    suppressed = True
                    break

            if not suppressed:
                # Accept this peak
                radius = max(min_radius * 0.5, d)  # Use distance value as radius estimate
                accepted.append((cx, cy, radius))
                self._nms_debug_points.append((cx, cy, radius, True))
            else:
                self._nms_debug_points.append((cx, cy, d, False))

        return accepted

    def _eval(self, cnt, min_circ=None):
        """Evaluate contour against filters. Returns dict or None."""
        area = cv2.contourArea(cnt)
        if area < self.min_area or area > self.max_area:
            return None

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            return None
        circ = 4 * math.pi * area / (peri * peri)
        if circ < (min_circ if min_circ is not None else self.min_circularity):
            return None

        x, y, w, h = cv2.boundingRect(cnt)
        asp = w / h if h > 0 else 0
        if asp < self.min_aspect or asp > self.max_aspect:
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        return {
            "cx": float(cx), "cy": float(cy),
            "radius": float(radius), "area": float(area),
            "circularity": float(circ), "bbox": (x, y, w, h),
            "contour": cnt, "was_split": False,
        }


# ============================================================================
# CENTROID TRACKER (Multi-Object)
# ============================================================================

class CentroidTracker:
    """
    Multi-object tracker using centroid distance association.
    Designed for high-count scenarios (hundreds of balls).
    """

    def __init__(self, max_disappeared=8, max_distance=80, trail_length=30):
        self.next_id = 0
        self.objects = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_length = trail_length

    def reset(self):
        self.next_id = 0
        self.objects = OrderedDict()

    def _register(self, detection):
        obj = TrackedObject(self.next_id, detection, self.trail_length)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def _deregister(self, obj_id):
        del self.objects[obj_id]

    def remove_objects(self, obj_ids):
        """
        Remove specific objects from tracking.

        Use this when balls enter goals or should otherwise be removed from
        active tracking (e.g., to prevent bounce-out balls from being double-counted).

        Args:
            obj_ids: Iterable of object IDs to remove
        """
        for oid in obj_ids:
            if oid in self.objects:
                del self.objects[oid]

    def update(self, detections):
        """Update tracker with new frame detections."""
        if len(detections) == 0:
            for oid in list(self.objects.keys()):
                self.objects[oid].disappeared += 1
                if self.objects[oid].disappeared > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_xy = np.array([(self.objects[oid].cx, self.objects[oid].cy)
                            for oid in obj_ids])
        det_xy = np.array([(d["cx"], d["cy"]) for d in detections])

        dist_mat = np.linalg.norm(
            obj_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :], axis=2
        )

        matched_o, matched_d = set(), set()
        flat = np.argsort(dist_mat, axis=None)
        rows, cols = np.unravel_index(flat, dist_mat.shape)

        for r, c in zip(rows, cols):
            if r in matched_o or c in matched_d:
                continue
            if dist_mat[r, c] > self.max_distance:
                break
            self.objects[obj_ids[r]].update(detections[c])
            matched_o.add(r)
            matched_d.add(c)

        for r in range(len(obj_ids)):
            if r not in matched_o:
                oid = obj_ids[r]
                self.objects[oid].disappeared += 1
                if self.objects[oid].disappeared > self.max_disappeared:
                    self._deregister(oid)

        for c in range(len(detections)):
            if c not in matched_d:
                self._register(detections[c])

        return self.objects


class TrackedObject:
    """A single tracked ball with position history."""

    def __init__(self, obj_id, detection, trail_length=30):
        self.id = obj_id
        self.cx = detection["cx"]
        self.cy = detection["cy"]
        self.radius = detection["radius"]
        self.area = detection["area"]
        self.bbox = detection["bbox"]
        self.disappeared = 0
        self.age = 0
        self.trail = deque(maxlen=trail_length)
        self.trail.append((self.cx, self.cy))
        self.vx = 0.0
        self.vy = 0.0
        self.shot_id = None
        self.robot_id = None
        self.shot_result = None

    def update(self, detection):
        old_cx, old_cy = self.cx, self.cy
        self.cx = detection["cx"]
        self.cy = detection["cy"]
        self.radius = detection["radius"]
        self.area = detection["area"]
        self.bbox = detection["bbox"]
        self.disappeared = 0
        self.age += 1
        self.trail.append((self.cx, self.cy))
        self.vx = self.cx - old_cx
        self.vy = self.cy - old_cy

    @property
    def speed(self):
        return math.sqrt(self.vx**2 + self.vy**2)

    @property
    def is_moving(self):
        return self.speed > 2.0


# ============================================================================
# ROBOT DETECTOR (Dynamic Robot Tracking)
# ============================================================================

# Optional OCR dependency
_EASYOCR_AVAILABLE = False
_ocr_reader = None

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    pass


class TrackedRobot:
    """
    A tracked robot with position, identity, and alliance info.

    Attributes:
        id: Unique tracking ID
        cx, cy: Current centroid position
        bbox: Bounding box (x, y, w, h)
        alliance: 'red' or 'blue'
        team_number: Team number (e.g., '2491') or None if unknown
        identity: Full identity string like 'red_2491' or 'blue_unknown'
        ocr_attempted: Whether OCR has been attempted
        manual_correction: Whether identity was manually corrected
    """

    def __init__(self, obj_id, detection, alliance):
        self.id = obj_id
        self.cx = detection["cx"]
        self.cy = detection["cy"]
        self.bbox = detection["bbox"]
        self.area = detection["area"]
        self.alliance = alliance
        self.team_number = None
        self.ocr_attempted = False
        self.manual_correction = False
        self.disappeared = 0
        self.age = 0

    @property
    def identity(self):
        """Return identity string like 'red_2491' or 'blue_unknown'."""
        num = self.team_number if self.team_number else "unknown"
        return f"{self.alliance}_{num}"

    def update(self, detection):
        """Update position from new detection."""
        self.cx = detection["cx"]
        self.cy = detection["cy"]
        self.bbox = detection["bbox"]
        self.area = detection["area"]
        self.disappeared = 0
        self.age += 1

    def set_team_number(self, team_number, manual=False):
        """Set team number (from OCR or manual correction)."""
        self.team_number = str(team_number) if team_number else None
        if manual:
            self.manual_correction = True


# ============================================================================
# OC-SORT TRACKER (Kalman + Hungarian Algorithm)
# ============================================================================

class KalmanBoxTracker:
    """
    Kalman filter for tracking a single bounding box.

    State vector: [x, y, w, h, vx, vy, vw, vh]
        - (x, y): center of bounding box
        - (w, h): width and height
        - (vx, vy, vw, vh): velocities

    This is a linear constant-velocity model. The Kalman filter predicts
    where the object will be in the next frame and corrects based on
    actual observations.
    """
    _count = 0  # Global ID counter

    def __init__(self, bbox, alliance=None):
        """
        Initialize tracker with first detection.

        Args:
            bbox: (x, y, w, h) bounding box
            alliance: 'red' or 'blue' for robot tracking
        """
        # 8 state dimensions, 4 measurement dimensions
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix (constant velocity model)
        # State: [x, y, w, h, vx, vy, vw, vh]
        # x_new = x + vx, y_new = y + vy, etc.
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix (we observe x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance (tuned for robot motion)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        self.kf.processNoiseCov[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        # Initial state covariance (high uncertainty)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        self.kf.errorCovPost[4:, 4:] *= 100.0  # Very uncertain about initial velocity

        # Initialize state with first observation
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        # Tracking metadata
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.alliance = alliance
        self.team_number = None
        self.manual_correction = False
        self.hits = 1  # Total observations
        self.age = 0  # Frames since creation
        self.time_since_update = 0  # Frames since last observation

        # OC-SORT: store recent observations for ORU
        self.observations = deque(maxlen=10)
        self.observations.append(bbox)
        self.last_observation = bbox

        # Velocity from last observation (for OCM)
        self.observed_velocity = np.array([0.0, 0.0], dtype=np.float32)

    def predict(self):
        """
        Advance state by one frame using Kalman prediction.

        Returns:
            Predicted bounding box (x, y, w, h)
        """
        # OC-SORT: Observation-Centric Momentum (OCM)
        # Blend predicted velocity with last observed velocity
        if self.time_since_update > 0:
            alpha = 0.5  # Momentum weight
            state = self.kf.statePost.flatten()
            # Use weighted average of predicted and observed velocity
            state[4] = alpha * self.observed_velocity[0] + (1 - alpha) * state[4]
            state[5] = alpha * self.observed_velocity[1] + (1 - alpha) * state[5]
            self.kf.statePost = state.reshape(-1, 1)

        state = self.kf.predict().flatten()
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(state)

    def update(self, bbox):
        """
        Correct state with new observation.

        Args:
            bbox: Observed bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        # Compute observed velocity before update
        if self.last_observation is not None:
            old_x, old_y, old_w, old_h = self.last_observation
            old_cx, old_cy = old_x + old_w / 2, old_y + old_h / 2
            self.observed_velocity = np.array([cx - old_cx, cy - old_cy], dtype=np.float32)

        # OC-SORT: Observation-Centric Re-update (ORU)
        # If recovering from occlusion, re-update with virtual observations
        if self.time_since_update > 1:
            self._apply_oru(bbox)

        self.kf.correct(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.observations.append(bbox)
        self.last_observation = bbox

    def _apply_oru(self, current_bbox):
        """
        Observation-Centric Re-update: Smooth trajectory after occlusion.

        Creates virtual observations between last observation and current,
        then re-runs Kalman updates to get a smoother trajectory estimate.
        """
        if len(self.observations) == 0 or self.time_since_update <= 1:
            return

        # Get last real observation
        last = self.observations[-1]
        x1, y1, w1, h1 = last
        x2, y2, w2, h2 = current_bbox

        # Linearly interpolate virtual observations for missing frames
        n_missing = self.time_since_update - 1
        for i in range(1, n_missing + 1):
            t = i / (n_missing + 1)
            vx = x1 + t * (x2 - x1)
            vy = y1 + t * (y2 - y1)
            vw = w1 + t * (w2 - w1)
            vh = h1 + t * (h2 - h1)

            # Apply virtual measurement with higher noise
            vcx, vcy = vx + vw / 2, vy + vh / 2
            measurement = np.array([[vcx], [vcy], [vw], [vh]], dtype=np.float32)

            # Temporarily increase measurement noise for virtual observations
            old_noise = self.kf.measurementNoiseCov.copy()
            self.kf.measurementNoiseCov *= 2.0
            self.kf.correct(measurement)
            self.kf.measurementNoiseCov = old_noise

    def get_state(self):
        """
        Get current bounding box estimate.

        Returns:
            Bounding box (x, y, w, h)
        """
        state = self.kf.statePost.flatten()
        return self._state_to_bbox(state)

    def _state_to_bbox(self, state):
        """Convert state vector to bounding box."""
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        # Ensure non-negative dimensions
        w = max(1, w)
        h = max(1, h)
        x = cx - w / 2
        y = cy - h / 2
        return (int(x), int(y), int(w), int(h))

    @property
    def identity(self):
        """Return identity string like 'red_2491' or 'blue_unknown'."""
        num = self.team_number if self.team_number else "unknown"
        return f"{self.alliance}_{num}"

    def set_team_number(self, team_number, manual=False):
        """Set team number (from OCR or manual correction)."""
        self.team_number = str(team_number) if team_number else None
        if manual:
            self.manual_correction = True


def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1, bbox2: Bounding boxes as (x, y, w, h)

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def iou_distance(bboxes1, bboxes2):
    """
    Compute IoU distance matrix between two sets of bounding boxes.

    Args:
        bboxes1: List of (x, y, w, h) bounding boxes
        bboxes2: List of (x, y, w, h) bounding boxes

    Returns:
        Distance matrix where dist[i,j] = 1 - IoU(bboxes1[i], bboxes2[j])
    """
    n, m = len(bboxes1), len(bboxes2)
    dist = np.ones((n, m), dtype=np.float32)

    for i, b1 in enumerate(bboxes1):
        for j, b2 in enumerate(bboxes2):
            dist[i, j] = 1.0 - iou(b1, b2)

    return dist


class OCSORTTracker:
    """
    OC-SORT multi-object tracker for robust robot tracking.

    Features:
        - Kalman filter for motion prediction through occlusions
        - Hungarian algorithm for globally optimal matching
        - Observation-Centric Re-update (ORU) for trajectory smoothing
        - Observation-Centric Momentum (OCM) to prevent velocity drift
        - IoU-based matching (better than centroid distance for boxes)

    Usage:
        tracker = OCSORTTracker(config)
        robots = tracker.update(detections)
    """

    def __init__(self, config):
        """
        Initialize OC-SORT tracker.

        Args:
            config: Full config dict (reads robot_tracking section)
        """
        self.config = config
        track_cfg = config.get("robot_tracking", {})

        self.max_age = track_cfg.get("max_age", 15)
        self.min_hits = track_cfg.get("min_hits", 3)
        self.iou_threshold = track_cfg.get("iou_threshold", 0.3)

        self.trackers = []  # List of KalmanBoxTracker
        self.frame_count = 0

        # Stats for debugging
        self._stats = {
            "matches": 0,
            "unmatched_tracks": 0,
            "unmatched_dets": 0,
            "id_switches": 0,
        }

    def update_config(self, config):
        """Update tracker parameters without resetting state."""
        self.config = config
        track_cfg = config.get("robot_tracking", {})
        self.max_age = track_cfg.get("max_age", 15)
        self.min_hits = track_cfg.get("min_hits", 3)
        self.iou_threshold = track_cfg.get("iou_threshold", 0.3)

    def update(self, detections):
        """
        Update tracker with new frame detections.

        Args:
            detections: List of detection dicts with 'bbox', 'alliance', etc.

        Returns:
            OrderedDict of track_id -> TrackedRobot-like objects
        """
        self.frame_count += 1

        # Convert detections to bbox list
        det_bboxes = [d["bbox"] for d in detections]
        det_alliances = [d.get("alliance", "unknown") for d in detections]

        # Predict new locations for all existing trackers
        predicted_bboxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_bboxes.append(pred)

        # Handle empty cases
        if len(detections) == 0:
            # No detections - just return confirmed tracks
            self._cleanup_dead_tracks()
            return self._get_output()

        if len(self.trackers) == 0:
            # No existing tracks - create new ones
            for i, det in enumerate(detections):
                self._create_tracker(det)
            return self._get_output()

        # Build IoU distance matrix
        cost_matrix = iou_distance(predicted_bboxes, det_bboxes)

        # Apply alliance constraint: set high cost for mismatched alliances
        for i, t in enumerate(self.trackers):
            for j, det_alliance in enumerate(det_alliances):
                if t.alliance != det_alliance:
                    cost_matrix[i, j] = 1.0  # Max distance = no match possible

        # Hungarian matching
        if _SCIPY_AVAILABLE:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            # Fallback to greedy matching if scipy not available
            row_indices, col_indices = self._greedy_matching(cost_matrix)

        # Process matches
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] > (1.0 - self.iou_threshold):
                # IoU too low - not a valid match
                continue
            self.trackers[r].update(det_bboxes[c])
            matched_tracks.add(r)
            matched_dets.add(c)
            self._stats["matches"] += 1

        # Handle unmatched tracks (went missing)
        for i in range(len(self.trackers)):
            if i not in matched_tracks:
                self._stats["unmatched_tracks"] += 1
                # Track already predicted, just increment time_since_update (done in predict)

        # Handle unmatched detections (new objects)
        for j in range(len(detections)):
            if j not in matched_dets:
                self._create_tracker(detections[j])
                self._stats["unmatched_dets"] += 1

        # Remove dead tracks
        self._cleanup_dead_tracks()

        return self._get_output()

    def _greedy_matching(self, cost_matrix):
        """Fallback greedy matching when scipy is not available."""
        n, m = cost_matrix.shape
        rows, cols = [], []
        matched_r, matched_c = set(), set()

        # Sort all pairs by cost
        flat = np.argsort(cost_matrix, axis=None)
        row_idx, col_idx = np.unravel_index(flat, cost_matrix.shape)

        for r, c in zip(row_idx, col_idx):
            if r in matched_r or c in matched_c:
                continue
            rows.append(r)
            cols.append(c)
            matched_r.add(r)
            matched_c.add(c)

        return rows, cols

    def _create_tracker(self, detection):
        """Create new Kalman tracker for detection."""
        bbox = detection["bbox"]
        alliance = detection.get("alliance", "unknown")
        tracker = KalmanBoxTracker(bbox, alliance=alliance)
        self.trackers.append(tracker)
        return tracker

    def _cleanup_dead_tracks(self):
        """Remove tracks that have been missing too long."""
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

    def _get_output(self):
        """
        Convert internal tracker list to TrackedRobot-like output dict.

        Only returns confirmed tracks (enough hits and recently updated).
        """
        output = OrderedDict()

        for t in self.trackers:
            # Only return confirmed tracks
            if t.hits >= self.min_hits and t.time_since_update == 0:
                # Create a TrackedRobot-like object
                robot = _TrackerOutputWrapper(t)
                output[t.id] = robot

        return output

    def get_tracker_by_id(self, track_id):
        """Get tracker by ID for manual corrections."""
        for t in self.trackers:
            if t.id == track_id:
                return t
        return None

    def apply_correction(self, track_id, team_number):
        """Apply team number correction to a tracker."""
        t = self.get_tracker_by_id(track_id)
        if t:
            t.set_team_number(team_number, manual=True)

    def get_stats(self):
        """Get tracking statistics for debugging."""
        return {
            "active_tracks": len(self.trackers),
            "confirmed_tracks": sum(1 for t in self.trackers if t.hits >= self.min_hits),
            "frame_count": self.frame_count,
            **self._stats
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            "matches": 0,
            "unmatched_tracks": 0,
            "unmatched_dets": 0,
            "id_switches": 0,
        }


class _TrackerOutputWrapper:
    """
    Wrapper to make KalmanBoxTracker compatible with TrackedRobot interface.

    This allows OCSORTTracker to be a drop-in replacement for the old
    centroid-based robot tracking.
    """

    def __init__(self, tracker):
        self._tracker = tracker

    @property
    def id(self):
        return self._tracker.id

    @property
    def cx(self):
        bbox = self._tracker.get_state()
        return bbox[0] + bbox[2] / 2

    @property
    def cy(self):
        bbox = self._tracker.get_state()
        return bbox[1] + bbox[3] / 2

    @property
    def bbox(self):
        return self._tracker.get_state()

    @property
    def area(self):
        bbox = self._tracker.get_state()
        return bbox[2] * bbox[3]

    @property
    def alliance(self):
        return self._tracker.alliance

    @property
    def team_number(self):
        return self._tracker.team_number

    @property
    def identity(self):
        return self._tracker.identity

    @property
    def manual_correction(self):
        return self._tracker.manual_correction

    @property
    def disappeared(self):
        return self._tracker.time_since_update

    @property
    def age(self):
        return self._tracker.age


class RobotDetector:
    """
    Detects and tracks FRC robots by bumper color.

    Features:
        - HSV color detection for red and blue bumpers
        - Multi-object tracking across frames
        - Optional OCR for reading team numbers from bumpers
        - Manual correction support
        - Position-based robot lookup for shot attribution

    Usage:
        detector = RobotDetector(config)
        robots = detector.detect_and_track(frame)
        robot_id = detector.get_robot_at_position(x, y)
    """

    def __init__(self, config):
        self.config = config
        robot_cfg = config.get("robot_detection", {})

        # Alliance to track
        self.track_alliance = robot_cfg.get("track_alliance", "both")

        # HSV ranges for bumper colors
        hsv_red = robot_cfg.get("hsv_red", {})
        self.hsv_red1_lower = np.array([
            hsv_red.get("h_low1", 0),
            hsv_red.get("s_low", 100),
            hsv_red.get("v_low", 80)
        ])
        self.hsv_red1_upper = np.array([
            hsv_red.get("h_high1", 10),
            hsv_red.get("s_high", 255),
            hsv_red.get("v_high", 255)
        ])
        self.hsv_red2_lower = np.array([
            hsv_red.get("h_low2", 170),
            hsv_red.get("s_low", 100),
            hsv_red.get("v_low", 80)
        ])
        self.hsv_red2_upper = np.array([
            hsv_red.get("h_high2", 180),
            hsv_red.get("s_high", 255),
            hsv_red.get("v_high", 255)
        ])

        hsv_blue = robot_cfg.get("hsv_blue", {})
        self.hsv_blue_lower = np.array([
            hsv_blue.get("h_low", 100),
            hsv_blue.get("s_low", 100),
            hsv_blue.get("v_low", 80)
        ])
        self.hsv_blue_upper = np.array([
            hsv_blue.get("h_high", 130),
            hsv_blue.get("s_high", 255),
            hsv_blue.get("v_high", 255)
        ])

        # Contour filters for bumpers
        self.min_area = robot_cfg.get("min_bumper_area", 500)
        self.max_area = robot_cfg.get("max_bumper_area", 15000)
        self.min_aspect = robot_cfg.get("min_aspect_ratio", 1.5)
        self.max_aspect = robot_cfg.get("max_aspect_ratio", 8.0)

        # OCR
        self.ocr_enabled = robot_cfg.get("ocr_enabled", True) and _EASYOCR_AVAILABLE

        # Determine tracker type from config
        track_type_cfg = config.get("robot_tracking", {})
        self.tracker_type = track_type_cfg.get("tracker", "centroid")

        # Legacy centroid tracking config (for fallback)
        track_cfg = robot_cfg.get("tracking", {})
        self.max_distance = track_cfg.get("max_distance", 150)
        self.max_disappeared = track_cfg.get("max_frames_missing", 15)

        # Initialize appropriate tracker
        if self.tracker_type == "ocsort":
            self._ocsort = OCSORTTracker(config)
            self.robots = OrderedDict()  # Will be updated by _ocsort
            self.next_id = 0  # Not used with ocsort
            print(f"[TRACKER] Using OC-SORT (Kalman + Hungarian)")
        else:
            self._ocsort = None
            self.next_id = 0
            self.robots = OrderedDict()  # id -> TrackedRobot
            print(f"[TRACKER] Using centroid tracker (legacy)")

        # Note: Corrections are session-only and not loaded from config.
        # Team numbers should only be assigned via OCR or manual correction
        # during the current session, as tracking IDs are ephemeral.

        # Debug
        self._debug_red_mask = None
        self._debug_blue_mask = None

    def update_config(self, config):
        """
        Update detection parameters from config WITHOUT resetting tracking state.

        This preserves robot identities and tracking across parameter changes,
        which is essential for interactive tuning.
        """
        self.config = config
        robot_cfg = config.get("robot_detection", {})

        # Alliance to track
        self.track_alliance = robot_cfg.get("track_alliance", "both")

        # HSV ranges for bumper colors
        hsv_red = robot_cfg.get("hsv_red", {})
        self.hsv_red1_lower = np.array([
            hsv_red.get("h_low1", 0),
            hsv_red.get("s_low", 100),
            hsv_red.get("v_low", 80)
        ])
        self.hsv_red1_upper = np.array([
            hsv_red.get("h_high1", 10),
            hsv_red.get("s_high", 255),
            hsv_red.get("v_high", 255)
        ])
        self.hsv_red2_lower = np.array([
            hsv_red.get("h_low2", 170),
            hsv_red.get("s_low", 100),
            hsv_red.get("v_low", 80)
        ])
        self.hsv_red2_upper = np.array([
            hsv_red.get("h_high2", 180),
            hsv_red.get("s_high", 255),
            hsv_red.get("v_high", 255)
        ])

        hsv_blue = robot_cfg.get("hsv_blue", {})
        self.hsv_blue_lower = np.array([
            hsv_blue.get("h_low", 100),
            hsv_blue.get("s_low", 100),
            hsv_blue.get("v_low", 80)
        ])
        self.hsv_blue_upper = np.array([
            hsv_blue.get("h_high", 130),
            hsv_blue.get("s_high", 255),
            hsv_blue.get("v_high", 255)
        ])

        # Contour filters for bumpers
        self.min_area = robot_cfg.get("min_bumper_area", 500)
        self.max_area = robot_cfg.get("max_bumper_area", 15000)
        self.min_aspect = robot_cfg.get("min_aspect_ratio", 1.5)
        self.max_aspect = robot_cfg.get("max_aspect_ratio", 8.0)

        # OCR
        self.ocr_enabled = robot_cfg.get("ocr_enabled", True) and _EASYOCR_AVAILABLE

        # Update OC-SORT tracker config if active
        if self._ocsort is not None:
            self._ocsort.update_config(config)

        # Legacy centroid tracking params (but NOT the tracking state itself)
        track_cfg = robot_cfg.get("tracking", {})
        self.max_distance = track_cfg.get("max_distance", 150)
        self.max_disappeared = track_cfg.get("max_frames_missing", 15)

    def detect_bumpers(self, frame):
        """
        Detect bumpers in frame by color.

        Returns:
            list: Detection dicts with 'cx', 'cy', 'bbox', 'area', 'alliance'
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []

        # Red bumpers (two HSV ranges since red wraps around 0/180)
        if self.track_alliance in ("both", "red"):
            mask1 = cv2.inRange(hsv, self.hsv_red1_lower, self.hsv_red1_upper)
            mask2 = cv2.inRange(hsv, self.hsv_red2_lower, self.hsv_red2_upper)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            self._debug_red_mask = red_mask
            detections.extend(self._find_bumper_contours(red_mask, "red"))

        # Blue bumpers
        if self.track_alliance in ("both", "blue"):
            blue_mask = cv2.inRange(hsv, self.hsv_blue_lower, self.hsv_blue_upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            self._debug_blue_mask = blue_mask
            detections.extend(self._find_bumper_contours(blue_mask, "blue"))

        return detections

    def _find_bumper_contours(self, mask, alliance):
        """Find bumper-shaped contours in mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0

            # Bumpers are typically wide and short
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Get centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            detections.append({
                "cx": float(cx),
                "cy": float(cy),
                "bbox": (x, y, w, h),
                "area": float(area),
                "alliance": alliance,
                "contour": cnt,
            })

        return detections

    def track(self, detections):
        """
        Update tracking with new detections.

        Uses OC-SORT (Kalman + Hungarian) if configured, otherwise
        falls back to greedy centroid matching.
        """
        # Use OC-SORT tracker if available
        if self._ocsort is not None:
            self.robots = self._ocsort.update(detections)
            return self.robots

        # Legacy centroid-based tracking
        if len(detections) == 0:
            for rid in list(self.robots.keys()):
                self.robots[rid].disappeared += 1
                if self.robots[rid].disappeared > self.max_disappeared:
                    del self.robots[rid]
            return self.robots

        if len(self.robots) == 0:
            for det in detections:
                self._register(det)
            return self.robots

        # Build distance matrix
        robot_ids = list(self.robots.keys())
        robot_xy = np.array([(self.robots[rid].cx, self.robots[rid].cy)
                              for rid in robot_ids])
        det_xy = np.array([(d["cx"], d["cy"]) for d in detections])

        dist_mat = np.linalg.norm(
            robot_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :], axis=2
        )

        # Greedy matching
        matched_r, matched_d = set(), set()
        flat = np.argsort(dist_mat, axis=None)
        rows, cols = np.unravel_index(flat, dist_mat.shape)

        for r, c in zip(rows, cols):
            if r in matched_r or c in matched_d:
                continue
            if dist_mat[r, c] > self.max_distance:
                break
            # Only match same alliance
            if self.robots[robot_ids[r]].alliance != detections[c]["alliance"]:
                continue
            self.robots[robot_ids[r]].update(detections[c])
            matched_r.add(r)
            matched_d.add(c)

        # Handle disappeared
        for r in range(len(robot_ids)):
            if r not in matched_r:
                rid = robot_ids[r]
                self.robots[rid].disappeared += 1
                if self.robots[rid].disappeared > self.max_disappeared:
                    del self.robots[rid]

        # Register new detections
        for c in range(len(detections)):
            if c not in matched_d:
                self._register(detections[c])

        return self.robots

    def _register(self, detection):
        """
        Register a new robot.

        New robots start with team_number=None ("unknown") until identified
        via OCR or manual correction. We intentionally do NOT auto-apply
        corrections because tracking IDs are ephemeral and can be reused
        when robots leave and re-enter the frame.
        """
        robot = TrackedRobot(self.next_id, detection, detection["alliance"])
        self.robots[self.next_id] = robot
        self.next_id += 1
        return robot

    def detect_and_track(self, frame):
        """
        Detect bumpers and update tracking in one call.

        Args:
            frame: BGR image

        Returns:
            OrderedDict of robot_id -> TrackedRobot
        """
        detections = self.detect_bumpers(frame)
        return self.track(detections)

    def attempt_ocr(self, frame, robot):
        """
        Attempt to read team number from bumper using OCR.

        Args:
            frame: BGR image
            robot: TrackedRobot to OCR

        Returns:
            str: Team number if found, None otherwise
        """
        if not self.ocr_enabled or robot.ocr_attempted or robot.manual_correction:
            return None

        global _ocr_reader
        if _ocr_reader is None:
            try:
                _ocr_reader = easyocr.Reader(['en'], gpu=_GPU_AVAILABLE)
            except Exception as e:
                print(f"[OCR] Failed to initialize: {e}")
                self.ocr_enabled = False
                return None

        robot.ocr_attempted = True

        # Crop to bumper region with padding
        x, y, w, h = robot.bbox
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        try:
            results = _ocr_reader.readtext(crop, allowlist='0123456789')
            for (bbox, text, confidence) in results:
                # FRC team numbers are 1-5 digits
                if text.isdigit() and 1 <= len(text) <= 5 and confidence > 0.5:
                    robot.set_team_number(text)
                    return text
        except Exception as e:
            print(f"[OCR] Error: {e}")

        return None

    def apply_correction(self, robot_id, team_number):
        """
        Manually correct a robot's team number (session-only).

        This correction only applies to the current tracking session.
        If the robot disappears and reappears, it will need to be
        re-identified. This is intentional because tracking IDs are
        ephemeral and cannot reliably persist across robot occlusions.

        Args:
            robot_id: Robot tracking ID
            team_number: Correct team number
        """
        if self._ocsort is not None:
            self._ocsort.apply_correction(robot_id, team_number)
        elif robot_id in self.robots:
            self.robots[robot_id].set_team_number(team_number, manual=True)

    def get_tracker_stats(self):
        """
        Get tracking statistics for debugging/display.

        Returns:
            dict with tracker type, active tracks, stats, etc.
        """
        if self._ocsort is not None:
            stats = self._ocsort.get_stats()
            stats["tracker_type"] = "ocsort"
            return stats
        else:
            visible = sum(1 for r in self.robots.values() if r.disappeared == 0)
            return {
                "tracker_type": "centroid",
                "active_tracks": len(self.robots),
                "confirmed_tracks": visible,
                "frame_count": 0,
            }

    def get_robot_at_position(self, x, y, max_distance=None):
        """
        Find the robot nearest to a position.

        Args:
            x, y: Position to search from
            max_distance: Maximum distance to consider (default: self.max_distance)

        Returns:
            str: Robot identity (e.g., 'red_2491') or 'unknown' if none found
        """
        if max_distance is None:
            max_distance = self.max_distance

        min_dist = float("inf")
        nearest = None

        for robot in self.robots.values():
            if robot.disappeared > 0:
                continue
            dist = math.sqrt((x - robot.cx)**2 + (y - robot.cy)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                nearest = robot

        return nearest.identity if nearest else "unknown"

    def get_all_robots(self):
        """Get all currently visible robots."""
        return {rid: r for rid, r in self.robots.items() if r.disappeared == 0}


# ============================================================================
# YOLO-BASED ROBOT DETECTOR
# ============================================================================

# Check for ultralytics availability
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass


class YOLORobotDetector:
    """
    Detects and tracks FRC robots using a YOLO model trained on bumper detection.

    This is a drop-in replacement for RobotDetector that uses ML-based detection
    instead of HSV color filtering. Requires a trained model from 08_train_bumper_model.py.

    Features:
        - YOLO-based bumper detection (more robust than HSV)
        - Automatic alliance classification from class names
        - Bounding box expansion for full robot body estimation
        - Same tracker integration (OC-SORT or centroid)
        - Same interface as RobotDetector

    Usage:
        detector = YOLORobotDetector(config)
        robots = detector.detect_and_track(frame)
        robot_id = detector.get_robot_at_position(x, y)
    """

    def __init__(self, config):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        self.config = config
        yolo_cfg = config.get("yolo_robot_detection", {})

        # Load YOLO model
        model_path = yolo_cfg.get("model_path", "models/bumper_detector.pt")
        if not os.path.isabs(model_path):
            # Make relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}\n"
                "Train a model with: python 08_train_bumper_model.py"
            )

        print(f"[YOLO] Loading model: {model_path}")
        self.model = _YOLO(model_path)
        self.conf_threshold = yolo_cfg.get("confidence_threshold", 0.5)
        self.bbox_expansion = yolo_cfg.get("bbox_expansion_factor", 1.5)

        # Check if model has bumper classes
        self.class_names = self.model.names
        print(f"[YOLO] Classes: {list(self.class_names.values())}")

        # Determine tracker type from config
        track_type_cfg = config.get("robot_tracking", {})
        self.tracker_type = track_type_cfg.get("tracker", "centroid")

        # Legacy tracking params (for centroid fallback)
        self.max_distance = 150
        self.max_disappeared = 15

        # Initialize tracker
        if self.tracker_type == "ocsort":
            self._ocsort = OCSORTTracker(config)
            self.robots = OrderedDict()
            self.next_id = 0
            print(f"[YOLO] Using OC-SORT tracker")
        else:
            self._ocsort = None
            self.next_id = 0
            self.robots = OrderedDict()
            print(f"[YOLO] Using centroid tracker")

        # Warmup model
        print("[YOLO] Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("[YOLO] Ready!")

    def _get_alliance_from_class(self, class_id):
        """Determine alliance from class name."""
        class_name = self.class_names.get(class_id, "").lower()
        if "red" in class_name:
            return "red"
        elif "blue" in class_name:
            return "blue"
        return "unknown"

    def _expand_bbox_upward(self, bbox, frame_height):
        """Expand bumper bbox upward to approximate full robot body."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        expanded_y1 = max(0, y1 - int(height * self.bbox_expansion))
        return (x1, expanded_y1, x2, y2)

    def detect_bumpers(self, frame):
        """
        Detect bumpers using YOLO.

        Returns:
            list: Detection dicts with 'cx', 'cy', 'bbox', 'area', 'alliance'
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        frame_height = frame.shape[0]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)
                alliance = self._get_alliance_from_class(cls_id)

                # Skip unknown alliance
                if alliance == "unknown":
                    continue

                # Calculate center and dimensions
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                area = w * h

                # Store both bumper bbox and expanded robot bbox
                bumper_bbox = (x1, y1, w, h)
                robot_bbox = self._expand_bbox_upward((x1, y1, x2, y2), frame_height)

                detections.append({
                    "cx": cx,
                    "cy": cy,
                    "bbox": bumper_bbox,
                    "robot_bbox": robot_bbox,
                    "area": area,
                    "alliance": alliance,
                    "confidence": conf,
                })

        return detections

    def track(self, detections):
        """Update tracking with new detections."""
        if self._ocsort is not None:
            # Use OC-SORT tracker
            self.robots = self._ocsort.update(detections)
            return self.robots

        # Legacy centroid tracking (same as RobotDetector)
        if not self.robots:
            for det in detections:
                self._register(det)
            return self.robots

        # Build distance matrix
        robot_ids = list(self.robots.keys())
        robot_xy = np.array([(self.robots[rid].cx, self.robots[rid].cy)
                              for rid in robot_ids])
        det_xy = np.array([(d["cx"], d["cy"]) for d in detections])

        if len(det_xy) == 0:
            # No detections - increment disappeared for all
            for rid in list(self.robots.keys()):
                self.robots[rid].disappeared += 1
                if self.robots[rid].disappeared > self.max_disappeared:
                    del self.robots[rid]
            return self.robots

        dist_mat = np.linalg.norm(
            robot_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :], axis=2
        )

        # Greedy matching
        matched_r, matched_d = set(), set()
        flat = np.argsort(dist_mat, axis=None)
        rows, cols = np.unravel_index(flat, dist_mat.shape)

        for r, c in zip(rows, cols):
            if r in matched_r or c in matched_d:
                continue
            if dist_mat[r, c] > self.max_distance:
                break
            # Only match same alliance
            if self.robots[robot_ids[r]].alliance != detections[c]["alliance"]:
                continue
            self.robots[robot_ids[r]].update(detections[c])
            matched_r.add(r)
            matched_d.add(c)

        # Handle disappeared
        for r in range(len(robot_ids)):
            if r not in matched_r:
                rid = robot_ids[r]
                self.robots[rid].disappeared += 1
                if self.robots[rid].disappeared > self.max_disappeared:
                    del self.robots[rid]

        # Register new detections
        for c in range(len(detections)):
            if c not in matched_d:
                self._register(detections[c])

        return self.robots

    def _register(self, detection):
        """Register a new robot."""
        robot = TrackedRobot(self.next_id, detection, detection["alliance"])
        self.robots[self.next_id] = robot
        self.next_id += 1
        return robot

    def detect_and_track(self, frame):
        """
        Detect bumpers and update tracking in one call.

        Args:
            frame: BGR image

        Returns:
            OrderedDict of robot_id -> TrackedRobot
        """
        detections = self.detect_bumpers(frame)
        return self.track(detections)

    def get_robot_at_position(self, x, y, max_distance=None):
        """
        Find the robot nearest to a position.

        Uses expanded robot bbox for better shot attribution.

        Args:
            x, y: Position to search from
            max_distance: Maximum distance to consider

        Returns:
            str: Robot identity (e.g., 'red_2491') or 'unknown' if none found
        """
        if max_distance is None:
            max_distance = self.max_distance

        min_dist = float("inf")
        nearest = None

        for robot in self.robots.values():
            if robot.disappeared > 0:
                continue
            dist = math.sqrt((x - robot.cx)**2 + (y - robot.cy)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                nearest = robot

        return nearest.identity if nearest else "unknown"

    def get_tracker_stats(self):
        """Get tracking statistics for debugging/display."""
        if self._ocsort is not None:
            stats = self._ocsort.get_stats()
            stats["tracker_type"] = "yolo+ocsort"
            return stats
        else:
            visible = sum(1 for r in self.robots.values() if r.disappeared == 0)
            return {
                "tracker_type": "yolo+centroid",
                "active_tracks": len(self.robots),
                "confirmed_tracks": visible,
                "frame_count": 0,
            }

    def get_all_robots(self):
        """Get all currently visible robots."""
        return {rid: r for rid, r in self.robots.items() if r.disappeared == 0}


def draw_robots(frame, robots, show_ids=True, show_bbox=True):
    """
    Draw tracked robots on frame.

    Args:
        frame: BGR image
        robots: Dict of robot_id -> TrackedRobot
        show_ids: Show identity labels
        show_bbox: Show bounding boxes

    Returns:
        Annotated frame
    """
    out = frame.copy()

    for rid, robot in robots.items():
        if robot.disappeared > 0:
            continue

        # Color based on alliance
        color = (0, 0, 255) if robot.alliance == "red" else (255, 0, 0)

        # Bounding box
        if show_bbox:
            x, y, w, h = robot.bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

        # Centroid
        cx, cy = int(robot.cx), int(robot.cy)
        cv2.circle(out, (cx, cy), 5, color, -1)

        # Label
        if show_ids:
            label = robot.identity
            if robot.manual_correction:
                label += " [M]"
            cv2.putText(out, label, (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return out


# ============================================================================
# DRAWING HELPERS
# ============================================================================

def draw_detections(frame, detections, color=(0, 255, 255), thickness=2):
    """Draw detection circles and centroids."""
    out = frame.copy()
    for det in detections:
        cx, cy, r = int(det["cx"]), int(det["cy"]), int(det["radius"])
        c = (0, 200, 255) if det.get("was_split") else color
        cv2.circle(out, (cx, cy), r, c, thickness)
        cv2.circle(out, (cx, cy), 2, c, -1)
    return out


def draw_tracked_objects(frame, objects, config=None):
    """Draw tracked objects with IDs, trails, and color coding."""
    out = frame.copy()
    draw_trails = True
    draw_ids = True
    if config and "output" in config:
        draw_trails = config["output"].get("draw_trails", True)
        draw_ids = config["output"].get("draw_ids", True)

    for obj_id, obj in objects.items():
        if obj.disappeared > 0:
            continue
        if obj.shot_result == "scored":
            color = (0, 255, 0)
        elif obj.shot_result == "missed":
            color = (0, 0, 255)
        elif obj.is_moving:
            color = (0, 255, 255)
        else:
            color = (128, 128, 128)
        cx, cy = int(obj.cx), int(obj.cy)
        if draw_trails and len(obj.trail) > 1:
            pts = list(obj.trail)
            for i in range(1, len(pts)):
                a = i / len(pts)
                tc = tuple(int(c * a) for c in color)
                cv2.line(out, (int(pts[i-1][0]), int(pts[i-1][1])),
                         (int(pts[i][0]), int(pts[i][1])), tc, 2)
        cv2.circle(out, (cx, cy), int(obj.radius), color, 2)
        if draw_ids:
            label = f"#{obj.id}"
            if obj.robot_id is not None:
                label += f" R{obj.robot_id}"
            cv2.putText(out, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def draw_zones(frame, config):
    """Draw goal regions and robot zones. Supports both polygon and rectangle formats."""
    out = frame.copy()

    # Draw goal regions
    for region in config.get("goal_regions", {}).get("regions", []):
        name = region.get("name", "GOAL")
        color = (0, 255, 0)

        # Check for polygon format first
        if "polygon" in region and region["polygon"]:
            out = draw_polygon(out, region["polygon"], color, 2, label=name)
        elif "x" in region:
            # Legacy rectangle format
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.putText(out, name, (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw robot zones
    for robot in config.get("robot_zones", {}).get("robots", []):
        x, y, w, h = robot["x"], robot["y"], robot["w"], robot["h"]
        c = tuple(robot.get("color", [255, 165, 0]))
        cv2.rectangle(out, (x, y), (x+w, y+h), c, 2)
        cv2.putText(out, robot.get("name", "ROBOT"), (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return out


def create_debug_view(frame, mask, detections, detector=None):
    """
    2x2 debug view:
        TL: detections on frame     TR: binary mask
        BL: masked color             BR: contours OR distance transform heatmap with NMS/watershed overlay
    """
    h, w = frame.shape[:2]
    qw, qh = w // 2, h // 2

    tl = cv2.resize(draw_detections(frame, detections), (qw, qh))
    tr = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (qw, qh))
    bl = cv2.resize(cv2.bitwise_and(frame, frame, mask=mask), (qw, qh))

    # Check if NMS splitting is active
    nms_active = (detector is not None
                  and hasattr(detector, '_split_config')
                  and detector._split_config.get("enabled", False))

    # Check if watershed is active (legacy)
    ws_active = (detector is not None and detector.use_watershed)

    # Show distance transform heatmap if any splitting is active
    if detector is not None and detector._debug_dist is not None and detector._debug_dist.max() > 0:
        dist_n = cv2.normalize(detector._debug_dist, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(np.uint8(dist_n), cv2.COLORMAP_JET)

        # Watershed markers (legacy)
        if ws_active and detector._debug_markers is not None:
            heat[detector._debug_markers == -1] = [255, 255, 255]

        # NMS visualization
        if nms_active:
            # Draw candidate peaks (yellow circles - before NMS)
            for cx, cy, r in getattr(detector, '_debug_nms_candidates', []):
                cv2.circle(heat, (int(cx), int(cy)), 3, (0, 255, 255), -1)
                cv2.circle(heat, (int(cx), int(cy)), int(r), (0, 255, 255), 1)

            # Draw accepted peaks (green circles - after NMS)
            for cx, cy, r in getattr(detector, '_debug_nms_accepted', []):
                cv2.circle(heat, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                cv2.circle(heat, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        # Draw final detections
        for det in detections:
            cx, cy = int(det["cx"]), int(det["cy"])
            if not nms_active:
                # Legacy watershed: show green dots
                cv2.circle(heat, (cx, cy), 4, (0, 255, 0), -1)
                cv2.circle(heat, (cx, cy), int(det["radius"]), (0, 255, 0), 1)

        # Status text
        if nms_active:
            label = f"NMS splits: {detector._debug_split_count}"
        else:
            label = f"WS splits: {detector._debug_split_count}"
        cv2.putText(heat, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Timing info
        timing = getattr(detector, '_last_detect_time', 0)
        if timing > 0:
            cv2.putText(heat, f"{timing:.1f}ms", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        br = cv2.resize(heat, (qw, qh))
    else:
        # No splitting active - show contours
        ci = np.zeros_like(frame)
        for det in detections:
            cv2.drawContours(ci, [det["contour"]], -1, (0, 255, 255), 2)
            cv2.circle(ci, (int(det["cx"]), int(det["cy"])), 3, (0, 0, 255), -1)
        br = cv2.resize(ci, (qw, qh))

    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])


# ============================================================================
# ZONE / REGION UTILITIES
# ============================================================================

def point_in_rect(px, py, rect):
    return (rect["x"] <= px <= rect["x"] + rect["w"] and
            rect["y"] <= py <= rect["y"] + rect["h"])


def point_in_polygon(px, py, polygon):
    """
    Check if point is inside polygon using ray casting algorithm.

    Args:
        px, py: Point coordinates
        polygon: List of [x, y] vertex coordinates

    Returns:
        bool: True if point is inside polygon
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def draw_polygon(frame, polygon, color, thickness=2, label=None, filled=False):
    """
    Draw polygon on frame with optional label.

    Args:
        frame: Image to draw on
        polygon: List of [x, y] vertex coordinates
        color: BGR color tuple
        thickness: Line thickness (ignored if filled=True)
        label: Optional text label to draw at polygon centroid
        filled: If True, fill the polygon with semi-transparent color

    Returns:
        Frame with polygon drawn
    """
    out = frame.copy()
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

    if filled:
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
        cv2.polylines(out, [pts], True, color, max(1, thickness // 2))
    else:
        cv2.polylines(out, [pts], True, color, thickness)

    if label:
        cx = int(np.mean([p[0] for p in polygon]))
        cy = int(np.mean([p[1] for p in polygon]))
        cv2.putText(out, label, (cx - len(label) * 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out


def select_zones_interactive(video_path, zone_type="goal"):
    """Interactively select rectangular zones on the first frame."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Cannot read first frame")
    zones = []
    print(f"[ZONES] Select {zone_type} zones. ENTER after each, ESC when done.")
    while True:
        display = frame.copy()
        for z in zones:
            cv2.rectangle(display, (z["x"], z["y"]),
                          (z["x"]+z["w"], z["y"]+z["h"]), (0, 255, 0), 2)
            cv2.putText(display, z.get("name", ""), (z["x"]+5, z["y"]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        r = cv2.selectROI(f"Select {zone_type} zone (ESC=done)",
                           display, fromCenter=False)
        if r[2] == 0 or r[3] == 0:
            break
        zones.append({
            "name": f"{zone_type}_{len(zones)+1}",
            "x": int(r[0]), "y": int(r[1]),
            "w": int(r[2]), "h": int(r[3]),
        })
        print(f"[ZONES] Added: {zones[-1]}")
    cv2.destroyAllWindows()
    return zones


# ============================================================================
# HUD OVERLAY
# ============================================================================

def draw_hud(frame, stats, frame_num=0, total_frames=0):
    """Draw heads-up display with tracking stats."""
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    y = 20
    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"Detected: {stats.get('balls_detected', 0)}",
        f"Tracked: {stats.get('balls_tracked', 0)}",
        f"Moving: {stats.get('balls_moving', 0)}",
    ]
    if "shots_total" in stats:
        lines.append(f"Shots: {stats['shots_total']}  "
                     f"In: {stats.get('shots_scored', 0)}  "
                     f"Miss: {stats.get('shots_missed', 0)}")
    for line in lines:
        cv2.putText(out, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
    if total_frames > 0:
        bar_y = h - 8
        cv2.rectangle(out, (0, bar_y),
                      (int(w * frame_num / total_frames), h),
                      (0, 200, 255), -1)
    return out
