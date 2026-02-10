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

        # --- Step 2: morphology on GPU ---
        m = gpu_mask_2d > 0

        se_open = cp.asarray(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self._open_k, self._open_k)).astype(bool))
        m = cp_ndimage.binary_opening(m, structure=se_open)

        se_close = cp.asarray(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self._close_k, self._close_k)).astype(bool))
        m = cp_ndimage.binary_closing(m, structure=se_close)

        if self.dilate_iters > 0:
            se_dilate = cp.asarray(
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self._dilate_k, self._dilate_k)).astype(bool))
            m = cp_ndimage.binary_dilation(m, structure=se_dilate,
                                            iterations=self.dilate_iters)

        # --- Step 3: download back to CPU ---
        return cp.asnumpy(m.astype(cp.uint8)) * 255

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
    """Draw goal regions and robot zones."""
    out = frame.copy()
    for region in config.get("goal_regions", {}).get("regions", []):
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(out, region.get("name", "GOAL"), (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
