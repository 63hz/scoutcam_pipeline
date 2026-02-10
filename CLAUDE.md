# FRC Ball Tracker - Development Guide

## Project Purpose

Motion tracking system for FRC scouting that detects yellow foam balls, tracks trajectories, and attributes shots to robots. Designed to help scouting teams identify performant robots for alliance selection by determining which robot made each successful/unsuccessful shot.

## Architecture Overview

```
frc_tracker_config.json     # Single source of truth for all parameters
         ↓
frc_tracker_utils.py        # Core library (BallDetector, CentroidTracker, TrackedObject)
         ↓
01_hsv_tuning.py           # Interactive HSV/morphology parameter tuning
02_detection_test.py       # Detection validation across full video
03_tracking_sandbox.py     # Multi-ball tracking visualization
04_zones_and_shots.py      # Shot detection & robot attribution
05_full_pipeline.py        # Production two-pass video processing
```

## Key Components

### BallDetector (frc_tracker_utils.py)
- HSV color space detection for yellow balls
- Morphological operations (open/close/dilate) for noise reduction
- Contour filtering by area, circularity, aspect ratio
- NMS-based cluster splitting for touching/overlapping balls (preferred)
- Legacy watershed splitting available as fallback
- Optional GPU acceleration via CuPy/CUDA
- Per-frame timing diagnostics

### GPU Diagnostics (frc_tracker_utils.py)
- `get_gpu_status()`: Returns detailed GPU info (device, VRAM, CUDA version)
- `print_gpu_status()`: Prints formatted GPU status to console
- Automatic detection of CuPy availability and CUDA kernel compilation
- Per-frame timing with periodic summary output

### CentroidTracker (frc_tracker_utils.py)
- Multi-object tracking via centroid distance association
- Greedy matching algorithm (not optimal - Hungarian algorithm recommended for upgrade)
- Tracks 100+ balls simultaneously
- Per-object state: position, velocity, trails, age

### ShotDetector (04_zones_and_shots.py)
- Detects shot events from tracked balls based on:
  - Upward velocity (vy < threshold)
  - Sufficient speed
  - Proximity to robot zone
  - Minimum consecutive flight frames
- Attributes shots to nearest robot zone
- Resolves outcomes (scored/missed) when ball enters/misses goal region

## Configuration System

All parameters in `frc_tracker_config.json`. Key sections:
- `roi`: Region of interest (exclude chirons/overlays)
- `hsv_yellow`: Color detection range (h_low, h_high, s_low, s_high, v_low, v_high)
- `morphology`: Noise reduction kernels
- `contour_filter`: Shape/size filtering
- `cluster_splitting`: NMS-based splitting for touching balls (enabled, min_ball_radius, peak_threshold, area_multiplier)
- `watershed`: Legacy watershed splitting (deprecated, use cluster_splitting)
- `tracking`: max_distance, max_frames_missing, trail_length
- `shot_detection`: velocity thresholds, proximity settings
- `goal_regions` / `robot_zones`: Defined areas for attribution

## Development Patterns

### Adding New Features
1. Add configuration parameters to `frc_tracker_config.json`
2. Implement core logic in `frc_tracker_utils.py` if reusable
3. Add interactive tuning if needed in appropriate numbered script
4. Update `05_full_pipeline.py` for production use

### Testing Changes
1. Use `01_hsv_tuning.py` for detection parameter changes
2. Use `02_detection_test.py` to validate across full video
3. Use `03_tracking_sandbox.py` for tracking algorithm changes
4. Use `04_zones_and_shots.py` for shot detection logic

### Code Style
- No hardcoded parameters - everything in JSON config
- Use `load_config()` / `save_config()` from utils
- Add trackbars for interactive parameter adjustment
- Include debug visualization options (press 'd' in most scripts)

## Game Day Workflow

```
1. Record test clip at venue
2. Run 01_hsv_tuning.py → adjust for lighting conditions
3. Save config (press 's')
4. Verify with 02_detection_test.py
5. Define zones with 04_zones_and_shots.py if needed
6. Run 05_full_pipeline.py on match recordings
7. Review annotated video + shot CSV
```

## Known Limitations & Upgrade Paths

| Current | Upgrade Path |
|---------|--------------|
| Greedy centroid matching | Hungarian algorithm (scipy.optimize.linear_sum_assignment) |
| No occlusion handling | Kalman filtering for trajectory prediction |
| HSV-only detection | YOLO/ML alternative for robustness |
| Batch processing only | Live camera input support |
| Single camera | Multi-camera triangulation |

## Cluster Splitting Methods

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| NMS (default) | Fast, handles 3+ ball clusters, crop-before-process optimization | May split noise | Dense clusters, most scenarios |
| Watershed (legacy) | Well-known algorithm | Fails on 3+ balls, slow on large blobs | Fallback if NMS has issues |
| None | Fastest | Misses overlapping balls | Sparse ball scenarios |

## Dependencies

**Required:** opencv-python, numpy
**Optional:** matplotlib, pandas, cupy-cuda12x (GPU), scipy

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| frc_tracker_utils.py | 826 | Core detection, tracking, drawing utilities |
| 01_hsv_tuning.py | 298 | HSV tuning with live trackbars |
| 02_detection_test.py | 202 | Full-video detection statistics |
| 03_tracking_sandbox.py | 326 | Tracking visualization and tuning |
| 04_zones_and_shots.py | 533 | Shot detection, zones, attribution |
| 05_full_pipeline.py | 302 | Two-pass production pipeline |

## Debugging Tips

- Press 'd' in most scripts to toggle debug quad view
- Debug view shows: detections | mask | masked color | distance transform heatmap
- In NMS mode: yellow dots = candidates before suppression, green dots = accepted peaks
- GPU status is printed at startup in 01_hsv_tuning.py
- Timing info shown in HUD (ms per frame)
- Check `max_distance` if fast shots lose tracking
- Lower `V_low` in dim venues, raise `S_low` if too many false positives
- Use motion filter (`use_motion_filter: true`) to reduce stationary ball noise
- If NMS splits too aggressively, increase `min_ball_radius` or `peak_threshold`

---

## Session Progress Log

### Session: 2026-02-10 - GPU Diagnostics + NMS Cluster Splitting

**Completed:**
1. ✅ Added GPU diagnostics (`get_gpu_status()`, `print_gpu_status()`)
2. ✅ Added per-frame timing to BallDetector with periodic summaries
3. ✅ Replaced watershed with NMS-based cluster splitting
   - New `_detect_with_nms_splitting()` method
   - New `_split_cluster_nms()` using distance transform + greedy NMS
   - Crop-before-process optimization (only process blob ROI)
4. ✅ Added separate `max_cluster_area` config (distinct from single-ball `max_area`)
5. ✅ Updated `01_hsv_tuning.py` with:
   - GPU status display at startup
   - NMS trackbars (Split, Radius, Peak%, AreaMul%, Max Cluster)
   - Increased slider ranges (Max Area: 0-2000, Max Cluster: 0-10000)
   - Timing display in HUD
6. ✅ Updated debug view to show NMS candidates (yellow) vs accepted (green)

**Current tuned values (from user testing on kettering1.mkv):**
- `min_ball_radius`: 5
- `peak_threshold`: 0.57
- `area_multiplier`: 1.47
- `max_cluster_area`: 5000
- `min_area`: 47, `max_area`: 407

**Status:** Detection tuning complete via `01_hsv_tuning.py`. Ready to proceed to next pipeline stage.

**Next Steps:**
1. Run `02_detection_test.py` to validate detection across full video
2. Run `03_tracking_sandbox.py` to test multi-ball tracking
3. Define goal regions and robot zones via `04_zones_and_shots.py`
4. Run full pipeline via `05_full_pipeline.py`

**Known Issues:**
- Very tightly packed clusters (balls significantly overlapping) may not split perfectly - distance transform peaks merge when overlap is too great
- Possible future enhancement: expected-count mode that forces N peaks based on blob area
