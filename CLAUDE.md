# FRC Ball Tracker - Development Guide

## Project Purpose

Motion tracking system for FRC scouting that detects yellow foam balls, tracks trajectories, and attributes shots to robots. Designed to help scouting teams identify performant robots for alliance selection by determining which robot made each successful/unsuccessful shot.

## Architecture Overview

```
frc_tracker_config.json     # Single source of truth for all parameters
         ↓
frc_tracker_utils.py        # Core library (BallDetector, CentroidTracker, TrackedObject, YOLORobotDetector)
         ↓
01_hsv_tuning.py           # Interactive HSV/morphology parameter tuning
02_detection_test.py       # Detection validation across full video
03_tracking_sandbox.py     # Multi-ball tracking visualization
04_zones_and_shots.py      # Shot detection & robot attribution
05_full_pipeline.py        # Production two-pass video processing
06_robot_tuning.py         # Robot bumper HSV tuning
07_yolo_poc.py             # YOLO robot detection testing (supports custom models)
08_train_bumper_model.py   # Train custom YOLO model for bumper detection
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
- `tracking`: max_distance, max_frames_missing, trail_length (for ball tracking)
- `robot_tracking`: tracker type (ocsort/centroid), max_age, min_hits, iou_threshold
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

| Current | Status |
|---------|--------|
| Greedy centroid matching (balls) | Upgrade available: Hungarian algorithm for robots via OC-SORT |
| No occlusion handling (balls) | Upgrade available: Kalman filtering for robots via OC-SORT |
| Batch processing only | Future: Live camera input support |
| Single camera | Future: Multi-camera triangulation |

**Robot Detection Options (Implemented):**
- **YOLO-based** (recommended): Custom-trained model detects red/blue bumpers directly
  - Train with `08_train_bumper_model.py` on labeled bumper datasets
  - Configure via `yolo_robot_detection.enabled = true`
  - Model stored in `models/bumper_detector.pt`
- **HSV-based** (fallback): Color-based bumper detection
  - Tune with `06_robot_tuning.py`
  - Configure via `robot_detection.enabled = true`

**Robot Tracking Upgrades (Implemented):**
- OC-SORT tracker with Kalman filter and Hungarian algorithm
- IoU-based matching (better than centroid distance)
- Observation-Centric Re-update (ORU) for trajectory smoothing
- Observation-Centric Momentum (OCM) to prevent velocity drift
- Configure via `robot_tracking.tracker` = "ocsort" or "centroid"

## Cluster Splitting Methods

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| NMS (default) | Fast, handles 3+ ball clusters, crop-before-process optimization | May split noise | Dense clusters, most scenarios |
| Watershed (legacy) | Well-known algorithm | Fails on 3+ balls, slow on large blobs | Fallback if NMS has issues |
| None | Fastest | Misses overlapping balls | Sparse ball scenarios |

## Dependencies

**Required:** opencv-python, numpy
**Recommended:** scipy (for Hungarian algorithm in OC-SORT tracker), ultralytics (YOLO robot detection)
**Optional:** matplotlib, pandas, cupy-cuda12x (GPU acceleration for ball detection), easyocr (team number OCR)

**Install for full functionality:**
```bash
pip install opencv-python numpy scipy ultralytics
pip install cupy-cuda12x  # Optional: GPU-accelerated ball detection
```

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| frc_tracker_utils.py | ~2600 | Core detection, tracking, OC-SORT, RobotDetector, YOLORobotDetector, drawing utilities |
| 01_hsv_tuning.py | ~300 | HSV tuning with live trackbars |
| 02_detection_test.py | ~200 | Full-video detection statistics |
| 03_tracking_sandbox.py | ~330 | Tracking visualization and tuning |
| 04_zones_and_shots.py | ~700 | Shot detection, polygon zones, robot integration |
| 05_full_pipeline.py | ~370 | Two-pass production pipeline with YOLO support |
| 06_robot_tuning.py | ~350 | Robot bumper HSV tuning and ID correction |
| 07_yolo_poc.py | ~580 | YOLO robot detection testing with custom model support |
| 08_train_bumper_model.py | ~180 | Train custom YOLO model on bumper datasets |

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

### Session: 2026-02-10 - Pipeline Improvements (Polygons, Bounce-out, Robots)

**Completed:**
1. ✅ **Bounce-out handling**: Ball entering goal = 1 scored shot, prevents double-counting if ball bounces out
   - Added `scored_ball_ids` set to ShotDetector
   - `update()` now returns IDs to remove from tracker
2. ✅ **Terminate tracking on goal entry**: Balls removed from tracker when they enter goals
   - Added `remove_objects()` method to CentroidTracker
3. ✅ **Polygon goal regions**: Full polygon support for irregularly-shaped goals
   - Added `point_in_polygon()` and `draw_polygon()` utilities
   - New interactive polygon selection UI (press 'G' in 04_zones_and_shots.py)
   - Backwards compatible with rectangle format
4. ✅ **Dynamic robot tracking**: Detect and track robots by bumper color
   - New `RobotDetector` class with HSV-based bumper detection (red & blue)
   - Multi-object tracking across frames
   - Optional OCR for team number reading (requires easyocr)
   - Manual correction support via click interface
   - New `06_robot_tuning.py` script for HSV tuning
5. ✅ **Occlusion-safe attribution**: Shot attribution locked at launch time (was already correct)
   - Added documentation comments to ShotDetector and ShotEvent classes
   - RobotDetector integration for dynamic robot lookup

**New Controls in 04_zones_and_shots.py:**
- `G` - Define polygon goal regions (click vertices, ENTER to finish)
- `g` - Define rectangle goal regions (legacy)
- `n` - Step forward one frame
- `b` - Step backward one frame

**New Script: 06_robot_tuning.py:**
- Separate trackbar window (like 01_hsv_tuning.py)
- Interactive HSV tuning for robot bumper detection
- Toggle red/blue bumper display (r/u keys)
- Debug view showing color masks (press 'd')
- Click-to-correct mode for manual team number assignment (session-only)
- Comprehensive inline help (press 'h' or '?')

**New Config Section:**
- `robot_detection`: HSV ranges for red/blue bumpers, contour filters, tracking params

**Important Design Decisions:**
- Team number corrections are SESSION-ONLY (not persisted to config)
- Tracking IDs are ephemeral and cannot reliably persist across robot occlusions
- Robots default to "unknown" until manually identified or OCR succeeds
- `update_config()` now preserves tracking state (doesn't reset IDs)

**Status:** All pipeline improvements complete. Ready for testing.

### Session: 2026-02-10 - YOLO Migration POC

**Context:**
After evaluating the current HSV-based robot tracking approach, identified key limitations:
- 15-25% GPU utilization (3090 underutilized)
- Cannot recover robot IDs after occlusion
- Bumper ROI doesn't capture full robot body for shot attribution

**Decision:** Implement YOLO + modern tracker (BoT-SORT/ByteTrack) as alternative approach.

**Created:**
1. ✅ `07_yolo_poc.py` - Standalone proof of concept script
   - YOLO detection with ultralytics library
   - Toggle between no tracker, ByteTrack, or BoT-SORT
   - HSV post-processing to classify alliance color from detections
   - FPS benchmarking on 3090

**New Script: 07_yolo_poc.py:**
- Standalone POC - does NOT modify existing code
- Tests YOLO detection quality on FRC footage
- Controls: SPACE=pause, T=cycle tracker, C=confidence, D=debug, +/-=speed
- Combines YOLO detection with HSV color classification for alliance

**New Dependency (optional):**
- `ultralytics`: ML-based object detection (~500MB with PyTorch)
- Install: `pip install ultralytics`

**Status:** POC created. Run `07_yolo_poc.py` to evaluate YOLO performance.

**Next Steps:**
1. Run POC to verify YOLO can detect robots in FRC footage
2. Benchmark FPS (expect 100-150 FPS on 3090)
3. Test BoT-SORT occlusion recovery
4. If successful, create `ml_tracker_utils.py` for production integration

### Session: 2026-02-10 - OC-SORT Robot Tracker Implementation

**Context:**
YOLO POC revealed pretrained COCO models don't recognize FRC robots. Training custom models requires labeled data. New direction: keep HSV bumper detection but upgrade tracker from simple centroid matching to OC-SORT style tracking.

**Completed:**
1. ✅ **KalmanBoxTracker class** (~100 lines)
   - 8-state Kalman filter: [x, y, w, h, vx, vy, vw, vh]
   - Constant velocity motion model
   - OpenCV KalmanFilter for prediction/correction
   - Stores recent observations for OC-SORT features

2. ✅ **OCSORTTracker class** (~180 lines)
   - Hungarian algorithm matching via `scipy.optimize.linear_sum_assignment`
   - IoU-based cost matrix (better than centroid distance for boxes)
   - Alliance constraint: only match same-color robots
   - Observation-Centric Re-update (ORU): smooth trajectory after occlusion
   - Observation-Centric Momentum (OCM): blend predicted/observed velocity
   - Falls back to greedy matching if scipy unavailable

3. ✅ **RobotDetector integration**
   - Configurable tracker type: `"ocsort"` or `"centroid"`
   - Seamless drop-in replacement via `_TrackerOutputWrapper`
   - `get_tracker_stats()` for debugging/display
   - Prints tracker type at initialization

4. ✅ **Config section added**
   ```json
   "robot_tracking": {
       "tracker": "ocsort",
       "max_age": 15,
       "min_hits": 3,
       "iou_threshold": 0.3,
       "use_oru": true,
       "use_ocm": true
   }
   ```

5. ✅ **06_robot_tuning.py updated**
   - HUD shows tracker type and stats
   - Displays active/confirmed tracks and match count for OC-SORT

**Key Design Decisions:**
- Kalman filter uses OpenCV's `cv2.KalmanFilter` (no new dependencies)
- Hungarian algorithm requires scipy (~30MB, commonly installed)
- Falls back to greedy matching if scipy unavailable
- OC-SORT wrapper class ensures same interface as `TrackedRobot`
- Config-driven tracker selection for easy A/B testing

**New Dependencies:**
- `scipy` (required for optimal Hungarian matching)
- Install: `pip install scipy`

**Files Modified:**
| File | Change |
|------|--------|
| `frc_tracker_utils.py` | +KalmanBoxTracker, +OCSORTTracker, +iou(), +iou_distance(), +_TrackerOutputWrapper, modified RobotDetector |
| `frc_tracker_config.json` | +robot_tracking config section |
| `06_robot_tuning.py` | +tracker stats in HUD |

**Status:** OC-SORT implementation complete. Ready for testing.

**Next Steps:**
1. Run `06_robot_tuning.py` with new tracker
2. Test robot ID persistence through collisions
3. Compare before/after on same video clips
4. Tune `max_age` and `iou_threshold` for FRC scenarios

### Session: 2026-02-11 - Custom YOLO Model Training & Pipeline Integration

**Context:**
Previous YOLO POC showed pretrained COCO models don't recognize FRC robots. Solution: train a custom YOLO model on labeled FRC bumper datasets from Roboflow.

**Completed:**

1. ✅ **Dataset Acquisition & Merging**
   - Downloaded 3 Roboflow datasets: robot-bumpers-3, bumpers1, bumpers2
   - Created merge script to combine datasets with filename prefixes
   - Fixed data.yaml relative paths for ultralytics compatibility
   - **Combined dataset:** 1,601 train / 442 valid / 192 test images
   - Classes: `blue_bumper`, `red_bumper`

2. ✅ **Training Script: `08_train_bumper_model.py`**
   - Configurable epochs, batch size, image size, base model
   - `--data` argument to select dataset (default: bumpers-merged)
   - `--validate` mode for testing existing models
   - `--resume` for continuing interrupted training
   - Outputs to `models/bumper_detector.pt`

3. ✅ **YOLORobotDetector Class** (~250 lines in frc_tracker_utils.py)
   - Drop-in replacement for HSV-based RobotDetector
   - Loads custom trained model from `models/bumper_detector.pt`
   - Auto-detects alliance from class names (blue_bumper/red_bumper)
   - Bbox expansion for full robot body estimation
   - Same interface: `detect_and_track()`, `get_robot_at_position()`
   - Integrates with existing OC-SORT/centroid trackers

4. ✅ **07_yolo_poc.py Updates**
   - `--model` CLI argument (auto-detects custom model if exists)
   - `--expand` for bbox expansion factor
   - Press 'E' to toggle expansion visualization
   - Auto-classifies alliance from model class names

5. ✅ **05_full_pipeline.py Updates**
   - Checks `yolo_robot_detection.enabled` first
   - Falls back to HSV if YOLO fails/disabled
   - Stores robot states in Pass 1, draws in Pass 2
   - Fixed bare expression bug on line 310

6. ✅ **CuPy/CUDA 12.9 Compatibility Fix**
   - Issue: CUDA 12.9 headers incompatible with CuPy's runtime-compiled morphology kernels
   - Fix: Hybrid approach - GPU for HSV masking, CPU for morphology (OpenCV)
   - Installed `cupy-cuda12x` for precompiled kernels
   - Performance: ~55 fps on full pipeline

**New Config Section:**
```json
"yolo_robot_detection": {
    "enabled": true,
    "model_path": "models/bumper_detector.pt",
    "confidence_threshold": 0.5,
    "bbox_expansion_factor": 1.5,
    "use_tracker": "ocsort"
}
```

**Training Results:**
- Model: `models/bumper_detector.pt` (trained on merged dataset)
- Classes: blue_bumper, red_bumper
- Training logs: `runs/train/bumper/`

**Pipeline Test Results (kettering1.mkv):**
- 5,219 frames processed
- ~55 fps (Pass 1), GPU path active
- 302 shots detected
- Robots attributed as red_unknown/blue_unknown (team numbers require manual assignment or OCR)
- Output: `kettering1_annotated.mp4`, `kettering1_annotated_shots.csv`

**Files Modified:**
| File | Change |
|------|--------|
| `frc_tracker_utils.py` | +YOLORobotDetector class, fixed CuPy morphology for CUDA 12.9 |
| `frc_tracker_config.json` | +yolo_robot_detection section, updated goal_regions |
| `05_full_pipeline.py` | +YOLO support, +robot drawing in Pass 2, fixed line 310 |
| `07_yolo_poc.py` | +--model/--expand args, +expansion viz, +auto alliance from class |

**Files Created:**
| File | Purpose |
|------|---------|
| `08_train_bumper_model.py` | YOLO training script |
| `models/bumper_detector.pt` | Trained bumper detection model |
| `bumpers-merged/` | Combined training dataset |

**Status:** Full pipeline operational with YOLO robot detection. Ready for production use.

**Workflow for New Venues:**
```
1. Run 01_hsv_tuning.py → tune ball detection for lighting
2. Run 04_zones_and_shots.py → define goal regions (press 'g')
3. Run 05_full_pipeline.py → process match recordings
4. Review kettering1_annotated.mp4 and _shots.csv
```

**Future Enhancements:**
- Fine-tune YOLO model on venue-specific data if needed
- Add team number OCR or manual assignment workflow
- Consider OC-SORT tracker for robot tracking (currently using centroid)
