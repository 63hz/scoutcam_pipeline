# FRC Ball Tracker - Sandbox

Multi-ball tracking pipeline for FRC "many ball shooter" games.  
Detects yellow foam balls, tracks trajectories, attributes shots to robots, and determines scoring outcomes.

## Architecture

```
frc_tracker/
├── frc_tracker_config.json   # All tunable parameters (persisted)
├── frc_tracker_utils.py      # Core components (detector, tracker, drawing)
├── 01_hsv_tuning.py          # Interactive HSV + morphology + contour tuning
├── 02_detection_test.py      # Validate detection across video, gather stats
├── 03_tracking_sandbox.py    # Multi-ball tracking with trails + velocity
├── 04_zones_and_shots.py     # Goal/robot zones + shot detection + attribution
├── 05_full_pipeline.py       # Two-pass pipeline → annotated output video + CSV
└── README.md                 # This file
```

## Workflow (Sequential)

### Step 1: HSV Tuning (`01_hsv_tuning.py`)
- Select ROI to crop out chirons/overlays
- Tune HSV color range for yellow ball detection via trackbars
- Adjust morphological operations and contour filters
- Step through frames to verify robustness across lighting changes
- **Save config** when satisfied (press `s`)

### Step 2: Detection Validation (`02_detection_test.py`)
- Run detection across full video with saved config
- Review detection counts per frame, processing time
- Identify problem areas (false positives, missed detections)
- Go back to Step 1 if needed

### Step 3: Tracking (`03_tracking_sandbox.py`)
- Multi-ball centroid tracking with trail visualization
- Tune `max_distance` (how far a ball can move between frames)
- Tune `max_frames_missing` (how long to keep tracking a disappeared ball)
- Toggle motion-only filter, velocity vectors, trails
- **Save config** when tracking looks stable

### Step 4: Zones & Shots (`04_zones_and_shots.py`)
- Define goal regions (where scored balls end up)
- Define robot zones (where robots are positioned/shooting from)
- Run shot detection: ball launches → trajectory → score/miss
- Review attribution and accuracy
- Export shot log CSV

### Step 5: Full Pipeline (`05_full_pipeline.py`)
- Two-pass processing for retroactive shot coloring
- Pass 1: Analyze entire video, detect all shots, determine outcomes
- Pass 2: Re-render with color coding from shot START (green=will score, red=will miss)
- Outputs: annotated MP4 + shot log CSV

## Spyder Setup

### Critical: External Terminal
Scripts 01-04 use OpenCV's `highgui` windows with trackbars, which need a real display.

**In Spyder:**
1. Go to `Run → Configuration per file`
2. Select `Execute in an external system terminal`
3. This applies per-script, so set it for each interactive script

Alternatively, run from terminal: `python 01_hsv_tuning.py path/to/video.mp4`

### Cell Mode
Scripts 02 and 03 have `# %%` cell markers for Spyder's cell execution mode.
After running, inspect variables in Spyder's Variable Explorer.

## Game Day Quick-Tune Workflow

1. Record a short test clip at the venue
2. Run `01_hsv_tuning.py` with the test clip
3. Adjust HSV range for venue lighting (main thing that changes)
4. Save config
5. Verify with `02_detection_test.py`
6. Run `05_full_pipeline.py` on match recordings

The config JSON is the single source of truth — tune once, use everywhere.

## Config Parameters Quick Reference

| Section | Key | What it does |
|---------|-----|--------------|
| `hsv_yellow` | `h_low/h_high` | Hue range (most important for lighting changes) |
| `hsv_yellow` | `s_low` | Saturation floor (raise if picking up non-yellow objects) |
| `hsv_yellow` | `v_low` | Value/brightness floor (lower for dim venues) |
| `morphology` | `open_kernel` | Removes small noise (increase if noisy) |
| `morphology` | `close_kernel` | Fills gaps in detections (increase if balls look fragmented) |
| `contour_filter` | `min_area` | Minimum blob size in pixels (adjust for camera distance) |
| `contour_filter` | `min_circularity` | How round a blob must be (lower = more permissive) |
| `tracking` | `max_distance` | Max pixels a ball can move between frames (increase for fast shots) |
| `tracking` | `max_frames_missing` | Frames before dropping a lost track |
| `shot_detection` | `min_upward_velocity` | vy threshold for shot detection (negative = upward) |
| `shot_detection` | `min_speed` | Minimum speed to qualify as a shot |

## Design Notes for Claude Code Migration

The codebase is structured for easy refactoring:
- `frc_tracker_utils.py` contains all reusable components with clean interfaces
- Config is JSON-based, no hardcoded parameters
- `BallDetector`, `CentroidTracker`, and `ShotDetector` are self-contained classes
- Detection → Tracking → Shot Detection is a clean pipeline with per-frame state
- The two-pass architecture in script 05 could become a single-pass streaming pipeline for live camera by removing retroactive coloring

### Likely improvements for production:
- Replace greedy centroid matching with Hungarian algorithm (scipy.optimize.linear_sum_assignment)
- Add Kalman filter prediction for tracking through occlusions
- GPU-accelerate HSV conversion + morphology with CUDA (if available)
- Add robot tracking via bumper color detection (separate from ball detection)
- Support multiple camera angles
- Live dashboard with Streamlit or similar
- YOLO-based detection as alternative/complement to HSV for robustness

## Dependencies

```
pip install opencv-python numpy
```

Optional for analysis:
```
pip install matplotlib pandas
```
