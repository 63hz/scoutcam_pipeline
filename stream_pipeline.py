# -*- coding: utf-8 -*-
"""
Streaming Pipeline for Real-Time FRC Ball Tracking
====================================================
Producer-consumer threading architecture for real-time video processing.

ARCHITECTURE:
    Thread 1: Decode     → Queue (10 frames)
    Thread 2: Detect     → Queue (10 frames)  [YOLO ball + robot detection]
    Thread 3: Track      → Queue (100 frames) [Sequential - cannot parallelize]
    Thread 4: Render     → Queue (10 frames)  [After delayed buffer]
    Thread 5: Encode     → Disk               [NVENC]
    Main:     Display    → Window             [cv2.imshow, non-blocking]

KEY DESIGN CHOICES:
    - 90-frame delay buffer allows retroactive shot coloring
    - Shot callbacks for real-time CSV logging
    - Frame dropping if processing can't keep up (live mode only)
    - Graceful shutdown with poison pill pattern

Author: Clay / Claude sandbox
"""

import os
import sys
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, apply_roi, draw_hud, draw_zones,
    BallDetector, CentroidTracker,
    VideoSource, FileSource, LiveCameraSource,
    NVENCWriter, create_video_writer,
    YOLORobotDetector, RobotDetector,
)

# Import shot detector
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "zones_and_shots",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_zones_and_shots.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ShotDetector = _mod.ShotDetector


# Try to import YOLO ball detector
try:
    from frc_tracker_utils import YOLOBallDetector, create_ball_detector
    _YOLO_BALL_AVAILABLE = True
except ImportError:
    _YOLO_BALL_AVAILABLE = False


# Sentinel value for shutdown
_POISON_PILL = object()


@dataclass
class FrameState:
    """State associated with a single frame in the pipeline."""
    frame_num: int
    timestamp: float
    frame: Optional[np.ndarray] = None
    roi_frame: Optional[np.ndarray] = None
    detections: Optional[List[dict]] = None
    object_states: Optional[Dict[int, dict]] = None
    robot_states: Optional[Dict[int, dict]] = None
    shot_info: Optional[Dict[int, dict]] = None
    rendered_frame: Optional[np.ndarray] = None


class PipelineStage(threading.Thread):
    """Base class for pipeline stages."""

    def __init__(self, name: str, input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__(name=name, daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._total_time = 0.0

    def stop(self):
        """Signal the stage to stop."""
        self._stop_event.set()

    def run(self):
        """Main thread loop."""
        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _POISON_PILL:
                # Propagate poison pill and exit
                if self.output_queue is not None:
                    self.output_queue.put(_POISON_PILL)
                break

            t0 = time.perf_counter()
            result = self.process(item)
            self._total_time += time.perf_counter() - t0
            self._frame_count += 1

            if result is not None and self.output_queue is not None:
                self.output_queue.put(result)

    def process(self, item: FrameState) -> Optional[FrameState]:
        """Process a single frame. Override in subclasses."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Return timing statistics."""
        if self._frame_count > 0:
            avg_ms = (self._total_time / self._frame_count) * 1000
            fps = self._frame_count / self._total_time if self._total_time > 0 else 0
        else:
            avg_ms = 0
            fps = 0
        return {
            "name": self.name,
            "frames": self._frame_count,
            "avg_ms": avg_ms,
            "fps": fps,
        }


class DecodeStage(PipelineStage):
    """Decodes frames from video source."""

    def __init__(self, source: VideoSource, config: dict,
                 output_queue: queue.Queue):
        super().__init__("Decode", None, output_queue)
        self.source = source
        self.config = config
        self.roi_cfg = config.get("roi", {"x": 0, "y": 0, "w": 1920, "h": 1080})

    def run(self):
        """Override run to read from source directly."""
        frame_num = 0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            ret, frame = self.source.read()
            if not ret:
                # End of video
                self.output_queue.put(_POISON_PILL)
                break

            roi_frame = apply_roi(frame, self.roi_cfg)

            state = FrameState(
                frame_num=frame_num,
                timestamp=time.time(),
                frame=frame,
                roi_frame=roi_frame,
            )

            self._total_time += time.perf_counter() - t0
            self._frame_count += 1
            frame_num += 1

            try:
                self.output_queue.put(state, timeout=1.0)
            except queue.Full:
                # Drop frame if queue full (live mode)
                if self.source.is_live():
                    continue
                else:
                    self.output_queue.put(state)


class DetectStage(PipelineStage):
    """Runs ball and robot detection."""

    def __init__(self, config: dict,
                 input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__("Detect", input_queue, output_queue)
        self.config = config

        # Initialize ball detector
        yolo_ball_cfg = config.get("yolo_ball_detection", {})
        if _YOLO_BALL_AVAILABLE and yolo_ball_cfg.get("enabled", False):
            try:
                self.ball_detector = create_ball_detector(config)
            except Exception as e:
                print(f"[DETECT] Ball YOLO failed: {e}, using HSV")
                self.ball_detector = BallDetector(config)
        else:
            self.ball_detector = BallDetector(config)

        # Initialize robot detector
        self.robot_detector = None
        yolo_robot_cfg = config.get("yolo_robot_detection", {})
        hsv_robot_cfg = config.get("robot_detection", {})

        if yolo_robot_cfg.get("enabled", False):
            try:
                self.robot_detector = YOLORobotDetector(config)
            except Exception as e:
                print(f"[DETECT] Robot YOLO failed: {e}")
                if hsv_robot_cfg.get("enabled", False):
                    self.robot_detector = RobotDetector(config)
        elif hsv_robot_cfg.get("enabled", False):
            self.robot_detector = RobotDetector(config)

    def process(self, state: FrameState) -> FrameState:
        """Detect balls and robots in frame."""
        # Ball detection
        state.detections = self.ball_detector.detect(state.roi_frame)

        # Robot detection
        if self.robot_detector is not None:
            self.robot_detector.detect_and_track(state.roi_frame)

        return state


class TrackStage(PipelineStage):
    """
    Runs tracking and shot detection.

    This stage MUST be sequential - tracking state depends on previous frames.
    """

    def __init__(self, config: dict, robot_detector,
                 input_queue: queue.Queue, output_queue: queue.Queue,
                 on_shot_launched: Callable = None,
                 on_shot_resolved: Callable = None):
        super().__init__("Track", input_queue, output_queue)
        self.config = config
        self.robot_detector = robot_detector

        # Initialize tracker
        track_cfg = config.get("tracking", {})
        self.tracker = CentroidTracker(
            max_disappeared=track_cfg.get("max_frames_missing", 8),
            max_distance=track_cfg.get("max_distance", 80),
            trail_length=track_cfg.get("trail_length", 30),
        )

        # Initialize shot detector
        # FPS will be set later when we know it
        self.shot_detector = None
        self._fps = 30.0
        self.on_shot_launched = on_shot_launched
        self.on_shot_resolved = on_shot_resolved

    def set_fps(self, fps: float):
        """Set FPS and initialize shot detector."""
        self._fps = fps
        self.shot_detector = ShotDetector(
            self.config,
            robot_detector=self.robot_detector,
            fps=fps
        )
        if self.on_shot_launched or self.on_shot_resolved:
            self.shot_detector.set_callbacks(
                on_shot_launched=self.on_shot_launched,
                on_shot_resolved=self.on_shot_resolved
            )

    def process(self, state: FrameState) -> FrameState:
        """Track objects and detect shots."""
        # Initialize shot detector if not done
        if self.shot_detector is None:
            self.set_fps(self._fps)

        # Update tracker
        objects = self.tracker.update(state.detections)

        # Update shot detection
        ids_to_remove = self.shot_detector.update(objects, state.frame_num)

        # Remove scored balls
        if ids_to_remove:
            self.tracker.remove_objects(ids_to_remove)

        # Save state for rendering
        state.object_states = {}
        for oid, obj in objects.items():
            if obj.disappeared == 0:
                state.object_states[oid] = {
                    "cx": obj.cx, "cy": obj.cy,
                    "radius": obj.radius,
                    "trail": list(obj.trail),
                    "is_moving": obj.is_moving,
                    "vx": obj.vx, "vy": obj.vy,
                    "shot_id": getattr(obj, "shot_id", None),
                    "robot_id": getattr(obj, "robot_id", None),
                    "classification": getattr(obj, "classification", None),
                }

        # Save robot states
        if self.robot_detector is not None:
            state.robot_states = {}
            for rid, robot in self.robot_detector.robots.items():
                if robot.disappeared == 0:
                    state.robot_states[rid] = {
                        "cx": robot.cx, "cy": robot.cy,
                        "bbox": robot.bbox,
                        "robot_bbox": getattr(robot, "robot_bbox", None),
                        "alliance": robot.alliance,
                        "identity": robot.identity,
                    }

        # Build shot info for current frame
        state.shot_info = {}
        for shot in self.shot_detector.shots:
            state.shot_info[shot.obj_id] = {
                "shot_id": shot.shot_id,
                "result": shot.result or "in_flight",
                "robot": shot.robot_name,
                "launch_frame": shot.launch_frame,
                "classification": shot.classification,
            }

        return state

    def get_shot_stats(self) -> dict:
        """Return shot detection statistics."""
        if self.shot_detector is not None:
            return self.shot_detector.get_stats()
        return {}


class RenderStage(PipelineStage):
    """
    Renders annotated frames with delayed buffer for retroactive coloring.

    Uses a ring buffer to delay rendering until shot outcomes are known.
    """

    def __init__(self, config: dict, buffer_size: int,
                 input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__("Render", input_queue, output_queue)
        self.config = config
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.shot_results = {}  # obj_id -> shot info (updated as shots resolve)

    def process(self, state: FrameState) -> Optional[FrameState]:
        """Buffer frame and render delayed frame."""
        # Update shot results from current state
        if state.shot_info:
            self.shot_results.update(state.shot_info)

        # Add to buffer
        self.buffer.append(state)

        # If buffer not full, don't output yet
        if len(self.buffer) < self.buffer_size:
            return None

        # Render oldest frame in buffer
        oldest = self.buffer[0]
        oldest.rendered_frame = self._render_frame(oldest)

        return oldest

    def flush(self) -> List[FrameState]:
        """Flush remaining frames from buffer."""
        results = []
        while self.buffer:
            state = self.buffer.popleft()
            state.rendered_frame = self._render_frame(state)
            results.append(state)
        return results

    def _render_frame(self, state: FrameState) -> np.ndarray:
        """Render a single frame with annotations."""
        frame = state.roi_frame.copy()
        config = self.config
        tracer_mode = config.get("output", {}).get("tracer_mode", "all")

        # Draw zones
        frame = draw_zones(frame, config)

        # Draw robots
        if state.robot_states:
            for rid, rs in state.robot_states.items():
                x, y, w, h = rs["bbox"]
                color = (0, 0, 255) if rs["alliance"] == "red" else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, rs["identity"], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracked objects
        if state.object_states:
            for oid, obj in state.object_states.items():
                cx, cy = int(obj["cx"]), int(obj["cy"])
                radius = int(obj["radius"])

                # Get shot info
                shot_info = self.shot_results.get(oid)
                is_shot = shot_info is not None
                is_field_pass = is_shot and shot_info.get("classification") == "field_pass"

                # Apply tracer mode
                if tracer_mode == "none":
                    continue
                if tracer_mode == "shots_only" and not is_shot:
                    continue

                # Determine color
                if is_shot:
                    if is_field_pass:
                        color = (255, 255, 0)  # Cyan
                    elif shot_info["result"] == "scored":
                        color = (0, 255, 0)  # Green
                    elif shot_info["result"] == "missed":
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 165, 255)  # Orange (in-flight)
                elif obj["is_moving"]:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (60, 60, 60)  # Gray

                # Draw trail
                trail = obj["trail"]
                if len(trail) > 1:
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        tc = tuple(int(c * alpha) for c in color)
                        pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                        pt2 = (int(trail[i][0]), int(trail[i][1]))
                        cv2.line(frame, pt1, pt2, tc, 2)

                # Draw ball
                cv2.circle(frame, (cx, cy), radius, color, 2)

                # Label
                if is_shot:
                    label = f"S{shot_info['shot_id']}"
                    if shot_info["robot"] != "unknown":
                        label += f":{shot_info['robot']}"
                    cv2.putText(frame, label, (cx + 8, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame


class EncodeStage(PipelineStage):
    """Writes frames to video file."""

    def __init__(self, writer, input_queue: queue.Queue):
        super().__init__("Encode", input_queue, None)
        self.writer = writer

    def process(self, state: FrameState) -> None:
        """Write frame to video."""
        if state.rendered_frame is not None:
            self.writer.write(state.rendered_frame)
        return None


class StreamingCSVLogger:
    """Real-time CSV logging for shot events."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", newline="")
        self._file.write("shot_id,classification,robot,launch_frame,launch_x,launch_y,result,result_frame,goal_name\n")
        self._file.flush()
        self._shot_rows = {}  # shot_id -> row data

    def on_shot_launched(self, shot, frame_num):
        """Called when shot is detected."""
        row = {
            "shot_id": shot.shot_id,
            "classification": shot.classification,
            "robot": shot.robot_name,
            "launch_frame": shot.launch_frame,
            "launch_x": shot.launch_x,
            "launch_y": shot.launch_y,
            "result": "",
            "result_frame": "",
            "goal_name": "",
        }
        self._shot_rows[shot.shot_id] = row

    def on_shot_resolved(self, shot, frame_num):
        """Called when shot is resolved."""
        if shot.shot_id in self._shot_rows:
            row = self._shot_rows[shot.shot_id]
            row["result"] = shot.result or "unresolved"
            row["result_frame"] = shot.result_frame or ""
            row["goal_name"] = shot.goal_name or ""

            # Write to CSV
            line = (f"{row['shot_id']},{row['classification']},{row['robot']},"
                    f"{row['launch_frame']},{row['launch_x']:.0f},{row['launch_y']:.0f},"
                    f"{row['result']},{row['result_frame']},{row['goal_name']}\n")
            self._file.write(line)
            self._file.flush()

    def close(self):
        """Close the CSV file."""
        self._file.close()


class RealTimePipeline:
    """
    Orchestrates the streaming pipeline.

    Usage:
        pipeline = RealTimePipeline(source, config, output_path)
        pipeline.start()
        pipeline.run_display_loop()  # Blocks until done
        pipeline.stop()
    """

    def __init__(self, source, config: dict, output_path: str = None,
                 enable_display: bool = True, enable_recording: bool = True,
                 buffer_size: int = 90):
        self.source = source
        self.config = config
        self.output_path = output_path
        self.enable_display = enable_display
        self.enable_recording = enable_recording
        self.buffer_size = buffer_size

        # Queues
        self.decode_queue = queue.Queue(maxsize=10)
        self.detect_queue = queue.Queue(maxsize=10)
        self.track_queue = queue.Queue(maxsize=100)
        self.render_queue = queue.Queue(maxsize=10) if enable_recording else None

        # CSV logger
        self.csv_logger = None
        if output_path:
            csv_path = os.path.splitext(output_path)[0] + "_shots.csv"
            self.csv_logger = StreamingCSVLogger(csv_path)

        # Stages
        self.decode_stage = DecodeStage(source, config, self.decode_queue)

        self.detect_stage = DetectStage(config, self.decode_queue, self.detect_queue)

        self.track_stage = TrackStage(
            config,
            self.detect_stage.robot_detector,
            self.detect_queue, self.track_queue,
            on_shot_launched=self.csv_logger.on_shot_launched if self.csv_logger else None,
            on_shot_resolved=self.csv_logger.on_shot_resolved if self.csv_logger else None,
        )
        self.track_stage.set_fps(source.get_fps())

        self.render_stage = RenderStage(
            config, buffer_size,
            self.track_queue, self.render_queue
        )

        # Video writer
        self.writer = None
        self.encode_stage = None
        if enable_recording and output_path:
            info = source.get_info()
            roi_cfg = config.get("roi", {"w": info["width"], "h": info["height"]})
            size = (roi_cfg["w"], roi_cfg["h"])
            self.writer = create_video_writer(output_path, info["fps"], size, config)
            self.encode_stage = EncodeStage(self.writer, self.render_queue)

        # State
        self._running = False
        self._paused = False
        self._display_frame = None
        self._display_lock = threading.Lock()

    def start(self):
        """Start all pipeline stages."""
        self._running = True

        self.decode_stage.start()
        self.detect_stage.start()
        self.track_stage.start()
        self.render_stage.start()
        if self.encode_stage:
            self.encode_stage.start()

        print("[PIPELINE] Started all stages")

    def stop(self):
        """Stop all pipeline stages."""
        self._running = False

        # Signal decode to stop (it will propagate poison pills)
        self.decode_stage.stop()

        # Wait for stages
        self.decode_stage.join(timeout=2)
        self.detect_stage.join(timeout=2)
        self.track_stage.join(timeout=2)
        self.render_stage.join(timeout=2)
        if self.encode_stage:
            self.encode_stage.join(timeout=2)

        # Flush render buffer
        remaining = self.render_stage.flush()
        for state in remaining:
            if self.writer and state.rendered_frame is not None:
                self.writer.write(state.rendered_frame)

        # Release resources
        if self.writer:
            self.writer.release()
        if self.csv_logger:
            self.csv_logger.close()
        self.source.release()

        print("[PIPELINE] Stopped all stages")
        self._print_stats()

    def _print_stats(self):
        """Print pipeline statistics."""
        print("\n" + "=" * 60)
        print("  PIPELINE STATISTICS")
        print("=" * 60)

        for stage in [self.decode_stage, self.detect_stage,
                      self.track_stage, self.render_stage]:
            stats = stage.get_stats()
            print(f"  {stats['name']:12s}: {stats['frames']:5d} frames, "
                  f"{stats['avg_ms']:.1f}ms avg, {stats['fps']:.0f} fps")

        shot_stats = self.track_stage.get_shot_stats()
        if shot_stats:
            print(f"\n  Shots: {shot_stats.get('shots_total', 0)} total, "
                  f"{shot_stats.get('shots_scored', 0)} scored, "
                  f"{shot_stats.get('shots_missed', 0)} missed")

        print("=" * 60 + "\n")

    def run_display_loop(self, window_name: str = "FRC Ball Tracker"):
        """
        Run the main display loop (blocks until done or quit).

        Controls:
            SPACE - Pause/Resume
            Q/ESC - Quit
            D     - Toggle debug info
        """
        if not self.enable_display:
            # Just wait for pipeline to finish
            while self._running and self.track_stage.is_alive():
                time.sleep(0.1)
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        show_debug = False

        while self._running:
            # Get frame from track queue for display
            try:
                state = self.track_queue.get(timeout=0.05)
            except queue.Empty:
                # Check if pipeline is done
                if not self.track_stage.is_alive():
                    break
                continue

            if state is _POISON_PILL:
                # Propagate to render
                if self.render_queue:
                    self.render_queue.put(state)
                break

            # Pass to render stage
            if self.render_queue:
                self.render_queue.put(state)

            # Also display (render locally for display)
            display_frame = self._quick_render(state)

            # Add HUD
            shot_stats = self.track_stage.get_shot_stats()
            hud_stats = {
                "balls_detected": len(state.detections) if state.detections else 0,
                "balls_tracked": len(state.object_states) if state.object_states else 0,
                **shot_stats,
            }
            info = self.source.get_info()
            frame_count = info.get("frame_count", 0)
            display_frame = draw_hud(display_frame, hud_stats, state.frame_num, frame_count)

            if show_debug:
                # Add debug info
                cv2.putText(display_frame, f"Q: {self.track_queue.qsize()}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self._paused = not self._paused
            elif key == ord('d'):
                show_debug = not show_debug

        cv2.destroyAllWindows()

    def _quick_render(self, state: FrameState) -> np.ndarray:
        """Quick render for display (without buffer delay)."""
        frame = state.roi_frame.copy()
        config = self.config

        # Draw zones
        frame = draw_zones(frame, config)

        # Draw robots
        if state.robot_states:
            for rid, rs in state.robot_states.items():
                x, y, w, h = rs["bbox"]
                color = (0, 0, 255) if rs["alliance"] == "red" else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw tracked objects
        if state.object_states:
            for oid, obj in state.object_states.items():
                cx, cy = int(obj["cx"]), int(obj["cy"])
                radius = int(obj["radius"])

                # Color based on movement
                shot_id = obj.get("shot_id")
                if shot_id is not None:
                    color = (0, 165, 255)  # Orange for shots
                elif obj["is_moving"]:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (80, 80, 80)  # Gray

                cv2.circle(frame, (cx, cy), radius, color, 2)

        return frame


def run_streaming_pipeline(source, config: dict, output_path: str = None,
                           enable_display: bool = True,
                           enable_recording: bool = True,
                           buffer_size: int = 90):
    """
    Convenience function to run the streaming pipeline.

    Args:
        source: VideoSource instance or path/index to open
        config: Configuration dict
        output_path: Output video path (None to disable recording)
        enable_display: Show live display window
        enable_recording: Write output video
        buffer_size: Frame buffer size for delayed rendering

    Returns:
        Shot statistics dict
    """
    # Open source if needed
    if not isinstance(source, (FileSource, LiveCameraSource)):
        source = VideoSource.open(source, config)

    pipeline = RealTimePipeline(
        source, config, output_path,
        enable_display=enable_display,
        enable_recording=enable_recording,
        buffer_size=buffer_size,
    )

    pipeline.start()
    pipeline.run_display_loop()
    pipeline.stop()

    return pipeline.track_stage.get_shot_stats()


if __name__ == "__main__":
    # Test the pipeline
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stream_pipeline.py <video.mkv> [output.mp4]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    config = load_config()
    source = VideoSource.open(video_path, config)

    stats = run_streaming_pipeline(
        source, config, output_path,
        enable_display=True,
        enable_recording=output_path is not None,
    )

    print(f"\nFinal stats: {stats}")
