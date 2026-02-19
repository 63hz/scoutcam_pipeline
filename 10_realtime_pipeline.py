# -*- coding: utf-8 -*-
"""
10 - Real-Time FRC Ball Tracking Pipeline
==========================================
Streaming pipeline for real-time ball tracking with simultaneous
display, logging, and recording.

FEATURES:
    - 60+ fps sustained processing on RTX 3090
    - Live display with ~1.5s latency for shot resolution
    - Simultaneous CSV logging and video recording
    - Support for video files, local cameras, and RTSP streams

MODES:
    --stream     Real-time streaming pipeline (default)
    --batch      Traditional two-pass batch processing
    --benchmark  FPS measurement mode (no display/recording)

USAGE:
    python 10_realtime_pipeline.py video.mkv [--output out.mp4]
    python 10_realtime_pipeline.py --camera 0  # Local camera
    python 10_realtime_pipeline.py rtsp://... --output stream.mp4

CONTROLS:
    SPACE - Pause/Resume
    Q/ESC - Quit
    D     - Toggle debug view
    S     - Save config
    R     - Toggle recording (live mode)

Author: Clay / Claude sandbox
"""

import sys
import os
import argparse
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import (
    load_config, save_config, VideoSource, create_video_writer,
    print_gpu_status,
)
from stream_pipeline import RealTimePipeline, run_streaming_pipeline


def run_batch_mode(video_path: str, output_path: str, config: dict):
    """
    Run traditional two-pass batch processing.

    This is the original 05_full_pipeline.py behavior for comparison.
    """
    # Import the original pipeline
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "full_pipeline",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_full_pipeline.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("\n" + "=" * 60)
    print("  BATCH MODE (Two-Pass)")
    print("=" * 60)

    mod.run_pipeline(video_path, output_path)


def run_benchmark_mode(video_path: str, config: dict):
    """
    Benchmark mode - measure FPS without display or recording.
    """
    from stream_pipeline import (
        VideoSource, DecodeStage, DetectStage, TrackStage,
        _POISON_PILL
    )
    import queue

    print("\n" + "=" * 60)
    print("  BENCHMARK MODE")
    print("=" * 60)
    print_gpu_status()

    source = VideoSource.open(video_path, config)
    info = source.get_info()

    print(f"  Input: {video_path}")
    print(f"  Resolution: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    print(f"  Frames: {info['frame_count']}")
    print("=" * 60)

    # Create simplified pipeline (no render/encode)
    decode_queue = queue.Queue(maxsize=10)
    detect_queue = queue.Queue(maxsize=10)
    track_queue = queue.Queue(maxsize=100)

    decode_stage = DecodeStage(source, config, decode_queue)
    detect_stage = DetectStage(config, decode_queue, detect_queue)
    track_stage = TrackStage(config, detect_stage.robot_detector,
                             detect_queue, track_queue)
    track_stage.set_fps(info["fps"])

    # Start stages
    t_start = time.time()
    decode_stage.start()
    detect_stage.start()
    track_stage.start()

    # Consume track output
    frame_count = 0
    while True:
        try:
            state = track_queue.get(timeout=1.0)
        except queue.Empty:
            if not track_stage.is_alive():
                break
            continue

        if state is _POISON_PILL:
            break

        frame_count += 1
        if frame_count % 500 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed
            print(f"  Processed {frame_count} frames ({fps:.0f} fps)")

    # Wait for stages
    decode_stage.join(timeout=2)
    detect_stage.join(timeout=2)
    track_stage.join(timeout=2)
    source.release()

    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Frames: {frame_count}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Average FPS: {frame_count / total_time:.1f}")
    print()

    for stage in [decode_stage, detect_stage, track_stage]:
        stats = stage.get_stats()
        print(f"  {stats['name']:12s}: {stats['avg_ms']:.1f}ms avg, "
              f"{stats['fps']:.0f} fps theoretical")

    shot_stats = track_stage.get_shot_stats()
    if shot_stats:
        print(f"\n  Shots: {shot_stats.get('shots_total', 0)} detected")

    print("=" * 60 + "\n")


def run_stream_mode(source, output_path: str, config: dict,
                    enable_display: bool = True,
                    enable_recording: bool = True):
    """
    Run real-time streaming pipeline.
    """
    print("\n" + "=" * 60)
    print("  STREAMING MODE")
    print("=" * 60)
    print_gpu_status()

    info = source.get_info()
    print(f"  Source: {info.get('path', info.get('source', 'unknown'))}")
    print(f"  Resolution: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    if info.get("frame_count", 0) > 0:
        print(f"  Frames: {info['frame_count']}")
    print(f"  Display: {'ON' if enable_display else 'OFF'}")
    print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
    if output_path:
        print(f"  Output: {output_path}")
    print("=" * 60)

    # Calculate buffer size based on fps (~1.5 seconds)
    buffer_size = int(info["fps"] * 1.5)

    stats = run_streaming_pipeline(
        source, config, output_path,
        enable_display=enable_display,
        enable_recording=enable_recording,
        buffer_size=buffer_size,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    if output_path:
        csv_path = os.path.splitext(output_path)[0] + "_shots.csv"
        print(f"  Output: {output_path}")
        print(f"  CSV: {csv_path}")
    print(f"\n  Shot Summary:")
    print(f"    Total: {stats.get('shots_total', 0)}")
    print(f"    Scored: {stats.get('shots_scored', 0)}")
    print(f"    Missed: {stats.get('shots_missed', 0)}")
    print(f"    Passes: {stats.get('field_passes', 0)}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time FRC Ball Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 10_realtime_pipeline.py video.mkv
  python 10_realtime_pipeline.py video.mkv --output annotated.mp4
  python 10_realtime_pipeline.py --camera 0 --output live.mp4
  python 10_realtime_pipeline.py rtsp://192.168.1.100/stream1
  python 10_realtime_pipeline.py video.mkv --benchmark
  python 10_realtime_pipeline.py video.mkv --batch --output annotated.mp4
"""
    )

    # Input source
    parser.add_argument(
        "source", nargs="?",
        help="Video file path, RTSP URL, or omit for --camera"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera device index (e.g., 0 for default camera)"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output video path (auto-generates if not specified)"
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live display window"
    )

    # Mode
    parser.add_argument(
        "--stream", action="store_true", default=True,
        help="Use real-time streaming pipeline (default)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use traditional two-pass batch processing"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark mode (no display/recording)"
    )

    # Config
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config file path"
    )

    args = parser.parse_args()

    # Determine source
    if args.camera is not None:
        source_spec = args.camera
    elif args.source:
        source_spec = args.source
    else:
        print("Error: Must specify video file, RTSP URL, or --camera")
        parser.print_help()
        sys.exit(1)

    # Load config
    config = load_config(args.config) if args.config else load_config()

    # Determine output path
    output_path = args.output
    if output_path is None and not args.no_record and not args.benchmark:
        if isinstance(source_spec, str) and not source_spec.startswith("rtsp://"):
            base = os.path.splitext(os.path.basename(source_spec))[0]
            output_path = f"{base}_realtime.mp4"
        elif args.camera is not None:
            output_path = f"camera_{args.camera}_capture.mp4"
        else:
            output_path = "stream_capture.mp4"

    # Run appropriate mode
    if args.benchmark:
        if isinstance(source_spec, int):
            print("Error: Benchmark mode requires a video file")
            sys.exit(1)
        run_benchmark_mode(source_spec, config)

    elif args.batch:
        if isinstance(source_spec, int) or (isinstance(source_spec, str) and
                                             source_spec.startswith("rtsp://")):
            print("Error: Batch mode requires a video file")
            sys.exit(1)
        run_batch_mode(source_spec, output_path, config)

    else:
        # Streaming mode (default)
        source = VideoSource.open(source_spec, config)
        run_stream_mode(
            source, output_path, config,
            enable_display=not args.no_display,
            enable_recording=not args.no_record,
        )


if __name__ == "__main__":
    main()
