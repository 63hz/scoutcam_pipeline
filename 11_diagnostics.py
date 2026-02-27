# -*- coding: utf-8 -*-
"""
11 - Tracking Quality Diagnostics
===================================
Headless diagnostic tool that evaluates detection, tracking, and shot quality.
Outputs structured JSON for automated analysis — no GUI required.

USAGE:
    python 11_diagnostics.py video.mkv                           # All frames, JSON to stdout
    python 11_diagnostics.py video.mkv --frames 0-500 --pretty   # First 500 frames, readable
    python 11_diagnostics.py video.mkv --compare centroid,kalman  # A/B comparison
    python 11_diagnostics.py video.mkv --override ball_tracking.tracker=kalman --pretty
    python 11_diagnostics.py video.mkv --no-shots --no-robots     # Tracking-only (fastest)
    python 11_diagnostics.py video.mkv -o results.json            # Save to file
    python 11_diagnostics.py video.mkv --timeseries frames.csv    # Per-frame CSV

Author: Clay / Claude sandbox
"""

import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_tracker_utils import load_config, DiagnosticRunner


def apply_overrides(config, overrides):
    """Apply dot-notation overrides to config dict.

    Example: 'ball_tracking.tracker=kalman' sets config["ball_tracking"]["tracker"] = "kalman"
    """
    for override in overrides:
        if "=" not in override:
            print(f"[WARN] Skipping invalid override (no '='): {override}",
                  file=sys.stderr)
            continue
        key, value = override.split("=", 1)
        parts = key.strip().split(".")

        # Auto-convert numeric values
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                # Keep as string; handle booleans
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

        # Walk into nested dict, creating intermediate dicts if needed
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value


def run_single(video_path, config, args, label=None):
    """Run a single diagnostic pass and return results dict."""
    runner = DiagnosticRunner(
        config, video_path, label=label,
        enable_shots=not args.no_shots,
        enable_robots=not args.no_robots,
    )

    start_frame = 0
    end_frame = None
    if args.frames:
        parts = args.frames.split("-")
        start_frame = int(parts[0])
        if len(parts) > 1 and parts[1]:
            end_frame = int(parts[1])

    results = runner.run(
        start_frame=start_frame,
        end_frame=end_frame,
        quiet=args.quiet,
    )

    # Write timeseries CSV if requested
    if args.timeseries:
        ts = runner.get_timeseries()
        if ts:
            with open(args.timeseries, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ts[0].keys())
                writer.writeheader()
                writer.writerows(ts)
            if not args.quiet:
                print(f"[DIAG] Timeseries written to {args.timeseries}",
                      file=sys.stderr)

    return results


def run_comparison(video_path, tracker_names, args):
    """Run A/B comparison across multiple tracker configs."""
    base_config = load_config()
    if args.override:
        apply_overrides(base_config, args.override)

    all_results = {}
    for name in tracker_names:
        if not args.quiet:
            print(f"\n[DIAG] === Running tracker: {name} ===", file=sys.stderr)
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("ball_tracking", {})["tracker"] = name
        all_results[name] = run_single(video_path, cfg, args, label=name)

    # Compute deltas between first two trackers
    names = list(all_results.keys())
    deltas = {}
    if len(names) >= 2:
        a_tracking = all_results[names[0]]["tracking"]
        b_tracking = all_results[names[1]]["tracking"]
        compare_keys = [
            ("id_creation_rate", True),   # lower is better
            ("short_lived_tracks_pct", True),
            ("match_rate", False),        # higher is better
            ("velocity_p99", True),
            ("avg_track_lifespan", False),
            ("simultaneous_peak", None),  # informational
        ]

        improved_count = 0
        total_scored = 0
        for key, lower_is_better in compare_keys:
            a_val = a_tracking.get(key, 0)
            b_val = b_tracking.get(key, 0)
            if a_val != 0:
                delta_pct = round((b_val - a_val) / abs(a_val) * 100, 1)
            else:
                delta_pct = 0.0

            improved = None
            if lower_is_better is not None:
                total_scored += 1
                if lower_is_better:
                    improved = b_val < a_val
                else:
                    improved = b_val > a_val
                if improved:
                    improved_count += 1

            deltas[key] = {
                "a": a_val, "b": b_val,
                "delta_pct": delta_pct,
                "improved": improved,
            }

        winner = names[1] if improved_count > total_scored / 2 else names[0]
        verdict = (f"{winner} is better "
                   f"({improved_count}/{total_scored} metrics improved)")
    else:
        verdict = "Need 2+ trackers to compare"

    return {
        "mode": "comparison",
        "results": all_results,
        "comparison": {
            "deltas": deltas,
            "verdict": verdict,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tracking/detection/shot quality (headless)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frames", metavar="START-END",
                        help="Frame range, e.g. 0-500")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write JSON to file instead of stdout")
    parser.add_argument("--timeseries", metavar="FILE",
                        help="Write per-frame CSV")
    parser.add_argument("--compare", metavar="TRACKERS",
                        help="Comma-separated tracker names for A/B test")
    parser.add_argument("--override", action="append", metavar="KEY=VALUE",
                        help="Dot-notation config override (repeatable)")
    parser.add_argument("--no-shots", action="store_true",
                        help="Skip shot detection (faster)")
    parser.add_argument("--no-robots", action="store_true",
                        help="Skip robot detection (faster)")
    parser.add_argument("--pretty", action="store_true",
                        help="Indent JSON output")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stderr progress")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        tracker_names = [t.strip() for t in args.compare.split(",")]
        results = run_comparison(args.video, tracker_names, args)
    else:
        config = load_config()
        if args.override:
            apply_overrides(config, args.override)
        results = run_single(args.video, config, args)

    # Output
    indent = 2 if args.pretty else None
    json_str = json.dumps(results, indent=indent)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
            f.write("\n")
        if not args.quiet:
            print(f"[DIAG] Results written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
