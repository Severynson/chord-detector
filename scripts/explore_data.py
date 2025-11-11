#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
import wave
from statistics import mean, stdev, quantiles

TRAIN_ROOT = os.path.join("data", "sound", "Train")


def get_wav_duration_seconds(path: str) -> float | None:
    """Return duration in seconds for a PCM WAV file, or None if unreadable."""
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return None
            return frames / float(rate)
    except Exception:
        return None  # unreadable / not a standard PCM WAV


def describe(durations: list[float]) -> dict:
    """Compute basic stats with safe handling for small samples."""
    if not durations:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "q1": None,
            "median": None,
            "q3": None,
            "stdev": None,
        }
    dsorted = sorted(durations)
    n = len(dsorted)
    q1, median, q3 = quantiles(dsorted, n=4, method="inclusive")
    stats = {
        "count": n,
        "min": dsorted[0],
        "max": dsorted[-1],
        "mean": mean(dsorted),
        "q1": q1,
        "median": median,
        "q3": q3,
        # Use sample standard deviation when n > 1, else None
        "stdev": stdev(dsorted) if n > 1 else None,
    }
    return stats


def fmt(x):
    return f"{x:.3f}s" if isinstance(x, (int, float)) else "—"


def analyze_folder(folder_path: str) -> tuple[dict, list[tuple[str, float]]]:
    durations = []
    per_file = []
    for entry in os.scandir(folder_path):
        if entry.is_file() and entry.name.lower().endswith(".wav"):
            dur = get_wav_duration_seconds(entry.path)
            if dur is not None:
                durations.append(dur)
                per_file.append((entry.name, dur))
            else:
                # Skip corrupted / unsupported wavs silently; could log if desired
                pass
    return describe(durations), per_file


def main():
    parser = argparse.ArgumentParser(
        description="Analyze WAV durations under data/sound/Train/<folders>."
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        help="Specific subfolders under data/sound/Train to analyze. "
        "If omitted, analyzes all immediate subfolders.",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Also list each file's duration per folder.",
    )
    args = parser.parse_args()

    if not os.path.isdir(TRAIN_ROOT):
        print(f"Error: '{TRAIN_ROOT}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.folders and len(args.folders) > 0:
        subfolders = [os.path.join(TRAIN_ROOT, f) for f in args.folders]
    else:
        # All immediate subdirectories in TRAIN_ROOT
        subfolders = [e.path for e in os.scandir(TRAIN_ROOT) if e.is_dir()]

    if not subfolders:
        print("No folders to analyze.")
        sys.exit(0)

    overall_durations = []
    print(f"Analyzing WAV durations under: {TRAIN_ROOT}\n")

    for folder in sorted(subfolders):
        if not os.path.isdir(folder):
            print(f"- Skipping (not a directory): {folder}")
            continue

        stats, per_file = analyze_folder(folder)
        folder_name = os.path.basename(folder)
        print(f"Folder: {folder_name}")
        print(f"  Files analyzed: {stats['count']}")
        print(f"  Min:    {fmt(stats['min'])}")
        print(f"  Q1:     {fmt(stats['q1'])}")
        print(f"  Median: {fmt(stats['median'])}")
        print(f"  Q3:     {fmt(stats['q3'])}")
        print(f"  Max:    {fmt(stats['max'])}")
        print(f"  Mean:   {fmt(stats['mean'])}")
        print(f"  StdDev: {fmt(stats['stdev'])}")
        if args.list_files and per_file:
            for name, dur in sorted(per_file):
                print(f"    - {name}: {dur:.3f}s")
        print()

        overall_durations.extend(d for _, d in per_file)

    # Overall stats across all analyzed folders
    overall = describe(overall_durations)
    print("Overall (all folders combined)")
    print(f"  Files analyzed: {overall['count']}")
    print(f"  Min:    {fmt(overall['min'])}")
    print(f"  Q1:     {fmt(overall['q1'])}")
    print(f"  Median: {fmt(overall['median'])}")
    print(f"  Q3:     {fmt(overall['q3'])}")
    print(f"  Max:    {fmt(overall['max'])}")
    print(f"  Mean:   {fmt(overall['mean'])}")
    print(f"  StdDev: {fmt(overall['stdev'])}")


if __name__ == "__main__":
    main()

# •	Analyze specific subfolders:
#       python analyze_wavs.py --folders cat dog bird
# •	Analyze all immediate subfolders:
#       python analyze_wavs.py
# •	Also list each file’s duration:
#       python analyze_wavs.py --list-files
