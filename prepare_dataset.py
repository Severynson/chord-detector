#!/usr/bin/env python3
"""
prepare_dataset_annotated_only.py
---------------------------------
Create training tensors from audio + time-interval chord labels,
USING ONLY EXPLICITLY ANNOTATED INTERVALS from the CSV. Unlabeled
regions are ignored (not converted to 'N').

Input:
  - WAV files in: data/sound/*.wav
  - CSV labels   in: data/labels/<same_stem>.csv
    columns: start,end,chord  (times may be MM:SS.s / HH:MM:SS.s / seconds)

Output (default: data/processed):
  - features/<stem>.npz  -> arrays: X (T_kept, F), y (T_kept,)
  - labels.json          -> { "label_to_index": {...}, "index_to_label": [...] }
  - manifest.tsv         -> summary of produced files and shapes

Deps:
  pip install librosa soundfile numpy pandas tqdm
"""

import os
import json
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# ======================
# Config
# ======================

AUDIO_DIR = "data/sound"
LABEL_DIR = "data/labels"
OUT_DIR = "data/processed"

# Feature settings
FEATURE_TYPE = "mel"  # "mel" or "cqt"
SR = 44100  # resample target (Hz)
FPS = 100  # frames per second (hop ~ SR/FPS)
N_FFT = 2048  # STFT window size (samples) for mel
N_MELS = 128  # mel bins
MEL_FMIN = 30.0
MEL_FMAX = None  # None = SR/2

# CQT settings (used if FEATURE_TYPE == "cqt")
CQT_BINS_PER_OCTAVE = 24
CQT_N_BINS = 84
CQT_FMIN = 32.7031956626  # C1

# Labeling behavior
USE_FRAME_CENTER = True  # label frame by its center time; else use frame start
ONLY_USE_ANNOTATED = True  # keep ONLY frames inside any CSV interval
EXCLUDE_LABELS: Set[str] = set()  # e.g., {"N"} to drop explicit N-labeled frames too

# Misc
ALLOW_MISSING_LABELS = True  # skip WAVs with missing CSV (warn)
RMS_NORMALIZE = True  # simple per-file RMS normalization (target ~0.1)

# Split behavior
USE_IMPLICIT_TRAIN_TEST_SPLIT = True # If True, use ' train'/' val' suffixes

# ======================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_wav_mono(path: str, sr: int) -> np.ndarray:
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    y = y - np.mean(y)  # remove DC
    if RMS_NORMALIZE:
        rms = np.sqrt(np.mean(y**2) + 1e-12)
        if rms > 0:
            y = y / rms * 0.1
    return y


def compute_mel(y: np.ndarray, sr: int, fps: int) -> Tuple[np.ndarray, int]:
    hop_length = int(sr / fps)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=hop_length,
        n_mels=N_MELS,
        fmin=MEL_FMIN,
        fmax=(sr / 2 if MEL_FMAX is None else MEL_FMAX),
        center=False,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T, N_FFT  # (T, F), window used


def compute_cqt(y: np.ndarray, sr: int, fps: int) -> Tuple[np.ndarray, Optional[int]]:
    hop_length = int(sr / fps)
    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
        center=False,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_db.T, None  # (T, F), window unknown/unused


@dataclass
class Interval:
    start: float
    end: float
    label: str


def _parse_time_to_seconds(x) -> float:
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Unrecognized time format: {x}")
    if len(parts) == 3:
        h, m, sec = parts
        return 3600 * h + 60 * m + sec
    if len(parts) == 2:
        m, sec = parts
        return 60 * m + sec
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"Unrecognized time format: {x}")


def load_intervals(csv_path: str) -> List[Interval]:
    df = pd.read_csv(csv_path)
    colmap = {c.lower().strip(): c for c in df.columns}
    req = ["start", "end", "chord"]
    for r in req:
        if r not in colmap:
            raise ValueError(
                f"CSV {csv_path} must contain columns: {req}, got {list(df.columns)}"
            )
    s_col, e_col, l_col = colmap["start"], colmap["end"], colmap["chord"]

    intervals: List[Interval] = []
    for _, row in df.iterrows():
        s = _parse_time_to_seconds(row[s_col])
        e = _parse_time_to_seconds(row[e_col])
        lab = str(row[l_col]).strip()
        if e > s:
            intervals.append(Interval(s, e, lab))
    intervals.sort(key=lambda x: x.start)
    return intervals


def frame_times(n_frames: int, sr: int, fps: int, window: Optional[int]) -> np.ndarray:
    """
    Return the time (in SECONDS) for each frame.
    hop_seconds = 1.0 / fps  (NOT sr / fps)
    If USE_FRAME_CENTER is True, add half hop (or half window for STFT) as center offset.
    """
    hop_s = 1.0 / fps              # <-- seconds per frame
    t = np.arange(n_frames) * hop_s
    if USE_FRAME_CENTER:
        if window is None:
            # e.g., for CQT where we don't track a fixed window; use half-hop center
            t = t + (hop_s / 2.0)
        else:
            # for STFT/mel with known window length in samples
            t = t + (window / (2.0 * sr))
    return t


def make_keep_mask_and_labels(
    ts: np.ndarray, intervals: List[Interval]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build boolean mask of frames to keep (inside any interval) and their labels.
    Only annotated portions are kept. Unlabeled time is dropped.
    """
    T = ts.shape[0]
    keep = np.zeros(T, dtype=bool)
    labels = np.empty(T, dtype=object)

    if not intervals:
        return keep, labels

    # Sweep through intervals
    i = 0
    for iv in intervals:
        # mark frames where ts in [iv.start, iv.end)
        while i < T and ts[i] < iv.start:
            i += 1
        j = i
        while j < T and ts[j] < iv.end:
            keep[j] = True
            labels[j] = iv.label
            j += 1
        i = j  # continue from here
    return keep, labels


def process_file(
    wav_path: str,
    csv_path: str,
    out_dir: str,
    feature_type: str,
    sr: int,
    fps: int,
    manifest_rows: List[str],
    label_map: Dict[str, int],
) -> Dict[str, int]:
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    y = load_wav_mono(wav_path, sr)

    # Features
    if feature_type == "mel":
        X, window = compute_mel(y, sr, fps)
    elif feature_type == "cqt":
        X, window = compute_cqt(y, sr, fps)
    else:
        raise ValueError("FEATURE_TYPE must be 'mel' or 'cqt'")

    n_frames = X.shape[0]
    ts = frame_times(n_frames, sr, fps, window)

    # Intervals -> keep mask & raw string labels
    intervals = load_intervals(csv_path)
    keep, raw_labels = make_keep_mask_and_labels(ts, intervals)

    # Optionally drop specific labels (e.g., EXCLUDE_LABELS={'N'})
    if EXCLUDE_LABELS:
        drop_mask = np.zeros_like(keep)
        # mark as drop any kept frames whose label is excluded
        drop_mask[keep] = np.array(
            [lab in EXCLUDE_LABELS for lab in raw_labels[keep]], dtype=bool
        )
        keep = keep & (~drop_mask)

    # After filtering, if nothing left -> skip this file
    if not np.any(keep):
        tqdm.write(f"[WARN] No annotated frames kept for '{stem}', skipping.")
        return label_map

    X_kept = X[keep]  # (T_kept, F)
    labels_kept = raw_labels[keep]  # (T_kept,)

    # Update label map with ONLY labels we actually kept
    for lab in np.unique(labels_kept):
        if lab not in label_map:
            label_map[lab] = len(label_map)

    y_idx = np.array([label_map[lab] for lab in labels_kept], dtype=np.int64)

    # Save
    out_features = os.path.join(out_dir, "features")
    ensure_dir(out_features)
    out_path = os.path.join(out_features, f"{stem}.npz")
    np.savez_compressed(out_path, X=X_kept, y=y_idx)

    manifest_rows.append(
        f"{stem}\t{feature_type}\t{SR}\t{FPS}\t{X_kept.shape[0]}\t{X_kept.shape[1]}"
    )
    return label_map


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "features"))

    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    if not wav_files:
        print(f"No WAV files found in {AUDIO_DIR}")
        return

    manifest_rows = ["stem\tfeat\tsr\tfps\tn_frames_kept\tn_feats"]
    label_map: Dict[str, int] = {}

    for wav in tqdm(wav_files, desc="Processing audio"):
        stem = os.path.splitext(os.path.basename(wav))[0]
        csv = os.path.join(LABEL_DIR, f"{stem}.csv")

        # Skip if label CSV is missing
        if not os.path.exists(csv):
            tqdm.write(f"[WARN] Missing labels for '{stem}', skipping (expected {csv})")
            continue

        try:
            label_map = process_file(
                wav_path=wav,
                csv_path=csv,
                out_dir=OUT_DIR,
                feature_type=FEATURE_TYPE,
                sr=SR,
                fps=FPS,
                manifest_rows=manifest_rows,
                label_map=label_map,
            )
        except Exception as e:
            tqdm.write(f"[WARN] Failed processing '{stem}': {e}. Skipping.")
            continue

    if len(manifest_rows) == 1:
        print("No files were processed. Check CSV alignment and labels.")
        return

    # Save label maps
    index_to_label = [None] * len(label_map)
    for lab, idx in label_map.items():
        index_to_label[idx] = lab
    labels_json = {"label_to_index": label_map, "index_to_label": index_to_label}
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    # NEW: Create train/val split map if using implicit split
    if USE_IMPLICIT_TRAIN_TEST_SPLIT:
        train_stems = []
        val_stems = []
        # Get stems from the original wav files list to handle cases where processing fails
        all_wav_stems = [os.path.splitext(os.path.basename(f))[0] for f in wav_files]

        for stem in all_wav_stems:
            # Check for space before train/val to avoid matching parts of names
            if stem.endswith(" train"):
                train_stems.append(stem)
            elif stem.endswith(" val"):
                val_stems.append(stem)

        split_map = {"train": train_stems, "val": val_stems}
        split_map_path = os.path.join(OUT_DIR, "split_map.json")
        with open(split_map_path, "w", encoding="utf-8") as f:
            json.dump(split_map, f, indent=2)
        print(f"Saved implicit train/val split map to {split_map_path}")


    # Save manifest
    with open(os.path.join(OUT_DIR, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_rows) + "\n")

    print(f"Done. Wrote {len(manifest_rows)-1} files to {OUT_DIR}.")
    print(f"- Features: {os.path.join(OUT_DIR, 'features')}")
    print(f"- Labels map: {os.path.join(OUT_DIR, 'labels.json')}")
    print(f"- Manifest: {os.path.join(OUT_DIR, 'manifest.tsv')}")


if __name__ == "__main__":
    main()
