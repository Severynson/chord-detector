#!/usr/bin/env python3

# Run as "python3 -m scripts.prepare_dataset"
"""
prepare_dataset.py
---------------------------------
Create training tensors from audio data. Supports multiple modes:
- "implicit_split": Uses ' train'/' val' suffixes on files in AUDIO_DIR.
- "random_split": Randomly splits files from AUDIO_DIR.
- "presplit_folders": Uses data/train and data/test folders, deriving labels from sub-folder names.

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

from src.config import (
    DATASET_TYPE, AUDIO_DIR, LABEL_DIR, TRAIN_AUDIO_DIR, TEST_AUDIO_DIR, PROCESSED_DIR,
    FEATURE_TYPE, SAMPLE_RATE, FPS, N_FFT, N_MELS, MEL_FMIN, MEL_FMAX,
    CQT_BINS_PER_OCTAVE, CQT_N_BINS, CQT_FMIN
)

# For legacy config files
USE_FRAME_CENTER = True
EXCLUDE_LABELS: Set[str] = set()
RMS_NORMALIZE = True

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
        y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS,
        fmin=MEL_FMIN, fmax=(sr / 2 if MEL_FMAX is None else MEL_FMAX),
        center=False, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T, N_FFT

def compute_cqt(y: np.ndarray, sr: int, fps: int) -> Tuple[np.ndarray, Optional[int]]:
    hop_length = int(sr / fps)
    C = librosa.cqt(
        y=y, sr=sr, hop_length=hop_length, fmin=CQT_FMIN,
        n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE, center=False,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_db.T, None

@dataclass
class Interval:
    start: float
    end: float
    label: str

def _parse_time_to_seconds(x) -> float:
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "": return 0.0
    try: return float(s)
    except ValueError: pass
    parts = s.split(":")
    try: parts = [float(p) for p in parts]
    except ValueError: raise ValueError(f"Unrecognized time format: {x}")
    if len(parts) == 3: h, m, sec = parts; return 3600 * h + 60 * m + sec
    if len(parts) == 2: m, sec = parts; return 60 * m + sec
    if len(parts) == 1: return float(parts[0])
    raise ValueError(f"Unrecognized time format: {x}")

def load_intervals_from_csv(csv_path: str) -> List[Interval]:
    df = pd.read_csv(csv_path)
    colmap = {c.lower().strip(): c for c in df.columns}
    req = ["start", "end", "chord"]
    for r in req:
        if r not in colmap:
            raise ValueError(f"CSV {csv_path} must contain columns: {req}, got {list(df.columns)}")
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
    hop_s = 1.0 / fps
    t = np.arange(n_frames) * hop_s
    if USE_FRAME_CENTER:
        if window is None: t = t + (hop_s / 2.0)
        else: t = t + (window / (2.0 * sr))
    return t

def make_keep_mask_and_labels(ts: np.ndarray, intervals: List[Interval]) -> Tuple[np.ndarray, np.ndarray]:
    T = ts.shape[0]
    keep = np.zeros(T, dtype=bool)
    labels = np.empty(T, dtype=object)
    if not intervals: return keep, labels
    i = 0
    for iv in intervals:
        while i < T and ts[i] < iv.start: i += 1
        j = i
        while j < T and ts[j] < iv.end:
            keep[j] = True
            labels[j] = iv.label
            j += 1
        i = j
    return keep, labels

def process_file_from_csv(
    wav_path: str, label_map: Dict[str, int], manifest_rows: List[str]
) -> Optional[Dict[str, int]]:
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    csv_path = os.path.join(LABEL_DIR, f"{stem}.csv")
    if not os.path.exists(csv_path):
        tqdm.write(f"[WARN] Missing labels for '{stem}', skipping (expected {csv_path})")
        return None

    y = load_wav_mono(wav_path, SAMPLE_RATE)
    if FEATURE_TYPE == "mel": X, window = compute_mel(y, SAMPLE_RATE, FPS)
    elif FEATURE_TYPE == "cqt": X, window = compute_cqt(y, SAMPLE_RATE, FPS)
    else: raise ValueError("FEATURE_TYPE must be 'mel' or 'cqt'")

    ts = frame_times(X.shape[0], SAMPLE_RATE, FPS, window)
    intervals = load_intervals_from_csv(csv_path)
    keep, raw_labels = make_keep_mask_and_labels(ts, intervals)

    if EXCLUDE_LABELS:
        drop_mask = np.zeros_like(keep)
        drop_mask[keep] = np.array([lab in EXCLUDE_LABELS for lab in raw_labels[keep]], dtype=bool)
        keep = keep & (~drop_mask)

    if not np.any(keep):
        tqdm.write(f"[WARN] No annotated frames kept for '{stem}', skipping.")
        return label_map

    X_kept, labels_kept = X[keep], raw_labels[keep]
    for lab in np.unique(labels_kept):
        if lab not in label_map: label_map[lab] = len(label_map)
    
    y_idx = np.array([label_map[lab] for lab in labels_kept], dtype=np.int64)

    out_features = os.path.join(PROCESSED_DIR, "features")
    ensure_dir(out_features)
    np.savez_compressed(os.path.join(out_features, f"{stem}.npz"), X=X_kept, y=y_idx)
    manifest_rows.append(f"{stem}\t{FEATURE_TYPE}\t{SAMPLE_RATE}\t{FPS}\t{X_kept.shape[0]}\t{X_kept.shape[1]}")
    return label_map

def process_file_presplit(
    wav_path: str, label: str, label_map: Dict[str, int], manifest_rows: List[str]
) -> Dict[str, int]:
    stem = os.path.splitext(os.path.basename(wav_path))[0]

    y = load_wav_mono(wav_path, SAMPLE_RATE)
    if FEATURE_TYPE == "mel": X, _ = compute_mel(y, SAMPLE_RATE, FPS)
    elif FEATURE_TYPE == "cqt": X, _ = compute_cqt(y, SAMPLE_RATE, FPS)
    else: raise ValueError("FEATURE_TYPE must be 'mel' or 'cqt'")

    if label not in label_map: label_map[label] = len(label_map)
    y_idx = np.full(X.shape[0], dtype=np.int64, fill_value=label_map[label])

    out_features = os.path.join(PROCESSED_DIR, "features")
    ensure_dir(out_features)
    np.savez_compressed(os.path.join(out_features, f"{stem}.npz"), X=X, y=y_idx)
    manifest_rows.append(f"{stem}\t{FEATURE_TYPE}\t{SAMPLE_RATE}\t{FPS}\t{X.shape[0]}\t{X.shape[1]}")
    return label_map

def main():
    ensure_dir(PROCESSED_DIR)
    ensure_dir(os.path.join(PROCESSED_DIR, "features"))

    manifest_rows = ["stem\tfeat\tsr\tfps\tn_frames_kept\tn_feats"]
    label_map: Dict[str, int] = {}
    train_stems, val_stems = [], []

    if DATASET_TYPE == "presplit_folders":
        print("Processing in 'presplit_folders' mode.")
        train_chord_dirs = sorted([d for d in glob.glob(os.path.join(TRAIN_AUDIO_DIR, '*')) if os.path.isdir(d)])
        val_chord_dirs = sorted([d for d in glob.glob(os.path.join(TEST_AUDIO_DIR, '*')) if os.path.isdir(d)])

        if not train_chord_dirs and not val_chord_dirs:
            print(f"No chord folders found in {TRAIN_AUDIO_DIR} or {TEST_AUDIO_DIR}")
            return

        for chord_dir in tqdm(train_chord_dirs, desc="Processing train folders"):
            chord_label = os.path.basename(chord_dir)
            wav_files = sorted(glob.glob(os.path.join(chord_dir, '*.wav')))
            for wav_path in wav_files:
                stem = os.path.splitext(os.path.basename(wav_path))[0]
                train_stems.append(stem)
                try: label_map = process_file_presplit(wav_path, chord_label, label_map, manifest_rows)
                except Exception as e: tqdm.write(f"[WARN] Failed processing '{wav_path}': {e}. Skipping.")
        
        for chord_dir in tqdm(val_chord_dirs, desc="Processing val folders"):
            chord_label = os.path.basename(chord_dir)
            wav_files = sorted(glob.glob(os.path.join(chord_dir, '*.wav')))
            for wav_path in wav_files:
                stem = os.path.splitext(os.path.basename(wav_path))[0]
                val_stems.append(stem)
                try: label_map = process_file_presplit(wav_path, chord_label, label_map, manifest_rows)
                except Exception as e: tqdm.write(f"[WARN] Failed processing '{wav_path}': {e}. Skipping.")

    else: # Handles "implicit_split" and "random_split"
        print(f"Processing in '{DATASET_TYPE}' mode.")
        wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
        if not wav_files:
            print(f"No WAV files found in {AUDIO_DIR}")
            return

        for wav in tqdm(wav_files, desc="Processing audio"):
            try:
                updated_map = process_file_from_csv(wav, label_map, manifest_rows)
                if updated_map: label_map = updated_map
            except Exception as e: tqdm.write(f"[WARN] Failed processing '{wav}': {e}. Skipping.")

        if DATASET_TYPE == "implicit_split":
            all_wav_stems = [os.path.splitext(os.path.basename(f))[0] for f in wav_files]
            for stem in all_wav_stems:
                if stem.endswith(" train"): train_stems.append(stem)
                elif stem.endswith(" val"): val_stems.append(stem)

    if len(manifest_rows) == 1:
        print("No files were processed. Check data directories and settings.")
        return

    # --- Finalize --- 
    # Save label map
    index_to_label = [None] * len(label_map)
    for lab, idx in label_map.items(): index_to_label[idx] = lab
    labels_json = {"label_to_index": label_map, "index_to_label": index_to_label}
    with open(os.path.join(PROCESSED_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    # Save split map if applicable
    if train_stems or val_stems:
        split_map = {"train": train_stems, "val": val_stems}
        split_map_path = os.path.join(PROCESSED_DIR, "split_map.json")
        with open(split_map_path, "w", encoding="utf-8") as f:
            json.dump(split_map, f, indent=2)
        print(f"Saved split map to {split_map_path}")

    # Save manifest
    with open(os.path.join(PROCESSED_DIR, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_rows) + "\n")

    print(f"Done. Wrote {len(manifest_rows)-1} files to {PROCESSED_DIR}.")
    print(f"- Features: {os.path.join(PROCESSED_DIR, 'features')}")
    print(f"- Labels map: {os.path.join(PROCESSED_DIR, 'labels.json')}")
    print(f"- Manifest: {os.path.join(PROCESSED_DIR, 'manifest.tsv')}")

if __name__ == "__main__":
    main()
