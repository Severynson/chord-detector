import os, json, glob, random
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

class ChordFramesDataset(Dataset):
    """
    Builds fixed-length windows from each per-file NPZ.
    Each item: (X_win [T,F], y_win [T]) -> tensors.
    """

    def __init__(self, features_dir: str, labels_json_path: str,
                 window_frames: int = 100, hop_frames: int = 50,
                 split: str = "train", split_ratio: float = 0.8,
                 seed: int = 42, majority_label_threshold: float = 0.0):
        super().__init__()
        self.features_dir = features_dir
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self.majority_label_threshold = majority_label_threshold

        with open(labels_json_path, "r", encoding="utf-8") as f:
            labmap = json.load(f)
        self.label_to_index: Dict[str, int] = labmap["label_to_index"]
        self.index_to_label: List[str] = labmap["index_to_label"]
        self.num_classes = len(self.index_to_label)

        # Collect files
        all_npz = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        if not all_npz:
            raise FileNotFoundError(f"No .npz found in {features_dir}")

        # Deterministic split by file
        rng = random.Random(seed)
        files = all_npz[:]
        rng.shuffle(files)
        split_point = int(len(files) * split_ratio)
        if split == "train":
            self.files = files[:split_point]
        elif split == "val":
            self.files = files[split_point:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        # Build index of (file_path, start_idx)
        self.index: List[Tuple[str, int]] = []
        for fpath in self.files:
            with np.load(fpath) as npz:
                T = npz["X"].shape[0]
            # slide windows
            start = 0
            while start + window_frames <= T:
                self.index.append((fpath, start))
                start += hop_frames

        # Optional filtering by majority label dominance
        if self.majority_label_threshold > 0.0:
            kept = []
            for fpath, start in self.index:
                with np.load(fpath) as npz:
                    y = npz["y"]
                y_win = y[start:start+window_frames]
                # compute dominant label fraction
                vals, counts = np.unique(y_win, return_counts=True)
                frac = counts.max() / y_win.size
                if frac >= self.majority_label_threshold:
                    kept.append((fpath, start))
            self.index = kept

        if not self.index:
            raise RuntimeError("No training windows found. Consider reducing WINDOW_FRAMES, changing HOP_FRAMES, or checking your processed data.")

        # Small cache for speed
        self._cache_path = None
        self._cache_X = None
        self._cache_y = None

    def __len__(self):
        return len(self.index)

    def _load_npz(self, path: str):
        # simple one-file cache to avoid reloading for consecutive windows from same file
        if self._cache_path != path:
            with np.load(path) as npz:
                self._cache_X = npz["X"].astype(np.float32)   # (T, F)
                self._cache_y = npz["y"].astype(np.int64)     # (T,)
            self._cache_path = path
        return self._cache_X, self._cache_y

    def __getitem__(self, idx):
        path, start = self.index[idx]
        X_all, y_all = self._load_npz(path)
        X = X_all[start:start+self.window_frames]   # (T,F)
        y = y_all[start:start+self.window_frames]   # (T,)

        # to torch
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y
