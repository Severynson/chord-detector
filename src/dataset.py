import os, json, glob, random
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

def spec_augment(spec: torch.Tensor, n_freq_masks: int = 1, n_time_masks: int = 1, freq_mask_param: int = 27, time_mask_param: int = 40):
    """
    Apply SpecAugment to a spectrogram.
    spec: (T, F) tensor
    """
    spec = spec.clone() # Make a copy
    T, F = spec.shape
    
    # Frequency masking
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, F - f)
        spec[:, f0:f0+f] = 0

    # Time masking
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, T - t)
        spec[t0:t0+t, :] = 0
        
    return spec

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
        self.split = split

        with open(labels_json_path, "r", encoding="utf-8") as f:
            labmap = json.load(f)
        self.label_to_index: Dict[str, int] = labmap["label_to_index"]
        self.index_to_label: List[str] = labmap["index_to_label"]
        self.num_classes = len(self.index_to_label)

        # Check for an explicit split map from prepare_dataset.py
        split_map_path = os.path.join(os.path.dirname(self.features_dir), "split_map.json")
        use_explicit_split = os.path.exists(split_map_path)

        if use_explicit_split:
            print(f"INFO: Found split_map.json, using explicit train/val files.")
            with open(split_map_path, "r", encoding="utf-8") as f:
                split_map = json.load(f)
            
            if split not in split_map:
                raise ValueError(f"Split '{split}' not found in {split_map_path}. Available splits: {list(split_map.keys())}")

            stems = split_map[split]
            self.files = [os.path.join(self.features_dir, f"{stem}.npz") for stem in stems]
            
            # Verify that the files actually exist
            for fpath in self.files:
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"File '{fpath}' from split_map.json not found.")
        else:
            # Fallback to original random split logic
            print("INFO: No split_map.json found, using random file-based split.")
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

        # If training, perform automatic downsampling of the majority class
        if self.split == 'train' and len(self.index) > 0:
            self._balance_dataset()

        if not self.index:
            raise RuntimeError("No training windows found. Consider reducing WINDOW_FRAMES, changing HOP_FRAMES, or checking your processed data.")

        # Small cache for speed
        self._cache_path = None
        self._cache_X = None
        self._cache_y = None

    def _balance_dataset(self):
        print("Performing automatic downsampling for training set...")
        # Get the dominant label for each window
        window_labels = []
        for fpath, start in self.index:
            with np.load(fpath) as npz:
                y = npz["y"]
            y_win = y[start:start+self.window_frames]
            dominant_label = np.bincount(y_win).argmax()
            window_labels.append(dominant_label)

        label_counts = np.bincount(window_labels, minlength=self.num_classes)
        majority_class_idx = label_counts.argmax()
        
        # Heuristic: target count is the count of the second most frequent class
        sorted_counts = sorted(label_counts, reverse=True)
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

        # Get indices for majority and other classes
        majority_windows_indices = [i for i, lab in enumerate(window_labels) if lab == majority_class_idx]
        other_windows_indices = [i for i, lab in enumerate(window_labels) if lab != majority_class_idx]

        # Randomly sample from the majority class windows
        if len(majority_windows_indices) > target_count:
            print(f"Downsampling majority class ({self.index_to_label[majority_class_idx]}) from {len(majority_windows_indices)} to {target_count} samples.")
            sampled_majority_indices = random.sample(majority_windows_indices, target_count)
        else:
            sampled_majority_indices = majority_windows_indices

        # Combine and create the new balanced index
        final_indices_to_keep = sorted(other_windows_indices + sampled_majority_indices)
        self.index = [self.index[i] for i in final_indices_to_keep]
        print(f"Original windows: {len(window_labels)}. After downsampling: {len(self.index)} windows.")


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

        if self.split == 'train':
            X = spec_augment(X)

        return X, y
