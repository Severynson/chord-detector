import os, json, glob, math, random
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
PROCESSED_DIR      = "data/processed"
FEATURES_DIR       = os.path.join(PROCESSED_DIR, "features")
LABELS_JSON_PATH   = os.path.join(PROCESSED_DIR, "labels.json")

WINDOW_FRAMES      = 100     # ~1.0 s if FPS=100
HOP_FRAMES         = 50      # 50% overlap
BATCH_SIZE         = 16
EPOCHS             = 15
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
NUM_WORKERS        = 2
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: filter windows that are label-messy (kept here as False because your frames are already annotated-only)
MAJORITY_LABEL_THRESHOLD = 0.0   # fraction (0..1). If >0, will keep only windows where some label occupies >= threshold*WINDOW_FRAMES

# -------------------------
# Dataset
# -------------------------

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
        X = torch.from_numpy(X)  # (T,F)
        y = torch.from_numpy(y)  # (T,)

        return X, y

# -------------------------
# Model: CRNN (Conv2D -> BiGRU -> per-frame logits)
# -------------------------

class CRNN(nn.Module):
    """
    Input:  (B, T, F)
    -> reshape to (B, 1, F, T)
    Conv blocks pool over F (freq) only, keep T aligned
    -> (B, C, F', T)
    Collapse freq: permute to (T, B, C*F')
    BiGRU -> (T, B, H*2)
    Linear per frame -> (T, B, C)
    Return (B, T, C)
    """
    def __init__(self, n_mels: int, n_classes: int,
                 conv_channels: List[int] = [32, 64, 128],
                 rnn_hidden: int = 128,
                 rnn_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        chs = [1] + conv_channels
        convs = []
        f = n_mels
        for i in range(len(conv_channels)):
            convs += [
                nn.Conv2d(chs[i], chs[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU(inplace=True),
                # pool along freq only; keep time resolution
                nn.MaxPool2d(kernel_size=(2,1))
            ]
            f = math.ceil(f / 2)
        self.conv = nn.Sequential(*convs)
        conv_out_dim = conv_channels[-1] * f  # channels * reduced freq

        self.rnn = nn.GRU(input_size=conv_out_dim,
                          hidden_size=rnn_hidden,
                          num_layers=rnn_layers,
                          bidirectional=True,
                          batch_first=False,
                          dropout=0.0 if rnn_layers == 1 else dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x):  # x: (B,T,F)
        B, T, F = x.shape
        x = x.unsqueeze(1)          # (B,1,T,F)? we want (B,1,F,T)
        x = x.permute(0,1,3,2)      # (B,1,F,T)
        x = self.conv(x)            # (B,C,F',T)
        x = x.permute(3,0,1,2)      # (T,B,C,F')
        T2, B2, Cc, Fp = x.shape
        x = x.reshape(T2, B2, Cc*Fp) # (T,B,conv_out_dim)
        x, _ = self.rnn(x)           # (T,B,2H)
        x = self.dropout(x)
        x = self.classifier(x)       # (T,B,C)
        x = x.permute(1,0,2)         # (B,T,C)
        return x

# -------------------------
# Utilities
# -------------------------

def compute_class_weights(dset: Dataset, num_classes: int) -> torch.Tensor:
    # quick pass to count labels
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(min(len(dset), 2000)):  # cap to speed up
        _, y = dset[i]
        vals, c = np.unique(y.numpy(), return_counts=True)
        counts[vals] += c
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    w = torch.tensor(weights, dtype=torch.float32)
    return w

def accuracy_per_frame(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits: (B,T,C), y: (B,T)
    pred = logits.argmax(dim=-1)
    correct = (pred == y).float().sum().item()
    total = y.numel()
    return correct / max(total, 1)

# -------------------------
# Train / Eval
# -------------------------

def run():
    # Datasets & loaders
    train_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="train", split_ratio=0.85, seed=123,
        majority_label_threshold=MAJORITY_LABEL_THRESHOLD
    )
    val_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="val", split_ratio=0.85, seed=123,
        majority_label_threshold=MAJORITY_LABEL_THRESHOLD
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    n_mels = train_ds[0][0].shape[1]
    n_classes = train_ds.num_classes
    print(f"Features per frame (F): {n_mels}, Classes: {n_classes}, Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    model = CRNN(n_mels=n_mels, n_classes=n_classes,
                 conv_channels=[32,64,128], rnn_hidden=128, rnn_layers=2, dropout=0.2).to(DEVICE)

    class_weights = compute_class_weights(train_ds, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)   # per-frame CE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        t_loss, t_acc, t_batches = 0.0, 0.0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            X = X.to(DEVICE)              # (B,T,F)
            y = y.to(DEVICE)              # (B,T)

            logits = model(X)             # (B,T,C)
            loss = criterion(logits.reshape(-1, n_classes), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc = accuracy_per_frame(logits, y)
            t_loss += loss.item()
            t_acc += acc
            t_batches += 1

        t_loss /= max(t_batches, 1)
        t_acc  /= max(t_batches, 1)

        # ---- val ----
        model.eval()
        v_loss, v_acc, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits.reshape(-1, n_classes), y.reshape(-1))
                acc = accuracy_per_frame(logits, y)
                v_loss += loss.item()
                v_acc += acc
                v_batches += 1
        v_loss /= max(v_batches, 1)
        v_acc  /= max(v_batches, 1)

        print(f"Epoch {epoch:02d} | train loss {t_loss:.4f} acc {t_acc:.3f}  ||  val loss {v_loss:.4f} acc {v_acc:.3f}")

        # save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "n_mels": n_mels,
                "n_classes": n_classes,
                "label_to_index": train_ds.label_to_index,
                "index_to_label": train_ds.index_to_label,
            }, f"checkpoints/crnn_best.pt")
            print(f"  ↳ saved checkpoints/crnn_best.pt (val_acc={v_acc:.3f})")

if __name__ == "__main__":
    run()