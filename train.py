import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.loss import GuitarChordDistanceLoss

from src.config import (
    FEATURES_DIR,
    LABELS_JSON_PATH,
    WINDOW_FRAMES,
    HOP_FRAMES,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    NUM_WORKERS,
    DEVICE,
    CHECKPOINT_PATH,
)
from src.dataset import ChordFramesDataset
from src.model import CRNN, ChordRecognitionWithSmoothing
from src.utils import compute_class_weights, accuracy_per_frame


def run():
    # Datasets & loaders
    train_ds = ChordFramesDataset(
        FEATURES_DIR,
        LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES,
        hop_frames=HOP_FRAMES,
        split="train",
        split_ratio=0.85,
        seed=123,
    )
    val_ds = ChordFramesDataset(
        FEATURES_DIR,
        LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES,
        hop_frames=HOP_FRAMES,
        split="val",
        split_ratio=0.85,
        seed=123,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    n_mels = train_ds[0][0].shape[1]
    n_classes = train_ds.num_classes
    print(
        f"Features per frame (F): {n_mels}, Classes: {n_classes}, Train windows: {len(train_ds)}, Val windows: {len(val_ds)}"
    )

    base_model = CRNN(
        n_mels=n_mels,
        n_classes=n_classes,
        conv_channels=[32, 64, 128],
        rnn_hidden=128,
        rnn_layers=2,
        dropout=0.3,
        use_attention=True,
    )
    model = ChordRecognitionWithSmoothing(base_model, smoothing_window=5).to(DEVICE)

    class_weights = compute_class_weights(train_ds, n_classes).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)   # per-frame CE
    criterion = GuitarChordDistanceLoss(
        label_to_index=train_ds.label_to_index,
        alpha=0.3,  # Start with 30% distance-aware
        root_weight=0.7,  # Root matters more than quality
        temperature=2.0,  # Smooth penalty
        noise_distance=2.0,  # Max distance for noise
        class_weights=class_weights,
    ).to(DEVICE)

    # TODO remove after testing
    criterion.print_example_distances()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        t_loss, t_acc, t_batches = 0.0, 0.0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            X = X.to(DEVICE)  # (B,T,F)
            y = y.to(DEVICE)  # (B,T)

            logits = model(X)  # (B,T,C)
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
        t_acc /= max(t_batches, 1)

        # ---- val ----
        model.eval()
        v_loss, v_acc, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                # Get logits for loss (bypass smoothing wrapper)
                logits = model.model(X)
                loss = criterion(logits.reshape(-1, n_classes), y.reshape(-1))

                # Get smoothed probs for accuracy
                probs = model(X, apply_smoothing=True)
                acc = accuracy_per_frame(probs, y)

                v_loss += loss.item()
                v_acc += acc
                v_batches += 1

        v_loss /= max(v_batches, 1)
        v_acc /= max(v_batches, 1)

        print(
            f"Epoch {epoch:02d} | train loss (CE) {t_loss:.4f} acc {t_acc:.3f}  ||  val loss (CE) {v_loss:.4f} acc {v_acc:.3f}"
        )

        # save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model_state": model.model.state_dict(),
                    "n_mels": n_mels,
                    "n_classes": n_classes,
                    "label_to_index": train_ds.label_to_index,
                    "index_to_label": train_ds.index_to_label,
                },
                CHECKPOINT_PATH,
            )
            print(f"  ↳ saved {CHECKPOINT_PATH} (val_acc={v_acc:.3f})")


if __name__ == "__main__":
    run()
