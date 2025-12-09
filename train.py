import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.config import (
    FEATURES_DIR, LABELS_JSON_PATH, WINDOW_FRAMES, HOP_FRAMES, BATCH_SIZE, EPOCHS, LR, 
    WEIGHT_DECAY, NUM_WORKERS, DEVICE, CHECKPOINT_PATH,
    ROOT_TO_INT, QUALITY_TO_INT
)
from src.dataset import ChordFramesDataset
from src.model import CRNN, ChordRecognitionWithSmoothing
from src.utils import accuracy_per_frame

def run_phase1(args):
    """
    Phase 1: train everything jointly (encoder + both heads).
    """
    print("Running Phase 1: Joint training")
    # Datasets & loaders
    train_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="train", split_ratio=0.85, seed=123
    )
    val_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="val", split_ratio=0.85, seed=123
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    n_mels = train_ds[0][0].shape[1]
    n_roots = len(ROOT_TO_INT)
    n_qualities = len(QUALITY_TO_INT)
    
    print(f"Features: {n_mels}, Roots: {n_roots}, Qualities: {n_qualities}, Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    base_model = CRNN(n_mels=n_mels, n_roots=n_roots, n_qualities=n_qualities,
                      conv_channels=[32, 64, 128], rnn_hidden=128, rnn_layers=2, dropout=0.3, use_attention=True, quality_tower_dim=None)
    model = ChordRecognitionWithSmoothing(base_model, smoothing_window=5).to(DEVICE)

    criterion_root = nn.CrossEntropyLoss()
    criterion_quality = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_root_acc = 0.0
    best_val_qual_acc = 0.0
    update_root = True
    update_quality = True
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        t_loss, t_acc_root, t_acc_qual, t_batches = 0.0, 0.0, 0.0, 0
        for X, y_root, y_quality in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            X = X.to(DEVICE)
            y_root = y_root.to(DEVICE)
            y_quality = y_quality.to(DEVICE)

            root_logits, quality_logits = model(X)
            
            loss_root = criterion_root(root_logits.reshape(-1, n_roots), y_root.reshape(-1))
            loss_quality = criterion_quality(quality_logits.reshape(-1, n_qualities), y_quality.reshape(-1))
            
            loss_to_backprop = 0
            if update_root:
                loss_to_backprop += loss_root
            if update_quality:
                loss_to_backprop += loss_quality
            
            if loss_to_backprop > 0:
                optimizer.zero_grad(set_to_none=True)
                loss_to_backprop.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss = loss_root + loss_quality
            acc_root = accuracy_per_frame(root_logits, y_root)
            acc_quality = accuracy_per_frame(quality_logits, y_quality)
            
            t_loss += loss.item()
            t_acc_root += acc_root
            t_acc_qual += acc_quality
            t_batches += 1

        t_loss /= max(t_batches, 1)
        t_acc_root /= max(t_batches, 1)
        t_acc_qual /= max(t_batches, 1)

        # ---- val ----
        model.eval()
        v_loss, v_acc_root, v_acc_qual, v_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for X, y_root, y_quality in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                X = X.to(DEVICE)
                y_root = y_root.to(DEVICE)
                y_quality = y_quality.to(DEVICE)

                root_logits, quality_logits = model.model(X) # Get raw logits from base model
                root_probs, quality_probs = model(X, apply_smoothing=True) # Get smoothed probabilities

                loss_root = criterion_root(root_logits.reshape(-1, n_roots), y_root.reshape(-1))
                loss_quality = criterion_quality(quality_logits.reshape(-1, n_qualities), y_quality.reshape(-1))
                loss = loss_root + loss_quality
                
                acc_root = accuracy_per_frame(root_probs, y_root)
                acc_quality = accuracy_per_frame(quality_probs, y_quality)

                v_loss += loss.item()
                v_acc_root += acc_root
                v_acc_qual += acc_quality
                v_batches += 1
        v_loss /= max(v_batches, 1)
        v_acc_root /= max(v_batches, 1)
        v_acc_qual /= max(v_batches, 1)

        print(f"Epoch {epoch:02d} | Train Loss: {t_loss:.4f}, Root Acc: {t_acc_root:.3f}, Qual Acc: {t_acc_qual:.3f} || Val Loss: {v_loss:.4f}, Root Acc: {v_acc_root:.3f}, Qual Acc: {v_acc_qual:.3f}")

        improved_root = v_acc_root > best_val_root_acc
        improved_qual = v_acc_qual > best_val_qual_acc

        if improved_root:
            best_val_root_acc = v_acc_root
        update_root = improved_root

        if improved_qual:
            best_val_qual_acc = v_acc_qual
        update_quality = improved_qual

        # Early stopping logic for phase 1: based only on root accuracy
        if improved_root:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # save best based on validation accuracy of the root
        if improved_root:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": model.model.state_dict(),
                "n_mels": n_mels,
                "n_roots": n_roots,
                "n_qualities": n_qualities,
                "root_to_int": ROOT_TO_INT,
                "quality_to_int": QUALITY_TO_INT,
            }, CHECKPOINT_PATH)
            print(f"  ↳ saved {CHECKPOINT_PATH} (val_root_acc={v_acc_root:.3f}, val_qual_acc={v_acc_qual:.3f})")
        
        if epochs_without_improvement >= 2:
            print(f"Early stopping at epoch {epoch} (Phase 1) due to no improvement in root accuracy for 2 consecutive epochs.")
            break

def run_phase2(args):
    """
    Phase 2: freeze shared_encoder and root_head, add a small extra block 
    before quality head, and train only quality-related params.
    """
    print("Running Phase 2: Fine-tuning quality head")
    
    # Datasets & loaders
    train_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="train", split_ratio=0.85, seed=123
    )
    val_ds = ChordFramesDataset(
        FEATURES_DIR, LABELS_JSON_PATH,
        window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES,
        split="val", split_ratio=0.85, seed=123
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    n_mels = train_ds[0][0].shape[1]
    n_roots = len(ROOT_TO_INT)
    n_qualities = len(QUALITY_TO_INT)

    print(f"Features: {n_mels}, Roots: {n_roots}, Qualities: {n_qualities}, Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    # Create model with quality tower
    base_model = CRNN(n_mels=n_mels, n_roots=n_roots, n_qualities=n_qualities,
                      conv_channels=[32, 64, 128], rnn_hidden=128, rnn_layers=2, 
                      dropout=0.3, use_attention=True, quality_tower_dim=128)

    # Load weights from phase 1
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Please run phase 1 training first.")
        return
        
    checkpoint = torch.load(CHECKPOINT_PATH)
    phase1_state_dict = checkpoint['model_state']
    
    # Remove quality_head weights from the state dict
    for key in list(phase1_state_dict.keys()):
        if 'quality_head' in key:
            del phase1_state_dict[key]
            
    base_model.load_state_dict(phase1_state_dict, strict=False)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Freeze layers
    for name, param in base_model.named_parameters():
        if 'quality_tower' in name or 'quality_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = ChordRecognitionWithSmoothing(base_model, smoothing_window=5).to(DEVICE)

    # Optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_quality = nn.CrossEntropyLoss()
    
    best_val_qual_acc = 0.0
    finetuned_checkpoint_path = CHECKPOINT_PATH.replace('.pt', '_finetuned.pt')
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        t_loss, t_acc_qual, t_batches = 0.0, 0.0, 0
        for X, y_root, y_quality in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [finetune]"):
            X, y_quality = X.to(DEVICE), y_quality.to(DEVICE)

            _, quality_logits = model(X)
            loss = criterion_quality(quality_logits.reshape(-1, n_qualities), y_quality.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            
            acc_quality = accuracy_per_frame(quality_logits, y_quality)
            
            t_loss += loss.item()
            t_acc_qual += acc_quality
            t_batches += 1

        t_loss /= max(t_batches, 1)
        t_acc_qual /= max(t_batches, 1)

        # ---- val ----
        model.eval()
        v_loss, v_acc_qual, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y_root, y_quality in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                X, y_quality = X.to(DEVICE), y_quality.to(DEVICE)

                _, quality_logits = model.model(X) # Get raw logits
                loss = criterion_quality(quality_logits.reshape(-1, n_qualities), y_quality.reshape(-1))

                _, quality_probs = model(X, apply_smoothing=True) # Get smoothed probabilities for accuracy
                acc_quality = accuracy_per_frame(quality_probs, y_quality)

                v_loss += loss.item()
                v_acc_qual += acc_quality
                v_batches += 1
        v_loss /= max(v_batches, 1)
        v_acc_qual /= max(v_batches, 1)

        print(f"Epoch {epoch:02d} | Train Loss: {t_loss:.4f}, Qual Acc: {t_acc_qual:.3f} || Val Loss: {v_loss:.4f}, Qual Acc: {v_acc_qual:.3f}")

        if v_acc_qual > best_val_qual_acc:
            best_val_qual_acc = v_acc_qual
            epochs_without_improvement = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": model.model.state_dict(),
                "n_mels": n_mels,
                "n_roots": n_roots,
                "n_qualities": n_qualities,
                "root_to_int": ROOT_TO_INT,
                "quality_to_int": QUALITY_TO_INT,
            }, finetuned_checkpoint_path)
            print(f"  ↳ saved {finetuned_checkpoint_path} (val_qual_acc={v_acc_qual:.3f})")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= 2:
            print(f"Early stopping at epoch {epoch} (Phase 2) due to no improvement in quality accuracy for 2 consecutive epochs.")
            break


def main():
    parser = argparse.ArgumentParser(description="Chord Recognition Training")
    parser.add_argument('--finetune-quality', action='store_true', help='Run phase 2: fine-tuning of the quality head.')
    args = parser.parse_args()

    if args.finetune_quality:
        run_phase2(args)
    else:
        run_phase1(args)

if __name__ == "__main__":
    main()
