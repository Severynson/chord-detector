import torch
import os

PROCESSED_DIR      = "data/processed"
FEATURES_DIR       = os.path.join(PROCESSED_DIR, "features")
LABELS_JSON_PATH   = os.path.join(PROCESSED_DIR, "labels.json")
CHECKPOINT_PATH    = "checkpoints/crnn_best.pt"

WINDOW_FRAMES      = 50
HOP_FRAMES         = 25
BATCH_SIZE         = 16
EPOCHS             = 15
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
NUM_WORKERS        = 2
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MAJORITY_LABEL_THRESHOLD = 0.5

# Real-time inference config
SAMPLE_RATE        = 22050
FPS                = 100
INFERENCE_HOP_SEC  = 0.2
INFERENCE_WIN_SEC  = 1.0
N_FFT              = 2048
N_MELS             = 128
HOP_LENGTH         = int(SAMPLE_RATE / FPS)
