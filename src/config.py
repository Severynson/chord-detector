import torch
import os

# --- Data Config ---
# One of: "implicit_split", "random_split", "presplit_folders"
DATASET_TYPE = "presplit_folders"

# Directories for all modes
PROCESSED_DIR = "data/processed"
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")
LABELS_JSON_PATH = os.path.join(PROCESSED_DIR, "labels.json")

# Directories for "implicit_split" and "random_split"
AUDIO_DIR = "data/sound"
LABEL_DIR = "data/labels"

# Directories for "presplit_folders"
TRAIN_AUDIO_DIR = "data/sound/Train"
TEST_AUDIO_DIR = "data/sound/Test"


# --- Feature Settings ---
FEATURE_TYPE = "mel"  # "mel" or "cqt"
SAMPLE_RATE = 22050  # resample target (Hz)
FPS = 100  # frames per second (hop ~ SR/FPS)
N_FFT = 2048  # STFT window size (samples) for mel
N_MELS = 128  # mel bins
MEL_FMIN = 30.0
MEL_FMAX = None  # None = SR/2
HOP_LENGTH = int(SAMPLE_RATE / FPS)

# CQT settings (used if FEATURE_TYPE == "cqt")
CQT_BINS_PER_OCTAVE = 24
CQT_N_BINS = 84
CQT_FMIN = 32.7031956626  # C1


# --- Model & Training Config ---
CHECKPOINT_PATH = "checkpoints/crnn_best.pt"
WINDOW_FRAMES = 50
HOP_FRAMES = 25
BATCH_SIZE = 16 if os.environ.get("ENV") == "local" else 700
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2 if os.environ.get("ENV") == "local" else 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Real-time inference config ---
INFERENCE_HOP_SEC = 0.2
INFERENCE_WIN_SEC = 1.0
# --- Chord Definitions ---
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_TO_INT = {**{pitch: i for i, pitch in enumerate(PITCH_CLASSES)}, 'N': 12}
INT_TO_ROOT = {**{i: pitch for i, pitch in enumerate(PITCH_CLASSES)}, 12: 'N'}

QUALITIES = ['maj', 'min', 'other']
QUALITY_TO_INT = {quality: i for i, quality in enumerate(QUALITIES)}
INT_TO_QUALITY = {i: quality for i, quality in enumerate(QUALITIES)}

# Helper to map original chord names to (root, quality) integer tuples
CHORD_TO_ROOT_QUALITY = {}
for i, root in enumerate(PITCH_CLASSES):
    # Major chords
    CHORD_TO_ROOT_QUALITY[root] = (ROOT_TO_INT[root], QUALITY_TO_INT['maj'])
    CHORD_TO_ROOT_QUALITY[root + 'm'] = (ROOT_TO_INT[root], QUALITY_TO_INT['min'])

CHORD_TO_ROOT_QUALITY['Noise'] = (ROOT_TO_INT['N'], QUALITY_TO_INT['other'])
# Special case for some datasets
CHORD_TO_ROOT_QUALITY['N'] = (ROOT_TO_INT['N'], QUALITY_TO_INT['other'])
