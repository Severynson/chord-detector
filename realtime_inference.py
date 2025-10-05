import sounddevice as sd
import numpy as np
import torch
import librosa
from collections import deque
import time

from src.model import CRNN, ChordRecognitionWithSmoothing
from src.config import (
    DEVICE, CHECKPOINT_PATH, SAMPLE_RATE, INFERENCE_HOP_SEC, 
    INFERENCE_WIN_SEC, N_FFT, N_MELS, HOP_LENGTH
)

def main():
    # --- Load Model ---
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        n_mels = ckpt["n_mels"]
        n_classes = ckpt["n_classes"]
        index_to_label = ckpt["index_to_label"]
        
        # Ensure architecture matches the one used for training
        base_model = CRNN(n_mels=n_mels, n_classes=n_classes,
                          conv_channels=[32, 64, 128], rnn_hidden=128, 
                          rnn_layers=2, dropout=0.3, use_attention=True)
        model = ChordRecognitionWithSmoothing(base_model, smoothing_window=5).to(DEVICE)
        model.model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("Model loaded successfully.")

    except FileNotFoundError:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Please train the model first.")
        return

    # --- Real-time Audio Processing ---
    win_len_samples = int(INFERENCE_WIN_SEC * SAMPLE_RATE)
    hop_len_samples = int(INFERENCE_HOP_SEC * SAMPLE_RATE)

    audio_buffer = deque(maxlen=win_len_samples)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.extend(indata[:, 0])

    print("Starting audio stream...")
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=hop_len_samples
    )
    stream.start()

    print("Listening for chords... Press Ctrl+C to stop.")
    try:
        while True:
            if len(audio_buffer) == win_len_samples:
                # Get audio window
                y = np.array(audio_buffer)

                # --- Feature Extraction ---
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS
                ).T
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # --- Inference ---
                with torch.no_grad():
                    features = torch.from_numpy(mel_spec).float().unsqueeze(0).to(DEVICE) # (1, T, F)
                    probs = model(features) # (1, T, C) -> now returns smoothed probabilities
                    
                    # --- Get Prediction ---
                    # Simple approach: average probabilities over time and take argmax
                    pred_idx = probs.mean(dim=1).argmax(dim=1).item()
                    pred_label = index_to_label[pred_idx]
                    
                    print(f"Predicted chord: {pred_label}")

            time.sleep(INFERENCE_HOP_SEC)

    except KeyboardInterrupt:
        print("\nStopping..." )
    finally:
        stream.stop()
        stream.close()
        print("Audio stream closed.")

if __name__ == "__main__":
    main()
