import sounddevice as sd
import numpy as np
import torch
import librosa
from collections import deque
import time

from src.model import CRNN, ChordRecognitionWithSmoothing
from src.config import (
    DEVICE,
    CHECKPOINT_PATH,
    SAMPLE_RATE,
    INFERENCE_HOP_SEC,
    INFERENCE_WIN_SEC,
    N_FFT,
    N_MELS,
    HOP_LENGTH,
    MEL_FMIN,
)


def main():
    print("Available audio input devices:")
    print(sd.query_devices())
    print("-" * 20)
    try:
        device_id_str = input("Enter the device ID to use: ")
        device_id = int(device_id_str)
    except (ValueError, TypeError):
        print("Invalid input. Please enter a number. Exiting.")
        return

    # Load Model
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        n_mels = ckpt["n_mels"]
        n_classes = ckpt["n_classes"]
        index_to_label = ckpt["index_to_label"]

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
        model.model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Please train the model first."
        )
        return

    # Real-time Audio Processing
    win_len_samples = int(INFERENCE_WIN_SEC * SAMPLE_RATE)
    hop_len_samples = int(INFERENCE_HOP_SEC * SAMPLE_RATE)
    audio_buffer = deque(maxlen=win_len_samples)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.extend(indata[:, 0])

    stream = sd.InputStream(
        device=device_id,
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=hop_len_samples,
    )
    stream.start()

    print("\nListening for chords... Press Ctrl+C to stop.")
    try:
        while True:
            if len(audio_buffer) == win_len_samples:
                y = np.array(audio_buffer, dtype=np.float32)

                # Normalization
                y = y - np.mean(y)
                rms = np.sqrt(np.mean(y**2) + 1e-12)
                if rms > 0:
                    y = y / rms * 0.1

                # Feature Extraction
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                    fmin=MEL_FMIN,
                    fmax=SAMPLE_RATE / 2,
                    center=False,
                    power=2.0,
                ).T
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Inference
                with torch.no_grad():
                    features = (
                        torch.from_numpy(mel_spec_db).float().unsqueeze(0).to(DEVICE)
                    )
                    probs = model(features)
                    pred_idx = probs.mean(dim=1).argmax(dim=1).item()
                    pred_label = index_to_label[pred_idx]
                    print(f"Predicted chord: {pred_label}")

            time.sleep(INFERENCE_HOP_SEC)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
