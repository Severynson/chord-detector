import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
import torch

from PyQt5 import QtCore, QtGui, QtWidgets

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


# -----------------------------
# Worker: audio + inference
# -----------------------------
class ChordRecognizer(QtCore.QObject):
    """
    Handles audio capture and model inference.
    Emits `chord_predicted` with the predicted label string.
    """
    chord_predicted = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.stream = None
        self.audio_buffer = None
        self.timer = None
        self.running = False

        # ---- Load model ----
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        n_mels = ckpt["n_mels"]
        n_classes = ckpt["n_classes"]
        self.index_to_label = ckpt["index_to_label"]

        base_model = CRNN(
            n_mels=n_mels,
            n_classes=n_classes,
            conv_channels=[32, 64, 128],
            rnn_hidden=128,
            rnn_layers=2,
            dropout=0.3,
            use_attention=True,
        )
        self.model = ChordRecognitionWithSmoothing(base_model, smoothing_window=5).to(
            DEVICE
        )
        self.model.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Precompute window sizes
        self.win_len_samples = int(INFERENCE_WIN_SEC * SAMPLE_RATE)
        self.hop_len_samples = int(INFERENCE_HOP_SEC * SAMPLE_RATE)

    def start(self, device_id: int):
        if self.running:
            return

        self.audio_buffer = deque(maxlen=self.win_len_samples)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, flush=True)
            # mono
            self.audio_buffer.extend(indata[:, 0])

        self.stream = sd.InputStream(
            device=device_id,
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=self.hop_len_samples,
        )
        self.stream.start()
        self.running = True

        # timer in GUI thread to run inference regularly
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._process_audio)
        self.timer.start(int(INFERENCE_HOP_SEC * 1000))

    def stop(self):
        self.running = False
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.audio_buffer = None

    def _process_audio(self):
        if not self.running or self.audio_buffer is None:
            return

        if len(self.audio_buffer) < self.win_len_samples:
            return  # not enough data yet

        # Take a copy to avoid race conditions while callback continues filling
        y = np.array(self.audio_buffer, dtype=np.float32)

        # ---- Normalization ----
        y = y - np.mean(y)
        rms = np.sqrt(np.mean(y ** 2) + 1e-12)
        if rms > 0:
            y = y / rms * 0.1

        # ---- Feature Extraction ----
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

        # ---- Inference ----
        with torch.no_grad():
            features = (
                torch.from_numpy(mel_spec_db).float().unsqueeze(0).to(DEVICE)
            )  # (1,T,F)
            probs = self.model(features)  # (1,T,C) if your wrapper returns probs
            # average over time, then argmax over classes
            pred_idx = probs.mean(dim=1).argmax(dim=1).item()
            pred_label = self.index_to_label[pred_idx]

        # emit to GUI
        self.chord_predicted.emit(pred_label)


# -----------------------------
# Main Window UI
# -----------------------------
class ChordUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Guitar Chord Detector")
        self.resize(900, 600)

        # Paths
        self.assets_dir = Path("assets/UI/pictures/chords")
        self.bg_path = self.assets_dir / "Background.jpg"
        self.noise_gif_path = self.assets_dir / "Noise.gif"

        # ---- Central widget ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---- Top controls: device selector + start/stop ----
        controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(controls_layout)

        self.device_label = QtWidgets.QLabel("Audio device:")
        controls_layout.addWidget(self.device_label)

        self.device_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(self.device_combo, 1)

        self.refresh_button = QtWidgets.QPushButton("Refresh devices")
        controls_layout.addWidget(self.refresh_button)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setCheckable(True)
        controls_layout.addWidget(self.start_button)

        # ---- Predicted chord text ----
        self.chord_label = QtWidgets.QLabel("Predicted chord: —")
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.chord_label.setFont(font)
        self.chord_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.chord_label)

        # ---- Image display area ----
        self.display_label = QtWidgets.QLabel()
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)
        self.display_label.setMinimumSize(600, 350)
        self.display_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.display_label, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

        # ---- Load images ----
        self.background_pixmap = QtGui.QPixmap(str(self.bg_path))
        if self.background_pixmap.isNull():
            print(f"Warning: could not load background: {self.bg_path}")
        self.chord_pixmaps = {}  # label -> QPixmap
        # We will lazy-load chord images on first use

        # GIF movie for Noise
        self.noise_movie = QtGui.QMovie(str(self.noise_gif_path))
        if self.noise_movie.isValid():
            self.noise_movie.setCacheMode(QtGui.QMovie.CacheAll)
        else:
            print(f"Warning: could not load Noise.gif: {self.noise_gif_path}")

        # ---- Recognizer worker ----
        self.recognizer = ChordRecognizer()
        self.recognizer.chord_predicted.connect(self.on_chord_predicted)

        # hook up UI
        self.refresh_button.clicked.connect(self.populate_devices)
        self.start_button.toggled.connect(self.on_start_toggled)

        self.populate_devices()
        self.show_background_only()

    # -----------------------------
    # Device handling
    # -----------------------------
    def populate_devices(self):
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not query audio devices:\n{e}"
            )
            return

        for i, dev in enumerate(devices):
            name = dev["name"]
            self.device_combo.addItem(f"{i}: {name}", userData=i)

        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(sd.default.device[0] or 0)

    # -----------------------------
    # Start / Stop
    # -----------------------------
    def on_start_toggled(self, checked: bool):
        if checked:
            # start
            idx = self.device_combo.currentIndex()
            if idx < 0:
                QtWidgets.QMessageBox.warning(
                    self, "No device", "Select an audio device first."
                )
                self.start_button.setChecked(False)
                return

            device_id = self.device_combo.itemData(idx)
            try:
                self.recognizer.start(device_id)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error starting stream", str(e)
                )
                self.start_button.setChecked(False)
                return

            self.start_button.setText("Stop")
            self.statusBar().showMessage(f"Listening on device {device_id}...")
        else:
            # stop
            self.recognizer.stop()
            self.start_button.setText("Start")
            self.statusBar().showMessage("Stopped")
            self.show_background_only()
            self.chord_label.setText("Predicted chord: —")

    # -----------------------------
    # Drawing logic
    # -----------------------------
    def show_background_only(self):
        if not self.background_pixmap.isNull():
            scaled_bg = self.background_pixmap.scaled(
                self.display_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.display_label.setPixmap(scaled_bg)
        else:
            self.display_label.clear()

    def resizeEvent(self, event):
        # update background when window is resized
        super().resizeEvent(event)
        if isinstance(self.display_label.pixmap(), QtGui.QPixmap):
            self.show_background_only()

    def overlay_chord_on_background(self, chord_label: str):
        """
        Draws Background.jpg and overlays <chord_label>.png on top,
        then sets the result into display_label.
        """
        if self.background_pixmap.isNull():
            self.display_label.clear()
            return

        # base image
        base = self.background_pixmap.scaled(
            self.display_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        # load chord pixmap (lazy)
        if chord_label not in self.chord_pixmaps:
            chord_img_path = self.assets_dir / f"{chord_label}.png"
            pm = QtGui.QPixmap(str(chord_img_path))
            if pm.isNull():
                print(f"Warning: could not load chord image: {chord_img_path}")
            self.chord_pixmaps[chord_label] = pm

        chord_pm = self.chord_pixmaps[chord_label]

        # compose only if chord image loaded
        if not chord_pm.isNull():
            # scale chord image to something reasonable (e.g. 60% of height)
            target_height = int(base.height() * 0.6)
            chord_scaled = chord_pm.scaledToHeight(
                target_height, QtCore.Qt.SmoothTransformation
            )

            # paint
            composed = QtGui.QPixmap(base)  # copy
            painter = QtGui.QPainter(composed)
            # center chord horizontally
            x = (composed.width() - chord_scaled.width()) // 2
            y = (composed.height() - chord_scaled.height()) // 2
            painter.drawPixmap(x, y, chord_scaled)
            painter.end()

            self.display_label.setPixmap(composed)
        else:
            # if no chord image, just show background
            self.display_label.setPixmap(base)

    def show_noise_gif(self):
        """
        Shows Noise.gif instead of static background+chord.
        """
        if self.noise_movie.isValid():
            self.display_label.setMovie(self.noise_movie)
            self.noise_movie.start()
        else:
            # fallback: just clear or show background if gif not available
            self.show_background_only()

    # -----------------------------
    # Slot: prediction from worker
    # -----------------------------
    @QtCore.pyqtSlot(str)
    def on_chord_predicted(self, label: str):
        self.chord_label.setText(f"Predicted chord: {label}")

        if label == "Noise":
            self.show_noise_gif()
        else:
            # stop gif if currently playing
            if isinstance(self.display_label.movie(), QtGui.QMovie):
                self.display_label.movie().stop()
            self.overlay_chord_on_background(label)

    # -----------------------------
    # Cleanup
    # -----------------------------
    def closeEvent(self, event):
        self.recognizer.stop()
        event.accept()


# -----------------------------
# Entry point
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ChordUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()