# 🕵️🎼 chord-detector

Application detecting guitar chords in real time powered by Deep Learning.

![Demo](assets/demo/demo.gif)

⸻

# 🎯 Current Validation Metrics
 
**Accuracy:** `0.921`

⸻

# 🗂️ Dataset

The dataset used for training was recorded manually using a single guitar:
🎸 Fender FA-15 3/4 Acoustic

Recordings were made in a quiet room with an open window to allow minor background sounds, making the data high-quality yet realistic — since inference is expected to occur in similar environments, slight side sounds were intentionally admitted.

⸻

## 🎵 Recording Pattern

Each chord was recorded using the following strumming patterns:
1. ⬇️ Regular down  
2. ⬆️ Regular up  
3. ⚡ Fast down  
4. ⚡ Fast up  
5. 🐢 Slow down *(string by string)*  
6. 🐢 Slow up *(string by string)*  
7. ⬇️⬇️ ⬆️⬆️ ⬇️ ⬆️  
8. ⬇️⬆️⬇️⬆️⬇️⬆️  
9. ⬇️⬇️⬇️⬇️⬇️⬇️ 

**10–18)** ➡️ Repeat with another fingering (alternative voicing)  
**19–25)** ➡️ Repeat steps 1–7 again for the new shape

⸻

💾 Each sample was labeled and preprocessed into Mel-spectrograms before training the neural network.

⸻

# 🚀 How to Use (Inference)

To run real-time chord detection locally, execute:
```
python app.py
```
This will launch a **PyQt5 desktop application** for live chord recognition.

### 🪟 Main Interface

Once the app opens, you’ll see a window with:

- 🎙️ **Input device selector** — choose which microphone / audio interface to use  
- ▶️ **Start / Stop button** — control when real-time detection is running  
- 🎼 **Current chord display** — shows the most recently detected chord in large text  
- 📈 **Status / confidence area** — shows detection confidence or logs (if enabled)

### 🔊 Running real-time detection

1. Select your preferred **input device** from the dropdown  
2. Click **Start** to begin listening  
3. Play chords on your guitar near the microphone  
4. The **current detected chord** will update continuously in the UI  
5. Click **Stop** to end the session and release the audio device

The application performs **live inference** on the incoming audio stream, using the trained model to classify the current guitar chord in real time.

⸻

# 🧠 How to Train / Contribute

If you’d like to improve the model or experiment with your own dataset, follow these steps:

## 1️⃣ Prepare the Dataset

Run the dataset preparation script to convert raw audio samples into spectrograms:

python -m scripts.prepare_dataset

This script expects sound samples to be stored in:

data/train/<ChordName>/
data/test/<ChordName>/

Each folder should contain raw .wav recordings of the corresponding chord.
The script will generate spectrograms ready for the CRNN model.

⸻

## 🎯 Loss Function (GuitarChordDistanceLoss)

Instead of a plain cross-entropy loss, this project uses a **custom, distance-aware chord loss** defined in `src/loss.py` as `GuitarChordDistanceLoss`.

Key ideas:

- **Musical structure–aware**
  - Models **12 root notes** on a chromatic circle (A, A#, B, C, C#, D, D#, E, F, F#, G, G#).
  - Distances between chords use a **circular semitone distance** (shortest path around the 12-tone ring).

- **Major vs minor quality**
  - Penalizes confusing major/minor versions of the same root differently than confusing completely unrelated chords.

- **Noise class**
  - A dedicated **Noise** class is treated as *maximally distant* from all musical chords.

- **Hybrid objective**
  - Combines standard **cross-entropy** with a **distance-based penalty**:
    - `alpha` controls the tradeoff between classification accuracy and musical-distance awareness.
    - `root_weight` balances root note importance vs chord quality (maj/min).
    - `temperature` controls how “soft” or strict the distance penalty is.
    - `noise_distance` sets how far the Noise class is from any chord.

This encourages the model not only to be correct, but also to be **musically reasonable when wrong** (e.g., preferring C vs. Cm over C vs. F#).

⸻

## 2️⃣ Train the Model Locally

To train using your local machine:

python train.py

This will start the training process using your prepared dataset and save the resulting model weights upon completion.

⸻

## 3️⃣ Train on Google Colab (Optional)

If you’d like to leverage Google Colab’s GPU for faster training, use the included notebook:

train_colab.ipynb

## 🧩 Notes:
	•	You’ll need to set up your own environment variables (e.g., paths, credentials if needed).
	•	You may need to adjust branch names or repository URLs if you are working on a fork or different version.

⸻

# 🤝 Contributing

Contributions are welcome!
You can improve:
	•	The dataset (more chords, cleaner samples, varied instruments)
	•	The model architecture or training parameters
	•	The inference interface or performance optimizations

Feel free to open a pull request or start a discussion for proposed improvements.

⸻

# 🧾 License

This project is distributed under the MIT License — see LICENSE for details.