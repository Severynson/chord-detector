# 🕵️🎼 chord-detector

Application detecting guitar chords in real time powered by Deep Learning.

⸻

# 🎯 Current Validation Metrics

**Validation Loss (CE):** `6.1324`  
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

python app.py

Once started, the console interface will:
	•	Prompt you to select an input device (microphone) 🎙️
	•	Continuously display updated chord classifications every 0.1 seconds

This allows live chord recognition directly from your audio input.

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