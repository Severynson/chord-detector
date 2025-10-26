
import os
import librosa
import soundfile as sf
import argparse

def process_audio_files(data_path, folder_names):
    """
    Processes audio files in specified subfolders of the data_path.

    Args:
        data_path (str): The path to the 'Train' or 'Test' directory.
        folder_names (list): A list of folder names to process.
    """
    for folder_name in folder_names:
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found at {folder_path}")
            continue

        print(f"Processing files in: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Load the audio file
                    y, sr = librosa.load(file_path, sr=None)

                    # Calculate the number of samples to cut
                    cut_samples = int(0.5 * sr)

                    # Cut the first 0.5 seconds
                    y_cut = y[cut_samples:]

                    # Save the modified audio file, overwriting the original
                    sf.write(file_path, y_cut, sr)
                    print(f"  - Processed {filename}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut the first 0.5 seconds of .wav files in specified folders.")
    parser.add_argument('folders', nargs='+', help='A list of folder names to process.')
    args = parser.parse_args()

    base_sound_path = os.path.join('data', 'sound')
    train_path = os.path.join(base_sound_path, 'Train')
    test_path = os.path.join(base_sound_path, 'Test')

    print("--- Starting Audio Processing ---")
    process_audio_files(train_path, args.folders)
    process_audio_files(test_path, args.folders)
    print("--- Audio Processing Complete ---")
