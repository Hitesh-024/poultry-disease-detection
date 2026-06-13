import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def audio_to_spectrogram_image(audio_path, save_path):
    y, sr = librosa.load(audio_path, duration=10.0)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(2.24, 2.24))
    plt.axis("off")
    librosa.display.specshow(mel_db)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    wav_files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".wav")
    ]

    print(f"Found {len(wav_files)} wav files")

    for i, filename in enumerate(wav_files, start=1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(
            output_folder,
            filename.replace(".wav", ".png")
        )

        audio_to_spectrogram_image(input_path, output_path)
        print(f"{i}/{len(wav_files)} done: {filename}")

    print("All done.")


if __name__ == "__main__":
    input_folder = "path/to/audio_folder"
    output_folder = "path/to/output_folder"

    convert_folder(input_folder, output_folder)
