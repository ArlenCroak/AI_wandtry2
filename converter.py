import os
import librosa
import soundfile as sf

input_folder = "spell_dataset/wav"
output_folder = "output_wavs"

os.makedirs(output_folder, exist_ok=True)

target_sr = 16000

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load and resample
        audio, sr = librosa.load(input_path, sr=target_sr)

        # Save
        sf.write(output_path, audio, target_sr)

        print(f"Converted: {filename}")