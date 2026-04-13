import json
import random
import re
import pickle
from pathlib import Path

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent #folder were our script is
WAV_DIR = BASE_DIR / "spell_dataset" / "wav" #audio files
MODEL_OUT = BASE_DIR / "spell_cnn_lstm.keras" #trained model save
LABELS_OUT = BASE_DIR / "spell_labels.pkl" # stores the label names
CONFIG_OUT = BASE_DIR / "spell_model_config.json" #configuration setting

# =========================
# SETTINGS
# =========================
SAMPLE_RATE = 16000
CLIP_SECONDS = 5.0 #cut to exactly 5 seconds long
N_MFCC = 40 #Mel Frequency Cepstral Coefficients
N_FFT = 512 #Fast Fourier Transform chunk
HOP_LENGTH = 160 #How many samples to move forward between each FFT chunk

TEST_SIZE = 0.2
RANDOM_SEED = 42 #radom_split
BATCH_SIZE = 10
EPOCHS = 1000

# Match filenames like:
# lumos_20260410_110552_434309.wav
# wingardium leviosa_20260410_110558_169478.wav
FILENAME_PATTERN = re.compile(r"^(?P<label>.+)_\d{8}_\d{6}_\d+\.wav$", re.IGNORECASE) #regex pattern that matches audio filenames

# =========================
# REPRODUCIBILITY
# =========================
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================
def normalize_label(label: str) -> str:
    return str(label).strip().lower()

def extract_label_from_filename(filename: str) -> str | None:
    """
    Extract label from filenames like:
    lumos_20260410_110552_434309.wav
    wingardium leviosa_20260410_110558_169478.wav
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return normalize_label(match.group("label"))

def load_audio_fixed_length(wav_path: str, sr: int, clip_seconds: float) -> np.ndarray:
    """
    Load audio and force it to exactly clip_seconds long.
    Too short -> pad with zeros
    Too long  -> truncate
    """
    target_len = int(sr * clip_seconds)

    audio, _ = librosa.load(wav_path, sr=sr, mono=True)

    if len(audio) < target_len:
        pad_amount = target_len - len(audio)
        audio = np.pad(audio, (0, pad_amount))
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)

def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns shape: (time_steps, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8) #1e-8 prevents dividing by 0
    return mfcc.T.astype(np.float32)  # (time, n_mfcc)

def find_wav_files(wav_dir: Path):
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV folder not found: {wav_dir}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in: {wav_dir}")

    return wav_files

def build_dataset_from_wavs(wav_files):
    X = []
    y = []
    bad_files = []
    skipped_files = []

    for wav_path in wav_files:
        label = extract_label_from_filename(wav_path.name)

        if label is None:
            skipped_files.append(str(wav_path))
            continue

        try:
            audio = load_audio_fixed_length(str(wav_path), SAMPLE_RATE, CLIP_SECONDS)
            mfcc = extract_mfcc(audio, SAMPLE_RATE)

            X.append(mfcc)
            y.append(label)

        except Exception as e:
            print(f"Skipping {wav_path} because of error: {e}")
            bad_files.append(str(wav_path))

    if len(X) == 0:
        raise RuntimeError("No usable audio files found after scanning wav folder.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y, bad_files, skipped_files

def build_model(time_steps: int, n_mfcc: int, num_classes: int):
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_mfcc)),

        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_config(time_steps: int, classes):
    config = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "time_steps": int(time_steps),
        "classes": list(classes)
    }

    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def main():
    print(f"Using wav folder: {WAV_DIR}")
    wav_files = find_wav_files(WAV_DIR)
    print(f"Found {len(wav_files)} wav files")

    X, y_text, bad_files, skipped_files = build_dataset_from_wavs(wav_files)

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files with unexpected names")
        for sf in skipped_files[:10]:
            print("  ", sf)
        if len(skipped_files) > 10:
            print("  ...")

    if bad_files:
        print(f"\nSkipped {len(bad_files)} unreadable/bad files")
        for bf in bad_files[:10]:
            print("  ", bf)
        if len(bad_files) > 10:
            print("  ...")

    print(f"\nDataset shape: X={X.shape}, y={y_text.shape}")
    print("Example input shape per sample:", X[0].shape)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    print("\nClasses:")
    for i, c in enumerate(le.classes_):
        count = int(np.sum(y_text == c))
        print(f"  {i}: {c} ({count} samples)")

    num_classes = len(le.classes_)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    class_counts = np.bincount(y)
    use_stratify = np.all(class_counts >= 2)

    if not use_stratify:
        print("\nWarning: at least one class has fewer than 2 samples.")
        print("Proceeding without stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if use_stratify else None
    )

    time_steps = X.shape[1]

    model = build_model(time_steps, N_MFCC, num_classes)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, pred_labels, target_names=le.classes_))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred_labels))

    # Save outputs
    model.save(MODEL_OUT)

    with open(LABELS_OUT, "wb") as f:
        pickle.dump(le, f)

    save_config(time_steps, le.classes_)

    print(f"\nSaved model to: {MODEL_OUT}")
    print(f"Saved labels to: {LABELS_OUT}")
    print(f"Saved config to: {CONFIG_OUT}")

if __name__ == "__main__":
    main()