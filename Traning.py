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
BASE_DIR = Path(__file__).resolve().parent
WAV_DIR = BASE_DIR / "spell_dataset" / "wav"  # folder containing all  recordings
MODEL_OUT = BASE_DIR / "spell_cnn_lstm.keras" # trained model saved
LABELS_OUT = BASE_DIR / "spell_labels.pkl" # label encoder saved
CONFIG_OUT = BASE_DIR / "spell_model_config.json" # audio settings saved

# =========================
# SETTINGS
# =========================
SAMPLE_RATE = 16000 # samples per second
CLIP_SECONDS = 5.0  # every audio clip is forced to exactly 5 seconds
N_MFCC = 40 # number of MFCC coefficients to extract per frame
N_FFT = 512  # size of the FFT window (resolution)
HOP_LENGTH = 160  # how many samples to move forward between each FFT window

TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 40

FILENAME_PATTERN = re.compile(r"^(?P<label>.+)_\d{8}_\d{6}_\d+\.wav$", re.IGNORECASE)

""""
Regex pattern to extract the spell name from filenames like:
lumos_20260410_110552_434309.wav          -> "lumos"
The pattern captures everything before the timestamp as the label
"""


# =========================
# REPRODUCIBILITY -> Set random seeds everywhere so results are the same each run
# =========================
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# HELPERS
# =========================

def fix_len(audio: np.ndarray, sr: int) -> np.ndarray: #Forces an audio array to be exactly CLIP_SECONDS long
    target_len = int(sr * CLIP_SECONDS)
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]

def normalize_label(label: str) -> str: # Strips whitespace and converts to lowercase so
    return str(label).strip().lower()

def extract_label_from_filename(filename: str) -> str | None: # Pulls the spell name out of a filename using regex.
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return normalize_label(match.group("label"))

def load_audio_fixed_length(wav_path: str, sr: int, clip_seconds: float) -> np.ndarray:
    """Load audio and force it to exactly clip_seconds long."""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    return fix_len(audio.astype(np.float32), sr)

def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns mfcc in shape: (time_steps, 3*n_mfcc)
    Includes MFCC + delta + delta-delta for richer speech features.
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    """
    Delta MFCCs capture how the sound is changing over time, not just what it sounds like at a single moment.
    Delta is the first derivative — the rate of change between frames. Like going from a position to a velocity.
    Delta-delta is the second derivative — the rate of change of the change. Like going from velocity to acceleration.
    """
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])  # (3*N_MFCC, time)
    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8) #+1e-8 so you do not devide by 0
    return combined.T.astype(np.float32)  # (time, 3*N_MFCC)

def fix_len(audio: np.ndarray, sr: int) -> np.ndarray: #fixes the len of video after augmentation
    target_len = int(sr * CLIP_SECONDS)
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]

def find_wav_files(wav_dir: Path): #Raises clear errors if the folder doesn't exist or is empty
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV folder not found: {wav_dir}")
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in: {wav_dir}")
    return wav_files

def augment_audio(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Returns a list of augmented versions of the input audio."""
    augmented = []

    # Add background noise
    noise = np.random.randn(len(audio)).astype(np.float32)
    noise_factor = random.uniform(0.003, 0.008)
    augmented.append(audio + noise_factor * noise)

    # Time stretch
    stretched_slow = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 0.9))
    augmented.append(fix_len(stretched_slow.astype(np.float32), sr))
    stretched_fast = librosa.effects.time_stretch(audio, rate=random.uniform(1.1, 1.2))
    augmented.append(fix_len(stretched_fast.astype(np.float32), sr))

    # 4. Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(1, 3)).astype(np.float32))
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-3, -1)).astype(np.float32))

    # 6. Volume
    augmented.append((audio * random.uniform(1.2, 1.5)).astype(np.float32))
    augmented.append((audio * random.uniform(0.4, 0.7)).astype(np.float32))

    # 8. Time shift
    shiftforward = int(random.uniform(0.05, 0.2) * sr)
    augmented.append(fix_len(np.roll(audio, shiftforward).astype(np.float32), sr))
    shiftbackward = int(random.uniform(0.05, 0.2) * sr)
    augmented.append(fix_len(np.roll(audio, -shiftbackward).astype(np.float32), sr))

    return augmented # 9 modified versions per original clip

def build_model(time_steps: int, n_features: int, num_classes: int):
    """
        a CNN-LSTM hybrid model
        - Conv1D layers: scan the time series for local patterns (like phonemes)
        - BatchNormalization: keeps values in a healthy range during training
        - MaxPooling1D: reduces the time dimension
        - Dropout: randomly turns off neurons(overfitting).
        - Bidirectional LSTM: reads the sequence both forward and backward,
        - Dense layers: final classification layers that combine all features
        - Softmax output: converts raw scores to probabilities that sum to 1,
    """

    model = models.Sequential([
        layers.Input(shape=(time_steps, n_features)),

        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_config(time_steps: int, n_features: int, classes): #Saves the audio processing settings to a JSON file
    config = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "time_steps": int(time_steps),
        "n_features": int(n_features),
        "classes": list(classes)
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def main():
    print(f"Using wav folder: {WAV_DIR}")
    wav_files = find_wav_files(WAV_DIR)
    print(f"Found {len(wav_files)} wav files")

    # =========================
    # COLLECT LABELED FILE PATHS
    # =========================
    wav_files_labeled = []
    labels_for_split = []

    # Fit label encoder on all labels first
    all_labels = []
    for wav_path in sorted(WAV_DIR.glob("*.wav")):
        label = extract_label_from_filename(wav_path.name)
        if label is not None:
            all_labels.append(label)

    le = LabelEncoder()
    le.fit(all_labels)

    print("\nClasses:")
    for i, c in enumerate(le.classes_):
        count = all_labels.count(c)
        print(f"  {i}: {c} ({count} samples)")

    num_classes = len(le.classes_)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    for wav_path in sorted(WAV_DIR.glob("*.wav")):
        label = extract_label_from_filename(wav_path.name)
        if label is None:
            continue
        try:
            label_enc = le.transform([label])[0]
            wav_files_labeled.append(wav_path)
            labels_for_split.append(label_enc)
        except ValueError:
            continue

    labels_for_split = np.array(labels_for_split)

    # =========================
    # SPLIT FILE PATHS (not arrays)
    # =========================
    class_counts = np.bincount(labels_for_split)
    use_stratify = np.all(class_counts >= 2)

    if not use_stratify:
        print("\nWarning: at least one class has fewer than 2 samples.")

    train_paths, test_paths, y_train_paths, y_test_paths = train_test_split(
        wav_files_labeled,
        labels_for_split,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels_for_split if use_stratify else None
    )

    # =========================
    # BUILD CLEAN TEST SET
    # =========================
    print("\nBuilding test set (no augmentation)...")
    X_test_list, y_test_list = [], []
    for wav_path, label_enc in zip(test_paths, y_test_paths):
        audio = load_audio_fixed_length(str(wav_path), SAMPLE_RATE, CLIP_SECONDS)
        X_test_list.append(extract_mfcc(audio, SAMPLE_RATE))
        y_test_list.append(label_enc)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list)

    # =========================
    # BUILD TRAINING SET WITH AUGMENTATION
    # =========================
    print("Building training set (with augmentation)...")
    X_train_list, y_train_list = [], []
    for wav_path, label_enc in zip(train_paths, y_train_paths):
        audio = load_audio_fixed_length(str(wav_path), SAMPLE_RATE, CLIP_SECONDS)

        # Original
        X_train_list.append(extract_mfcc(audio, SAMPLE_RATE))
        y_train_list.append(label_enc)

        # Augmented versions
        for aug_audio in augment_audio(audio, SAMPLE_RATE):
            X_train_list.append(extract_mfcc(aug_audio, SAMPLE_RATE))
            y_train_list.append(label_enc)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list)

    print(f"\nTraining samples after augmentation: {len(X_train)}")
    print(f"Test samples (clean):                {len(X_test)}")

    print("\nTest set class distribution:")
    for i, c in enumerate(le.classes_):
        print(f"  {c}: {int(np.sum(y_test == i))} test samples")

    # =========================
    # CLASS WEIGHTS
    # =========================
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # =========================
    # BUILD MODEL
    # =========================
    time_steps = X_train.shape[1]
    n_features = X_train.shape[2]  # 3*N_MFCC = 120 with deltas

    print(f"\nInput shape: ({time_steps}, {n_features})")

    model = build_model(time_steps, n_features, num_classes)
    model.summary()

    # =========================
    # TRAIN
    # =========================
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
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
        class_weight=class_weight_dict,
        verbose=1
    )

    # =========================
    # EVALUATE
    # =========================
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    print("\nPrediction distribution:")
    for i, c in enumerate(le.classes_):
        print(f"  {c}: {int(np.sum(pred_labels == i))} predictions")

    print("\nClassification report:")
    print(classification_report(y_test, pred_labels, target_names=le.classes_, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred_labels))

    # =========================
    # SAVE
    # =========================
    model.save(MODEL_OUT)

    with open(LABELS_OUT, "wb") as f:
        pickle.dump(le, f)

    save_config(time_steps, n_features, le.classes_)

    print(f"\nSaved model to:  {MODEL_OUT}")
    print(f"Saved labels to: {LABELS_OUT}")
    print(f"Saved config to: {CONFIG_OUT}")


if __name__ == "__main__":
    main()