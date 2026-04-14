import json
import pickle
import time
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

from Traning import fix_len

# =========================
# PATHS
# =========================
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "spell_cnn_lstm.keras"
LABELS_PATH = BASE_DIR / "spell_labels.pkl"
CONFIG_PATH = BASE_DIR / "spell_model_config.json"

# =========================
# DETECTION SETTINGS
# =========================
CONFIDENCE_THRESHOLD = 0.85  # model must be at least this confident to announce a spell
COOLDOWN_SECONDS     = 2.0   # after detecting a spell, ignore detections for this long
STEP_SECONDS         = 0.3   # how often we run the model on the buffer

# =========================
# LOAD MODEL + CONFIG
# =========================
def load_everything(): #Load the trained model, label encoder, and audio config.

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading labels...")
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)

    print("Loading config...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print(f"\nLoaded successfully!")
    print(f"Listening for: {list(le.classes_.astype(str))}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"Sample rate: {config['sample_rate']} Hz")
    print(f"Clip length: {config['clip_seconds']} seconds\n")

    return model, le, config

# =========================
# AUDIO PROCESSING : imported the functions fix_len from traning.py
# =========================
def extract_mfcc(audio: np.ndarray, config: dict) -> np.ndarray: #thise one different then traning
    """
    Extract MFCC + delta + delta-delta features.
    Identical to extract_mfcc() in Traning.py.
    Output shape: (time_steps, 3 * n_mfcc)
    """
    sr         = config["sample_rate"]
    n_mfcc     = config["n_mfcc"]
    n_fft      = config["n_fft"]
    hop_length = config["hop_length"]

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])
    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)
    return combined.T.astype(np.float32)  # (time_steps, n_features)

def predict_spell(audio: np.ndarray, model, le, config) -> tuple[str, float]: #Run the model on a chunk of audio. returns (spell_name, confidence) where confidence is 0.0 to 1.0.

    sr         = config["sample_rate"] #for easier writing
    clip_secs  = config["clip_seconds"] #for easier writing
    target_len = int(sr * clip_secs)


    audio = fix_len(audio.astype(np.float32), target_len) # run fix_len
    mfcc = extract_mfcc(audio, config) # run extract_mfcc
    X = np.expand_dims(mfcc, axis=0)  # Add batch dimension: (time, features) -> (1, time, features)
    probs = model.predict(X, verbose=0)[0] # Run model, get probability for each spell class

    best_idx   = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    spell_name = le.classes_[best_idx]

    return spell_name, confidence

# =========================
# REAL-TIME LISTENER
# =========================
def listen(model, le, config): #Keeps the microphone running continuously using a rolling buffer.

    sr           = config["sample_rate"]
    clip_secs    = config["clip_seconds"]
    buffer_size  = int(sr * clip_secs)
    step_samples = int(sr * STEP_SECONDS)

    buffer = np.zeros(buffer_size, dtype=np.float32) # Rolling buffer starts as silence
    last_detection_time = 0.0 # Tracks when we last fired a detection for cooldown


    def audio_callback(indata, frames, time_info, status):
        """
        Called by sounddevice every time a new chunk arrives from the mic.
        Runs in a background thread automatically.
        indata shape: (frames, channels) — we use only channel 0 (mono).
        """
        nonlocal buffer
        chunk = indata[:, 0].astype(np.float32)

        # Roll buffer left by chunk size, write new chunk at the end
        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk

    print("=" * 50)
    print("Microphone is active. Say a spell!")
    print("Spells: " + " | ".join(le.classes_))
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    # Open the microphone stream
    # blocksize=step_samples means the callback fires every STEP_SECONDS
    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=step_samples,
        callback=audio_callback
    ):
        while True:
            # Wait one step interval before running the model
            time.sleep(STEP_SECONDS)

            now = time.time()

            # If still in cooldown, show countdown and skip
            if now - last_detection_time < COOLDOWN_SECONDS:
                remaining = COOLDOWN_SECONDS - (now - last_detection_time)
                print(f"  [cooldown {remaining:.1f}s]                              ", end="\r")
                continue

            # Snapshot the buffer so the mic callback can't modify it mid-prediction
            audio_chunk = buffer.copy()

            # Run the model on the last 5 seconds of audio
            spell, confidence = predict_spell(audio_chunk, model, le, config)

            # Draw a live confidence bar in the terminal
            # Example: [████████████░░░░░░░░] 60%  lumos
            filled = int(confidence * 20)
            bar    = "█" * filled + "░" * (20 - filled)
            print(f"  [{bar}] {confidence * 100:5.1f}%  {spell:<25}", end="\r")

            # Only announce if confidence clears the threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"\n✨ SPELL DETECTED: {spell.upper()} ({confidence * 100:.1f}% confident)\n")
                last_detection_time = now

# =========================
# MAIN
# =========================
def main():
    # Verify all required files exist before doing anything
    missing = [p for p in [MODEL_PATH, LABELS_PATH, CONFIG_PATH] if not p.exists()]
    if missing:
        print("ERROR: Missing files:")
        for p in missing:
            print(f"  {p}")
        print("\nMake sure you have run Traning.py first to generate these files.")
        return

    model, le, config = load_everything()

    try:
        listen(model, le, config)
    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye!")


if __name__ == "__main__":
    main()
