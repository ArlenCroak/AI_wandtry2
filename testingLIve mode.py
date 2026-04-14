import json
import pickle
import time
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import serial

from Traning import fix_len

# =========================
# PATHS
# =========================
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "spell_cnn_lstm.keras"
LABELS_PATH = BASE_DIR / "spell_labels.pkl"
CONFIG_PATH = BASE_DIR / "spell_model_config.json"

# =========================
# SERIAL SETTINGS
# =========================
SERIAL_PORT = "/dev/ttyUSB0"   # change if needed
BAUD_RATE = 115200

SPELL_TO_COMMAND = {
    "lumos": "light",
    "wingardium leviosa": "lift",
    "aguamenti": "splash"
}

# =========================
# DETECTION SETTINGS
# =========================
CONFIDENCE_THRESHOLD = 0.95
NOTHING_MARGIN       = 0.25
RMS_GATE             = 0.01
COOLDOWN_SECONDS     = 2.0
STEP_SECONDS         = 0.3
SHOW_TOP3            = True

# =========================
# LOAD MODEL + CONFIG
# =========================
def load_everything():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading labels...")
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)

    print("Loading config...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    classes = [str(c) for c in le.classes_]

    print(f"\nLoaded successfully!")
    print(f"Listening for: {classes}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"Nothing margin: {NOTHING_MARGIN:.2f}")
    print(f"RMS gate: {RMS_GATE}")
    print(f"Sample rate: {config['sample_rate']} Hz")
    print(f"Clip length: {config['clip_seconds']} seconds\n")

    if "nothing" not in classes:
        raise ValueError("Your label set does not contain a 'nothing' class.")

    return model, le, config

# =========================
# AUDIO PROCESSING
# =========================
def extract_mfcc(audio: np.ndarray, config: dict) -> np.ndarray:
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
    return combined.T.astype(np.float32)

def predict_probs(audio: np.ndarray, model, config) -> np.ndarray:
    sr         = config["sample_rate"]
    clip_secs  = config["clip_seconds"]
    target_len = int(sr * clip_secs)

    audio = fix_len(audio.astype(np.float32), target_len)
    mfcc = extract_mfcc(audio, config)
    X = np.expand_dims(mfcc, axis=0)
    probs = model.predict(X, verbose=0)[0]
    return probs

def top_k_predictions(probs: np.ndarray, le, k: int = 3):
    idxs = np.argsort(probs)[::-1][:k]
    return [(str(le.classes_[i]), float(probs[i])) for i in idxs]

# =========================
# REAL-TIME LISTENER
# =========================
def listen(model, le, config, ser):
    sr           = config["sample_rate"]
    clip_secs    = config["clip_seconds"]
    buffer_size  = int(sr * clip_secs)
    step_samples = int(sr * STEP_SECONDS)

    buffer = np.zeros(buffer_size, dtype=np.float32)
    last_detection_time = 0.0

    classes = [str(c) for c in le.classes_]
    nothing_idx = classes.index("nothing")

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer

        if status:
            print(status)

        chunk = indata[:, 0].astype(np.float32)

        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk

    print("=" * 60)
    print("Microphone is active. Say a spell!")
    print("Spells: " + " | ".join(classes))
    print("Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=step_samples,
        callback=audio_callback
    ):
        while True:
            time.sleep(STEP_SECONDS)
            now = time.time()

            if now - last_detection_time < COOLDOWN_SECONDS:
                remaining = COOLDOWN_SECONDS - (now - last_detection_time)
                print(f"  [cooldown {remaining:.1f}s]                              ", end="\r")
                continue

            audio_chunk = buffer.copy()

            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            if rms < RMS_GATE:
                print(f"  [quiet] rms={rms:.4f} below gate {RMS_GATE:.4f}                     ", end="\r")
                continue

            t0 = time.time()
            probs = predict_probs(audio_chunk, model, config)
            pred_ms = (time.time() - t0) * 1000.0

            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            best_class = classes[best_idx]

            nothing_conf = float(probs[nothing_idx])
            margin = best_conf - nothing_conf

            filled = int(best_conf * 20)
            bar = "█" * filled + "░" * (20 - filled)

            top3_text = ""
            if SHOW_TOP3:
                top3 = top_k_predictions(probs, le, 3)
                top3_text = " | ".join([f"{name}:{conf:.2f}" for name, conf in top3])

            print(
                f"  [{bar}] {best_conf * 100:5.1f}%  {best_class:<20} "
                f"rms={rms:.4f} margin={margin:.2f} pred={pred_ms:.1f}ms"
                + (f" | {top3_text}" if SHOW_TOP3 else "")
                + "          ",
                end="\r"
            )

            if (
                best_class != "nothing"
                and best_conf >= CONFIDENCE_THRESHOLD
                and margin >= NOTHING_MARGIN
            ):
                print(
                    f"\n✨ SPELL DETECTED: {best_class.upper()} "
                    f"({best_conf * 100:.1f}% confident, margin={margin:.2f}, rms={rms:.4f}, {pred_ms:.1f} ms)"
                )

                if ser is not None and best_class in SPELL_TO_COMMAND:
                    cmd = SPELL_TO_COMMAND[best_class] + "\n"
                    try:
                        ser.write(cmd.encode("utf-8"))
                        ser.flush()
                        print(f"📤 Sent to Arduino: {cmd.strip()}\n")
                    except Exception as e:
                        print(f"⚠️ Serial send failed: {e}\n")
                else:
                    print("⚠️ No serial connection or no mapped command for this spell.\n")

                last_detection_time = now

# =========================
# MAIN
# =========================
def main():
    missing = [p for p in [MODEL_PATH, LABELS_PATH, CONFIG_PATH] if not p.exists()]
    if missing:
        print("ERROR: Missing files:")
        for p in missing:
            print(f"  {p}")
        print("\nMake sure you have run Traning.py first to generate these files.")
        return

    model, le, config = load_everything()

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud")
    except Exception as e:
        print(f"⚠️ Could not open serial on {SERIAL_PORT}: {e}")
        ser = None

    try:
        listen(model, le, config, ser)
    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye!")
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()