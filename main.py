import os
import json
import queue
from datetime import datetime
import tkinter as tk

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa


# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000
RECORD_SECONDS = 2.5
N_MFCC = 13

# Change these to your actual spell names
SPELLS = {
    "1": "lumos",
    "2": "expelliarmus",
    "3": "alohomora"
}

DATASET_DIR = "spell_dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
MFCC_DIR = os.path.join(DATASET_DIR, "mfcc")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.jsonl")
# -----------------------------------------


class SpellRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Harry Potter Spell Recorder")
        self.root.geometry("650x420")
        self.root.resizable(False, False)

        os.makedirs(WAV_DIR, exist_ok=True)
        os.makedirs(MFCC_DIR, exist_ok=True)

        self.is_recording = False
        self.message_queue = queue.Queue()

        self.status_var = tk.StringVar(value="Ready. Press 1, 2, or 3 to record a spell.")
        self.count_var = tk.StringVar(value=self.get_counts_text())

        title = tk.Label(
            root,
            text="Spell Data Recorder",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        instructions = tk.Label(
            root,
            text=(
                "Press one of these keys to record:\n"
                f"1 = {SPELLS['1']}\n"
                f"2 = {SPELLS['2']}\n"
                f"3 = {SPELLS['3']}\n\n"
                "Press ESC to quit."
            ),
            font=("Arial", 12),
            justify="left"
        )
        instructions.pack(pady=10)

        self.status_label = tk.Label(
            root,
            textvariable=self.status_var,
            font=("Arial", 12),
            wraplength=600,
            justify="left"
        )
        self.status_label.pack(pady=10)

        self.count_label = tk.Label(
            root,
            textvariable=self.count_var,
            font=("Arial", 11),
            justify="left"
        )
        self.count_label.pack(pady=5)

        self.log_box = tk.Text(root, height=10, width=75, state="disabled")
        self.log_box.pack(pady=10)

        root.bind("<KeyPress-1>", lambda event: self.start_recording("1"))
        root.bind("<KeyPress-2>", lambda event: self.start_recording("2"))
        root.bind("<KeyPress-3>", lambda event: self.start_recording("3"))
        root.bind("<Escape>", lambda event: root.destroy())

        self.root.after(100, self.process_queue)

    def log(self, text: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def get_counts_text(self):
        counts = {spell: 0 for spell in SPELLS.values()}

        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        label = item["label"]
                        if label in counts:
                            counts[label] += 1
                    except Exception:
                        pass

        return (
            "Saved samples:\n"
            + "\n".join([f"{label}: {count}" for label, count in counts.items()])
        )

    def start_recording(self, key):
        if self.is_recording:
            self.status_var.set("Already recording. Wait until it finishes.")
            return

        if key not in SPELLS:
            return

        self.is_recording = True
        label = SPELLS[key]
        self.status_var.set(f"Recording spell: {label} ... Speak now!")
        self.log(f"[{datetime.now().strftime('%H:%M:%S')}] Started recording: {label}")

        self.root.after(100, lambda: self.record_and_save(label))

    def record_and_save(self, label):
        try:
            num_samples = int(SAMPLE_RATE * RECORD_SECONDS)

            audio = sd.rec(
                frames=num_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            audio = audio.flatten()

            # normalize if not silent
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = f"{label}_{timestamp}"

            wav_path = os.path.join(WAV_DIR, base_name + ".wav")
            mfcc_path = os.path.join(MFCC_DIR, base_name + ".npy")

            # Save raw audio
            sf.write(wav_path, audio, SAMPLE_RATE)

            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC
            )

            # Optional: use a fixed-size feature vector for easier training later
            # Here we store mean and std over time
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            feature_vector = np.concatenate([mfcc_mean, mfcc_std])

            np.save(mfcc_path, feature_vector)

            metadata = {
                "label": label,
                "wav_path": wav_path,
                "mfcc_path": mfcc_path,
                "timestamp": timestamp,
                "sample_rate": SAMPLE_RATE,
                "record_seconds": RECORD_SECONDS,
                "n_mfcc": N_MFCC
            }

            with open(LABELS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata) + "\n")

            self.status_var.set(f"Saved recording for: {label}")
            self.count_var.set(self.get_counts_text())
            self.log(f"[{datetime.now().strftime('%H:%M:%S')}] Saved: {label}")

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            self.log(f"ERROR: {e}")

        finally:
            self.is_recording = False

    def process_queue(self):
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            self.log(msg)
        self.root.after(100, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = SpellRecorderApp(root)
    root.mainloop()