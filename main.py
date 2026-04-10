import os
import json
from datetime import datetime
import tkinter as tk

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000
N_MFCC = 13

SPELLS = {
    "1": "lumos",
    "2": "wingardium leviosa",
    "3": "Aguamenti"
}

DATASET_DIR = "spell_dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
IMG_DIR = os.path.join(DATASET_DIR, "images")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.jsonl")
# -----------------------------------------


class SpellRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Harry Potter Spell Recorder")
        self.root.geometry("720x520")
        self.root.resizable(False, False)

        os.makedirs(WAV_DIR, exist_ok=True)
        os.makedirs(IMG_DIR, exist_ok=True)

        self.is_recording = False
        self.current_label = None
        self.current_key = None
        self.stream = None
        self.audio_buffer = []

        self.status_var = tk.StringVar(value="Ready. Press 1, 2, or 3 to start recording.")
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
                "Press one of these keys:\n"
                f"1 = {SPELLS['1']}\n"
                f"2 = {SPELLS['2']}\n"
                f"3 = {SPELLS['3']}\n\n"
                "Press the same key again to stop recording.\n"
                "Press ESC to quit."
            ),
            font=("Arial", 12),
            justify="left"
        )
        instructions.pack(pady=10)

        status_label = tk.Label(
            root,
            textvariable=self.status_var,
            font=("Arial", 12),
            wraplength=680,
            justify="left"
        )
        status_label.pack(pady=10)

        count_label = tk.Label(
            root,
            textvariable=self.count_var,
            font=("Arial", 11),
            justify="left"
        )
        count_label.pack(pady=5)

        self.log_box = tk.Text(root, height=15, width=82, state="disabled")
        self.log_box.pack(pady=10)

        root.bind("<KeyPress-1>", lambda event: self.handle_key("1"))
        root.bind("<KeyPress-2>", lambda event: self.handle_key("2"))
        root.bind("<KeyPress-3>", lambda event: self.handle_key("3"))
        root.bind("<Escape>", lambda event: self.on_close())

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

        return "Saved samples:\n" + "\n".join(
            [f"{label}: {count}" for label, count in counts.items()]
        )

    def handle_key(self, key):
        if key not in SPELLS:
            return

        label = SPELLS[key]

        if not self.is_recording:
            self.start_recording(key, label)
        else:
            if key == self.current_key:
                self.stop_recording()
            else:
                self.status_var.set(
                    f"Currently recording {self.current_label}. Press {self.current_key} again to stop first."
                )

    def start_recording(self, key, label):
        try:
            self.audio_buffer = []
            self.current_label = label
            self.current_key = key
            self.is_recording = True

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self.audio_callback
            )
            self.stream.start()

            self.status_var.set(f"Recording {label}... Press {key} again to stop.")
            self.log(f"[{datetime.now().strftime('%H:%M:%S')}] START recording: {label}")

        except Exception as e:
            self.status_var.set(f"Error starting recording: {e}")
            self.log(f"ERROR starting recording: {e}")
            self.is_recording = False
            self.current_label = None
            self.current_key = None
            self.audio_buffer = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer.append(indata.copy())

    def stop_recording(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if not self.audio_buffer:
                self.status_var.set("No audio recorded.")
                self.log("WARNING: No audio captured.")
                return

            audio = np.concatenate(self.audio_buffer, axis=0).flatten()

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            label = self.current_label
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = f"{label}_{timestamp}"

            wav_path = os.path.join(WAV_DIR, base_name + ".wav")
            img_path = os.path.join(IMG_DIR, base_name + ".png")

            # Save raw audio
            sf.write(wav_path, audio, SAMPLE_RATE)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC
            )

            # Better contrast for image saving
            mfcc_db = librosa.power_to_db(np.abs(mfcc), ref=np.max)

            # Save MFCC image
            plt.figure(figsize=(4, 4))

            librosa.display.specshow(
                mfcc_db,
                sr=SAMPLE_RATE,
                x_axis='time'
            )
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            metadata = {
                "label": label,
                "wav_path": wav_path,
                "image_path": img_path,
                "timestamp": timestamp,
                "sample_rate": SAMPLE_RATE,
                "n_mfcc": N_MFCC
            }

            with open(LABELS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata) + "\n")

            self.status_var.set(f"Saved recording for {label}")
            self.count_var.set(self.get_counts_text())
            self.log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP recording: {label}")
            self.log(f"Saved WAV: {wav_path}")
            self.log(f"Saved image: {img_path}")

        except Exception as e:
            self.status_var.set(f"Error stopping recording: {e}")
            self.log(f"ERROR stopping recording: {e}")

        finally:
            self.is_recording = False
            self.current_label = None
            self.current_key = None
            self.audio_buffer = []

    def on_close(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpellRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()