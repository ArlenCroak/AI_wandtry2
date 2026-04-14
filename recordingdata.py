import os
from datetime import datetime
import tkinter as tk
import threading
import socket

import numpy as np
import soundfile as sf


# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000

ESP_IP = "192.168.4.1"
UDP_PORT = 12345

SPELLS = {
    "1": "lumos",
    "2": "wingardium leviosa",
    "3": "aguamenti",
    "0": "nothing"
}

DATASET_DIR = "spell_dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
# -----------------------------------------


class SpellRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Harry Potter Spell Recorder (WiFi Mic)")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        os.makedirs(WAV_DIR, exist_ok=True)

        self.is_recording = False
        self.current_label = None
        self.current_key = None
        self.audio_buffer = []

        self.lock = threading.Lock()

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", UDP_PORT))
        self.sock.sendto(b"hello", (ESP_IP, UDP_PORT))

        # start UDP thread
        threading.Thread(target=self.udp_listener, daemon=True).start()

        self.status_var = tk.StringVar(value="Ready. Press 0, 1, 2, or 3.")
        self.count_var = tk.StringVar(value=self.get_counts_text())
        self.big_label_var = tk.StringVar(value="READY")

        tk.Label(root, text="Spell Data Recorder (WiFi)", font=("Arial", 20, "bold")).pack(pady=10)

        tk.Label(
            root,
            textvariable=self.big_label_var,
            font=("Arial", 42, "bold"),
            fg="white",
            bg="black",
            width=20,
            height=2
        ).pack(pady=20)

        tk.Label(
            root,
            text=(
                "1 = lumos\n2 = wingardium leviosa\n3 = aguamenti\n0 = nothing\n\n"
                "Press same key again to stop\nESC to quit"
            ),
            font=("Arial", 12)
        ).pack()

        tk.Label(root, textvariable=self.status_var).pack(pady=10)
        tk.Label(root, textvariable=self.count_var).pack()

        self.log_box = tk.Text(root, height=16, width=100, state="disabled")
        self.log_box.pack(pady=10)

        root.bind("<KeyPress-0>", lambda e: self.handle_key("0"))
        root.bind("<KeyPress-1>", lambda e: self.handle_key("1"))
        root.bind("<KeyPress-2>", lambda e: self.handle_key("2"))
        root.bind("<KeyPress-3>", lambda e: self.handle_key("3"))
        root.bind("<Escape>", lambda e: self.on_close())

    # ---------------- UDP AUDIO ----------------
    def udp_listener(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            samples = np.frombuffer(data, dtype=np.int16)

            if samples.size == 0:
                continue

            with self.lock:
                if self.is_recording:
                    self.audio_buffer.append(samples.copy())

    # ---------------- UI LOG ----------------
    def log(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def get_counts_text(self):
        counts = {spell: 0 for spell in SPELLS.values()}
        if os.path.exists(WAV_DIR):
            for f in os.listdir(WAV_DIR):
                if f.endswith(".wav"):
                    for label in counts:
                        if f.startswith(label + "_"):
                            counts[label] += 1
        return "\n".join([f"{k}: {v}" for k, v in counts.items()])

    # ---------------- RECORDING ----------------
    def handle_key(self, key):
        if not self.is_recording:
            self.start_recording(key, SPELLS[key])
        elif key == self.current_key:
            self.stop_recording()

    def start_recording(self, key, label):
        with self.lock:
            self.audio_buffer = []
            self.is_recording = True

        self.current_label = label
        self.current_key = key

        self.status_var.set(f"Recording {label}...")
        self.big_label_var.set(label.upper())
        self.log(f"START: {label}")

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            if not self.audio_buffer:
                self.status_var.set("No audio!")
                return
            audio = np.concatenate(self.audio_buffer)

        # normalize
        audio = audio.astype(np.float32)
        audio /= (np.max(np.abs(audio)) + 1e-6)

        filename = f"{self.current_label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        path = os.path.join(WAV_DIR, filename)

        sf.write(path, audio, SAMPLE_RATE)

        self.status_var.set(f"Saved {filename}")
        self.big_label_var.set("READY")
        self.count_var.set(self.get_counts_text())

        self.log(f"Saved: {path}")

        self.current_label = None
        self.current_key = None
        self.audio_buffer = []

    def on_close(self):
        self.sock.close()
        self.root.destroy()


# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SpellRecorderApp(root)
    root.mainloop()