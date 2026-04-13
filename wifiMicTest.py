import socket
import threading
import wave
import time
import os

import numpy as np

ESP_IP = "192.168.4.1"
UDP_PORT = 12345
SAMPLE_RATE = 48000   # set this to match your USB config exactly
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16

SAVE_DIR = "recordings"

is_recording = False
recorded_chunks = []
recording_count = 0
packets_received_total = 0
packets_at_record_start = 0
lock = threading.Lock()


def save_wav(filename: str, audio_int16: np.ndarray):
    os.makedirs(SAVE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_DIR, filename)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    print(f"💾 Saved: {filepath}")


def start_recording():
    global is_recording, recorded_chunks, packets_at_record_start

    with lock:
        recorded_chunks = []
        packets_at_record_start = packets_received_total
        is_recording = True
        print("🔴 Recording started")


def stop_recording():
    global is_recording, recording_count

    with lock:
        is_recording = False

        new_packets = packets_received_total - packets_at_record_start

        if new_packets <= 0 or len(recorded_chunks) == 0:
            print("⚠️ No new audio received during this recording")
            return

        audio = np.concatenate(recorded_chunks).astype(np.int16)
        filename = f"esp_recording_{recording_count:03d}.wav"
        recording_count += 1
        save_wav(filename, audio)

        seconds = len(audio) / SAMPLE_RATE
        print(f"Saved {len(audio)} samples ({seconds:.2f} s)")


def toggle_recording():
    with lock:
        currently_recording = is_recording

    if currently_recording:
        stop_recording()
    else:
        start_recording()


def input_thread():
    print("\nPress ENTER to start/stop recording")
    print("Type 'q' then ENTER to quit\n")

    while True:
        cmd = input().strip().lower()

        if cmd == "q":
            print("Exiting...")
            os._exit(0)
        else:
            toggle_recording()


def main():
    global packets_received_total

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    # Tell ESP where to stream
    sock.sendto(b"hello", (ESP_IP, UDP_PORT))
    print(f"Sent hello to {ESP_IP}:{UDP_PORT}")

    threading.Thread(target=input_thread, daemon=True).start()

    last_debug = time.time()

    while True:
        data, addr = sock.recvfrom(4096)
        samples = np.frombuffer(data, dtype=np.int16)

        if samples.size == 0:
            continue

        with lock:
            packets_received_total += 1
            if is_recording:
                recorded_chunks.append(samples.copy())

        now = time.time()
        if now - last_debug > 2.0:
            print(f"Packets received so far: {packets_received_total}")
            last_debug = now


if __name__ == "__main__":
    main()