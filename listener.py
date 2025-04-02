import tkinter as tk
import sounddevice as sd
import numpy as np
import queue
from protocol import IncrementalDecoder, SAMPLE_RATE

class ListenerGUI:
    def __init__(self, master):
        self.master = master
        self.decoder = IncrementalDecoder()
        self.q = queue.Queue()
        self.running = False
        self.history = []

        master.title("Acoustic Modem Listener")

        self.title = tk.Label(master, text="🎧 Acoustic Listener", font=("Helvetica", 16))
        self.title.pack(pady=10)

        self.warning_label = tk.Label(
            master,
            text=(
                "⚠️ After clicking 'Start Listening', please wait at least 2 seconds\n"
                "before playing the sender’s message. This gives your microphone time\n"
                "to initialize and prevents the preamble from being missed."
            ),
            wraplength=420,
            justify="center",
            fg="darkred",
            font=("Helvetica", 11, "bold")
        )
        self.warning_label.pack(pady=10)
    

        self.output_label = tk.Label(master, text="Press Start to begin listening.", wraplength=400, justify="center")
        self.output_label.pack(pady=10)

        self.start_button = tk.Button(master, text="Start Listening", command=self.start_listening)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(master, text="Stop", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.history_label = tk.Label(master, text="Decoded Message History:")
        self.history_label.pack()

        self.history_box = tk.Text(master, height=10, width=50, state=tk.DISABLED, wrap=tk.WORD)
        self.history_box.pack(padx=10, pady=5)

        self.explanation = tk.Label(master, text=(
            "🔹 Checksum mismatch: Message may contain errors.\n"
            "   The checksum is a final character added during transmission that encodes a summary of the message.\n"
            "   It's calculated by taking the sum of the ASCII values of all characters, then converting that into a letter.\n"
            "   If the received message doesn't match this expected checksum, something likely got distorted in transmission.\n\n"
            "🔸 Uncertain characters: Some tones were noisy or ambiguous based on their frequency signature.\n"
            "   These are flagged using an entropy measure, which detects if a tone is too 'fuzzy' to confidently interpret.\n\n"
            "📦 Listening stops automatically once the end-of-message tone (postamble) is detected."
        ), wraplength=420, justify="left", fg="gray", font=("Courier", 12))
        self.explanation.pack(pady=10)

        self.master.after(100, self.process_queue)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self.decoder.feed_samples(mono)
        msg = self.decoder.get_message()
        self.q.put(msg)
        if self.decoder.is_done():
            self.running = False

    def start_listening(self):
        self.decoder = IncrementalDecoder()
        self.q = queue.Queue()
        self.running = True
        self.output_label.config(text="Listening...")

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=int(0.1 * SAMPLE_RATE),
            callback=self.audio_callback,
            dtype='float32'
        )
        self.stream.start()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_listening(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        self.output_label.config(text="Stopped.")

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_history(self, final_msg):
        self.history.insert(0, final_msg)
        self.history_box.config(state=tk.NORMAL)
        self.history_box.delete("1.0", tk.END)
        for idx, msg in enumerate(self.history[:10], 1):  # Show only last 10 messages
            self.history_box.insert(tk.END, f"{idx}. {msg}\n\n")
        self.history_box.config(state=tk.DISABLED)

    def process_queue(self):
        try:
            while not self.q.empty():
                msg = self.q.get_nowait()
                self.output_label.config(text="Partial decode:\n" + msg)
                if self.decoder.is_done():
                    self.update_history(msg)
                    self.stop_listening()
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

def main():
    root = tk.Tk()
    app = ListenerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
