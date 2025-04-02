import tkinter as tk
import sounddevice as sd
from protocol import (
    encode_text_to_waveform,
    SAMPLE_RATE,
    set_transmission_params,
)

# Supported characters
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"

class AcousticSenderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Acoustic Modem Sender")

        # GUI Title
        self.title_label = tk.Label(master, text="Acoustic Modem Transmitter", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # Message input prompt
        self.input_label = tk.Label(master, text="Enter your message:")
        self.input_label.pack()

        # Text entry field with a default message
        self.text_entry = tk.Entry(master, width=40, font=("Courier", 12))
        self.text_entry.pack(pady=5)
        self.text_entry.insert(0, "HELLO")

        # Amplitude control slider
        self.amp_label = tk.Label(master, text="Amplitude (volume):")
        self.amp_label.pack()
        self.amp_slider = tk.Scale(master, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=200)
        self.amp_slider.set(0.5)
        self.amp_slider.pack(pady=5)

        # Fade-in/out duration slider (in seconds)
        self.fade_label = tk.Label(master, text="Fade Duration (sec):")
        self.fade_label.pack()
        self.fade_slider = tk.Scale(master, from_=0.0, to=0.05, resolution=0.005, orient=tk.HORIZONTAL, length=200)
        self.fade_slider.set(0.01)
        self.fade_slider.pack(pady=5)

        # Button to transmit the message
        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=10)

        # Status message (e.g. transmission done, error)
        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack()

    def is_valid_message(self, text):
        """Checks if all characters are within the supported CHARSET."""
        return all(char in CHARSET for char in text.upper())

    def send_message(self):
        """Handles message validation, encoding, and playback."""
        text = self.text_entry.get().strip()

        # Don't allow empty messages
        if not text:
            self.status_label.config(text="Please enter a message to send.", fg="red")
            return

        # Validate message contents
        if not self.is_valid_message(text):
            self.status_label.config(
                text="Message contains unsupported characters.\nAllowed: A–Z, 0–9, space, period, comma, question mark, exclamation mark.",
                fg="red"
            )
            return

        # Update transmission parameters (volume and fade)
        amplitude = self.amp_slider.get()
        fade_time = self.fade_slider.get()
        set_transmission_params(amplitude, fade_time)

        # Show transmitting status
        self.status_label.config(text=f"Transmitting: '{text}'", fg="green")
        self.master.update()

        # Encode the message into audio and play it
        waveform = encode_text_to_waveform(text)
        sd.play(waveform, SAMPLE_RATE)
        sd.wait()

        # Show done status
        self.status_label.config(text="Done transmitting!", fg="blue")

# Launch the GUI
def main():
    root = tk.Tk()
    gui = AcousticSenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()