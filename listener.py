import argparse
import numpy as np
import sounddevice as sd
from protocol import decode_waveform_to_text, SAMPLE_RATE

def main():
    # Record for N seconds, then decode
    parser = argparse.ArgumentParser(description="Acoustic Modem Listener")
    parser.add_argument("--duration", type=float, default=10,
                        help="How many seconds to record (default 10s)")
    args = parser.parse_args()

    print(f"Listening for {args.duration} seconds...")
    recording = sd.rec(int(args.duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, blocking=True)
    recording = recording.flatten()  # make it 1D
    print("Decoding...")

    # Decode the recorded audio
    decoded_msg = decode_waveform_to_text(recording)
    print(f"Decoded message: '{decoded_msg}'")

if __name__ == "__main__":
    main()

