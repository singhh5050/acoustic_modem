import argparse
import sounddevice as sd
from protocol import encode_text_to_waveform, SAMPLE_RATE

def main():
    # Grab message from command-line arguments or default to "HELLO"
    parser = argparse.ArgumentParser(description="Acoustic Modem Sender")
    parser.add_argument("message", nargs="*", help="Message to transmit")
    args = parser.parse_args()

    text = " ".join(args.message) if args.message else "HELLO"
    print(f"Transmitting: '{text}'")

    # Encode text to waveform
    waveform = encode_text_to_waveform(text)

    # Play the audio
    sd.play(waveform, SAMPLE_RATE)
    sd.wait()  # wait until playback finishes

    print("Done transmitting!")

if __name__ == "__main__":
    main()
