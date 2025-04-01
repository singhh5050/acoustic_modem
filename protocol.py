import numpy as np

# Constants
SAMPLE_RATE = 44100
TONE_DURATION = 0.2  # I'm making each tone half a second
AMPLITUDE = 0.5      # Volume factor, let's keep it modest

# Characters we support!
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"
BASE_FREQ = 600   # Hz -> where we start the frequency map
STEP_SIZE = 60    # Hz offset between subsequent chars

# Building a freq map: each char -> freq, freq -> char
char_to_freq = {char: BASE_FREQ + i * STEP_SIZE for i, char in enumerate(CHARSET)}
freq_to_char = {v: k for k, v in char_to_freq.items()}

# Preamble tone: a special freq that we don't use in CHARSET
PREAMBLE_FREQ = 300  # Hz (just pick something outside your char range)
PREAMBLE_DURATION = 0.3  # short beep to locate start

# Tonebank (easy look up!). Precompute sine waves for each char
_samples_per_tone = int(SAMPLE_RATE * TONE_DURATION)
_time = np.linspace(0, TONE_DURATION, _samples_per_tone, endpoint=False)  # time axis for one tone
tonebank = {
    char: AMPLITUDE * np.sin(2 * np.pi * freq * _time)  # basic sine wave formula
    for char, freq in char_to_freq.items()
}

def generate_preamble():
    """
    Create a short sine wave that we'll prepend to every message.
    We'll look for this in the received audio to sync up.
    """
    preamble_samples = int(SAMPLE_RATE * PREAMBLE_DURATION)
    t = np.linspace(0, PREAMBLE_DURATION, preamble_samples, endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * PREAMBLE_FREQ * t)

def encode_text_to_waveform(text):
    """
    Attach the preamble tone + the message tones. 
    That way, the decoder can find the start easily (no big offset search).
    """
    # Convert text to a single concatenated waveform of character tones
    message_tones = [
        tonebank[char] for char in text.upper() if char in tonebank
    ]
    if not message_tones:
        # If no valid characters, just send silence after preamble
        return np.concatenate([generate_preamble(), np.zeros_like(_time)])
    
    message_wave = np.concatenate(message_tones)
    # Prepend the preamble wave
    return np.concatenate([generate_preamble(), message_wave])

def decode_waveform_to_text(waveform, tolerance=20):
    """
    Detects the preamble tone to figure out where the message actually starts,
    then decodes from there assuming perfect alignment.
    """

    def detect_freq(chunk):
        """
        Find the loudest frequency in this chunk using FFT.
        The whole point is to figure out which tone was most dominant.
        """
        windowed = np.hanning(len(chunk)) * chunk 
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(chunk), 1 / SAMPLE_RATE)
        return freqs[np.argmax(np.abs(fft_vals))]

    def match_freq_to_char(freq):
        """
        Map a frequency to the closest known character.
        If it's too far off, we just call it '�' and move on.
        """
        closest = min(freq_to_char.keys(), key=lambda f: abs(f - freq))
        return freq_to_char[closest] if abs(freq - closest) < tolerance else "�"

    def decode_aligned(signal):
        """
        Break the signal into tone-sized chunks, detect the frequency of each,
        and translate those into characters.
        Assumes we're starting cleanly at the first character.
        """
        chunks = [
            signal[i : i + _samples_per_tone]
            for i in range(0, len(signal), _samples_per_tone)
            if np.any(signal[i : i + _samples_per_tone])  # skip empty/silent blocks
        ]
        freqs = [detect_freq(c) for c in chunks]
        return ''.join(match_freq_to_char(f) for f in freqs)

    # === Step 1: Find the preamble ===
    # This is just a short beep we play before the real message
    # so the decoder knows where the message starts (like a sync marker).
    preamble_samples = int(SAMPLE_RATE * PREAMBLE_DURATION)
    search_limit = min(len(waveform), 5 * SAMPLE_RATE)  # search first few seconds max
    preamble_found_at = None
    step = preamble_samples // 2 if preamble_samples // 2 > 0 else 1

    for start_idx in range(0, search_limit - preamble_samples, step):
        chunk = waveform[start_idx : start_idx + preamble_samples]
        freq_detected = detect_freq(chunk)
        if abs(freq_detected - PREAMBLE_FREQ) < tolerance:
            preamble_found_at = start_idx
            break

    if preamble_found_at is None:
        # didn't hear the preamble — maybe noisy? maybe it never got played?
        return ""

    # === Step 2: Decode everything after the preamble ===
    message_start = preamble_found_at + preamble_samples
    message_signal = waveform[message_start:]
    return decode_aligned(message_signal)