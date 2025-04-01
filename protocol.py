import numpy as np

# Constants
SAMPLE_RATE = 44100
TONE_DURATION = 0.2  # Each tone is a fifth of a second
AMPLITUDE = 0.5      # Volume factor, keep it chill

# Characters we support!
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"
BASE_FREQ = 600      # Hz -> where we start the frequency map
STEP_SIZE = 60       # Hz offset between characters

# Frequency map: char -> freq and freq -> char
char_to_freq = {char: BASE_FREQ + i * STEP_SIZE for i, char in enumerate(CHARSET)}
freq_to_char = {v: k for k, v in char_to_freq.items()}

# Preamble tone: special "beep" to mark the start
PREAMBLE_FREQ = 300
PREAMBLE_DURATION = 0.3  # Short burst to help us sync

# Tonebank: cache sine waves for each character
_samples_per_tone = int(SAMPLE_RATE * TONE_DURATION)
_time = np.linspace(0, TONE_DURATION, _samples_per_tone, endpoint=False)
tonebank = {
    char: AMPLITUDE * np.sin(2 * np.pi * freq * _time)
    for char, freq in char_to_freq.items()
}

def generate_preamble():
    """Short beep before the message to help the receiver sync up."""
    preamble_samples = int(SAMPLE_RATE * PREAMBLE_DURATION)
    t = np.linspace(0, PREAMBLE_DURATION, preamble_samples, endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * PREAMBLE_FREQ * t)

def compute_checksum_char(text):
    """Compute a simple checksum: sum of ASCII values mod len(CHARSET)"""
    s = sum(ord(c) for c in text if c in CHARSET)
    return CHARSET[s % len(CHARSET)]

def encode_text_to_waveform(text):
    """Attach preamble + message tones + checksum tone."""
    text = text.upper()
    checksum_char = compute_checksum_char(text)

    message_tones = [tonebank[char] for char in text if char in tonebank]
    message_tones.append(tonebank[checksum_char])

    if not message_tones:
        return np.concatenate([generate_preamble(), np.zeros_like(_time)])

    message_wave = np.concatenate(message_tones)
    return np.concatenate([generate_preamble(), message_wave])

def decode_waveform_to_text(waveform, tolerance=20, entropy_threshold=5.0):
    """
    Decode audio waveform back to text with error checking:
    - Finds the preamble to sync.
    - Splits into tone chunks.
    - Rejects high-entropy (uncertain) chunks.
    - Checks the final character against a simple checksum.
    """
    def detect_freq(chunk):
        windowed = np.hanning(len(chunk)) * chunk
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(chunk), 1 / SAMPLE_RATE)
        return freqs[np.argmax(np.abs(fft_vals))]

    def chunk_entropy(chunk):
        windowed = np.hanning(len(chunk)) * chunk
        fft_vals = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_vals)
        total = np.sum(magnitudes)
        if total == 0:
            return float('inf')  # fully silent => treat as "infinite" uncertainty
        p = magnitudes / total
        return -np.sum(p * np.log2(p + 1e-12))

    def match_freq_to_char(freq):
        closest = min(freq_to_char.keys(), key=lambda f: abs(f - freq))
        return freq_to_char[closest] if abs(freq - closest) < tolerance else ""

    def decode_aligned(signal):
        # Split into tone-sized chunks, skipping empty ones
        chunks = [
            signal[i : i + _samples_per_tone]
            for i in range(0, len(signal), _samples_per_tone)
            if np.any(signal[i : i + _samples_per_tone])
        ]

        decoded_chars = []
        for chunk in chunks:
            ent = chunk_entropy(chunk)
            if ent > entropy_threshold:
                decoded_chars.append("")  # high entropy => uncertain => skip
            else:
                freq = detect_freq(chunk)
                decoded_chars.append(match_freq_to_char(freq))

        # If there's not at least one char + checksum, just return what we have
        if len(decoded_chars) < 2:
            return ''.join(decoded_chars)

        # Separate the checksum char from the actual message
        *message, checksum = decoded_chars
        expected = compute_checksum_char(''.join(message))

        if checksum != expected:
            return ''.join(message) + " [CHECKSUM MISMATCH]"
        else:
            return ''.join(message)

    # Search for preamble in the first ~5 seconds
    preamble_samples = int(SAMPLE_RATE * PREAMBLE_DURATION)
    search_limit = min(len(waveform), 5 * SAMPLE_RATE)
    preamble_found_at = None
    step = preamble_samples // 2 if preamble_samples // 2 > 0 else 1

    # Slide through in half-preamble steps to find the beep
    for start_idx in range(0, search_limit - preamble_samples, step):
        chunk = waveform[start_idx : start_idx + preamble_samples]
        freq_detected = detect_freq(chunk)
        if abs(freq_detected - PREAMBLE_FREQ) < tolerance:
            preamble_found_at = start_idx
            break

    # If we never found the preamble, decode as empty
    if preamble_found_at is None:
        return ""

    # Decode the portion after the preamble
    message_start = preamble_found_at + preamble_samples
    message_signal = waveform[message_start:]
    return decode_aligned(message_signal)