import numpy as np

# Constants
SAMPLE_RATE = 44100
TONE_DURATION = 0.2  # Each tone is a fifth of a second
AMPLITUDE = 0.5      # Volume factor

# Characters we support!
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"
BASE_FREQ = 600      # Hz -> where we start the frequency map
STEP_SIZE = 60       # Hz offset between characters

# Frequency map: char -> freq and freq -> char
char_to_freq = {char: BASE_FREQ + i * STEP_SIZE for i, char in enumerate(CHARSET)}
freq_to_char = {v: k for k, v in char_to_freq.items()}

# Pre- & post- amble tones: special "beeps" to mark the start and end
PREAMBLE_FREQ = 400
PREAMBLE_DURATION = 0.3  # Short burst to help us sync
POSTAMBLE_FREQ = 3000
POSTAMBLE_DURATION = 0.3

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

def generate_postamble():
    """Short beep after the message to signal the end."""
    postamble_samples = int(SAMPLE_RATE * POSTAMBLE_DURATION)
    t = np.linspace(0, POSTAMBLE_DURATION, postamble_samples, endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * POSTAMBLE_FREQ * t)

def compute_checksum_char(text):
    """Compute a simple checksum: sum of ASCII values mod len(CHARSET)"""
    s = sum(ord(c) for c in text if c in CHARSET)
    return CHARSET[s % len(CHARSET)]

def encode_text_to_waveform(text):
    """
    Attach preamble + message tones + (optional) checksum tone + postamble.

    - Filters out any characters not in CHARSET/tonebank.
    - If we have zero valid chars, skip checksum and just send preamble + short silence.
    """
    text = text.upper()
    valid_chars = [ch for ch in text if ch in tonebank]

    # If no recognized characters, send just preamble + silence
    if len(valid_chars) == 0:
        return np.concatenate([generate_preamble(), np.zeros_like(_time)])

    # Otherwise, compute checksum and build the message wave
    message_str = ''.join(valid_chars)
    checksum_char = compute_checksum_char(message_str)

    message_tones = [tonebank[ch] for ch in valid_chars]
    message_tones.append(tonebank[checksum_char])  # final chunk is your checksum
    message_wave = np.concatenate(message_tones)

    # Append postamble
    postamble_wave = generate_postamble()

    return np.concatenate([generate_preamble(), message_wave, postamble_wave])

    return np.concatenate([generate_preamble(), message_wave, postamble_wave])
def decode_waveform_to_text(waveform, tolerance=20, entropy_threshold=5.0):
    """
    Decode audio waveform back to text with principled error awareness:
    - Finds the preamble tone to synchronize to the message start.
    - Splits the post-preamble signal into fixed-size tone chunks.
    - Detects frequency and entropy for each chunk.
    - Tracks which chunks are high-entropy (i.e. uncertain), but still attempts to decode them.
    - Appends the indexes of uncertain chunks at the end of the message.
    - Uses the final decoded character as a checksum and flags mismatches.
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
        """
        Decode the signal assuming alignment. For each chunk, detect both frequency and entropy.
        If we detect the postamble tone, we stop decoding further chunks.
        """
        chunks = [
            signal[i : i + _samples_per_tone]
            for i in range(0, len(signal), _samples_per_tone)
            if np.any(signal[i : i + _samples_per_tone])  # skip truly empty blocks
        ]

        decoded_info = []
        
        for chunk in chunks:
            ent = chunk_entropy(chunk)
            freq = detect_freq(chunk)
            
            # If chunk matches postamble freq, we stop decoding right away
            if abs(freq - POSTAMBLE_FREQ) < tolerance:
                break
            
            # Otherwise proceed with normal char detection
            char = match_freq_to_char(freq)
            uncertain = (ent > entropy_threshold)
            decoded_info.append((char, uncertain))

        # If fewer than 2 characters => no checksum
        if len(decoded_info) < 2:
            message_chars = [char for char, _ in decoded_info]
            message_string = ''.join(message_chars)

            uncertain_char_positions = [
                i + 1 for i, (char, uncertain) in enumerate(decoded_info)
                if uncertain and char
            ]
            if uncertain_char_positions:
                friendly_positions = [f"#{i}" for i in uncertain_char_positions]
                message_string += f" [UNCERTAIN characters at positions {', '.join(friendly_positions)}]"
            return message_string

        # Separate message from checksum
        *message_info, checksum_info = decoded_info
        message_chars = [char for char, _ in message_info]
        message_string = ''.join(message_chars)

        # Checksum validation
        message_for_checksum = ''.join(c for c in message_chars if c)
        expected_cs = compute_checksum_char(message_for_checksum)
        actual_cs = checksum_info[0]

        result = message_string
        if actual_cs != expected_cs:
            result += " [CHECKSUM MISMATCH]"

        # Report uncertainty by character position (1-based)
        uncertain_positions = [
            i + 1 for i, (char, uncertain) in enumerate(message_info)
            if uncertain and char
        ]
        if uncertain_positions:
            friendly_positions = [f"#{i}" for i in uncertain_positions]
            result += f" [UNCERTAIN characters at positions {', '.join(friendly_positions)}]"

        return result

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