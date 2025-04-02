import numpy as np

# Constants
SAMPLE_RATE = 44100
TONE_DURATION = 0.33  # Each tone is a third of a second
AMPLITUDE = 0.5      # Volume factor

# Characters we support!
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"
BASE_FREQ = 500      # Hz -> where we start the frequency map
STEP_SIZE = 40       # Hz offset between characters

# Frequency map: char -> freq and freq -> char
char_to_freq = {char: BASE_FREQ + i * STEP_SIZE for i, char in enumerate(CHARSET)}
freq_to_char = {v: k for k, v in char_to_freq.items()}

# Pre- & post- amble tones: special "beeps" to mark the start and end
PREAMBLE_FREQ = 2250
PREAMBLE_DURATION = 0.5
POSTAMBLE_FREQ = 2500
POSTAMBLE_DURATION = 0.5

# Tonebank: cache sine waves for each character
_samples_per_tone = int(SAMPLE_RATE * TONE_DURATION)
_time = np.linspace(0, TONE_DURATION, _samples_per_tone, endpoint=False)
tonebank = {
    char: AMPLITUDE * np.sin(2 * np.pi * freq * _time)
    for char, freq in char_to_freq.items()
}

def set_transmission_params(amplitude, fade_time=0.01):
    """Update amplitude and apply fade-in/out to each tone."""
    global AMPLITUDE, tonebank
    AMPLITUDE = amplitude

    fade_samples = int(fade_time * SAMPLE_RATE)
    envelope = np.ones_like(_time)

    if fade_samples > 0 and 2 * fade_samples < len(_time):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        envelope[:fade_samples] = fade_in
        envelope[-fade_samples:] = fade_out
    # else: keep as ones (no fading)

    tonebank = {
        char: AMPLITUDE * np.sin(2 * np.pi * freq * _time) * envelope
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

class IncrementalDecoder:
    """
    A stateful, incremental decoder that processes new audio data chunk by chunk.
    It:
      - Accumulates samples in a buffer.
      - Searches for the preamble tone to synchronize.
      - Decodes full tone chunks incrementally (using _samples_per_tone).
      - Stops decoding when the postamble tone is detected.
      - Stores decoded characters (with uncertainty flags) for checksum validation.
    """
    def __init__(self, tolerance=20, entropy_threshold=10.0):
        self.tolerance = tolerance
        self.entropy_threshold = entropy_threshold
        self.buffer = np.array([], dtype=np.float32)
        self.last_decoded_index = 0  # Pointer into the buffer (in samples)
        self.preamble_found = False
        self.done = False
        self.decoded_info = []  # List of (character, uncertain flag)
    
    def feed_samples(self, new_samples):
        """Append new_samples to the internal buffer and process new complete chunks."""
        self.buffer = np.concatenate((self.buffer, new_samples))
        # Process complete chunks only from last_decoded_index to end.
        while self.last_decoded_index + _samples_per_tone <= len(self.buffer):
            chunk = self.buffer[self.last_decoded_index : self.last_decoded_index + _samples_per_tone]
            # If preamble not yet found, look for it.
            if not self.preamble_found:
                freq = self.detect_freq(chunk)
                if abs(freq - PREAMBLE_FREQ) < self.tolerance:
                    self.preamble_found = True
                self.last_decoded_index += _samples_per_tone
                continue
            
            # If preamble found, check for postamble.
            freq = self.detect_freq(chunk)
            if abs(freq - POSTAMBLE_FREQ) < self.tolerance:
                self.done = True
                self.last_decoded_index += _samples_per_tone
                break  # Stop processing further chunks
            
            # Otherwise, decode this chunk as a character.
            char = self.match_freq_to_char(freq)
            uncertain = (self.chunk_entropy(chunk) > self.entropy_threshold)
            self.decoded_info.append((char, uncertain))
            self.last_decoded_index += _samples_per_tone
    
    def detect_freq(self, chunk):
        """Detects the dominant frequency in a chunk using FFT and a Hanning window."""
        windowed = np.hanning(len(chunk)) * chunk
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(chunk), 1/SAMPLE_RATE)
        return freqs[np.argmax(np.abs(fft_vals))]
    
    def chunk_entropy(self, chunk):
        """Calculates Shannon entropy of the FFT magnitudes of a chunk."""
        windowed = np.hanning(len(chunk)) * chunk
        fft_vals = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_vals)
        total = np.sum(magnitudes)
        if total == 0:
            return float('inf')
        p = magnitudes / total
        return -np.sum(p * np.log2(p + 1e-12))
    
    def match_freq_to_char(self, freq):
        """Maps a detected frequency to the closest valid character."""
        closest = min(freq_to_char.keys(), key=lambda f: abs(f - freq))
        return freq_to_char[closest] if abs(freq - closest) < self.tolerance else ""
    
    def get_message(self):
        """
        Returns the decoded message based on the processed chunks.
        The last decoded character is treated as a checksum.
        Uncertain characters (flagged during decoding) are reported by their
        1-based positions in the final message.
        """
        if not self.decoded_info:
            return ""
        
        # If fewer than 2 characters decoded, no checksum is available.
        if len(self.decoded_info) < 2:
            msg = "".join(char for char, _ in self.decoded_info)
            uncertain_positions = [i+1 for i, (char, uncertain) in enumerate(self.decoded_info) if uncertain and char]
            if uncertain_positions:
                friendly = [f"#{pos}" for pos in uncertain_positions]
                msg += f" [UNCERTAIN characters at positions {', '.join(friendly)}]"
            return msg
        
        # Otherwise, assume the final chunk is the checksum.
        *message_info, checksum_info = self.decoded_info
        message_chars = [char for char, _ in message_info]
        message_str = "".join(message_chars)
        message_for_cs = "".join(c for c in message_chars if c)
        expected_cs = compute_checksum_char(message_for_cs)
        actual_cs = checksum_info[0]
        
        result = message_str
        if actual_cs != expected_cs:
            result += " [CHECKSUM MISMATCH]"
        
        uncertain_positions = [i+1 for i, (char, uncertain) in enumerate(message_info) if uncertain and char]
        if uncertain_positions:
            friendly = [f"#{pos}" for pos in uncertain_positions]
            result += f" [UNCERTAIN characters at positions {', '.join(friendly)}]"
        
        return result

    def is_done(self):
        """Returns True if the postamble has been detected and decoding is complete."""
        return self.done