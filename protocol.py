import numpy as np

# Constants
SAMPLE_RATE = 44100
TONE_DURATION = 0.5  # I'm making each tone half a second
AMPLITUDE = 0.5      # Volume factor, let's keep it modest

# Characters we support!
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"
BASE_FREQ = 600   # Hz -> where we start the frequency map
STEP_SIZE = 40    # Hz offset between subsequent chars

# Building a freq map: each char -> freq, freq -> char
char_to_freq = {char: BASE_FREQ + i * STEP_SIZE for i, char in enumerate(CHARSET)}
freq_to_char = {v: k for k, v in char_to_freq.items()}

# Tonebank (yay!). Precompute sine waves for each char
_samples_per_tone = int(SAMPLE_RATE * TONE_DURATION)
_time = np.linspace(0, TONE_DURATION, _samples_per_tone, endpoint=False)  # time axis for one tone
tonebank = {
    char: AMPLITUDE * np.sin(2 * np.pi * freq * _time)  # basic sine wave formula
    for char, freq in char_to_freq.items()
}

def encode_text_to_waveform(text):
    """
    Takes our text, finds each char's wave from the tonebank,
    and stitches it all together into one glorious waveform.
    If there's no recognized char, we skip it. 
    """
    # Upper-casing to match the keys (since we built everything uppercase)
    tones = [tonebank[char] for char in text.upper() if char in tonebank]
    # If no valid tones, return an empty array instead of failing
    return np.concatenate(tones) if tones else np.zeros_like(_time)

def decode_waveform_to_text(waveform, tolerance=20):
    """
    Attempt to decode the waveform back into text via sliding windows.
    We look for the offset that yields the smallest average entropy
    (meaning the signal is super-peaked around one freq per chunk).
    If we find no good slice, we return an empty string. 
    """

    def chunk_entropy(chunk):
        """
        Convert chunk -> windowed chunk -> FFT -> magnitude array -> probability distribution.
        Then compute Shannon entropy (base-2), i.e. how "spread out" is the power. 
        Lower entropy => more concentration in one frequency bin => good news.
        """
        windowed = np.hanning(len(chunk)) * chunk  # smooth edges for less spectral leakage
        fft_vals = np.fft.rfft(windowed)          # real FFT -> freq domain
        magnitudes = np.abs(fft_vals)             # absolute value -> we only want magnitudes

        total = np.sum(magnitudes)
        if total == 0:
            # If chunk is silent, let's just say no entropy here
            return 0

        # Probability distribution: each freq bin's magnitude is a "prob"
        p = magnitudes / total
        # Shannon entropy => sum(p*log2(p)), negative sign
        return -np.sum(p * np.log2(p + 1e-12))  # tiny offset to avoid log(0)

    def detect_freq(chunk):
        """
        Find the strongest frequency in a chunk by picking the index of
        the largest magnitude in the FFT result.
        """
        windowed = np.hanning(len(chunk)) * chunk
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(chunk), 1 / SAMPLE_RATE)
        return freqs[np.argmax(np.abs(fft_vals))]  # freq with the biggest amplitude

    def match_freq_to_char(freq):
        """
        We look for the closest known freq. If it's within tolerance, we take it;
        otherwise we call it '�' to indicate we couldn't match. 
        """
        closest = min(freq_to_char.keys(), key=lambda f: abs(f - freq))
        return freq_to_char[closest] if abs(freq - closest) < tolerance else "�"

    def decode_aligned(signal):
        """
        Decodes a signal under the assumption that it starts exactly at a chunk boundary.
        If it doesn't, we might get gibberish, which is why we do the sliding window approach
        to find the correct alignment in the first place.
        """
        chunks = [
            signal[i : i + _samples_per_tone]
            for i in range(0, len(signal), _samples_per_tone)
            if np.any(signal[i : i + _samples_per_tone])  # skip all-silent blocks
        ]
        freqs = [detect_freq(c) for c in chunks]
        return ''.join(match_freq_to_char(f) for f in freqs)

    # We'll track the best slice based on minimal average entropy
    best_slice = None
    best_average_entropy = float('inf')

    # Slide from offset=0 to offset=_samples_per_tone-1
    # and see which offset yields the biggest "confidence" (lowest entropy).
    for offset in range(_samples_per_tone):
        sliced = waveform[offset:]
        chunks = [
            sliced[i : i + _samples_per_tone]
            for i in range(0, len(sliced) - _samples_per_tone, _samples_per_tone)
            if np.any(sliced[i : i + _samples_per_tone])
        ]
        if not chunks:
            continue  # If we didn't get any chunks, just ignore this offset

        # Compute average chunk entropy for this offset
        avg_ent = np.mean([chunk_entropy(chunk) for chunk in chunks])

        # Lower is better => more "peaked" => more obviously one freq
        if avg_ent < best_average_entropy:
            best_average_entropy = avg_ent
            best_slice = sliced

    # If we found a best slice, decode it normally; else return ""
    return decode_aligned(best_slice) if best_slice is not None else ""