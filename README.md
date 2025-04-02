# Acoustic Modem Project

**Overview**  
This repository contains three Python files – **`protocol.py`**, **`sender.py`**, and **`listener.py`** – that work together to encode text as audio tones, send those tones through speakers, and decode them via a microphone. I designed this as a mini acoustic communication system that demonstrates signal processing, real-time streaming, and a user-friendly GUI.

---

## Project Requirements

1. **`protocol.py`**  
   - Defines how characters map to frequencies.  
   - Generates waveforms (including a preamble and a postamble).  
   - Decodes waveforms back into text.  
   - Includes a **checksum** for basic error detection.  
   - Implements an **IncrementalDecoder** for real-time (streaming) reception.

2. **`sender.py`**  
   - A **Tkinter GUI** for transmitting messages.  
   - Lets the user enter text, adjust amplitude, add optional fade-in/out, then plays the audio.  
   - Displays status updates when sending is in progress or done.

3. **`listener.py`**  
   - Another **Tkinter GUI** that starts an audio stream from the mic.  
   - Decodes partial messages in real time using a queue-based, ring-buffer approach.  
   - Shows the final message once a postamble is detected.  
   - Logs recent messages and explains uncertain characters and checksum mismatches.

---

## How It Works

### **Encoding**

1. **Character Set**  
  We support the characters in `CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!"`.  
  Each character maps to a frequency:  
  The frequency of a character equals the base frequency plus the character's position in our set multiplied by the step size.
2. **Preamble & Postamble**  
  A special tone (above or below normal character frequencies) signals the start (preamble) and end (postamble).
3. **Checksum for Error Detection**  
  Each message ends with a checksum character computed as:

    $\text{checksum} = \text{CHARSET}\left[\left(\sum_{i} \text{ord}(c_i)\right) \bmod |\text{CHARSET}|\right]$

    Where $\text{ord}(c_i)$ is the ASCII value of the *i*-th character and $|\texttt{CHARSET}|$ is the number of supported characters (43 in this case)


    This simple modulo-based approach allows the receiver to detect (but not correct) most single or multi-character errors.  
    It's lightweight, intuitive, and works well for short, human-readable transmissions.
  
4. **Fade-In/Out**  
  Optionally applies a short fade at the start/end of each tone to reduce sharp clicks or unwanted harmonics.

---

### **Decoding**

1. **Ring Buffer**  
   In `listener.py`, we accumulate mic samples in a ring buffer. The `IncrementalDecoder` processes chunks of `_samples_per_tone` each time and maps them back to characters. This is a technique I adapted from my *Data Structures* coursework, allowing the class to maintain state across time and decode one tone-sized chunk at a time, without reprocessing previously analyzed samples. It keeps latency low and supports true real-time decoding.

3. **Entropy-Based Noise Detection**  
  I used Shannon entropy — a concept from my *Probability for Machine Learning* class — to evaluate how concentrated the energy of each audio chunk is in the frequency domain.

    Given the normalized FFT magnitudes \( p_i \) of a chunk, entropy is computed as:
  
    $$H = -\sum_i p_i \log_2(p_i + \epsilon)$$
  
    A clean tone will have a dominant peak (low entropy), while a noisy chunk will exhibit a flatter spectrum (higher entropy).  
    If a chunk’s distribution is too spread out, we flag that character as **uncertain**.

4. **FFT + Frequency Matching in Real-Time**  
  Each audio chunk is windowed with a Hanning function and passed through a real FFT to extract its frequency spectrum.  
  The peak frequency is then mapped to the closest valid tone in our `CHARSET` frequency map. A configurable `tolerance` threshold ensures we only accept matches that are sufficiently close, which helps filter out noise and harmonics.
  We display partial decodes in the GUI as we go. Once the postamble frequency is detected, we finalize the message, check the checksum, and display the result.

6. **UI Explanation**  
   The **Sender GUI** lets users input a message and tweak amplitude/fade before sending.  
   The **Listener GUI** displays partial decodes live, explains uncertainty and checksum failures in plain English, and keeps a history of recent messages.

---

## Known Limitations & Future Ideas

- **Fixed Tone Duration**  
We pinned `_samples_per_tone` to `0.33s`. Changing it mid-run (e.g., a different setting on the listener) can cause misalignment while chunking.

- **No Auto Parameter Sync**  
Base frequency, step size, or tone duration changes on the sender require the listener to match. Otherwise, frequencies won’t decode properly. (Hence why there's no option to adjust these parameters in the GUI)

- **Basic Checksum**  
No advanced error correction is used, just a detection approach. Could add Hamming or Reed-Solomon codes for more robust correction.

- **High Frequencies**  
Some default values (preamble/postamble near 2250–2500 Hz and lower character values at 500-600 Hz) may be hard to detect on certain speakers/mics.

---

## Conclusion

This project demonstrates how to:
1. **Encode** text as sine waves with a preamble, postamble, and checksum.
2. **Capture & decode** partial waveforms in real time using a ring buffer.
3. Provide **user-friendly GUIs** for both sending and listening.

Hope you enjoy exploring the code as much as I did creating it!
