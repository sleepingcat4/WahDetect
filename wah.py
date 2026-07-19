import librosa
import numpy as np
from scipy.signal import correlate, find_peaks

def detect_wah(audio_file, template_file, threshold=0.7):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    template, _ = librosa.load(template_file, sr=16000, mono=True)

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    template = template / (np.max(np.abs(template)) + 1e-8)

    corr = correlate(audio, template, mode="valid")
    corr = corr / (np.max(corr) + 1e-8)

    peaks, _ = find_peaks(
        corr,
        height=threshold,
        distance=sr
    )

    times = peaks / sr

    return len(times)
