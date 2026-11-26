"""Convert measured impulse responses to a common log-frequency grid."""

import numpy as np


def log_spectra(impulses, sample_rate, frequencies, fft_length, magnitude_floor):
    """Evaluate full-length rectangular-window spectra without smoothing."""
    if fft_length < impulses.shape[-1]:
        raise ValueError("FFT length would truncate the measured impulse responses")
    fft_hz = np.fft.rfftfreq(fft_length, 1 / sample_rate)
    magnitudes = np.abs(np.fft.rfft(impulses, n=fft_length, axis=-1))
    db = 20 * np.log10(np.maximum(magnitudes, magnitude_floor))
    return np.stack([np.interp(frequencies, fft_hz, row) for row in db])
