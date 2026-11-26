"""RBJ peaking equalizers with gain, log frequency and log Q parameters."""

import numpy as np


def peaking_sos(parameters, sample_rate):
    """Return one normalized SOS row per peaking filter."""
    gain, log_frequency, log_q = np.asarray(parameters).reshape(-1, 3).T
    amplitude = 10 ** (gain / 40)
    angle = 2 * np.pi * np.exp(log_frequency) / sample_rate
    alpha = np.sin(angle) / (2 * np.exp(log_q))
    middle = -2 * np.cos(angle)
    ones = np.ones_like(gain)
    sos = np.column_stack((
        1 + alpha * amplitude, middle, 1 - alpha * amplitude,
        ones + alpha / amplitude, middle, 1 - alpha / amplitude,
    ))
    return sos / sos[:, 3, None]


def response_db(parameters, frequencies, sample_rate):
    """Evaluate the cascade by summing the biquad responses in dB."""
    sos = peaking_sos(parameters, sample_rate)
    z = np.exp(-2j * np.pi * np.asarray(frequencies) / sample_rate)
    powers = np.stack((np.ones_like(z), z, z**2))
    return 20 * np.sum(
        np.log10(np.abs(sos[:, :3] @ powers))
        - np.log10(np.abs(sos[:, 3:] @ powers)), axis=0,
    )


def response_and_jacobian(parameters, frequencies, sample_rate):
    """Return the dB cascade and its frequency-by-parameter Jacobian."""
    gain, log_frequency, log_q = np.asarray(parameters).reshape(-1, 3).T[:, :, None]
    center = 2 * np.pi * np.exp(log_frequency) / sample_rate
    omega = 2 * np.pi * np.asarray(frequencies)[None, :] / sample_rate
    amplitude = 10 ** (gain / 40)
    alpha = np.sin(center) / (2 * np.exp(log_q))
    # Factor 2*z out of each polynomial. The product of sines evaluates
    # cos(omega) - cos(center) accurately near DC and the filter center.
    real = -2 * np.sin((omega + center) / 2) * np.sin((omega - center) / 2)
    numerator_imag = 1j * np.sin(omega) * alpha * amplitude
    denominator_imag = 1j * np.sin(omega) * alpha / amplitude
    numerator = real + numerator_imag
    denominator = real + denominator_imag
    scale = 20 / np.log(10)
    response = scale * np.sum(np.log(np.abs(numerator)) - np.log(np.abs(denominator)), axis=0)
    d_real = center * np.sin(center)
    d_alpha_ratio = center * np.cos(center) / np.sin(center)
    numerator_ratio = numerator_imag / numerator
    denominator_ratio = denominator_imag / denominator
    d_gain = 0.5 * (numerator_ratio + denominator_ratio).real
    d_frequency = scale * (
        d_real * (1 / numerator - 1 / denominator)
        + d_alpha_ratio * (numerator_ratio - denominator_ratio)
    ).real
    d_q = scale * (denominator_ratio - numerator_ratio).real
    jacobian = np.stack((d_gain, d_frequency, d_q), axis=-1)
    return response, jacobian.transpose(1, 0, 2).reshape(len(frequencies), -1)
