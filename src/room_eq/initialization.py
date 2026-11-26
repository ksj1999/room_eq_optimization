"""Filter bounds and reproducible starting points."""

import numpy as np
from scipy.signal import find_peaks


def filter_setup(config):
    count = config["filter_count"]
    f_min, f_max = config["band_hz"]
    gain_min, gain_max = config["gain_bounds_db"]
    q_min, q_max = config["q_bounds"]
    lower = np.tile([gain_min, np.log(f_min), np.log(q_min)], count)
    upper = np.tile([gain_max, np.log(f_max), np.log(q_max)], count)
    centers = np.geomspace(f_min, f_max, count + 2)[1:-1]
    initial = np.column_stack((
        np.full(count, config["initial_gain_db"]),
        np.log(centers),
        np.full(count, np.log(config["initial_q"])),
    )).ravel()
    return lower, upper, initial


def seeded_starts(config):
    """Use a near-flat start followed by reproducible perturbed starts."""
    lower, upper, initial = filter_setup(config)
    yield "near_flat", initial
    for seed in config["start_seeds"]:
        rng = np.random.default_rng(seed)
        offsets = rng.normal(size=initial.size) * np.tile([2, 0.15, 0.4], initial.size // 3)
        yield f"seed_{seed}", np.clip(initial + offsets, lower, upper)


def peak_informed_start(responses, frequencies, config):
    """Place cuts at separated positive peaks of the median training response."""
    lower, upper, initial = filter_setup(config)
    log_hz = np.log(frequencies)
    median_db = np.median(responses, axis=0)
    peaks, _ = find_peaks(median_db, height=0)
    candidates = peaks[np.argsort(-median_db[peaks], kind="stable")]
    spacing = np.log(2) / 6
    count = config["filter_count"]
    selected = []
    for index in candidates:
        if len(selected) == count:
            break
        if all(abs(log_hz[index] - log_hz[other]) >= spacing for other in selected):
            selected.append(index)
    peak_count = len(selected)
    # Fill unused sections in the gaps left by the detected peaks.
    while len(selected) < count:
        anchors = [log_hz[0], log_hz[-1], *log_hz[selected]]
        distance = np.abs(log_hz[:, None] - anchors).min(axis=1)
        selected.append(int(distance.argmax()))
    filters = initial.reshape(-1, 3)
    filters[:, 1] = log_hz[selected]
    filters[:peak_count, 0] = -np.clip(median_db[selected[:peak_count]] / 2, 1, 5)
    return np.clip(initial, lower, upper), peak_count
