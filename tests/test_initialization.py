import numpy as np
import pytest

from room_eq.peak_ablation import candidate_starts


@pytest.mark.parametrize("has_peaks", [True, False])
def test_candidate_starts_preserve_bounds_spacing_and_controls(has_peaks):
    config = {
        "filter_count": 6, "band_hz": [50, 500], "gain_bounds_db": [-12, 6],
        "q_bounds": [0.3, 10], "initial_gain_db": -1, "initial_q": 1,
    }
    frequencies = np.geomspace(*config["band_hz"], 128)
    data = np.full((3, 128), -2.0)
    if has_peaks:
        data[:, [15, 17, 40]] = [12, 4, 6]
    starts = list(candidate_starts(data, frequencies, config))
    _, _, peak, detected = starts[0]
    assert detected == (2 if has_peaks else 0)
    if has_peaks:
        np.testing.assert_allclose(np.exp(peak[1:6:3]), frequencies[[15, 40]])
    for name, _, parameters, _ in starts:
        gain, log_frequency, log_q = parameters.reshape(-1, 3).T
        assert np.all((gain >= -12) & (gain < 0))
        assert np.all(np.exp(log_q) == 1)
        assert np.all(np.diff(np.sort(log_frequency)) >= np.log(2) / 6 - 1e-14)
        if name == "random_centers":
            np.testing.assert_array_equal(gain, peak[::3])
