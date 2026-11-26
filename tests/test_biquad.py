import numpy as np
from scipy.signal import sosfreqz

from room_eq.biquad import peaking_sos, response_db


def test_peaking_cascade_matches_scipy():
    sample_rate = 48000
    frequencies = np.geomspace(50, 500, 128)
    parameters = np.array([[-12, np.log(50), np.log(10)], [6, np.log(500), np.log(0.3)]])
    sos = peaking_sos(parameters, sample_rate)
    _, reference = sosfreqz(sos, worN=frequencies, fs=sample_rate)
    np.testing.assert_allclose(
        response_db(parameters, frequencies, sample_rate),
        20 * np.log10(np.abs(reference)),
        atol=1e-8,
    )
    assert all(np.all(np.abs(np.roots(section[3:])) < 1) for section in sos)


def test_zero_gain_is_unity():
    parameters = [[0, np.log(150), np.log(1.5)]]
    frequencies = np.geomspace(50, 500, 128)
    np.testing.assert_array_equal(response_db(parameters, frequencies, 48000), 0)
