import numpy as np
import pytest

from room_eq.data import ARCHIVE, IR_LENGTH, MeshRIR
from room_eq.preprocess import log_spectra
from room_eq.smoke import select_receivers


def test_impulse_spectrum_is_flat():
    impulse = np.zeros((1, 1024))
    impulse[0, 127] = 1
    frequencies = np.geomspace(50, 500, 16)
    np.testing.assert_allclose(log_spectra(impulse, 48000, frequencies, 1024, 1e-12), 0, atol=1e-12)


def test_receiver_split_is_disjoint_and_deterministic():
    receivers = np.array([[x, y, z] for z in (0, 0.05, 0.1) for x in range(3) for y in range(3)])
    train, test, height = select_receivers(receivers, 4, 5, 5403)
    repeat_train, repeat_test, _ = select_receivers(receivers, 4, 5, 5403)
    assert height == 0.05
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(9, 18))
    np.testing.assert_array_equal(train, repeat_train)
    np.testing.assert_array_equal(test, repeat_test)


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="MeshRIR archive is not installed")
def test_measured_mesh_rir_loads():
    dataset = MeshRIR(ARCHIVE)
    train, test, height = select_receivers(dataset.receivers, 9, 12, 5403)
    ids = np.concatenate((train, test))
    np.testing.assert_allclose(dataset.receivers[ids, 2], height)
    assert dataset.load(ids).shape == (21, IR_LENGTH)
