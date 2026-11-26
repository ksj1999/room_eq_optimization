import numpy as np
import pytest

from room_eq.biquad import response_and_jacobian, response_db
from room_eq.experiments import spatial_splits
from room_eq.objectives import SeatObjective
from room_eq.solvers import generalized_gauss_newton, lbfgsb, projected_gradient_norm


FREQUENCIES = np.geomspace(50, 500, 80)
PARAMETERS = np.array([-7, np.log(80), np.log(3), 2, np.log(285), np.log(0.7)])


def central_difference(function, x):
    steps = 1e-5 * np.eye(x.size)
    return np.stack([(function(x + h) - function(x - h)) / 2e-5 for h in steps], axis=-1)


def test_biquad_jacobian():
    response, jacobian = response_and_jacobian(PARAMETERS, FREQUENCIES, 48000)
    np.testing.assert_allclose(response, response_db(PARAMETERS, FREQUENCIES, 48000), atol=1e-8)
    numerical = central_difference(lambda x: response_and_jacobian(x, FREQUENCIES, 48000)[0], PARAMETERS)
    np.testing.assert_allclose(jacobian, numerical, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_objective_gradient_and_curvature(alpha):
    responses = np.random.default_rng(5403).normal(0, 4, size=(9, len(FREQUENCIES)))
    objective = SeatObjective(responses, FREQUENCIES, 48000, 0.01, alpha, 2)
    value, gradient, curvature = objective.evaluate(PARAMETERS, curvature=True)
    numerical = central_difference(lambda x: objective.evaluate(x)[0], PARAMETERS)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(curvature, curvature.T, atol=1e-12)
    assert np.linalg.eigvalsh(curvature).min() >= -1e-10
    eq, _ = response_and_jacobian(PARAMETERS, FREQUENCIES, 48000)
    if alpha == 1:
        expected = 0.5 * np.mean((responses + eq)**2) + 0.005 * np.sum(PARAMETERS[::3]**2)
        assert value == pytest.approx(expected)
    uncorrected_variance = responses.var(axis=0)
    np.testing.assert_allclose((responses + eq).var(axis=0), uncorrected_variance)
    unrestricted = responses - responses.mean(axis=0)
    assert np.mean(unrestricted**2) == pytest.approx(uncorrected_variance.mean())


def test_small_temperature_approaches_worst_seat():
    responses = np.stack((np.ones_like(FREQUENCIES), np.full_like(FREQUENCIES, 1e6)))
    objective = SeatObjective(responses, FREQUENCIES, 48000, 0, 0, 1e-300)
    eq, _ = response_and_jacobian(PARAMETERS, FREQUENCIES, 48000)
    value, gradient, curvature = objective.evaluate(PARAMETERS, curvature=True)
    assert value == pytest.approx(np.mean((responses[1] + eq)**2) / 2)
    assert np.isfinite(gradient).all() and np.isfinite(curvature).all()


def test_large_temperature_approaches_mean():
    responses = np.stack((np.ones_like(FREQUENCIES), np.full_like(FREQUENCIES, 32)))
    objective = SeatObjective(responses, FREQUENCIES, 48000, 0, 0, 1e100)
    mean = SeatObjective(responses, FREQUENCIES, 48000, 0)
    value, gradient = objective.evaluate(PARAMETERS)
    expected_value, expected_gradient = mean.evaluate(PARAMETERS)
    assert value == pytest.approx(expected_value)
    np.testing.assert_allclose(gradient, expected_gradient)


@pytest.mark.parametrize("solver", [lbfgsb, generalized_gauss_newton])
def test_solver_handles_active_bounds(solver):
    class Quadratic:
        def evaluate(self, x, curvature=False):
            delta = x - [-2, 0.25, 2]
            value = 0.5 * (delta @ delta)
            return (value, delta, np.eye(3)) if curvature else (value, delta)

    lower, upper = -np.ones(3), np.ones(3)
    fit = solver(Quadratic(), np.zeros(3), lower, upper, budget=100, tolerance=1e-7)
    assert fit.success
    np.testing.assert_allclose(fit.x, [-1, 0.25, 1], atol=1e-7)
    assert projected_gradient_norm(fit.x, fit.jac, lower, upper) <= 1e-7
    assert np.all(np.diff([row["objective"] for row in fit.history]) <= 0)


def test_spatial_splits_share_a_fixed_test_block():
    receiver_grid = np.array([(x, y, 0) for x in (-0.25, 0, 0.25) for y in (-0.25, 0, 0.25)])
    config = {
        "seed": 5403, "spatial_seeds": [101, 202], "training_counts": [2, 4],
        "min_test_x_m": 0.25, "max_training_x_m": 0,
    }
    splits = list(spatial_splits(receiver_grid, config))
    for _, train, test in splits:
        np.testing.assert_array_equal(test, [6, 7, 8])
        assert train[0] == 4
        assert np.min(receiver_grid[test, 0]) - np.max(receiver_grid[train, 0]) >= 0.25
    np.testing.assert_array_equal(splits[0][1], splits[1][1][:2])
