"""Mean and smooth worst-seat objectives for shared room equalization."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from room_eq.biquad import response_and_jacobian


@dataclass(frozen=True)
class SeatObjective:
    """Responses are in dB relative to the common target."""

    responses: np.ndarray
    frequencies: np.ndarray
    sample_rate: float
    regularization: float
    alpha: float = 1.0
    temperature: float = 1.0

    def __post_init__(self):
        if not 0 <= self.alpha <= 1 or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Expected alpha in [0, 1] and a positive finite temperature")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("Expected finite nonnegative gain regularization")

    def evaluate(self, parameters, curvature=False):
        eq, jacobian = response_and_jacobian(parameters, self.frequencies, self.sample_rate)
        weight = 1 / len(self.frequencies)
        residuals = self.responses + eq
        losses = 0.5 * weight * np.sum(residuals**2, axis=1)
        seat_gradients = weight * (residuals @ jacobian)
        value = float(losses.mean())
        gradient = seat_gradients.mean(axis=0)
        if self.alpha < 1:
            maximum = losses.max()
            with np.errstate(over="ignore"):
                scaled = (losses - maximum) / self.temperature
            normalization = logsumexp(scaled)
            probabilities = np.exp(scaled - normalization)
            # Use log1p when every scaled loss is close to zero so very large
            # temperatures do not round the normalized log mean to zero.
            log_mean = (
                np.log1p(np.expm1(scaled).mean())
                if scaled.min() > -0.5 else normalization - np.log(len(losses))
            )
            smooth_max = maximum + self.temperature * log_mean
            weighted_gradient = probabilities @ seat_gradients
            value = self.alpha * value + (1 - self.alpha) * smooth_max
            gradient = self.alpha * gradient + (1 - self.alpha) * weighted_gradient
        gains = parameters[::3]
        value += 0.5 * self.regularization * (gains @ gains)
        gradient[::3] += self.regularization * gains
        if not curvature:
            return value, gradient
        hessian = weight * (jacobian.T @ jacobian)
        gain_indices = np.arange(0, len(parameters), 3)
        hessian[gain_indices, gain_indices] += self.regularization
        if self.alpha < 1:
            centered = np.sqrt(probabilities[:, None]) * (seat_gradients - weighted_gradient)
            hessian += (1 - self.alpha) * (centered.T @ centered) / self.temperature
        return value, gradient, hessian
