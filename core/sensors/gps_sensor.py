from __future__ import annotations
import numpy as np
from core.reachability.pzonotope import PZonotope
from core.reachability.trajectory import Trajectory


class GpsSensor:
    """Base for the GPS sensors. Reports position only (Equation 6) with a
    bounded bias and Gaussian noise, as featured in Section 4.1.1."""

    def __init__(
        self,
        trajectory: Trajectory,
        bias_bound: float = 0.5,
        standard_deviation: float = 5.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            trajectory: ground truth the sensor reads from.
            bias_bound: half-width of the bounded bias, in metres, in each
                position dimension (+/-0.5 m in Section 4.1.1).
            standard_deviation: measurement noise sigma, in metres (5 m).
            rng: source of randomness, so Monte Carlo runs are reproducible.
        """
        self.trajectory = trajectory
        self.bias_bound = bias_bound
        self.standard_deviation = standard_deviation
        self.rng = rng if rng is not None else np.random.default_rng()

        # Drawn once and held for the whole run: this is a bias, not noise.
        self.bounded_bias = self.rng.uniform(
            low=-bias_bound, high=bias_bound, size=(2, 1)
        )

    @property
    def error_generator(self) -> np.ndarray:
        """Return G for this sensor's error p-zonotope, per Section 4.1.1"""
        return np.diag([self.bias_bound, self.bias_bound])

    @property
    def error_covariance(self) -> np.ndarray:
        """Return Sigma for this sensor's error p-zonotope (Z_k in Equation 7)"""
        return self.standard_deviation**2 * np.eye(2)

    @property
    def error_pzonotope(self) -> PZonotope:
        """Return the 2D R_k this sensor's measurements carry, constant in k"""
        return PZonotope(
            center=np.zeros((2, 1)),
            generator=self.error_generator,
            covariance=self.error_covariance,
        )

    def sample_measurement(self, t: float) -> np.ndarray:
        """Return the 2x1 z_k read at time t, biased and noisy but authentic"""
        true_position = self.trajectory.poll_position(t).reshape(2, 1)
        measurement_noise = self.rng.normal(
            scale=self.standard_deviation, size=(2, 1)
        )
        return true_position + self.bounded_bias + measurement_noise
