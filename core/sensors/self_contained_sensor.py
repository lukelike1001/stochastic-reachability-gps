from __future__ import annotations
import numpy as np
from core.reachability.trajectory import Trajectory


class SelfContainedSensor:
    """Simulates a self-contained sensor on a drone that calculates an
    acceleration value at a set frequency, such as 10Hz. Uses a Trajectory
    for the ground truth and feeds data to the Self-Contained SR-KF."""

    def __init__(
        self,
        trajectory: Trajectory,
        sample_period_s: float,
        Q: np.ndarray,
        process_bias_generator: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            trajectory: ground truth the sensor reads from.
            sample_period_s: t_s, the sampling period in seconds (0.1 for the
                10 Hz rate of Section 4.1.1).
            Q: 4x4 process noise covariance (Equation 47). Share the same
                array with the SR-KF so truth and filter agree.
            process_bias_generator: 4xm generator for the bounded bias of
                Section 4.1.1 (+/-0.1 m position, +/-0.01 m/s velocity).
            rng: source of randomness, so Monte Carlo runs are reproducible.
        """
        if process_bias_generator is None:
            process_bias_generator = np.diag([0.1, 0.1, 0.01, 0.01])

        self.trajectory = trajectory
        self.sample_period_s = sample_period_s
        self.Q = Q
        self.process_bias_generator = process_bias_generator
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_input(self, t: float) -> np.ndarray:
        """Return u_{k-1}, the 2x1 acceleration (a_x, a_y) read at time t"""
        acceleration = self.trajectory.poll_acceleration(t)
        return acceleration.reshape(2, 1)

    def sample_process_noise(self) -> np.ndarray:
        """Return the 4x1 w_{k-1}, the sensor's state-space error"""
        stochastic_noise = self.rng.multivariate_normal(
            mean=np.zeros(4), cov=self.Q
        ).reshape(4, 1)

        bias_coefficients = self.rng.uniform(
            low=-1.0, high=1.0, size=(self.process_bias_generator.shape[1], 1)
        )
        bounded_bias = self.process_bias_generator @ bias_coefficients

        return stochastic_noise + bounded_bias
