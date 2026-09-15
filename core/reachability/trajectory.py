from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryPoint:
    """A single sample of a Trajectory at some time t."""
    time_s: float
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)
    acceleration: np.ndarray  # shape (2,)


class Trajectory:
    """A trajectory representable by a linear double-integrator system
    (Section 4.1.1). Position and velocity can be integrated from an
    acceleration vector at any provided time t"""

    def __init__(
        self,
        a0: float,
        w0: float,
        phi0: float,
        p0: np.ndarray,
        v0: np.ndarray,
    ):
        """
        Args:
            a0: constant magnitude of the acceleration vector.
            w0: constant angular rate at which the acceleration vector
                rotates (radians/sec). Must be non-zero.
            phi0: initial phase angle of the acceleration vector (radians).
            p0: initial position (p_x, p_y) at t=0.
            v0: initial velocity (v_x, v_y) at t=0.
        """

        self.a0 = a0
        self.w0 = w0
        self.phi0 = phi0
        self.p0 = np.asarray(p0, dtype=float)
        self.v0 = np.asarray(v0, dtype=float)

    def poll_acceleration(self, t: float) -> np.ndarray:
        """Poll the drone's acceleration at a given time t"""
        angle = self.w0 * t + self.phi0
        return self.a0 * np.array([np.cos(angle), np.sin(angle)])

    def poll_velocity(self, t: float) -> np.ndarray:
        """Poll the drone's velocity at a given time t, integrated from a(t)"""
        angle = self.w0 * t + self.phi0
        return self.v0 + (self.a0 / self.w0) * np.array([
            np.sin(angle) - np.sin(self.phi0),
            -np.cos(angle) + np.cos(self.phi0),
        ])

    def poll_position(self, t: float) -> np.ndarray:
        """Poll the drone's position at a given time t, doubly integrated
        from a(t)"""
        angle = self.w0 * t + self.phi0
        oscillating_term = (self.a0 / self.w0**2) * np.array([
            np.cos(self.phi0) - np.cos(angle),
            np.sin(self.phi0) - np.sin(angle),
        ])
        linear_term = (self.a0 * t / self.w0) * np.array([
            np.sin(self.phi0),
            -np.cos(self.phi0),
        ])
        return self.p0 + self.v0 * t + oscillating_term - linear_term

    def poll(self, t: float) -> TrajectoryPoint:
        """Return the drone's position, velocity, and acceleration at time t."""
        return TrajectoryPoint(
            time_s=t,
            position=self.poll_position(t),
            velocity=self.poll_velocity(t),
            acceleration=self.poll_acceleration(t),
        )