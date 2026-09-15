from __future__ import annotations
import numpy as np
from core import constants
from core.reachability.trajectory import Trajectory


class GroundTruth:
    """The true vehicle states x_k of Equation 3, which only the simulator may
    read. Wraps a nominal Trajectory and propagates it with the same discrete
    A and B the filters use, plus a realization of the process noise.

    Exposes the same poll_ methods as Trajectory, so sensors can read either.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        t_s: float,
        step_count: int,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            trajectory: nominal path supplying the acceleration inputs.
            t_s: discrete sample period, in seconds.
            step_count: how many steps to propagate, so k runs 0..step_count.
            rng: source of randomness, so Monte Carlo runs are reproducible.
        """
        self.trajectory = trajectory
        self.t_s = t_s
        self.rng = rng if rng is not None else np.random.default_rng()

        A = constants.state_transition(t_s)
        B = constants.input_matrix(t_s)
        Gamma = constants.GAMMA
        Q = constants.process_noise_covariance(t_s)
        bias_generator = constants.PROCESS_BIAS_GENERATOR

        state = np.concatenate(
            [trajectory.poll_position(0.0), trajectory.poll_velocity(0.0)]
        ).reshape(4, 1)
        self.states = [state]
        for k in range(1, step_count + 1):
            u = trajectory.poll_acceleration((k - 1) * t_s).reshape(2, 1)
            noise = self.rng.multivariate_normal(np.zeros(4), Q).reshape(4, 1)
            bias = bias_generator @ self.rng.uniform(
                -1.0, 1.0, size=(bias_generator.shape[1], 1)
            )
            state = A @ state + B @ u + Gamma @ (noise + bias)
            self.states.append(state)
        self.states = np.hstack(self.states)  # (4, step_count + 1)

    def state_at(self, t: float) -> np.ndarray:
        """Return the 4x1 true state at time t, snapped to the nearest step"""
        return self.states[:, [int(round(t / self.t_s))]]

    def poll_position(self, t: float) -> np.ndarray:
        """Return the true (p_x, p_y) at time t"""
        return self.state_at(t)[0:2].ravel()

    def poll_velocity(self, t: float) -> np.ndarray:
        """Return the true (v_x, v_y) at time t"""
        return self.state_at(t)[2:4].ravel()

    def poll_acceleration(self, t: float) -> np.ndarray:
        """Return the commanded acceleration, shared by truth and filters"""
        return self.trajectory.poll_acceleration(t)
