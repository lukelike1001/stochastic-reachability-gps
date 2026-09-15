from __future__ import annotations
import numpy as np
from core import constants
from core.reachability.pzonotope import PZonotope


class SelfContainedSRKF:
    """Simple implementation of the Stochastic Reachability-Based Kalman Filter
    (SR-KF) as described in the Mina et al., (2023) paper. See more at Section
    3.3. Not to be confused with the nonlinear SR-EKF nor the fused SR-KF."""

    def __init__(
        self,
        t_s: float,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        initial_covariance: np.ndarray,
        initial_position_error: PZonotope,
    ):
        """
        Args:
            t_s: discrete sample period, in seconds.
            initial_position: (p_x, p_y) of the Chimera fix at t=0.
            initial_velocity: (v_x, v_y) at t=0.
            initial_covariance: 4x4 covariance of the initial state.
            initial_position_error: 2D error of that Chimera fix. It comes
                from a least-squares solve, so it is not the zero set.
        """
        self.t_s = t_s
        self.A = constants.state_transition(t_s)
        self.B = constants.input_matrix(t_s)
        self.H = constants.MEASUREMENT_MATRIX
        self.Gamma = constants.GAMMA
        self.Q = constants.process_noise_covariance(t_s)

        self.process_noise_pzonotope = PZonotope(
            center=np.zeros((4, 1)),
            generator=constants.PROCESS_BIAS_GENERATOR,
            covariance=self.Q,
        )

        self.hat_x = np.concatenate(
            [initial_position, initial_velocity]
        ).reshape(4, 1)
        self.hat_P = initial_covariance
        # GPS says nothing about velocity, so H.T leaves those rows at zero.
        self.tilde_X = initial_position_error.linear_map(self.H.T)

    @property
    def state_estimate(self) -> np.ndarray:
        """Return the 4x1 hat_x_k of Equation 19"""
        return self.hat_x

    @property
    def position_estimate(self) -> np.ndarray:
        """Return the 2x1 hat_p_k^self that the spoofing statistic uses"""
        return self.H @ self.hat_x

    @property
    def error_pzonotope(self) -> PZonotope:
        """Return the 4D tilde_X_k^self of Equation 20"""
        return self.tilde_X

    @property
    def position_error_pzonotope(self) -> PZonotope:
        """Return the 2D tilde_P_k^self that Equation 25 combines"""
        return self.tilde_X.linear_map(self.H)

    def step(self, u_prev: np.ndarray) -> None:
        """Advance one tick on an acceleration u_{k-1}.

        Everything moves together, so the estimate, its covariance and the
        error p-zonotope can never fall out of step with one another.
        """
        self.hat_x = self.A @ self.hat_x + self.B @ u_prev # Eq. 4, 19
        self.hat_P = self.A @ self.hat_P @ self.A.T + self.Q # Eq. 5
        self.tilde_X = self.tilde_X.linear_map(self.A).minkowski_sum(
            self.process_noise_pzonotope.linear_map(-self.Gamma)
        ) # Eq. 20

    def sync_to_chimera(
        self,
        chimera_position: np.ndarray,
        chimera_position_error: PZonotope,
    ) -> None:
        """Re-anchor position to a fresh Chimera fix at the start of an epoch.
        Velocity carries over untouched, since Chimera only authenticates
        position.

        NOTE: Unused by Figures 4 and 5, since the experiments only last for a
        single 6 second epoch."""
        self.hat_x[0:2] = chimera_position.reshape(2, 1)
        self.tilde_X = chimera_position_error.linear_map(self.H.T)
