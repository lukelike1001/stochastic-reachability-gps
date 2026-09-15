from __future__ import annotations
import numpy as np
from core import constants
from core.reachability.pzonotope import PZonotope


class FusedSRKF:
    """The fused (self-contained + GPS) SR filter of Figure 3. Propagates on
    the self-contained sensor like the SelfContainedSRKF, then corrects with
    the unauthenticated GPS measurement. See Sections 3.2 and 3.3."""

    def __init__(
        self,
        t_s: float,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        initial_covariance: np.ndarray,
        initial_position_error: PZonotope,
        gps_error_pzonotope: PZonotope,
    ):
        """
        Args:
            t_s: discrete sample period, in seconds.
            initial_position: (p_x, p_y) of the Chimera fix at t=0.
            initial_velocity: (v_x, v_y) at t=0.
            initial_covariance: 4x4 covariance of the initial state.
            initial_position_error: 2D error of that Chimera fix.
            gps_error_pzonotope: 2D R_k of Section 4.1.1, constant in k. Its
                covariance is the Z_k that Equation 7 needs.
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
        self.R = gps_error_pzonotope
        # Equation 7 takes only the Gaussian half. Section 3.2 models r_k as
        # N(0, Z_k), so the +/-0.5 m bias stays out of the gain and shows up
        # only in Equation 17.
        self.Z = gps_error_pzonotope.covariance

        self.hat_x = np.concatenate(
            [initial_position, initial_velocity]
        ).reshape(4, 1)
        self.hat_P = initial_covariance
        self.tilde_X = initial_position_error.linear_map(self.H.T)

    @property
    def state_estimate(self) -> np.ndarray:
        """Return the 4x1 hat_x_k of Equation 8"""
        return self.hat_x

    @property
    def position_estimate(self) -> np.ndarray:
        """Return the 2x1 fused position estimate"""
        return self.H @ self.hat_x

    @property
    def error_pzonotope(self) -> PZonotope:
        """Return the 4D tilde_X_k of Equation 17"""
        return self.tilde_X

    @property
    def position_error_pzonotope(self) -> PZonotope:
        """Return the 4D error set projected down to position"""
        return self.tilde_X.linear_map(self.H)

    def step(self, u_prev: np.ndarray, measurement: np.ndarray) -> None:
        """Advance one tick on an acceleration u_{k-1} and a GPS z_k

        The gain is what couples the point-valued track to the set-valued
        one, so it has to be formed before Equation 17 can run.
        """
        hat_x_predicted = self.A @ self.hat_x + self.B @ u_prev  # Eq. 4
        hat_P_predicted = self.A @ self.hat_P @ self.A.T + self.Q  # Eq. 5

        innovation_covariance = self.H @ hat_P_predicted @ self.H.T + self.Z
        K = hat_P_predicted @ self.H.T @ np.linalg.inv(innovation_covariance)  # Eq. 7

        innovation = measurement.reshape(2, 1) - self.H @ hat_x_predicted
        self.hat_x = hat_x_predicted + K @ innovation  # Eq. 8
        self.hat_P = (np.eye(4) - K @ self.H) @ hat_P_predicted  # Eq. 9

        # Equation 17, term by term. The middle sign follows the paper; it
        # makes no difference while the process noise is centred at zero.
        residual = np.eye(4) - K @ self.H
        self.tilde_X = (
            self.tilde_X.linear_map(residual @ self.A)
            .minkowski_sum(
                self.process_noise_pzonotope.linear_map(-residual @ self.Gamma)
            )
            .minkowski_sum(self.R.linear_map(K))
        )
