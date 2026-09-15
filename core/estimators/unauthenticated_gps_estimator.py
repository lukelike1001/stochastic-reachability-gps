from __future__ import annotations
import numpy as np
from core.reachability.pzonotope import PZonotope


class UnauthenticatedGpsEstimator:
    """The unauthenticated GPS p-zonotope estimator of Figure 3. Not a filter:
    Equation 21 takes the received measurement as the position estimate, so
    there is no state to propagate and no measurement update to apply."""

    def __init__(self, error_pzonotope: PZonotope, initial_measurement: np.ndarray):
        """
        Args:
            error_pzonotope: 2D R_k from Section 4.1.1, constant in k. This is
                the declared error model, not a realization, so it stays the
                same whether or not an attacker is present.
            initial_measurement: the 2x1 z_k received at t=0.
        """
        self.error = error_pzonotope
        self.hat_p = initial_measurement.reshape(2, 1)

    @property
    def position_estimate(self) -> np.ndarray:
        """Return the 2x1 hat_p_k^GPS that the spoofing statistic uses"""
        return self.hat_p

    @property
    def position_error_pzonotope(self) -> PZonotope:
        """Return the 2D tilde_P_k^GPS that Equation 25 combines

        Never grows. That is the whole asymmetry the detector runs on: the
        self-contained set widens with time while this one holds still.
        """
        return self.error

    def update(self, measurement: np.ndarray) -> None:
        """Take the latest 2x1 z_k, mirroring the SR-KF's step"""
        self.hat_p = measurement.reshape(2, 1)
