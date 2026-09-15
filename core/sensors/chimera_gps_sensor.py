from __future__ import annotations
import numpy as np
from core.reachability.pzonotope import PZonotope
from core.sensors.gps_sensor import GpsSensor


class ChimeraGpsSensor(GpsSensor):
    """GPS that Chimera has authenticated, so it can never be spoofed. Only
    read once every 6 seconds, which for Figures 4 and 5 means only at t=0."""

    def sample_authenticated_fix(self, t: float) -> np.ndarray:
        """Return the 2x1 authenticated position the SR-KFs initialize from"""
        return self.sample_measurement(t)

    def initial_error_pzonotope(self) -> PZonotope:
        """Return the 2D set describing how wrong that fix can be

        # NOTE: We're reusing the unauthenticated numbers for now. The paper
        # says the fix comes from a least squares solution over at least four
        # Chimera signals, but gives no values. Authentication proves a signal
        # is genuine, it doesn't make it more accurate.
        """
        return self.error_pzonotope

    def initial_state_covariance(self, velocity_variance: float = 0.0) -> np.ndarray:
        """Return the 4x4 P_0 that matches the fix, per Equation 5's recursion"""
        covariance = np.zeros((4, 4))
        covariance[0:2, 0:2] = self.error_covariance
        covariance[2:4, 2:4] = velocity_variance * np.eye(2)
        return covariance
