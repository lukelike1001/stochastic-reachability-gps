from __future__ import annotations
from typing import Callable
import numpy as np
from core.reachability.trajectory import Trajectory
from core.sensors.gps_sensor import GpsSensor


class UnauthenticatedGpsSensor(GpsSensor):
    """GPS received between Chimera authentications, so it may be spoofed.
    Feeds the spoofing statistic (Equation 21) and the fused SR-KF."""

    def __init__(
        self,
        trajectory: Trajectory,
        bias_bound: float = 0.5,
        standard_deviation: float = 5.0,
        spoof_offset: Callable[[float], np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            spoof_offset: maps a time to the 2x1 additive bias an attacker
                adds at that time. None for the nominal case of Figure 4.
                Figure 5 passes a ramp reaching 60 m by t=6.
        """
        super().__init__(trajectory, bias_bound, standard_deviation, rng)
        self.spoof_offset = spoof_offset

    def sample_measurement(self, t: float) -> np.ndarray:
        """Return the 2x1 z_k read at time t, spoofed if an offset was given"""
        measurement = super().sample_measurement(t)
        if self.spoof_offset is None:
            return measurement
        return measurement + self.spoof_offset(t).reshape(2, 1)
