from __future__ import annotations
import numpy as np
from core import constants
from core.estimators.self_contained_srkf import SelfContainedSRKF
from core.estimators.unauthenticated_gps_estimator import UnauthenticatedGpsEstimator
from core.reachability.pzonotope import PZonotope


class SpoofingDetector:
    """The SR spoofing detector of Figure 3. Asks whether the gap between the
    self-contained position estimate and the unauthenticated GPS one is still
    small enough to look authentic. See Section 3.4.

    Holds no state between steps, so there is no order to get wrong.
    """

    def __init__(self, confidence_sigma: float | None = None):
        """
        Args:
            confidence_sigma: the user-specified false alarm threshold of
                Figure 3, as a Mahalanobis radius. Defaults to whatever radius
                spends P_FA over the 2D spoofing statistic, which is 3.41 and
                not the 3.0 the figures draw their zonotopes at.
        """
        self.confidence_sigma = (
            constants.false_alarm_radius(2)
            if confidence_sigma is None
            else confidence_sigma
        )

    def spoofing_statistic(
        self,
        self_contained: SelfContainedSRKF,
        gps: UnauthenticatedGpsEstimator,
    ) -> np.ndarray:
        """Return the 2x1 q_k of Equation 21"""
        return self_contained.position_estimate - gps.position_estimate

    def statistic_pzonotope(
        self,
        self_contained: SelfContainedSRKF,
        gps: UnauthenticatedGpsEstimator,
    ) -> PZonotope:
        """Return the 2D Q_k of Equation 25, the overbound q_k should sit in"""
        return self_contained.position_error_pzonotope.minkowski_sum(
            -gps.position_error_pzonotope
        )

    def is_authentic(
        self,
        self_contained: SelfContainedSRKF,
        gps: UnauthenticatedGpsEstimator,
    ) -> bool:
        """Return d_k of Equation 26, where True means authentic

        Polarity follows the paper: d_k = 1 is an authentic decision, d_k = 0
        is a spoofed one.
        """
        return self.statistic_pzonotope(self_contained, gps).contains(
            self.spoofing_statistic(self_contained, gps), self.confidence_sigma
        )
