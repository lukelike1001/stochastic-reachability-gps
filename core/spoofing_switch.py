from __future__ import annotations
from core.estimators.fused_srkf import FusedSRKF
from core.estimators.self_contained_srkf import SelfContainedSRKF


class SpoofingSwitch:
    """The output SR filter switch of Figure 3. Picks which filter supplies
    the state estimate and error p-zonotope the user actually sees.

    Latches: Section 3.1 says that once the detector calls a spoof, the output
    relies on the self-contained filter "until it can re-authenticate the
    received GPS measurements", so a later authentic step does not switch it
    back. Only reset() does, at the next Chimera epoch.
    """

    def __init__(self):
        self.spoof_detected = False

    def select_estimator(
        self,
        is_authentic: bool,
        fused: FusedSRKF,
        self_contained: SelfContainedSRKF,
    ) -> FusedSRKF | SelfContainedSRKF:
        """Return whichever filter the user should trust this tick

        The fused filter keeps stepping either way, so it is ready to be
        trusted again once a fresh Chimera authentication arrives.
        """
        self.spoof_detected = self.spoof_detected or not is_authentic
        return self_contained if self.spoof_detected else fused

    def reset(self) -> None:
        """Clear the latch on a fresh Chimera authentication

        # NOTE: unused by Figures 4 and 5, where the 6 second run holds a
        # single epoch and no re-authentication ever arrives.
        """
        self.spoof_detected = False
