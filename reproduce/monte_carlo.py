from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from core import constants
from core.estimators.fused_srkf import FusedSRKF
from core.estimators.self_contained_srkf import SelfContainedSRKF
from core.estimators.unauthenticated_gps_estimator import UnauthenticatedGpsEstimator
from core.reachability.pzonotope import PZonotope
from core.reachability.trajectory import Trajectory
from core.sensors.chimera_gps_sensor import ChimeraGpsSensor
from core.sensors.self_contained_sensor import SelfContainedSensor
from core.sensors.unauthenticated_gps_sensor import UnauthenticatedGpsSensor
from core.simulation.ground_truth import GroundTruth
from core.spoofing_detector import SpoofingDetector
from core.spoofing_switch import SpoofingSwitch


@dataclass
class MonteCarloResult:
    """Everything the three panels of Figures 4 to 7 need"""
    times: np.ndarray
    correct_authentication_rate: np.ndarray
    missed_detection_rate: np.ndarray
    chimera_bounding_ratio: np.ndarray
    naive_bounding_ratio: np.ndarray
    mean_true_positions: np.ndarray
    mean_self_contained_positions: np.ndarray
    mean_gps_positions: np.ndarray
    mean_chimera_positions: np.ndarray
    self_contained_error_sets: list[PZonotope]
    gps_error_sets: list[PZonotope]
    chimera_error_sets: list[PZonotope]

    @property
    def correct_detection_rate(self) -> np.ndarray:
        """Return the CDR, which is just what the MDR leaves over"""
        return 1.0 - self.missed_detection_rate


class LinearMonteCarlo:
    """Runs the linear SR-KF pipeline of Figure 3 many times over, per the
    1000 trajectory Monte Carlo of Section 4.1.2."""

    def __init__(
        self,
        spoof_offset: Callable[[float], np.ndarray] | None = None,
        runs: int = constants.MONTE_CARLO_RUNS,
        duration_s: float = constants.CHIMERA_EPOCH_S,
        t_s: float = constants.SAMPLE_PERIOD_S,
        detection_radius: float | None = None,
        bounding_sigma: float = constants.CONFIDENCE_SIGMA,
        seed: int = 0,
    ):
        """
        Args:
            spoof_offset: None for the nominal figures, a ramp for the
                spoofed ones.
            detection_radius: threshold for Equation 26, defaulting to the
                radius that spends P_FA over a 2D statistic.
            bounding_sigma: confidence level for the panel (c) check, which
                the captions fix at 3 sigma. Deliberately not the same number.
        """
        self.spoof_offset = spoof_offset
        self.runs = runs
        self.t_s = t_s
        self.step_count = int(round(duration_s / t_s))
        self.detection_radius = (
            constants.false_alarm_radius(2) if detection_radius is None
            else detection_radius
        )
        self.bounding_sigma = bounding_sigma
        self.seed = seed

    def _single_run(self, seed: int) -> dict:
        """Play one trajectory through the whole pipeline"""
        rng = np.random.default_rng(seed)
        nominal = Trajectory(p0=np.zeros(2), **constants.TRUE_TRAJECTORY)
        truth = GroundTruth(nominal, self.t_s, self.step_count, rng)

        chimera = ChimeraGpsSensor(truth, rng=rng)
        gps_sensor = UnauthenticatedGpsSensor(
            truth, spoof_offset=self.spoof_offset, rng=rng
        )
        imu = SelfContainedSensor(
            truth, self.t_s, constants.process_noise_covariance(self.t_s), rng=rng
        )

        fix = chimera.sample_authenticated_fix(0.0).ravel()
        fix_error = chimera.initial_error_pzonotope()
        start = dict(
            t_s=self.t_s,
            initial_position=fix,
            initial_velocity=truth.poll_velocity(0.0),
            initial_covariance=chimera.initial_state_covariance(),
            initial_position_error=fix_error,
        )
        self_contained = SelfContainedSRKF(**start)
        fused = FusedSRKF(**start, gps_error_pzonotope=gps_sensor.error_pzonotope)
        naive = FusedSRKF(**start, gps_error_pzonotope=gps_sensor.error_pzonotope)
        gps = UnauthenticatedGpsEstimator(
            gps_sensor.error_pzonotope, gps_sensor.sample_measurement(0.0)
        )
        detector = SpoofingDetector(self.detection_radius)
        switch = SpoofingSwitch()

        record = {key: [] for key in (
            "authentic", "chimera_bounded", "naive_bounded",
            "true", "self_contained", "gps", "chimera",
            "self_contained_set", "gps_set", "chimera_set",
        )}

        for k in range(self.step_count + 1):
            if k > 0:
                u = imu.sample_input((k - 1) * self.t_s)
                measurement = gps_sensor.sample_measurement(k * self.t_s)
                self_contained.step(u)
                fused.step(u, measurement)
                naive.step(u, measurement)
                gps.update(measurement)

            authentic = detector.is_authentic(self_contained, gps)
            selected = switch.select_estimator(authentic, fused, self_contained)
            true_state = truth.state_at(k * self.t_s)

            record["authentic"].append(authentic)
            record["chimera_bounded"].append(
                self._bounds(selected, true_state)
            )
            record["naive_bounded"].append(self._bounds(naive, true_state))
            record["true"].append(true_state[0:2].ravel())
            record["self_contained"].append(self_contained.position_estimate.ravel())
            record["gps"].append(gps.position_estimate.ravel())
            record["chimera"].append(selected.position_estimate.ravel())
            record["self_contained_set"].append(
                self_contained.position_error_pzonotope
            )
            record["gps_set"].append(gps.position_error_pzonotope)
            record["chimera_set"].append(selected.position_error_pzonotope)

        return record

    def _bounds(self, estimator, true_state: np.ndarray) -> bool:
        """Return whether the true state sits inside the estimator's error set

        Equation 18 puts the set of possible true states at the estimate minus
        the error set, so the error the set has to cover is estimate minus
        truth.
        """
        error = estimator.state_estimate - true_state
        return estimator.error_pzonotope.confidence_zonotope(
            self.bounding_sigma
        ).contains_point(error)

    def run(self) -> MonteCarloResult:
        """Run every trajectory and average the panels out"""
        records = [self._single_run(self.seed + i) for i in range(self.runs)]

        def stack(key):
            return np.array([r[key] for r in records])

        authentic = stack("authentic")
        return MonteCarloResult(
            times=np.arange(self.step_count + 1) * self.t_s,
            correct_authentication_rate=authentic.mean(axis=0),
            missed_detection_rate=authentic.mean(axis=0),
            chimera_bounding_ratio=stack("chimera_bounded").mean(axis=0),
            naive_bounding_ratio=stack("naive_bounded").mean(axis=0),
            mean_true_positions=stack("true").mean(axis=0),
            mean_self_contained_positions=stack("self_contained").mean(axis=0),
            mean_gps_positions=stack("gps").mean(axis=0),
            mean_chimera_positions=stack("chimera").mean(axis=0),
            # The error sets carry no randomness, so one run speaks for all.
            self_contained_error_sets=records[0]["self_contained_set"],
            gps_error_sets=records[0]["gps_set"],
            chimera_error_sets=records[0]["chimera_set"],
        )


def load_or_run(
    cache_path: str, monte_carlo: LinearMonteCarlo, refresh: bool = False
) -> MonteCarloResult:
    """Return a cached result if there is one, otherwise run and cache it

    A full 1000 run sweep takes minutes, and panel (a) only looks right at that
    many runs, so styling work reuses the same simulation rather than paying
    for it again on every colour tweak. Pass refresh to force a rerun.
    """
    path = Path(cache_path)
    if path.exists() and not refresh:
        with path.open("rb") as handle:
            return pickle.load(handle)

    result = monte_carlo.run()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return result
