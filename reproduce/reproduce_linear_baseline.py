"""Reproduces Figure 4: the Chimera SR-KF under nominal, unspoofed conditions
for the linear double-integrator system of Section 4.1.1."""

from __future__ import annotations
import sys
from core import constants
from reproduce.figure_panels import ThreePanelFigure
from reproduce.monte_carlo import LinearMonteCarlo, load_or_run

OUTPUT_PATH = "figures/figure4_linear_baseline.png"
CACHE_PATH = "cache/figure4_linear_baseline.pkl"


def main() -> None:
    result = load_or_run(
        CACHE_PATH,
        LinearMonteCarlo(spoof_offset=None),
        refresh="--refresh" in sys.argv,
    )

    figure = ThreePanelFigure("Figure 4: Chimera SR-KF, nominal conditions")
    figure.plot_correct_authentication(
        result.times,
        result.correct_authentication_rate,
        required=1.0 - constants.FALSE_ALARM_PROBABILITY,
    )
    figure.plot_birds_eye(
        true_positions=result.mean_true_positions,
        sets={
            "self_contained": (
                result.mean_self_contained_positions,
                result.self_contained_error_sets,
            ),
            "gps": (result.mean_gps_positions, result.gps_error_sets),
            "chimera": (result.mean_chimera_positions, result.chimera_error_sets),
        },
        confidence_sigma=constants.CONFIDENCE_SIGMA,
    )
    figure.plot_bounding_ratio(
        result.times,
        result.chimera_bounding_ratio,
        result.naive_bounding_ratio,
        name="SR-KF",
    )
    figure.save(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    print(f"lowest CAR = {result.correct_authentication_rate.min():.4f}")


if __name__ == "__main__":
    main()
