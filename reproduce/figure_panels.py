from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
from core.reachability.pzonotope import PZonotope

# Read off Figures 4 to 7. Adjust here and every figure follows.
COLOURS = dict(
    car="#b163a8",
    required="black",
    mdr="tab:orange",
    cdr="#b163a8",
    self_contained="#316f8f",
    gps="#d0673c",
    chimera="#0fb050",
    truth="black",
    naive="#d0673c",
)

# The paper's own legend wording, so the panels read like the originals.
LABELS = dict(
    self_contained="Self-Contained Position Estimate",
    gps="GPS Position Estimate",
    chimera="Chimera Estimate",
)

EPOCH_SECONDS = 6.0
BIRDS_EYE_XLIM = (-20.0, 60.0)
BIRDS_EYE_YLIM = (-20.0, 120.0)
BOX_FACE_ALPHA = 0.1
BOX_EDGE_ALPHA = 0.75
# Figure 4(b) shows 11 boxes per series over the 6 second epoch, so one every
# 0.6 seconds, which is every sixth sample at 10 Hz.
BOX_STRIDE = 6


def confidence_vertices(
    pzonotope: PZonotope, confidence_sigma: float, centre: np.ndarray
) -> np.ndarray:
    """Return the polygon outlining a 2D p-zonotope at a confidence level

    Only does the drawing half. Building the confidence set is the model's
    job, over in PZonotope.confidence_zonotope.
    """
    generators = pzonotope.confidence_zonotope(confidence_sigma).reduce().generator

    # A 2D zonotope is the sum of its generators as line segments. Point them
    # all upwards, sort by angle, and walking them traces the boundary.
    generators = np.where(generators[1] < 0, -generators, generators)
    generators = generators[:, np.argsort(np.arctan2(generators[1], generators[0]))]

    start = centre.reshape(2) - generators.sum(axis=1)
    upper = start + 2 * np.cumsum(generators, axis=1).T
    lower = start + 2 * generators.sum(axis=1) - 2 * np.cumsum(generators, axis=1).T
    return np.vstack([start, upper, lower])


class ThreePanelFigure:
    """The (a) (b) (c) layout shared by Figures 4 to 7. Panel (a) is a
    detection rate, panel (b) a bird's eye view, panel (c) a bounding ratio."""

    def __init__(self, title: str | None = None):
        self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 4.2))
        if title:
            self.figure.suptitle(title)

    def plot_correct_authentication(
        self, times: np.ndarray, car: np.ndarray, required: float
    ) -> None:
        """Panel (a) for the nominal figures: CAR against its requirement"""
        ax = self.axes[0]
        ax.plot(times, car, color=COLOURS["car"], label="CAR")
        ax.axhline(
            required, color=COLOURS["required"], label="Minimum Required CAR"
        )
        ax.set_ylim(0.99, 1.0)
        ax.set_yticks(np.arange(0.99, 1.0001, 0.002))
        self._style_rate_axis(ax, "Ratio over Monte Carlo runs")

    def plot_detection_rates(
        self, times: np.ndarray, mdr: np.ndarray, cdr: np.ndarray
    ) -> None:
        """Panel (a) for the spoofed figures: MDR falling as CDR rises"""
        ax = self.axes[0]
        ax.plot(times, mdr, color=COLOURS["mdr"], label="MDR")
        ax.plot(times, cdr, color=COLOURS["cdr"], label="CDR")
        ax.set_ylim(-0.02, 1.02)
        self._style_rate_axis(ax, "Ratio across MC runs")

    def plot_birds_eye(
        self,
        true_positions: np.ndarray,
        sets: dict[str, tuple[np.ndarray, list[PZonotope]]],
        confidence_sigma: float,
        stride: int = BOX_STRIDE,
    ) -> None:
        """Panel (b): error zonotopes along the trajectory

        Args:
            sets: label -> (centres, per-step p-zonotopes). Section 4.1.2 draws
                these centred on the mean estimated trajectory rather than on
                each run's own estimate, purely so the picture is readable.
        """
        ax = self.axes[1]
        for label, (centres, pzonotopes) in sets.items():
            colour = COLOURS[label]
            for k in range(0, len(pzonotopes), stride):
                ax.add_patch(
                    Polygon(
                        confidence_vertices(
                            pzonotopes[k], confidence_sigma, centres[k]
                        ),
                        closed=True,
                        facecolor=to_rgba(colour, BOX_FACE_ALPHA),
                        edgecolor=to_rgba(colour, BOX_EDGE_ALPHA),
                        linewidth=0.8,
                    )
                )
            ax.plot([], [], color=colour, linewidth=1.5, label=LABELS[label])

        ax.plot(
            true_positions[:, 0],
            true_positions[:, 1],
            color=COLOURS["truth"],
            linewidth=1.5,
            label="True Position",
        )
        ax.set_xlabel("$x_1$ position")
        ax.set_ylabel("$x_2$ position")
        ax.set_xlim(*BIRDS_EYE_XLIM)
        ax.set_ylim(*BIRDS_EYE_YLIM)
        ax.set_xticks(np.arange(BIRDS_EYE_XLIM[0], BIRDS_EYE_XLIM[1] + 1, 20.0))
        ax.set_yticks(np.arange(BIRDS_EYE_YLIM[0], BIRDS_EYE_YLIM[1] + 1, 20.0))
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")

    def plot_bounding_ratio(
        self, times: np.ndarray, chimera: np.ndarray, naive: np.ndarray, name: str
    ) -> None:
        """Panel (c): how often the true state stays inside the error zonotope"""
        ax = self.axes[2]
        ax.plot(times, chimera, color=COLOURS["truth"], label=f"Chimera {name}")
        ax.plot(times, naive, color=COLOURS["naive"], label=f"Naively Fused {name}")
        ax.set_ylim(-0.02, 1.02)
        self._style_rate_axis(ax, "Ratio over Monte Carlo runs")

    def save(self, path: str) -> None:
        """Write the figure out and close it"""
        for ax, letter in zip(self.axes, "abc"):
            ax.set_title(f"({letter})", loc="left", fontsize=10)
        self.figure.tight_layout()
        self.figure.savefig(path, dpi=160)
        plt.close(self.figure)

    @staticmethod
    def _style_rate_axis(ax, ylabel: str) -> None:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, EPOCH_SECONDS)
        ax.set_xticks(np.arange(0.0, EPOCH_SECONDS + 0.1, 1.0))
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8)
