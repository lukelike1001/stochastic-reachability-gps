import numpy as np
from scipy.stats import chi2

# --- Provided by the paper -------------------------------------------
SAMPLE_PERIOD_S = 0.1            # 10 Hz, Section 4.1.1
CHIMERA_EPOCH_S = 6.0            # fast channel, Section 1
FALSE_ALARM_PROBABILITY = 0.003  # Section 4.1.2
MONTE_CARLO_RUNS = 1000          # Section 4.1.2

ACCELERATION_PSD = 0.1           # m^2 s^-3, Equation 47
PROCESS_BIAS_BOUNDS = np.array([0.1, 0.1, 0.01, 0.01])
GPS_BIAS_BOUND = 0.5
GPS_STANDARD_DEVIATION = 5.0
SPOOF_FINAL_ERROR_M = 60.0       # total error by t=6, Section 4.1.1


def process_noise_covariance(t_s: float) -> np.ndarray:
    """Return the 4x4 Q of Equation 47"""
    return ACCELERATION_PSD * np.block([
        [t_s**3 / 3 * np.eye(2), t_s**2 / 2 * np.eye(2)],
        [t_s**2 / 2 * np.eye(2), t_s * np.eye(2)],
    ])


def state_transition(t_s: float) -> np.ndarray:
    """Return the 4x4 A of the double integrator, per Section 4.1.1"""
    return np.block([
        [np.eye(2), t_s * np.eye(2)],
        [np.zeros((2, 2)), np.eye(2)],
    ])


def input_matrix(t_s: float) -> np.ndarray:
    """Return the 4x2 B mapping acceleration into the state"""
    return np.vstack([t_s**2 / 2 * np.eye(2), t_s * np.eye(2)])


# --- Inferred, see assumptions/ --------------------------------------
GAMMA = np.eye(4)                # filter_assumptions.md, Equation 47
MEASUREMENT_MATRIX = np.hstack([np.eye(2), np.zeros((2, 2))])
                                 # H, equation_and_figure_assumptions.md, Equation 6
PROCESS_BIAS_GENERATOR = np.diag(PROCESS_BIAS_BOUNDS)
                                 # G_w, filter_assumptions.md, redrawn every step
CONFIDENCE_SIGMA = 3.0           # the "3-sigma zonotopes" the figures draw


def false_alarm_radius(dimension: int) -> float:
    """Return the Mahalanobis radius that spends exactly P_FA in this dimension

    Not the same as CONFIDENCE_SIGMA. Three sigma spends 0.003 of the
    probability in one dimension, but 0.011 in two, so thresholding the 2D
    spoofing statistic at three sigma overspends the false alarm budget by
    nearly four times. Section 4.1.2 fixes the budget rather than the radius,
    so the radius is derived from it. See filter_assumptions.md.
    """
    return float(np.sqrt(chi2.ppf(1.0 - FALSE_ALARM_PROBABILITY, dimension)))

# Waypoints read off Figures 4(b) and 5(b), as (time_s, x, y).
# See equation_and_figure_assumptions.md; expect these to move while fitting.
# Centres of the 11 green "Chimera Estimate" boxes in Figure 4(b), one every
# 0.6 s. The caption says those centres coincide with the true trajectory.
TRUE_WAYPOINTS = [
    (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 4.0, 2.0), (1.8, 8.0, 4.0),
    (2.4, 18.0, 8.0), (3.0, 22.0, 10.0), (3.6, 28.0, 18.0), (4.2, 28.0, 22.0),
    (4.8, 30.0, 34.0), (5.4, 28.0, 48.0), (6.0, 25.0, 65.0),
]
SPOOFED_WAYPOINTS = [(0.0, 0.0, 0.0), (3.0, 45.0, 30.0), (6.0, 65.0, 80.0)]

# Least squares fit of the rotating-acceleration Trajectory to TRUE_WAYPOINTS.
# RMS residual is 2.5 m, which is about how precisely a box centre can be read
# off the figure, so the fit is as good as the data allows. Starts slow at
# 3.8 m/s and accelerates at 5.5 m/s^2.
TRUE_TRAJECTORY = dict(
    a0=-5.5453, w0=0.4702, phi0=-2.6440, v0=np.array([2.3287, -3.0240])
)
