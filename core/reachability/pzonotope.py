from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


class PZonotope:
    """Representation of a probabilistic zonotope (Equation 2) from Section 2.2:"""

    def __init__(self, center: np.ndarray, generator: np.ndarray, covariance: np.ndarray):
        """
        Args:
            center: c, shape (n, 1).
            generator: G, shape (n, m). Columns are generators used for the
                bounded-bias uncertainty. Use an (n, 0) array if there is no
                bias uncertainty to represent (degenerate case), not None.
            covariance: Sigma, shape (n, n), positive semi-definite. Encodes
                the stochastic (Gaussian) part. Use a zero matrix if there is
                no stochastic part to represent, not None.
        """
        n = center.shape[0]
        assert center.shape == (n, 1), f"center must be ({n}, 1), got {center.shape}"
        assert generator.shape[0] == n, f"generator must have {n} rows, got {generator.shape}"
        assert covariance.shape == (n, n), f"covariance must be ({n}, {n}), got {covariance.shape}"

        self.center = center
        self.generator = generator
        self.covariance = covariance

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    def linear_map(self, matrix: np.ndarray) -> PZonotope:
        """Apply a linear map M to this p-zonotope: M @ Z_p(c, G, Sigma) =
        Z_p(M @ c, M @ G, M @ Sigma @ M.T).

        Used for, e.g., A_k @ tilde_X_{k-1} in Equations 17/20, or projecting
        a 4D state p-zonotope down to 2D position via H_k (Equation 25).
        """
        return PZonotope(
            center=matrix @ self.center,
            generator=matrix @ self.generator,
            covariance=matrix @ self.covariance @ matrix.T,
        )

    def minkowski_sum(self, other: PZonotope) -> PZonotope:
        """Minkowski sum (⊕) of two p-zonotopes: centers add, generators
        concatenate (side by side), covariances add (independent sums).

        Used to combine terms in Equations 17/20/25.
        """
        assert self.dim == other.dim, "Cannot sum p-zonotopes of different dimension"
        return PZonotope(
            center=self.center + other.center,
            generator=np.hstack([self.generator, other.generator]),
            covariance=self.covariance + other.covariance,
        )

    def __neg__(self) -> PZonotope:
        """Negation (-Z): -Z_p(c, G, Sigma) = Z_p(-c, -G, Sigma).

        Covariance is unaffected by negation (Sigma describes spread, not
        direction). Enables Minkowski difference via self + (-other), as in
        Equation 25: Q_k = P_self ⊕ (-P_GPS).
        """
        return PZonotope(
            center=-self.center,
            generator=-self.generator,
            covariance=self.covariance,
        )

    def reduce(self, tolerance: float = 1e-9) -> PZonotope:
        """Merge parallel generators into one each, leaving the set unchanged

        Two parallel generators g and (alpha)g sweep the same line segment as
        a single generator of length (1 + |alpha|), so this is exact rather
        than an over-approximation. Equation 20 tacks on four generators per
        step, but the double integrator keeps them axis aligned, so hundreds
        collapse back to a handful. CORA's order reduction does the same job
        conservatively for the general case.
        """
        magnitudes = np.linalg.norm(self.generator, axis=0)
        kept = self.generator[:, magnitudes > tolerance]
        magnitudes = magnitudes[magnitudes > tolerance]
        if kept.shape[1] == 0:
            return PZonotope(self.center, np.zeros((self.dim, 0)), self.covariance)

        # Point every generator the same way so that g and -g group together.
        directions = kept / magnitudes
        leading = np.argmax(np.abs(directions) > tolerance, axis=0)
        signs = np.sign(directions[leading, np.arange(directions.shape[1])])
        directions = directions * signs

        keys, inverse = np.unique(
            np.round(directions / tolerance).astype(np.int64).T,
            axis=0,
            return_inverse=True,
        )
        merged = np.zeros(keys.shape[0])
        np.add.at(merged, inverse, magnitudes)

        return PZonotope(
            center=self.center,
            generator=(keys.T * tolerance) * merged,
            covariance=self.covariance,
        )

    def confidence_zonotope(self, confidence_sigma: float) -> PZonotope:
        """Return the bounded set covering this p-zonotope to a confidence level

        The "3-sigma confidence level zonotope" of Section 4.1.2. Turns the
        Gaussian part into generators, sigma along each principal axis of
        Sigma, and adds them to the bias generators. The result is a zonotope,
        so it is wider than the matching ellipsoid: its corners reach
        sigma * sqrt(dim) rather than sigma. That is what the figures draw and
        what the bounding ratio in panel (c) is measured against.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        spread = (
            eigenvectors
            @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
            * confidence_sigma
        )
        return PZonotope(
            center=self.center,
            generator=np.hstack([self.generator, spread]),
            covariance=np.zeros((self.dim, self.dim)),
        )

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Return whether the point lies inside this zonotope, ignoring Sigma

        Pair with confidence_zonotope, which folds the Gaussian part into the
        generators first.
        """
        reduced = self.reduce()
        target = (point.reshape(self.dim, 1) - reduced.center).ravel()
        scale = max(1.0, float(np.linalg.norm(target)))
        if reduced.generator.shape[1] == 0:
            return float(np.linalg.norm(target)) <= tolerance * scale

        solution = lsq_linear(
            reduced.generator, target, bounds=(-1.0, 1.0)
        )
        return float(np.sqrt(2.0 * solution.cost)) <= tolerance * scale

    def mahalanobis_distance(self, point: np.ndarray) -> float:
        """Return how many sigma this point sits from the zonotope of means

        This is how we read Q_k(q_k) in Equation 26. The paper never says
        what evaluating a p-zonotope at a point means, so we take it as the
        highest density any distribution in the set can give that point:

            sup over ||beta||_inf <= 1 of N(point; c + G beta, Sigma)

        Maximising that density is the same as minimising the Mahalanobis
        distance from the point to the zonotope Z(c, G), which is a box
        constrained least squares problem once Sigma is whitened away. A
        point inside the zonotope scores 0, since some mean sits right on it.
        """
        reduced = self.reduce()
        residual = point.reshape(self.dim, 1) - reduced.center

        # Whiten by Sigma so the metric becomes plain Euclidean distance.
        # Sigma can be singular, e.g. the initial error set says nothing about
        # velocity, so go through the eigenvalues rather than a Cholesky and
        # floor them. A direction with no variance then costs a great deal to
        # miss, which is what a deterministic direction should do.
        eigenvalues, eigenvectors = np.linalg.eigh(reduced.covariance)
        floor = max(eigenvalues.max(), 1.0) * 1e-12
        whitener = eigenvectors @ np.diag(
            1.0 / np.sqrt(np.maximum(eigenvalues, floor))
        ) @ eigenvectors.T

        whitened_generator = whitener @ reduced.generator
        whitened_residual = (whitener @ residual).ravel()

        if reduced.generator.shape[1] == 0:
            return float(np.linalg.norm(whitened_residual))

        # Orthogonal generators make the box constrained problem separable, so
        # every coefficient is just a clipped projection. The double integrator
        # keeps its generators axis aligned, so this is the usual case here and
        # it is far quicker than calling a solver.
        gram = whitened_generator.T @ whitened_generator
        diagonal = np.diag(gram)
        if np.abs(gram - np.diag(diagonal)).max() <= 1e-9 * max(diagonal.max(), 1.0):
            coefficients = np.clip(
                whitened_generator.T @ whitened_residual / diagonal, -1.0, 1.0
            )
            return float(
                np.linalg.norm(whitened_residual - whitened_generator @ coefficients)
            )

        solution = lsq_linear(
            whitened_generator, whitened_residual, bounds=(-1.0, 1.0)
        )
        return float(np.sqrt(2.0 * solution.cost))

    def contains(self, point: np.ndarray, confidence_sigma: float) -> bool:
        """Return whether the point falls inside the confidence-level zonotope

        Equation 26 compares against P_FA, but a density cannot be compared
        to a probability, so we threshold the distance instead. Section 4.1.2
        uses 3 sigma throughout. Also answers the panel (c) question of
        whether the true state stays bounded.

        Screens with two cheap bounds before paying for the exact distance,
        which matters because the fused error set carries hundreds of
        generators that no longer line up with the axes.
        """
        reduced = self.reduce()
        if reduced.generator.shape[1] == 0:
            return self.mahalanobis_distance(point) <= confidence_sigma

        residual = point.reshape(self.dim, 1) - reduced.center
        eigenvalues, eigenvectors = np.linalg.eigh(reduced.covariance)
        floor = max(eigenvalues.max(), 1.0) * 1e-12
        whitener = eigenvectors @ np.diag(
            1.0 / np.sqrt(np.maximum(eigenvalues, floor))
        ) @ eigenvectors.T
        generators = whitener @ reduced.generator
        target = (whitener @ residual).ravel()

        # Any feasible coefficient vector overestimates the true distance, so
        # a clipped projection landing inside settles it.
        squared_norms = np.einsum("ij,ij->j", generators, generators)
        coefficients = np.clip(generators.T @ target / squared_norms, -1.0, 1.0)
        gap = target - generators @ coefficients
        if np.linalg.norm(gap) <= confidence_sigma:
            return True

        # Any direction underestimates it, via the zonotope's support function.
        # The gap direction is the natural separating candidate to try.
        length = np.linalg.norm(gap)
        if length > 0:
            direction = gap / length
            support = np.abs(generators.T @ direction).sum()
            if direction @ target - support > confidence_sigma:
                return False

        return self.mahalanobis_distance(point) <= confidence_sigma