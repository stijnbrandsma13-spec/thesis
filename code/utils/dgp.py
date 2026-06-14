import numpy as np
from scipy.stats import multivariate_normal
from typing import Callable


class DataGenerator:
    """Generator for synthetic mixed logit data.

    Generates observations for the random-coefficient discrete choice model
    with a Type I extreme value error term, producing choice indicators ``y``
    and covariate arrays ``x``.

    Parameters
    ----------
    N : int
        Number of individuals.
    J : int
        Number of alternatives, excluding the outside good (alternative 0).
    K : int
        Number of covariates per individual per alternative.
    rng : np.random.Generator
        NumPy random number generator used for all stochastic draws.
    """

    def __init__(self, N: int, J: int, K: int, rng: np.random.Generator) -> None:
        self.N = N
        self.J = J
        self.K = K
        self.rng = rng
        self.x: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.epsilon: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def _default_discrete_distribution_generator(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the uniform distribution over {0, 1}^K.

        Returns
        -------
        support : np.ndarray of shape (2**K, K)
            All elements of the K-fold Cartesian product {0, 1}^K.
        probs : np.ndarray of shape (2**K,)
            Uniform probability 1 / 2^K for each support point.
        """
        support = np.array(np.meshgrid(*([np.array([0, 1])] * self.K))).T.reshape(-1, self.K)
        probs = np.full(2**self.K, 1 / 2**self.K)
        return support, probs

    def _discrete_sampler_generator(
        self, support: np.ndarray, probs: np.ndarray
    ) -> Callable[[tuple[int, ...]], np.ndarray]:
        """Build a sampler that draws from a discrete distribution.

        Parameters
        ----------
        support : np.ndarray of shape (S, K)
            Support points of the distribution.
        probs : np.ndarray of shape (S,)
            Probability mass for each support point.

        Returns
        -------
        Callable
            Function ``draw(size)`` that returns ``size[0]`` i.i.d. draws from
            the distribution, shaped ``(size[0], K)``.
        """
        rng = self.rng

        def draw(size: tuple[int, ...]) -> np.ndarray:
            indices = rng.choice(len(probs), size=size[0], p=probs)
            return support[indices]

        return draw

    def _generate_x(self, sampler: Callable) -> np.ndarray:
        """Draw the covariate array, setting the outside good (j=0) to zero.

        Parameters
        ----------
        sampler : Callable
            Function accepting ``size=(N, J, K)`` and returning an array of
            shape ``(N, J, K)``.

        Returns
        -------
        x : np.ndarray of shape (N, J+1, K)
            Covariates, with ``x[:, 0, :]`` fixed at zero.
        """
        x = np.zeros((self.N, self.J + 1, self.K))
        x[:, 1:, :] = sampler(size=(self.N, self.J, self.K))
        self.x = x
        return x

    def _generate_beta(self, sampler: Callable) -> np.ndarray:
        """Draw individual-specific random coefficients.

        Parameters
        ----------
        sampler : Callable
            Function accepting ``size=(N, K)`` and returning an array of
            shape ``(N, K)``.

        Returns
        -------
        beta : np.ndarray of shape (N, K)
        """
        beta = sampler(size=(self.N, self.K))
        self.beta = beta
        return beta

    def _generate_epsilon(self, sampler: Callable) -> np.ndarray:
        """Draw idiosyncratic error terms.

        Parameters
        ----------
        sampler : Callable
            Function accepting ``size=(N, J+1)`` and returning an array of
            shape ``(N, J+1)``.

        Returns
        -------
        epsilon : np.ndarray of shape (N, J+1)
        """
        epsilon = sampler(size=(self.N, self.J + 1))
        self.epsilon = epsilon
        return epsilon

    def _generate_latent_utility(
        self,
        x_sampler: Callable | None = None,
        beta_sampler: Callable | None = None,
        epsilon_sampler: Callable | None = None,
        beta_support: np.ndarray | None = None,
        beta_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute latent utilities u_ij = x_ij' beta_i + epsilon_ij.

        Falls back to default samplers when arguments are omitted: standard
        normal for x, uniform over {0,1}^K for beta, and Gumbel for epsilon.
        Providing ``beta_support`` and ``beta_probs`` overrides the default
        discrete beta distribution.

        Parameters
        ----------
        x_sampler : Callable, optional
        beta_sampler : Callable, optional
        epsilon_sampler : Callable, optional
        beta_support : np.ndarray, optional
        beta_probs : np.ndarray, optional

        Returns
        -------
        u : np.ndarray of shape (N, J+1)
        """
        if x_sampler is None:
            x_sampler = self.rng.normal
        if beta_sampler is None:
            if beta_support is None or beta_probs is None:
                support, probs = self._default_discrete_distribution_generator()
                beta_sampler = self._discrete_sampler_generator(support=support, probs=probs)
            else:
                beta_sampler = self._discrete_sampler_generator(support=beta_support, probs=beta_probs)
        if epsilon_sampler is None:
            epsilon_sampler = self.rng.gumbel

        x = self._generate_x(sampler=x_sampler) if self.x is None else self.x
        b = self._generate_beta(sampler=beta_sampler) if self.beta is None else self.beta
        e = self._generate_epsilon(sampler=epsilon_sampler) if self.epsilon is None else self.epsilon

        u = np.einsum('ijk,ik->ij', x, b) + e
        self.u = u
        return u

    def _generate_choices(
        self,
        x_sampler: Callable | None = None,
        beta_sampler: Callable | None = None,
        epsilon_sampler: Callable | None = None,
        beta_support: np.ndarray | None = None,
        beta_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Derive binary choice indicators from latent utilities.

        Parameters
        ----------
        x_sampler : Callable, optional
        beta_sampler : Callable, optional
        epsilon_sampler : Callable, optional
        beta_support : np.ndarray, optional
        beta_probs : np.ndarray, optional

        Returns
        -------
        y : np.ndarray of shape (N, J+1), dtype int
            ``y[i, j] = 1`` if individual ``i`` chooses alternative ``j``.
        """
        if self.u is None:
            u = self._generate_latent_utility(
                x_sampler=x_sampler,
                beta_sampler=beta_sampler,
                epsilon_sampler=epsilon_sampler,
                beta_support=beta_support,
                beta_probs=beta_probs,
            )
        else:
            u = self.u

        y = (u == u.max(axis=1, keepdims=True)).astype(int)
        self.y = y
        return y

    def generate(
        self,
        x_sampler: Callable | None = None,
        beta_sampler: Callable | None = None,
        epsilon_sampler: Callable | None = None,
        beta_support: np.ndarray | None = None,
        beta_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a dataset from the mixed logit model.

        Parameters
        ----------
        x_sampler : Callable, optional
            Sampler for covariates; defaults to standard normal.
        beta_sampler : Callable, optional
            Sampler for random coefficients; defaults to uniform on {0,1}^K.
        epsilon_sampler : Callable, optional
            Sampler for error terms; defaults to Gumbel.
        beta_support : np.ndarray, optional
            Support points to use with the default discrete beta sampler.
        beta_probs : np.ndarray, optional
            Probabilities to use with the default discrete beta sampler.

        Returns
        -------
        y : np.ndarray of shape (N, J+1)
            Choice indicators.
        x : np.ndarray of shape (N, J+1, K)
            Covariates.
        """
        self._generate_choices(
            x_sampler=x_sampler,
            beta_sampler=beta_sampler,
            epsilon_sampler=epsilon_sampler,
            beta_support=beta_support,
            beta_probs=beta_probs,
        )
        return self.y, self.x

    def reset(self) -> None:
        """Clear all cached draws, preserving the RNG state.

        Call before each Monte Carlo repetition to obtain a fresh sample
        without resetting the generator, so draws remain sequentially
        dependent across repetitions from the same seed.
        """
        self.x = None
        self.beta = None
        self.epsilon = None
        self.u = None
        self.y = None


# ---------------------------------------------------------------------------
# Covariate sampler
# ---------------------------------------------------------------------------

def make_heiss_x_sampler(
    rng: np.random.Generator,
) -> Callable[[tuple[int, ...]], np.ndarray]:
    """Build a covariate sampler following Heiss, Hetzenecker & Osterhaus (2022).

    Draws K=2 features independently:

    - feature 1 ~ U(0, 5)
    - feature 2 ~ U(-3, 1)

    All draws are routed through the supplied ``rng`` so the covariates are
    reproducible from the simulation seed. Pass the same generator that drives
    :class:`DataGenerator` to keep the entire DGP reproducible from one seed.

    Parameters
    ----------
    rng : np.random.Generator
        Generator used for every covariate draw.

    Returns
    -------
    Callable
        ``draw(size)`` returning an array of shape ``size``, where ``size`` is
        the ``(N, J, K)`` tuple passed by :class:`DataGenerator`. ``K`` (the
        last element) must equal 2.
    """
    def draw(size: tuple[int, ...]) -> np.ndarray:
        *base_dims, K = size
        if K != 2:
            raise ValueError(f"heiss_x_sampler requires K=2, got K={K}.")
        result = np.zeros(size)
        result[..., 0] = rng.uniform(0, 5, size=base_dims)
        result[..., 1] = rng.uniform(-3, 1, size=base_dims)
        return result

    return draw


# ---------------------------------------------------------------------------
# Support / probability constructors
# ---------------------------------------------------------------------------

def _make_grid(R: int, grid_range: tuple[float, float]) -> np.ndarray:
    """Build a 2-D square grid of ``R`` points over ``grid_range`` × ``grid_range``.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    grid_range : tuple of float
        (min, max) range applied to both dimensions.

    Returns
    -------
    grid : np.ndarray of shape (R, 2)
    """
    r = int(np.sqrt(R))
    grid_1d = np.linspace(*grid_range, r)
    b1, b2 = np.meshgrid(grid_1d, grid_1d)
    return np.column_stack((b1.ravel(), b2.ravel()))


def heiss_beta_support_probs(R: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bimodal support from Heiss, Hetzenecker & Osterhaus (2022).

    Mass is distributed uniformly over two quadrants of the grid:

    - region 1: beta_1 <= -0.5 and beta_2 <= -0.5
    - region 2: beta_1 >= -0.5 and beta_2 >= -0.5

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
        All grid points (used as estimator support).
    support : np.ndarray of shape (R, 2)
        Same as ``full_grid``.
    probs : np.ndarray of shape (R,)
        Uniform mass over valid points, zero elsewhere.
    """
    r = int(np.sqrt(R))
    grid_1d = np.linspace(-4.5, 3.5, r)
    beta_1, beta_2 = np.meshgrid(grid_1d, grid_1d)
    full_grid = np.column_stack((beta_1.ravel(), beta_2.ravel()))

    in_region_1 = (full_grid[:, 0] <= -0.5) & (full_grid[:, 1] <= -0.5)
    in_region_2 = (full_grid[:, 0] >= -0.5) & (full_grid[:, 1] >= -0.5)
    valid_mask = in_region_1 | in_region_2

    probs = np.zeros(len(full_grid))
    S = valid_mask.sum()
    if S > 0:
        probs[valid_mask] = 1.0 / S

    return full_grid, full_grid, probs


def beta_tight_normal_support_probs(
    R: int,
    mean: tuple[float, float] = (0.0, 0.0),
    cov: np.ndarray | None = None,
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tight (identity-covariance) bivariate normal mass over the full grid (baseline DGP).

    All grid points receive non-trivial mass, so there is no boundary problem.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    mean : tuple of float, optional
        Mean of the bivariate normal distribution.
    cov : np.ndarray of shape (2, 2), optional
        Covariance matrix; defaults to the 2×2 identity.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
        Normalised bivariate normal density evaluated at each grid point.
    """
    full_grid = _make_grid(R, grid_range)
    cov = np.eye(2) if cov is None else np.asarray(cov)
    probs = multivariate_normal.pdf(full_grid, mean=np.asarray(mean), cov=cov)
    probs /= probs.sum()
    return full_grid, full_grid, probs


# Hardcoded locations and (unequal) masses for the four types DGP.
# Drawn at random once and frozen, so the DGP is identical across runs,
# repetitions, and seeds. Each location is snapped to the nearest grid point,
# so the true distribution lives on the estimator grid for every R.
FOUR_TYPES_POINTS = np.array([
    [-3.2, 1.6],
    [-1.0, -2.9],
    [0.6, 2.3],
    [2.4, -0.7],
])
FOUR_TYPES_MASSES = np.array([0.37, 0.29, 0.21, 0.13])


def beta_four_types_support_probs(
    R: int,
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four fixed grid points carry unequal mass, all others are on the boundary.

    The four active points (``FOUR_TYPES_POINTS`` snapped to the grid)
    carry masses ``FOUR_TYPES_MASSES`` = (0.37, 0.29, 0.21, 0.13);
    the remaining ``R - 4`` points have structurally zero probability.
    Provides a clean interior/boundary split to study coverage across
    structural zeros.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    full_grid = _make_grid(R, grid_range)
    probs = np.zeros(len(full_grid))
    for point, mass in zip(FOUR_TYPES_POINTS, FOUR_TYPES_MASSES):
        idx = np.argmin(np.linalg.norm(full_grid - point, axis=1))
        if probs[idx] > 0:
            raise ValueError(
                "Two four-types points snapped to the same grid point; "
                "the grid is too coarse for the hardcoded locations."
            )
        probs[idx] = mass
    return full_grid, full_grid, probs


def beta_bimodal_support_probs(
    R: int,
    means: tuple[tuple[float, float], tuple[float, float]] = ((-2.2, -2.2), (1.3, 1.3)),
    covs: list[np.ndarray] | None = None,
    weights: tuple[float, float] = (0.5, 0.5),
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixture of two Gaussian clusters (bimodal DGP).

    Interior mass at the two modes; near-zero mass in the valley between
    them — the boundary problem appears in a localised sub-region.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    means : tuple of two (float, float) pairs, optional
        Means of the two mixture components.
    covs : list of two (2, 2) arrays, optional
        Covariance matrices, defaults to [[0.8,0.15],[0.15,0.8]]
    weights : tuple of two floats, optional
        Mixture weights; must sum to 1.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    full_grid = _make_grid(R, grid_range)
    covs = [np.array([[0.8,0.15],[0.15,0.8]])] * 2 if covs is None else [np.asarray(c) for c in covs]
    probs = sum(
        w * multivariate_normal.pdf(full_grid, mean=np.asarray(m), cov=c)
        for w, m, c in zip(weights, means, covs)
    )
    probs /= probs.sum()
    return full_grid, full_grid, probs


def beta_wide_normal_support_probs(
    R: int,
    mean: tuple[float, float] = (0.0, 0.0),
    variance: float = 4.0,
    corr: float = 0.0,
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wide (high-variance) Gaussian with small positive mass everywhere (near-boundary globally).

    Varying ``corr`` in {-0.7, 0.0, 0.7} allows studying grid-orientation
    effects on coverage.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    mean : tuple of float, optional
        Mean of the distribution.
    variance : float, optional
        Variance applied to both dimensions.
    corr : float, optional
        Correlation coefficient (off-diagonal entry of the correlation matrix).
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    full_grid = _make_grid(R, grid_range)
    cov = variance * np.array([[1.0, corr], [corr, 1.0]])
    probs = multivariate_normal.pdf(full_grid, mean=np.asarray(mean), cov=cov)
    probs /= probs.sum()
    return full_grid, full_grid, probs


def beta_almost_single_type_support_probs(
    R: int,
    center: tuple[float, float] = (0.0, 0.0),
    spike_mass: float = 0.9,
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most mass at one grid point, residual mass spread uniformly over the rest.

    Tests inference on a large interior mass surrounded by many near-zero
    (boundary) parameters.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    center : tuple of float, optional
        Coordinates of the spike; snapped to the nearest grid point.
    spike_mass : float, optional
        Probability mass at the spike point.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    full_grid = _make_grid(R, grid_range)
    spike_idx = np.argmin(np.linalg.norm(full_grid - np.asarray(center), axis=1))
    probs = np.full(len(full_grid), (1 - spike_mass) / (len(full_grid) - 1))
    probs[spike_idx] = spike_mass
    return full_grid, full_grid, probs


def beta_single_type_support_probs(
    R: int,
    center: tuple[float, float] = (0.0, 0.0),
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All mass at one grid point, exactly zero mass everywhere else.

    Degenerate version of the almost-single-type DGP: every individual shares
    the same coefficient vector, so all ``R`` parameters sit exactly on a
    boundary of the simplex (``R - 1`` at zero, one at one).

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    center : tuple of float, optional
        Coordinates of the spike; snapped to the nearest grid point.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    return beta_almost_single_type_support_probs(
        R, center=center, spike_mass=1.0, grid_range=grid_range
    )


def beta_strictly_uniform_support_probs(
    R: int,
    grid_range: tuple[float, float] = (-4.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly equal mass ``1 / R`` at every grid point.

    No parameter is on the boundary, but every parameter approaches it at
    rate ``1 / R`` as the grid grows: a globally near-boundary setting with
    no shape information in the true distribution.

    Parameters
    ----------
    R : int
        Total number of grid points; must be a perfect square.
    grid_range : tuple of float, optional
        (min, max) range for both grid dimensions.

    Returns
    -------
    full_grid : np.ndarray of shape (R, 2)
    support : np.ndarray of shape (R, 2)
    probs : np.ndarray of shape (R,)
    """
    full_grid = _make_grid(R, grid_range)
    probs = np.full(len(full_grid), 1.0 / len(full_grid))
    return full_grid, full_grid, probs


if __name__ == '__main__':
    test = DataGenerator(10, 2, 2, np.random.default_rng(10))
    y, x = test.generate()
    print(y.shape)
    print(x.shape)
    print(x[0])
    print(x[0][0])
    print(x[0].shape)
    print(test.beta)
