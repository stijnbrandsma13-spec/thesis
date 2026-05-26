import numpy as np
from math import sqrt
from scipy.special import softmax
from scipy.stats import norm, chi2
from scipy.optimize import nnls, minimize, brentq
from sklearn.linear_model import enet_path
import numdifftools as nd
import cvxpy as cp
from typing import Callable


def _solve_simplex_ls(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimise ||y - Z theta||^2 over the probability simplex.

    NNLS reparametrisation: pin theta_R = 1 - sum(theta_1..R-1). If the NNLS
    solution sums to more than one, walk a LASSO path and interpolate to the
    point where the coefficients sum to exactly one.
    """
    R = Z.shape[1]
    if R == 1:
        return np.ones(1)
    y_tilde = y - Z[:, -1]
    Z_tilde = Z[:, :-1] - Z[:, [-1]]
    theta_nnls, _ = nnls(Z_tilde, y_tilde)
    if theta_nnls.sum() <= 1.0:
        theta_partial = theta_nnls
    else:
        _, coefs, _ = enet_path(Z_tilde, y_tilde, l1_ratio=1, positive=True)
        sums = coefs.sum(axis=0)
        idx = np.searchsorted(sums, 1.0)
        if idx == 0:
            theta_partial = coefs[:, 0]
        elif idx >= coefs.shape[1]:
            theta_partial = coefs[:, -1]
        else:
            t = (1.0 - sums[idx - 1]) / (sums[idx] - sums[idx - 1])
            theta_partial = coefs[:, idx - 1] + t * (coefs[:, idx] - coefs[:, idx - 1])
    return np.append(theta_partial, 1.0 - theta_partial.sum())


def _solve_simplex_ls_pinned_coord(
    Z: np.ndarray, y: np.ndarray, r: int, phi0: float
) -> np.ndarray:
    """Minimise ||y - Z theta||^2 over the simplex with theta[r] = phi0.

    Eliminates theta[r] and rescales the remaining problem to a unit simplex
    of dimension R-1, so the NNLS+LASSO solver in :func:`_solve_simplex_ls`
    can be reused.
    """
    R = Z.shape[1]
    theta = np.zeros(R)
    theta[r] = phi0
    s = 1.0 - phi0
    if s <= 0:
        return theta
    mask = np.ones(R, dtype=bool)
    mask[r] = False
    y_red = y - Z[:, r] * phi0
    Z_red = Z[:, mask]
    # theta_red in s * simplex_{R-1}. Let u = theta_red / s, then u on the unit
    # simplex of size R-1 and the LS becomes ||y_red - (s*Z_red) u||^2.
    u = _solve_simplex_ls(s * Z_red, y_red)
    theta[mask] = s * u
    return theta


def _solve_simplex_ls_pinned_linear(
    Z: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    phi0: float,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Minimise ||y - Z theta||^2 over the simplex with c.T @ theta = phi0.

    Uses SLSQP. Used for QLR CIs on general linear functionals (e.g. CDF
    evaluations, mean of beta). Slower than the coordinate-pinned version;
    for per-support-point QLR CIs use :func:`_solve_simplex_ls_pinned_coord`.
    """
    R = Z.shape[1]
    if x0 is None:
        x0 = np.ones(R) / R
    ZtZ = Z.T @ Z
    Zty = Z.T @ y

    def obj(theta: np.ndarray) -> float:
        return 0.5 * theta @ ZtZ @ theta - Zty @ theta

    def grad(theta: np.ndarray) -> np.ndarray:
        return ZtZ @ theta - Zty

    constraints = [
        {'type': 'eq', 'fun': lambda t: t.sum() - 1.0,
         'jac': lambda t: np.ones_like(t)},
        {'type': 'eq', 'fun': lambda t: c @ t - phi0,
         'jac': lambda t: c},
    ]
    bounds = [(0.0, 1.0)] * R
    res = minimize(obj, x0, jac=grad, method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x


def _draw_multipliers(N: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Draw N iid bootstrap multipliers with mean 0 and variance 1."""
    if kind == 'rademacher':
        return rng.choice(np.array([-1.0, 1.0]), size=N)
    if kind == 'mammen':
        sqrt5 = sqrt(5.0)
        p = (sqrt5 + 1) / (2 * sqrt5)
        vals = np.array([-(sqrt5 - 1) / 2, (sqrt5 + 1) / 2])
        return rng.choice(vals, size=N, p=[p, 1 - p])
    if kind == 'normal':
        return rng.standard_normal(N)
    raise ValueError(f"Unknown multiplier kind: {kind!r}")


class FKRBEstimator:
    """Fox-Kim-Ryan-Bajari (FKRB) estimator for the mixed logit model.

    Estimates the probability mass function over a pre-specified discrete
    support of random coefficients. The estimation proceeds in two steps:

    1. An unconstrained OLS regression to obtain a clustered sandwich
       variance-covariance matrix, which is used for inference.
    2. A constrained least squares problem (simplex constraints) to obtain a
       valid probability vector as the point estimate.

    Parameters
    ----------
    beta_support : np.ndarray of shape (R, K)
        Pre-specified grid of random coefficient vectors.
    optimal_weighting : bool, optional
        If True, use a two-step feasible GLS: a first-stage unweighted
        constrained fit yields predicted probabilities p_hat, and the
        regression is re-weighted by 1 / sqrt(p_hat * (1 - p_hat)) (the
        model-implied row-wise standard deviation). All cached moments,
        sandwich variance, bootstrap, and QLR computations then operate
        on the weighted (Z, y). Under correct specification this is the
        optimally-weighted PSMD of Chen-Pouzo (2015) Theorem 4.3(2), so
        the QLR statistic is asymptotically chi-squared without
        studentisation.
    """

    def __init__(
        self,
        beta_support: np.ndarray,
        optimal_weighting: bool = False,
    ) -> None:
        self.beta_support = beta_support
        self.optimal_weighting = optimal_weighting
        self.theta: dict[str, np.ndarray] = {}
        self.se: np.ndarray | None = None
        self.vcov: np.ndarray | None = None
        self.ci: np.ndarray | None = None
        self._Z: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._N: int | None = None
        self._J: int | None = None
        self._sqrt_w: np.ndarray | None = None
        self.vcov_cp: np.ndarray | None = None
        self.se_cp: np.ndarray | None = None

    # def _nearest_psd(matrix: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    #     vals, vecs = np.linalg.eigh(matrix)
    #     return vecs @ np.diag(np.maximum(vals, epsilon)) @ vecs.T

    def _calculate_Z(self, x: np.ndarray) -> np.ndarray:
        """Compute the moment matrix Z used in the FKRB regression.

        For each individual-alternative pair (i, j) and support point r,
        Z[i*(J-1)+j, r] = g_j(x_i, beta^(r)), where g_j is the softmax
        choice probability. The outside good (j=0) is excluded.

        Parameters
        ----------
        x : np.ndarray of shape (N, J+1, K)
            Covariates; ``x[:, 0, :]`` is assumed zero (outside good).

        Returns
        -------
        Z : np.ndarray of shape (N*(J-1), R)
        """
        N, J, K = x.shape
        R = self.beta_support.shape[0]
        v = np.einsum('ijk,rk->ijr', x, self.beta_support)
        g = softmax(v, axis=1)
        Z = g[:, 1:, :].reshape(N * (J - 1), R)
        return Z

    def _unconstrained_estimator(
        self, Z: np.ndarray, y: np.ndarray, N: int, J: int, R: int
    ) -> np.ndarray:
        """Fit OLS and compute a clustered sandwich variance-covariance matrix.

        Observations are clustered at the individual level. A small-sample
        HC bias correction is applied to the variance estimator.

        Parameters
        ----------
        Z : np.ndarray of shape (N*(J-1), R)
            Moment matrix from :meth:`_calculate_Z`.
        y : np.ndarray of shape (N*(J-1),)
            Stacked choice indicators (outside good excluded, flattened).
        N : int
            Number of individuals.
        J : int
            Number of alternatives including the outside good.
        R : int
            Number of support points.

        Returns
        -------
        theta : np.ndarray of shape (R,)
            Unconstrained OLS estimate.
        """
        theta = np.linalg.lstsq(Z, y, rcond=None)[0]
        self.theta['unconstrained'] = theta

        u = y - Z @ theta
        u_clustered = u.reshape(N, J - 1)
        Z_clustered = Z.reshape(N, J - 1, R)

        scores = np.einsum('ijr,ij->ir', Z_clustered, u_clustered)
        omega_hat = scores.T @ scores

        ZprimeZinv = np.linalg.pinv(Z.T @ Z)
        bias_correction = (N / (N - 1)) * ((N * (J - 1) - 1) / (N * (J - 1) - R))
        vcov = bias_correction * ZprimeZinv @ omega_hat @ ZprimeZinv

        self.se = np.sqrt(np.diag(vcov))
        self.vcov = vcov
        return theta

    def _constrained_estimator(
        self, Z: np.ndarray, y: np.ndarray, R: int, NNL: bool = True
    ) -> np.ndarray:
        """Fit constrained least squares on the probability simplex.

        When ``NNL=True`` (default), the problem is solved via a LASSO-path
        reparametrisation: the sum-to-one constraint is absorbed by fixing
        theta_R = 1 - sum(theta_1, ..., theta_{R-1}), turning CLS into a
        non-negative least squares problem. If the unconstrained NNLS solution
        is already feasible, the LASSO path is skipped entirely.

        When ``NNL=False``, the problem is solved directly with CVXPY.

        Parameters
        ----------
        Z : np.ndarray of shape (N*(J-1), R)
        y : np.ndarray of shape (N*(J-1),)
        R : int
            Number of support points.
        NNL : bool, optional
            Use the NNLS/LASSO-path solver (True) or CVXPY (False).

        Returns
        -------
        theta : np.ndarray of shape (R,)
            Constrained estimate satisfying theta >= 0 and sum(theta) = 1.
        """
        if NNL:
            theta = _solve_simplex_ls(Z, y)
            self.theta['constrained'] = theta

        else:
            theta_cp = cp.Variable(R)
            objective = cp.Minimize(cp.sum_squares(y - Z @ theta_cp))
            constraints = [theta_cp >= 0, cp.sum(theta_cp) == 1]
            cp.Problem(objective, constraints).solve()
            theta = theta_cp.value
            self.theta['constrained'] = theta

        return theta

    def estimate(
        self,
        y: np.ndarray,
        x: np.ndarray,
        constrained: bool = True,
        estimate_both: bool = True,
        NNL: bool = True,
    ) -> np.ndarray:
        """Estimate the random coefficient distribution.

        By default, both estimators are computed: the unconstrained OLS (for
        inference) and the constrained simplex estimate (as point estimate).
        The returned array depends on the ``constrained`` flag.

        Parameters
        ----------
        y : np.ndarray of shape (N, J+1)
            Choice indicators as returned by :class:`DataGenerator`.
        x : np.ndarray of shape (N, J+1, K)
            Covariates.
        constrained : bool, optional
            Which estimate to return when ``estimate_both=True``.
        estimate_both : bool, optional
            Whether to compute both the constrained and unconstrained estimates.
            Should be ``True`` whenever confidence intervals are needed.
        NNL : bool, optional
            Passed to :meth:`_constrained_estimator`.

        Returns
        -------
        theta : np.ndarray of shape (R,)
        """
        N, J, K = x.shape
        R = self.beta_support.shape[0]

        Z = self._calculate_Z(x)
        y = y[:, 1:].ravel()

        if self.optimal_weighting:
            theta_first = _solve_simplex_ls(Z, y)
            p_hat = np.clip(Z @ theta_first, 1e-6, 1 - 1e-6)
            sqrt_w = 1.0 / np.sqrt(p_hat * (1.0 - p_hat))
            Z = sqrt_w[:, None] * Z
            y = sqrt_w * y
            self._sqrt_w = sqrt_w

        self._Z = Z
        self._y = y
        self._N = N
        self._J = J

        if estimate_both:
            if constrained:
                self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)
                theta = self._constrained_estimator(Z=Z, y=y, R=R, NNL=NNL)
            else:
                self._constrained_estimator(Z=Z, y=y, R=R, NNL=NNL)
                theta = self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)
        else:
            if constrained:
                theta = self._constrained_estimator(Z=Z, y=y, R=R, NNL=NNL)
            else:
                theta = self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)

        return theta

    def plug_in_estimate_functional(
        self,
        functional: Callable[[np.ndarray], float],
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        derivative: Callable[[np.ndarray], np.ndarray] | None = None,
        ci: bool = False,
        alpha: float = 0.05,
    ) -> float | tuple[float, np.ndarray]:
        """Plug-in estimate of a scalar functional, with optional delta-method CI.

        The point estimate is ``functional(theta_constrained)``. When ``ci=True``,
        a delta-method confidence interval is computed using the unconstrained
        OLS covariance matrix and then clipped to the feasible range of the
        functional over the probability simplex.

        Parameters
        ----------
        functional : Callable
            Maps a probability vector of shape (R,) to a scalar.
        clip_lower : float, optional
            Minimum feasible value of the functional over the simplex. Computed
            via SLSQP optimisation if not provided.
        clip_upper : float, optional
            Maximum feasible value of the functional over the simplex. Computed
            via SLSQP optimisation if not provided.
        derivative : Callable, optional
            Gradient of ``functional`` at a probability vector of shape (R,),
            returning an array of shape (R,). Estimated numerically via
            :func:`numdifftools.Gradient` if not provided.
        ci : bool, optional
            Whether to compute and return a confidence interval.
        alpha : float, optional
            Significance level for the confidence interval.

        Returns
        -------
        eta : float
            Plug-in point estimate.
        ci_bounds : np.ndarray of shape (2,)
            Clipped ``[lower, upper]`` confidence interval bounds. Only
            returned when ``ci=True``.
        """
        theta_constrained = self.theta['constrained']
        eta = functional(theta_constrained)

        if not ci:
            return eta

        R = self.beta_support.shape[0]
        if derivative is None:
            derivative = nd.Gradient(functional)

        theta_ols = self.theta['unconstrained']
        nabla_T = derivative(theta_ols)
        se = sqrt(nabla_T.T @ self.vcov @ nabla_T)
        z_score = norm.ppf(1 - alpha / 2)

        lower_bound = eta - z_score * se
        upper_bound = eta + z_score * se

        eq_cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        bnds = [(0.0, 1.0)] * R
        x0 = np.ones(R) / R

        if clip_lower is None:
            clip_lower = minimize(functional, x0, method='SLSQP', bounds=bnds, constraints=eq_cons).fun
        if clip_upper is None:
            clip_upper = -minimize(lambda x: -functional(x), x0, method='SLSQP', bounds=bnds, constraints=eq_cons).fun

        ci_bounds = np.clip([lower_bound, upper_bound], a_min=clip_lower, a_max=clip_upper)
        return eta, ci_bounds

    def _compute_cp_sandwich(self) -> np.ndarray:
        """Cluster-robust sandwich evaluated at the constrained (FKRB) residuals.

        This is the sieve-variance specialisation of CP2015 eq. (2.10) for the
        FKRB case (linear rho, trivial m_hat, identity Sigma): see eq. (CPsievevariance)
        in the paper. The variance is identical in form to the OLS sandwich
        in :meth:`_unconstrained_estimator` except the residuals are formed
        from theta_constrained rather than theta_unconstrained.
        """
        if self.vcov_cp is not None:
            return self.vcov_cp
        Z = self._Z
        y = self._y
        N = self._N
        J = self._J
        R = self.beta_support.shape[0]
        theta_c = self.theta['constrained']

        u = y - Z @ theta_c
        u_clustered = u.reshape(N, J - 1)
        Z_clustered = Z.reshape(N, J - 1, R)
        scores = np.einsum('ijr,ij->ir', Z_clustered, u_clustered)
        omega_hat = scores.T @ scores
        ZprimeZinv = np.linalg.pinv(Z.T @ Z)
        bias_correction = (N / (N - 1)) * ((N * (J - 1) - 1) / (N * (J - 1) - R))
        self.vcov_cp = bias_correction * ZprimeZinv @ omega_hat @ ZprimeZinv
        self.se_cp = np.sqrt(np.diag(self.vcov_cp))
        return self.vcov_cp

    def cp_wald_ci(
        self,
        alpha: float = 0.05,
        functional: Callable[[np.ndarray], float] | None = None,
        derivative: Callable[[np.ndarray], np.ndarray] | None = None,
        c: np.ndarray | None = None,
    ) -> np.ndarray | tuple[float, np.ndarray]:
        """Sieve Wald CI of Chen-Pouzo (2015, Thm. 4.2).

        Differs from the FKRB CI in two ways:
        (i) the sandwich variance is evaluated at the constrained (FKRB) fit
            rather than the unconstrained OLS fit;
        (ii) the resulting interval is NOT intersected with the feasible
             range of the functional.

        With no ``functional`` or ``c``, returns per-support-point CIs of
        shape (R, 2). With a linear functional via ``c`` (shape (R,)) or a
        general ``functional``, returns ``(eta_hat, ci_bounds)``.
        """
        if self._Z is None or self._y is None:
            raise RuntimeError("Call estimate() before cp_wald_ci().")
        V = self._compute_cp_sandwich()
        theta_c = self.theta['constrained']
        z_score = norm.ppf(1 - alpha / 2)

        scalar = (functional is not None) or (c is not None)
        if not scalar:
            se = np.sqrt(np.diag(V))
            lower = theta_c - z_score * se
            upper = theta_c + z_score * se
            return np.column_stack((lower, upper))

        if functional is not None:
            eta_hat = float(functional(theta_c))
            if derivative is None:
                derivative = nd.Gradient(functional)
            g = np.asarray(derivative(theta_c), dtype=float)
        else:
            c = np.asarray(c, dtype=float)
            eta_hat = float(c @ theta_c)
            g = c

        se_eta = sqrt(max(g @ V @ g, 0.0))
        ci_bounds = np.array([eta_hat - z_score * se_eta, eta_hat + z_score * se_eta])
        return eta_hat, ci_bounds

    def get_confidence_interval(self, alpha: float = 0.05) -> np.ndarray:
        """Compute per-support-point confidence intervals for theta.

        Uses the unconstrained OLS standard errors and clips each interval
        to [0, 1] (the feasible range for any probability).

        Parameters
        ----------
        alpha : float, optional
            Significance level.

        Returns
        -------
        ci : np.ndarray of shape (R, 2)
            ``ci[r, 0]`` and ``ci[r, 1]`` are the lower and upper bounds for
            the r-th support point.
        """
        theta_constrained = self.theta['constrained']
        z_score = norm.ppf(1 - alpha / 2)

        lower = theta_constrained - z_score * self.se
        upper = theta_constrained + z_score * self.se

        ci = np.clip(np.column_stack((lower, upper)), 0.0, 1.0)
        self.ci = ci
        return ci

    def bootstrap_ci(
        self,
        alpha: float = 0.05,
        B: int = 500,
        multiplier: str = 'rademacher',
        method: str = 'percentile',
        functional: Callable[[np.ndarray], float] | None = None,
        derivative: Callable[[np.ndarray], np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | tuple[float, np.ndarray]:
        """Generalised residual bootstrap CIs (Chen-Pouzo 2015 §4).

        For each replication, multiply the cluster-level fitted residuals by
        iid mean-zero variance-one multipliers, form bootstrap responses
        ``y* = Z theta_hat + u*``, and re-fit the simplex-constrained estimator.
        Confidence intervals are formed from the bootstrap distribution.

        Parameters
        ----------
        alpha : float, optional
            Significance level.
        B : int, optional
            Number of bootstrap replications.
        multiplier : {'rademacher', 'mammen', 'normal'}, optional
            Distribution of the iid multipliers.
        method : {'percentile', 'basic', 'studentized'}, optional
            Bootstrap CI construction.
        functional : Callable, optional
            If ``None``, returns per-support-point CIs of shape (R, 2). If
            given, returns ``(eta, ci_bounds)`` for the scalar functional.
        derivative : Callable, optional
            Gradient of ``functional`` at a probability vector of shape (R,);
            only used when ``method='studentized'``. Numerically approximated
            via :func:`numdifftools.Gradient` if not provided.
        rng : np.random.Generator, optional
            For reproducibility.

        Returns
        -------
        ci : np.ndarray of shape (R, 2) if ``functional`` is None
        (eta, ci_bounds) : (float, np.ndarray of shape (2,)) otherwise
        """
        if self._Z is None or self._y is None:
            raise RuntimeError("Call estimate() before bootstrap_ci().")
        if rng is None:
            rng = np.random.default_rng()

        Z = self._Z
        y = self._y
        N = self._N
        J = self._J
        R = self.beta_support.shape[0]
        theta_hat = self.theta['constrained']
        fitted = Z @ theta_hat
        u_hat_clustered = (y - fitted).reshape(N, J - 1)

        scalar = functional is not None
        if scalar:
            eta_hat = float(functional(theta_hat))
            if method == 'studentized':
                if derivative is None:
                    derivative = nd.Gradient(functional)
                grad_hat = derivative(self.theta['unconstrained'])
                se_hat = sqrt(grad_hat @ self.vcov @ grad_hat)
            boot_eta = np.empty(B)
            boot_se = np.empty(B) if method == 'studentized' else None
        else:
            boot_theta = np.empty((B, R))

        for b in range(B):
            omega = _draw_multipliers(N, multiplier, rng)
            u_star = (u_hat_clustered * omega[:, None]).ravel()
            y_star = fitted + u_star
            theta_b = _solve_simplex_ls(Z, y_star)

            if scalar:
                boot_eta[b] = functional(theta_b)
                if method == 'studentized':
                    # Sandwich SE on the same bootstrap sample, evaluated at
                    # the unconstrained OLS bootstrap fit.
                    theta_b_ols = np.linalg.lstsq(Z, y_star, rcond=None)[0]
                    u_b = (y_star - Z @ theta_b_ols).reshape(N, J - 1)
                    Zc = Z.reshape(N, J - 1, R)
                    scores_b = np.einsum('ijr,ij->ir', Zc, u_b)
                    omega_b = scores_b.T @ scores_b
                    ZpZinv = np.linalg.pinv(Z.T @ Z)
                    bc = (N / (N - 1)) * ((N * (J - 1) - 1) / (N * (J - 1) - R))
                    Vb = bc * ZpZinv @ omega_b @ ZpZinv
                    grad_b = derivative(theta_b_ols)
                    boot_se[b] = sqrt(max(grad_b @ Vb @ grad_b, 1e-300))
            else:
                boot_theta[b] = theta_b

        if scalar:
            if method == 'percentile':
                lo = np.quantile(boot_eta, alpha / 2)
                hi = np.quantile(boot_eta, 1 - alpha / 2)
            elif method == 'basic':
                lo = 2 * eta_hat - np.quantile(boot_eta, 1 - alpha / 2)
                hi = 2 * eta_hat - np.quantile(boot_eta, alpha / 2)
            elif method == 'studentized':
                t_stats = (boot_eta - eta_hat) / boot_se
                lo = eta_hat - se_hat * np.quantile(t_stats, 1 - alpha / 2)
                hi = eta_hat - se_hat * np.quantile(t_stats, alpha / 2)
            else:
                raise ValueError(f"Unknown method: {method!r}")
            return eta_hat, np.array([lo, hi])

        if method == 'percentile':
            lower = np.quantile(boot_theta, alpha / 2, axis=0)
            upper = np.quantile(boot_theta, 1 - alpha / 2, axis=0)
        elif method == 'basic':
            lower = 2 * theta_hat - np.quantile(boot_theta, 1 - alpha / 2, axis=0)
            upper = 2 * theta_hat - np.quantile(boot_theta, alpha / 2, axis=0)
        elif method == 'studentized':
            raise NotImplementedError(
                "studentized bootstrap not implemented for per-support-point CIs; "
                "use functional= or method='percentile'/'basic'"
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return np.column_stack([lower, upper])

    def qlr_ci(
        self,
        alpha: float = 0.05,
        functional: Callable[[np.ndarray], float] | None = None,
        derivative: Callable[[np.ndarray], np.ndarray] | None = None,
        c: np.ndarray | None = None,
        tol: float = 1e-4,
    ) -> np.ndarray | tuple[float, np.ndarray]:
        """Studentised sieve-QLR CIs (Chen-Pouzo 2015).

        For per-support-point inference (``functional`` and ``c`` both
        ``None``), inverts the studentised QLR test
        ``H_0: theta[r] = phi0`` over ``phi0 in [0, 1]`` for each r. For a
        linear functional ``phi(theta) = c.T @ theta``, pass ``c`` of shape
        (R,) and the inversion is done over the feasible range of phi.
        For a nonlinear functional, pass ``functional`` and optionally
        ``derivative``; the inversion uses SLSQP with an equality constraint
        on the functional value.

        The studentised statistic is

            T_n(phi0) = (RSS_H0(phi0) - RSS_unrestricted)
                       * (g'(Z'Z)^{-1} g) / (g' V_hat g),

        where ``g`` is the linear coefficient (or gradient of phi at the
        constrained estimate), ``V_hat`` is the cluster-robust sandwich
        variance from the unconstrained sieve estimator, and ``RSS_H0`` is
        the simplex-constrained RSS subject to the functional restriction.
        Under regularity, ``T_n -> chi^2_1`` under ``H_0``.

        Returns
        -------
        ci : np.ndarray of shape (R, 2) when no functional/c is supplied
        (eta, ci_bounds) : (float, np.ndarray of shape (2,)) otherwise
        """
        if self._Z is None or self._y is None:
            raise RuntimeError("Call estimate() before qlr_ci().")

        Z = self._Z
        y = self._y
        R = self.beta_support.shape[0]
        theta_hat = self.theta['constrained']
        rss_hat = float(((y - Z @ theta_hat) ** 2).sum())
        crit = chi2.ppf(1 - alpha, df=1)
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)

        scalar = (functional is not None) or (c is not None)

        if not scalar:
            ci = np.zeros((R, 2))
            for r in range(R):
                g = np.zeros(R)
                g[r] = 1.0
                scale = (g @ ZtZ_inv @ g) / max(g @ self.vcov @ g, 1e-300)

                def T(phi0: float, r=r, scale=scale) -> float:
                    th = _solve_simplex_ls_pinned_coord(Z, y, r, phi0)
                    return (float(((y - Z @ th) ** 2).sum()) - rss_hat) * scale

                ci[r] = self._invert_qlr(T, theta_hat[r], 0.0, 1.0, crit, tol)
            return ci

        # Scalar functional path
        if functional is not None:
            eta_hat = float(functional(theta_hat))
            if derivative is None:
                derivative = nd.Gradient(functional)
            g = np.asarray(derivative(self.theta['unconstrained']), dtype=float)
        else:
            c = np.asarray(c, dtype=float)
            eta_hat = float(c @ theta_hat)
            g = c

        scale = (g @ ZtZ_inv @ g) / max(g @ self.vcov @ g, 1e-300)

        eq_simplex = {'type': 'eq', 'fun': lambda t: t.sum() - 1.0}
        bnds = [(0.0, 1.0)] * R
        x0 = np.ones(R) / R
        if functional is not None:
            lo_feas = float(minimize(functional, x0, method='SLSQP',
                                     bounds=bnds, constraints=eq_simplex).fun)
            hi_feas = -float(minimize(lambda t: -functional(t), x0, method='SLSQP',
                                      bounds=bnds, constraints=eq_simplex).fun)
        else:
            lo_feas = float(minimize(lambda t: c @ t, x0, method='SLSQP',
                                     bounds=bnds, constraints=eq_simplex).fun)
            hi_feas = -float(minimize(lambda t: -c @ t, x0, method='SLSQP',
                                      bounds=bnds, constraints=eq_simplex).fun)

        if functional is not None:
            def T(phi0: float) -> float:
                cons = [
                    {'type': 'eq', 'fun': lambda t: t.sum() - 1.0},
                    {'type': 'eq', 'fun': lambda t, p=phi0: functional(t) - p},
                ]
                res = minimize(
                    lambda t: 0.5 * float(((y - Z @ t) ** 2).sum()),
                    x0, method='SLSQP', bounds=bnds, constraints=cons,
                )
                rss = float(((y - Z @ res.x) ** 2).sum())
                return (rss - rss_hat) * scale
        else:
            def T(phi0: float) -> float:
                th = _solve_simplex_ls_pinned_linear(Z, y, c, phi0)
                return (float(((y - Z @ th) ** 2).sum()) - rss_hat) * scale

        ci_bounds = self._invert_qlr(T, eta_hat, lo_feas, hi_feas, crit, tol)
        return eta_hat, ci_bounds

    @staticmethod
    def _invert_qlr(
        T: Callable[[float], float],
        center: float,
        lo_feas: float,
        hi_feas: float,
        crit: float,
        tol: float,
    ) -> np.ndarray:
        """Bisect each side of T(phi0) = crit using brentq.

        T(center) is subtracted from every evaluation: mathematically it
        equals zero (the pinned RSS coincides with the unrestricted RSS at
        phi0 = eta_hat), but the reparametrised pinned LS solver may leave
        a small positive residual that, after studentisation, can exceed
        the critical value and break the bracket.
        """
        center = float(np.clip(center, lo_feas, hi_feas))
        T_center = max(T(center), 0.0)

        def G(p: float) -> float:
            return T(p) - T_center - crit

        if center <= lo_feas + tol:
            lo = lo_feas
        elif G(lo_feas) <= 0:
            lo = lo_feas
        else:
            lo = brentq(G, lo_feas, center, xtol=tol)

        if center >= hi_feas - tol:
            hi = hi_feas
        elif G(hi_feas) <= 0:
            hi = hi_feas
        else:
            hi = brentq(G, center, hi_feas, xtol=tol)

        return np.array([lo, hi])

    def get_cdf(self) -> Callable[[np.ndarray, bool], float]:
        """Return a CDF function estimated from the support and probability mass.

        Returns
        -------
        cdf : Callable
            ``cdf(b, constrained=True)`` evaluates the estimated CDF at point
            ``b`` by summing the probability mass of all support points that
            are component-wise at most ``b``.
        """
        theta_unconstrained = self.theta['unconstrained']
        theta_constrained = self.theta['constrained']

        def cdf(b: np.ndarray, constrained: bool = True) -> float:
            theta = theta_constrained if constrained else theta_unconstrained
            mask = (self.beta_support <= b).all(axis=1)
            return theta[mask].sum()

        return cdf
