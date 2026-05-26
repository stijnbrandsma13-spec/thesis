from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from utils.dgp import (
    DataGenerator,
    heiss_x_sampler,
    beta_bimodal_support_probs,
    beta_bivariate_normal_support_probs,
    beta_concentrated_spike_support_probs,
    beta_diffuse_support_probs,
    beta_discrete_uniform_support_probs,
)
from utils.estimators import FKRBEstimator
from utils.visualization import plot_coverage_grid

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we only save figures
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

J = 4
K = 2
N_SET = (1_000, 10_000)
R_SET = (25, 225)
ALPHAS = (0.05, 0.5)
REPETITIONS = 250
SEED = 42

OPTIMAL_WEIGHTING = True
QLR_TOL = 1e-3

BETA_DISTRIBUTIONS: tuple[Callable, ...] = (
    beta_bimodal_support_probs,
    beta_bivariate_normal_support_probs,
    beta_concentrated_spike_support_probs,
    beta_diffuse_support_probs,
    beta_discrete_uniform_support_probs,
)

METHODS = ('fkrb', 'cp_wald', 'qlr')
METHOD_LABELS = {
    'fkrb':    'FKRB (constrained, OLS-residual sandwich, clipped)',
    'cp_wald': 'CP Wald (constrained-residual sandwich, no clip)',
    'qlr':     'CP QLR (studentised)',
}
QUANTITIES = ('theta', 'mean_1', 'mean_2')
QUANTITY_LABELS = {
    'theta': 'theta (support points)',
    'mean_1': 'mean(beta_1)',
    'mean_2': 'mean(beta_2)',
}

OUTPUT_DIR = Path(__file__).parent.parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
PROGRESS_LOG = OUTPUT_DIR / "run_progress.log"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean_functional(support: np.ndarray, probs: np.ndarray) -> float:
    return float(support @ probs)


def _acd(coverage: float, alpha: float) -> float:
    return coverage - (1 - alpha)


def _rcd(coverage: float, alpha: float) -> float:
    return (coverage - (1 - alpha)) / (1 - alpha)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    dgp_name: str
    alpha: float
    N: int
    R: int
    coverages: dict[str, dict[str, float]]
    elapsed_s: float
    full_grid: np.ndarray
    true_probs: np.ndarray
    theta_coverage_per_point: dict[str, np.ndarray]  # method -> (R,)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def run_simulation_block(
    beta_support_probs: Callable,
    N: int,
    R: int,
    rng: np.random.Generator,
) -> list[SimulationResult]:
    """Run REPETITIONS at (DGP, N, R), reusing the data and fit across ALPHAS.

    Returns one SimulationResult per alpha. Sharing the FKRB fit across
    alphas roughly halves the wall time relative to recomputing it.
    """
    full_grid, support, true_probs = beta_support_probs(R)
    support_first = full_grid[:, 0]
    support_second = full_grid[:, 1]
    true_mean_first = _mean_functional(support_first, true_probs)
    true_mean_second = _mean_functional(support_second, true_probs)

    generator = DataGenerator(N=N, J=J, K=K, rng=rng)

    # cov[alpha][method][quantity] -> (REPETITIONS,)
    cov = {
        a: {m: {q: np.zeros(REPETITIONS) for q in QUANTITIES} for m in METHODS}
        for a in ALPHAS
    }
    theta_hits = {
        a: {m: np.zeros((REPETITIONS, full_grid.shape[0])) for m in METHODS}
        for a in ALPHAS
    }
    t0 = time.time()

    for i in range(REPETITIONS):
        generator.reset()
        y, x = generator.generate(
            x_sampler=heiss_x_sampler,
            beta_support=support,
            beta_probs=true_probs,
        )

        estimator = FKRBEstimator(
            beta_support=full_grid,
            optimal_weighting=OPTIMAL_WEIGHTING,
        )
        estimator.estimate(y=y, x=x)

        for alpha in ALPHAS:
            # --- FKRB CI (constrained centre, OLS-residual sandwich, clipped) ---
            ci_fkrb = estimator.get_confidence_interval(alpha=alpha)
            _, ci_fkrb_m1 = estimator.plug_in_estimate_functional(
                lambda probs, s=support_first: _mean_functional(s, probs),
                ci=True, alpha=alpha,
            )
            _, ci_fkrb_m2 = estimator.plug_in_estimate_functional(
                lambda probs, s=support_second: _mean_functional(s, probs),
                ci=True, alpha=alpha,
            )

            # --- CP Wald (constrained-residual sandwich, no clip) ---
            ci_cp = estimator.cp_wald_ci(alpha=alpha)
            _, ci_cp_m1 = estimator.cp_wald_ci(c=support_first, alpha=alpha)
            _, ci_cp_m2 = estimator.cp_wald_ci(c=support_second, alpha=alpha)

            # --- Sieve QLR (studentised) ---
            ci_qlr = estimator.qlr_ci(alpha=alpha, tol=QLR_TOL)
            _, ci_qlr_m1 = estimator.qlr_ci(c=support_first, alpha=alpha, tol=QLR_TOL)
            _, ci_qlr_m2 = estimator.qlr_ci(c=support_second, alpha=alpha, tol=QLR_TOL)

            fkrb_hits = (true_probs >= ci_fkrb[:, 0]) & (true_probs <= ci_fkrb[:, 1])
            theta_hits[alpha]['fkrb'][i] = fkrb_hits
            cov[alpha]['fkrb']['theta'][i] = np.mean(fkrb_hits)
            cov[alpha]['fkrb']['mean_1'][i] = (true_mean_first >= ci_fkrb_m1[0]) & (true_mean_first <= ci_fkrb_m1[1])
            cov[alpha]['fkrb']['mean_2'][i] = (true_mean_second >= ci_fkrb_m2[0]) & (true_mean_second <= ci_fkrb_m2[1])

            cp_hits = (true_probs >= ci_cp[:, 0]) & (true_probs <= ci_cp[:, 1])
            theta_hits[alpha]['cp_wald'][i] = cp_hits
            cov[alpha]['cp_wald']['theta'][i] = np.mean(cp_hits)
            cov[alpha]['cp_wald']['mean_1'][i] = (true_mean_first >= ci_cp_m1[0]) & (true_mean_first <= ci_cp_m1[1])
            cov[alpha]['cp_wald']['mean_2'][i] = (true_mean_second >= ci_cp_m2[0]) & (true_mean_second <= ci_cp_m2[1])

            qlr_hits = (true_probs >= ci_qlr[:, 0]) & (true_probs <= ci_qlr[:, 1])
            theta_hits[alpha]['qlr'][i] = qlr_hits
            cov[alpha]['qlr']['theta'][i] = np.mean(qlr_hits)
            cov[alpha]['qlr']['mean_1'][i] = (true_mean_first >= ci_qlr_m1[0]) & (true_mean_first <= ci_qlr_m1[1])
            cov[alpha]['qlr']['mean_2'][i] = (true_mean_second >= ci_qlr_m2[0]) & (true_mean_second <= ci_qlr_m2[1])

    elapsed = time.time() - t0

    results = []
    for alpha in ALPHAS:
        coverages = {
            m: {q: float(np.mean(cov[alpha][m][q])) for q in QUANTITIES}
            for m in METHODS
        }
        theta_coverage_per_point = {
            m: theta_hits[alpha][m].mean(axis=0) for m in METHODS
        }
        results.append(SimulationResult(
            dgp_name=beta_support_probs.__name__,
            alpha=alpha,
            N=N,
            R=R,
            coverages=coverages,
            elapsed_s=elapsed,
            full_grid=full_grid,
            true_probs=true_probs,
            theta_coverage_per_point=theta_coverage_per_point,
        ))
    return results


# ---------------------------------------------------------------------------
# Output / logging
# ---------------------------------------------------------------------------

def _format_coverage_row(label: str, coverage: float, alpha: float) -> str:
    nominal = 1 - alpha
    return (
        f"  {label:<44}"
        f"  coverage={coverage:.4f}  (nominal {nominal:.4f})"
        f"  ACD={_acd(coverage, alpha):+.4f}"
        f"  RCD={_rcd(coverage, alpha):+.4f}"
    )


def _format_result(result: SimulationResult) -> str:
    sep = "-" * 104
    header = (
        f"DGP : {result.dgp_name}\n"
        f"alpha={result.alpha}   N={result.N:,}   R={result.R}"
        f"   ({result.elapsed_s:.1f}s for both alphas combined)"
    )
    rows = []
    for q in QUANTITIES:
        for m in METHODS:
            label = f"[{m:>7}] {QUANTITY_LABELS[q]}"
            rows.append(_format_coverage_row(label, result.coverages[m][q], result.alpha))
        rows.append("")
    body = "\n".join(rows).rstrip()
    return f"{sep}\n{header}\n{sep}\n{body}"


def save_coverage_figures(result: SimulationResult, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = f"{result.alpha:.2f}".replace(".", "p")
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        plot_coverage_grid(
            grid=result.full_grid,
            coverage_per_point=result.theta_coverage_per_point[method],
            alpha=result.alpha,
            true_probs=result.true_probs,
            title=(
                f"{result.dgp_name}  [{method}]  "
                f"$1-\\alpha={1 - result.alpha:.2f}$  $N={result.N}$  $R={result.R}$"
            ),
            ax=ax,
        )
        fname = (
            f"coverage_{result.dgp_name}_alpha{alpha_tag}"
            f"_N{result.N}_R{result.R}_{method}.png"
        )
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        plt.close(fig)


def write_log(results: list[SimulationResult], log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dgp_names = "  ".join(d.__name__ for d in BETA_DISTRIBUTIONS)

    header = "\n".join([
        "=" * 104,
        "FKRB Simulation -- Coverage Report",
        "=" * 104,
        f"Generated   : {timestamp}",
        f"Seed        : {SEED}",
        f"Repetitions : {REPETITIONS}",
        f"N values    : {N_SET}",
        f"R values    : {R_SET}",
        f"Alpha       : {ALPHAS}",
        f"DGPs        : {dgp_names}",
        f"Methods     : " + "; ".join(f"{m} ({METHOD_LABELS[m]})" for m in METHODS),
        f"Weighting   : {'optimal (feasible GLS)' if OPTIMAL_WEIGHTING else 'identity'}",
        f"QLR tol     : {QLR_TOL}",
        "=" * 104,
    ])

    body = "\n\n".join(_format_result(r) for r in results)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"{header}\n\n{body}\n", encoding="utf-8")
    logging.info("Log written to %s", log_path)


def _setup_logging() -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")  # truncate prior run

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear any pre-existing handlers (e.g. from jupyter re-runs)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(PROGRESS_LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    rng = np.random.default_rng(SEED)
    results: list[SimulationResult] = []
    total = len(BETA_DISTRIBUTIONS) * len(N_SET) * len(R_SET)
    done = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"simulation_{timestamp}.log"

    logging.info(
        "Starting simulation: %d (DGP, N, R) blocks x %d reps x %d alphas; %d methods",
        total, REPETITIONS, len(ALPHAS), len(METHODS),
    )

    for beta_dist in BETA_DISTRIBUTIONS:
        for N in N_SET:
            for R in R_SET:
                done += 1
                logging.info(
                    "[%d/%d] %-45s  N=%6d  R=%4d  starting (%d reps)",
                    done, total, beta_dist.__name__, N, R, REPETITIONS,
                )
                block = run_simulation_block(beta_dist, N, R, rng)
                results.extend(block)
                summary = "  ".join(
                    f"a={r.alpha:.2f}: "
                    + " / ".join(
                        f"{m}={r.coverages[m]['theta']:.3f}" for m in METHODS
                    )
                    for r in block
                )
                logging.info(
                    "[%d/%d] %-45s  N=%6d  R=%4d  done (%.1fs)  theta: %s",
                    done, total, beta_dist.__name__, N, R, block[0].elapsed_s,
                    summary,
                )
                # Write log incrementally after each block completes.
                write_log(results, log_path)
                for r in block:
                    save_coverage_figures(r, FIGURES_DIR)

    logging.info("Done. %d configurations completed.", total)


if __name__ == "__main__":
    main()
