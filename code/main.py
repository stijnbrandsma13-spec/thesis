from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import chain, repeat
from pathlib import Path
from typing import Callable

import numpy as np

from utils.dgp import (
    DataGenerator,
    make_heiss_x_sampler,
    beta_almost_single_type_support_probs,
    beta_bimodal_support_probs,
    beta_four_types_support_probs,
    beta_single_type_support_probs,
    beta_strictly_uniform_support_probs,
    beta_tight_normal_support_probs,
    beta_wide_normal_support_probs,
)
from utils.estimators import FKRBEstimator
from utils.results_io import (
    SimulationResult,
    acd,
    coverage_ci,
    rcd,
    save_result,
    write_manifest,
    write_summary_csv,
)

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

# Diagnostic snapshots: in every (N, R) block the first repetition's estimated
# weights are plotted as PMF/CDF bar charts for both the unconstrained OLS and
# the constrained FKRB estimator, so they line up with the true PMF/CDF figures
# (which are produced for each R).

# Worker processes for the embarrassingly-parallel Monte Carlo repetitions.
# Physical cores, not SMT threads: the per-repetition work is FP-heavy and
# gains little from hyperthreading.
MAX_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Order matters: it fixes the column order of the coverage tables (via the
# run manifest) and matches the order of presentation in the paper.
BETA_DISTRIBUTIONS: tuple[Callable, ...] = (
    beta_bimodal_support_probs,
    beta_tight_normal_support_probs,
    beta_wide_normal_support_probs,
    beta_strictly_uniform_support_probs,
    beta_almost_single_type_support_probs,
    beta_single_type_support_probs,
    beta_four_types_support_probs,
)

METHODS = ('fkrb', 'cp_wald', 'qlr')
METHOD_LABELS = {
    'fkrb':    'FKRB (OLS centre, OLS-residual sandwich, clipped)',
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
# Figures are grouped by role: true PMF/CDF of the DGPs, the estimated (OLS /
# FKRB) PMF/CDF, and the per-point coverage maps.
ESTIMATED_FIGURES_DIR = FIGURES_DIR / "estimated"
COVERAGE_FIGURES_DIR = FIGURES_DIR / "coverage"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
PROGRESS_LOG = OUTPUT_DIR / "run_progress.log"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean_functional(support: np.ndarray, probs: np.ndarray) -> float:
    return float(support @ probs)


def _in_interval(value: float, ci: np.ndarray) -> bool:
    return bool((value >= ci[0]) & (value <= ci[1]))


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

_THREAD_LIMITER = None


def _limit_worker_threads() -> None:
    """Worker-process initializer: pin BLAS thread pools to one thread.

    Each worker is one parallel lane; letting every worker spin up its own
    multi-threaded BLAS pool would oversubscribe the CPU. The controller is
    kept alive in a module global so the limit persists for the worker's
    lifetime.
    """
    global _THREAD_LIMITER
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _run_one_repetition(
    seed_seq: np.random.SeedSequence,
    beta_support_probs: Callable,
    N: int,
    R: int,
    return_estimates: bool = False,
    estimates_only: bool = False,
) -> dict:
    """Run a single Monte Carlo repetition; return CI hit indicators.

    Executed in a worker process. Draws a dataset from an independent RNG
    stream seeded by ``seed_seq``, fits the FKRB estimator once, and checks
    every METHODS x QUANTITIES x ALPHAS confidence interval against the
    truth. Returns ``hits[alpha][method]`` with a boolean array of shape
    (R,) under ``'theta'`` and scalars under ``'mean_1'``/``'mean_2'``.
    With ``return_estimates=True``, the estimated weight vectors are added
    under ``'estimates'`` (keys ``'ols'`` and ``'fkrb'``) for diagnostic
    PMF/CDF plots. ``estimates_only=True`` returns just that ``'estimates'``
    dict and skips every CI computation -- a fast path for regenerating the
    diagnostic figures.
    """
    rng = np.random.default_rng(seed_seq)
    full_grid, support, true_probs = beta_support_probs(R)
    support_first = full_grid[:, 0]
    support_second = full_grid[:, 1]
    true_mean_first = _mean_functional(support_first, true_probs)
    true_mean_second = _mean_functional(support_second, true_probs)

    # mean(beta_k) = support_k . probs is linear in probs, so its delta-method
    # gradient is exactly support_k (constant) and its feasible range over the
    # simplex is [min(support_k), max(support_k)] (attained at the vertices).
    mean_1_clip = (float(support_first.min()), float(support_first.max()))
    mean_2_clip = (float(support_second.min()), float(support_second.max()))

    generator = DataGenerator(N=N, J=J, K=K, rng=rng)
    x_sampler = make_heiss_x_sampler(rng)
    y, x = generator.generate(
        x_sampler=x_sampler,
        beta_support=support,
        beta_probs=true_probs,
    )

    estimator = FKRBEstimator(
        beta_support=full_grid,
        optimal_weighting=OPTIMAL_WEIGHTING,
    )
    estimator.estimate(y=y, x=x)

    if estimates_only:
        # Diagnostic-figure fast path: the PMF/CDF bar charts only need the
        # weight vectors, so skip the (costly, R-heavy) QLR/Wald CI inversions.
        return {'estimates': {
            'ols': estimator.theta['unconstrained'],
            'fkrb': estimator.theta['constrained'],
        }}

    # --- Sieve QLR (studentised) ---
    # The QLR statistic is alpha-independent, so invert once for all alphas;
    # the setup (RSS, scale, memoised T evaluations) is shared internally.
    qlr_theta = estimator.qlr_ci(alpha=ALPHAS, tol=QLR_TOL)
    qlr_m1 = estimator.qlr_ci(c=support_first, alpha=ALPHAS, tol=QLR_TOL)
    qlr_m2 = estimator.qlr_ci(c=support_second, alpha=ALPHAS, tol=QLR_TOL)

    hits: dict = {a: {m: {} for m in METHODS} for a in ALPHAS}
    for alpha in ALPHAS:
        # --- FKRB CI (OLS centre, OLS-residual sandwich, clipped) ---
        ci_fkrb = estimator.get_confidence_interval(alpha=alpha)
        _, ci_fkrb_m1 = estimator.plug_in_estimate_functional(
            lambda probs, s=support_first: _mean_functional(s, probs),
            derivative=lambda probs, s=support_first: s,
            clip_lower=mean_1_clip[0], clip_upper=mean_1_clip[1],
            ci=True, alpha=alpha,
        )
        _, ci_fkrb_m2 = estimator.plug_in_estimate_functional(
            lambda probs, s=support_second: _mean_functional(s, probs),
            derivative=lambda probs, s=support_second: s,
            clip_lower=mean_2_clip[0], clip_upper=mean_2_clip[1],
            ci=True, alpha=alpha,
        )

        # --- CP Wald (constrained-residual sandwich, no clip) ---
        ci_cp = estimator.cp_wald_ci(alpha=alpha)
        _, ci_cp_m1 = estimator.cp_wald_ci(c=support_first, alpha=alpha)
        _, ci_cp_m2 = estimator.cp_wald_ci(c=support_second, alpha=alpha)

        ci_qlr = qlr_theta[alpha]
        _, ci_qlr_m1 = qlr_m1[alpha]
        _, ci_qlr_m2 = qlr_m2[alpha]

        hits[alpha]['fkrb']['theta'] = (
            (true_probs >= ci_fkrb[:, 0]) & (true_probs <= ci_fkrb[:, 1])
        )
        hits[alpha]['fkrb']['mean_1'] = _in_interval(true_mean_first, ci_fkrb_m1)
        hits[alpha]['fkrb']['mean_2'] = _in_interval(true_mean_second, ci_fkrb_m2)

        hits[alpha]['cp_wald']['theta'] = (
            (true_probs >= ci_cp[:, 0]) & (true_probs <= ci_cp[:, 1])
        )
        hits[alpha]['cp_wald']['mean_1'] = _in_interval(true_mean_first, ci_cp_m1)
        hits[alpha]['cp_wald']['mean_2'] = _in_interval(true_mean_second, ci_cp_m2)

        hits[alpha]['qlr']['theta'] = (
            (true_probs >= ci_qlr[:, 0]) & (true_probs <= ci_qlr[:, 1])
        )
        hits[alpha]['qlr']['mean_1'] = _in_interval(true_mean_first, ci_qlr_m1)
        hits[alpha]['qlr']['mean_2'] = _in_interval(true_mean_second, ci_qlr_m2)

    if return_estimates:
        hits['estimates'] = {
            'ols': estimator.theta['unconstrained'],
            'fkrb': estimator.theta['constrained'],
        }

    return hits


def run_simulation_block(
    beta_support_probs: Callable,
    N: int,
    R: int,
    rep_seeds: list[np.random.SeedSequence],
    executor: ProcessPoolExecutor,
    collect_estimates: bool = False,
) -> tuple[list[SimulationResult], dict | None]:
    """Run REPETITIONS at (DGP, N, R) in parallel across worker processes.

    Each repetition draws from its own independent seed stream (spawned from
    the master SEED), fits once, and evaluates the CIs for every alpha.
    Returns one SimulationResult per alpha, plus — when
    ``collect_estimates=True`` — the first repetition's estimated weight
    vectors (``{'ols': ..., 'fkrb': ...}``) for diagnostic plotting.
    """
    full_grid, _, true_probs = beta_support_probs(R)

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

    rep_results = executor.map(
        _run_one_repetition,
        rep_seeds,
        repeat(beta_support_probs),
        repeat(N),
        repeat(R),
        chain([collect_estimates], repeat(False)),
    )
    estimates: dict | None = None
    for i, hits in enumerate(rep_results):
        if i == 0:
            estimates = hits.pop('estimates', None)
        for alpha in ALPHAS:
            for m in METHODS:
                th = hits[alpha][m]['theta']
                theta_hits[alpha][m][i] = th
                cov[alpha][m]['theta'][i] = np.mean(th)
                cov[alpha][m]['mean_1'][i] = hits[alpha][m]['mean_1']
                cov[alpha][m]['mean_2'][i] = hits[alpha][m]['mean_2']

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
        # Raw per-repetition hit indicators: everything saved to disk is
        # recomputable from these (coverage, per-point maps, MC std errors).
        rep_hits = {
            m: {
                'theta': theta_hits[alpha][m].astype(bool),
                'mean_1': cov[alpha][m]['mean_1'].astype(bool),
                'mean_2': cov[alpha][m]['mean_2'].astype(bool),
            }
            for m in METHODS
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
            rep_hits=rep_hits,
        ))
    return results, estimates


def plot_estimated_distributions(
    dgp_name: str,
    full_grid: np.ndarray,
    estimates: dict,
    N: int,
    R: int,
) -> None:
    """Save diagnostic PMF/CDF bar charts of the estimated weights.

    One PMF and one CDF figure per estimator (OLS and FKRB), in the same
    style as the true-distribution figures produced by
    ``utils/visualization.py``.
    """
    # Lazy import: workers re-import this module under Windows spawn, and
    # they have no business loading PyVista/VTK.
    from utils.visualization import plot_cdf_3D, plot_pmf_3D, pretty_dgp_name

    ESTIMATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    short_name = dgp_name.removesuffix('_support_probs')
    pretty = pretty_dgp_name(dgp_name)
    x_range = (float(full_grid[:, 0].min()), float(full_grid[:, 0].max()))
    y_range = (float(full_grid[:, 1].min()), float(full_grid[:, 1].max()))

    for est_name, probs in estimates.items():
        for kind, plot_fn in (('pmf', plot_pmf_3D), ('cdf', plot_cdf_3D)):
            path = ESTIMATED_FIGURES_DIR / f'{kind}_{est_name}_{short_name}_N{N}_R{R}.png'
            plot_fn(
                full_grid, probs,
                x_range=x_range, y_range=y_range,
                save_path=str(path), show=False,
                title=f'{est_name.upper()} estimated {kind.upper()} of {pretty}',
            )
            logging.info("Saved estimated-distribution figure %s", path.name)


def regenerate_estimated_distribution_figures(
    n_values: tuple[int, ...] = N_SET,
    r_values: tuple[int, ...] = R_SET,
    rep_index: int = 0,
) -> None:
    """Rebuild the diagnostic estimated-PMF/CDF figures without a full run.

    Replays the exact ``SeedSequence(SEED)`` block chain that :func:`main`
    uses, so for each DGP it reproduces repetition ``rep_index``'s OLS and FKRB
    weight estimates at every requested sample size in ``n_values`` and grid
    size in ``r_values``, then writes the bar charts via
    :func:`plot_estimated_distributions`. The figures therefore match the most
    recent full run for the same ``SEED`` and parameter sets, but cost a single
    fit per ``(DGP, N, R)`` instead of ``REPETITIONS``.
    """
    root_seed = np.random.SeedSequence(SEED)
    block_seeds = root_seed.spawn(len(BETA_DISTRIBUTIONS) * len(N_SET) * len(R_SET))
    block = 0
    for beta_dist in BETA_DISTRIBUTIONS:
        for N in N_SET:
            for R in R_SET:
                idx, block = block, block + 1
                if N not in n_values or R not in r_values:
                    continue
                rep_seeds = block_seeds[idx].spawn(REPETITIONS)
                out = _run_one_repetition(
                    rep_seeds[rep_index], beta_dist, N, R, estimates_only=True,
                )
                full_grid, _, _ = beta_dist(R)
                plot_estimated_distributions(
                    beta_dist.__name__, full_grid, out['estimates'], N, R,
                )


# ---------------------------------------------------------------------------
# Output / logging
# ---------------------------------------------------------------------------

def _format_coverage_row(
    label: str,
    coverage: float,
    alpha: float,
    hits: np.ndarray | None = None,
) -> str:
    nominal = 1 - alpha
    ci = ""
    if hits is not None:
        lo, hi, _ = coverage_ci(hits)
        ci = f"  CI95=[{lo:.4f}, {hi:.4f}]"
    return (
        f"  {label:<44}"
        f"  coverage={coverage:.4f}  (nominal {nominal:.4f})"
        f"  ACD={acd(coverage, alpha):+.4f}"
        f"  RCD={rcd(coverage, alpha):+.4f}"
        f"{ci}"
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
            hits = (result.rep_hits or {}).get(m, {}).get(q)
            rows.append(_format_coverage_row(
                label, result.coverages[m][q], result.alpha, hits,
            ))
        rows.append("")
    body = "\n".join(rows).rstrip()
    return f"{sep}\n{header}\n{sep}\n{body}"


def write_log(results: list[SimulationResult], log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dgp_names = "  ".join(d.__name__ for d in BETA_DISTRIBUTIONS)

    header = "\n".join([
        "=" * 104,
        "FKRB Simulation -- Coverage Report",
        "=" * 104,
        f"Generated   : {timestamp}",
        f"Seed        : {SEED} (per-repetition streams via SeedSequence.spawn)",
        f"Repetitions : {REPETITIONS}",
        f"Workers     : {MAX_WORKERS}",
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

    # Reproducible parallel seeding: each (DGP, N, R) block gets a child
    # SeedSequence, which in turn spawns one independent stream per
    # repetition. Results are identical for a given SEED regardless of
    # worker count or scheduling order.
    root_seed = np.random.SeedSequence(SEED)
    results: list[SimulationResult] = []
    total = len(BETA_DISTRIBUTIONS) * len(N_SET) * len(R_SET)
    block_seeds = root_seed.spawn(total)
    done = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"simulation_{timestamp}.log"
    run_dir = RESULTS_DIR / f"run_{timestamp}"

    write_manifest(run_dir, {
        "timestamp": timestamp,
        "seed": SEED,
        "repetitions": REPETITIONS,
        "workers": MAX_WORKERS,
        "J": J,
        "K": K,
        "N_set": list(N_SET),
        "R_set": list(R_SET),
        "alphas": list(ALPHAS),
        "dgps": [d.__name__ for d in BETA_DISTRIBUTIONS],
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "quantities": list(QUANTITIES),
        "quantity_labels": QUANTITY_LABELS,
        "optimal_weighting": OPTIMAL_WEIGHTING,
        "qlr_tol": QLR_TOL,
    })
    logging.info("Results will be stored in %s", run_dir)

    logging.info(
        "Starting simulation: %d (DGP, N, R) blocks x %d reps x %d alphas; "
        "%d methods; %d worker processes",
        total, REPETITIONS, len(ALPHAS), len(METHODS), MAX_WORKERS,
    )

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_limit_worker_threads,
    ) as executor:
        for beta_dist in BETA_DISTRIBUTIONS:
            for N in N_SET:
                for R in R_SET:
                    rep_seeds = block_seeds[done].spawn(REPETITIONS)
                    done += 1
                    logging.info(
                        "[%d/%d] %-45s  N=%6d  R=%4d  starting (%d reps)",
                        done, total, beta_dist.__name__, N, R, REPETITIONS,
                    )
                    collect = True
                    block, estimates = run_simulation_block(
                        beta_dist, N, R, rep_seeds, executor,
                        collect_estimates=collect,
                    )
                    if estimates is not None:
                        plot_estimated_distributions(
                            beta_dist.__name__, block[0].full_grid,
                            estimates, N, R,
                        )
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
                    # Persist incrementally after each block completes, so a
                    # crashed or interrupted run keeps everything finished
                    # so far (postprocess.py works on partial runs too).
                    write_log(results, log_path)
                    for r in block:
                        save_result(r, run_dir)
                    write_summary_csv(results, run_dir / "coverage_summary.csv")

    logging.info("Done. %d configurations completed.", total)

    # Figures and LaTeX tables are derived entirely from the saved results;
    # they can be regenerated any time with `python code/postprocess.py`.
    from postprocess import generate_figures, generate_tables
    logging.info("Generating figures and tables from %s", run_dir)
    generate_tables(results, manifest={"repetitions": REPETITIONS,
                                       "methods": list(METHODS),
                                       "dgps": [d.__name__ for d in BETA_DISTRIBUTIONS]},
                    tables_dir=TABLES_DIR)
    generate_figures(results, figures_dir=COVERAGE_FIGURES_DIR)


if __name__ == "__main__":
    main()
