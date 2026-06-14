"""Persist and reload Monte Carlo simulation results.

Each simulation run gets its own timestamped directory under
``output/results``::

    output/results/run_20260611_213000/
        manifest.json          # run-level parameters (seed, reps, grids, ...)
        coverage_summary.csv   # flat per-cell summary (easy to eyeball / import)
        beta_bimodal_support_probs_N1000_R25_alpha0p05.npz
        ...

One ``.npz`` file holds everything needed to recreate the plots and tables
for one (DGP, N, R, alpha) cell: the support grid, the true probabilities,
and the per-repetition CI hit indicators for every method and quantity.
Aggregates (coverage, per-point coverage) are stored too, but they are
derivable from the raw hits, so new statistics (e.g. MC standard errors)
can be computed later without re-running the simulation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "output" / "results"


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------

# Confidence level for the CIs *on the empirical coverage estimates* (fixed,
# independent of the simulation's alpha). The bootstrap seed is fixed so that
# re-running postprocess on a saved run reproduces identical intervals.
COVERAGE_CI_LEVEL = 0.95
N_BOOTSTRAP = 9_999
BOOTSTRAP_SEED = 20260612


def acd(coverage: float, alpha: float) -> float:
    """Absolute Coverage Deviation: coverage minus nominal level."""
    return coverage - (1 - alpha)


def rcd(coverage: float, alpha: float) -> float:
    """Relative Coverage Deviation: ACD as a fraction of the nominal level."""
    return (coverage - (1 - alpha)) / (1 - alpha)


def wilson_ci(
    hits: np.ndarray, level: float = COVERAGE_CI_LEVEL
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion from iid hit indicators."""
    from scipy.stats import norm

    hits = np.asarray(hits, dtype=float)
    n = hits.shape[0]
    p = hits.mean()
    z = norm.ppf(1 - (1 - level) / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float(centre - half), float(centre + half))


def bootstrap_coverage_ci(
    theta_hits: np.ndarray,
    level: float = COVERAGE_CI_LEVEL,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the doubly averaged theta coverage.

    The reported theta coverage averages the (reps, R) hit matrix over both
    repetitions and grid points. Hits are correlated across grid points
    within a repetition, so a binomial interval is invalid; resampling whole
    repetitions (rows) preserves that dependence (a cluster bootstrap with
    the repetition as the resampling unit).
    """
    per_rep = np.asarray(theta_hits, dtype=float).mean(axis=1)
    reps = per_rep.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, reps, size=(n_boot, reps))
    boot_means = per_rep[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def coverage_ci(hits: np.ndarray) -> tuple[float, float, str]:
    """95% CI on an empirical coverage estimate from per-repetition hits.

    1-D hits (scalar functionals, iid Bernoulli across repetitions) get a
    Wilson interval; 2-D hits (theta: reps x grid points) get the
    repetition-level percentile bootstrap.
    """
    hits = np.asarray(hits)
    if hits.ndim == 1:
        lo, hi = wilson_ci(hits)
        return (lo, hi, "wilson")
    lo, hi = bootstrap_coverage_ci(hits)
    return (lo, hi, "bootstrap")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    dgp_name: str
    alpha: float
    N: int
    R: int
    coverages: dict[str, dict[str, float]]  # method -> quantity -> coverage
    elapsed_s: float
    full_grid: np.ndarray
    true_probs: np.ndarray
    theta_coverage_per_point: dict[str, np.ndarray]  # method -> (R,)
    # method -> quantity -> per-repetition hit indicators; 'theta' is
    # (reps, R), scalar functionals are (reps,). Optional so old call sites
    # that only need aggregates keep working.
    rep_hits: dict[str, dict[str, np.ndarray]] | None = None


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def result_filename(result: SimulationResult) -> str:
    return (
        f"{result.dgp_name}_N{result.N}_R{result.R}"
        f"_alpha{alpha_tag(result.alpha)}.npz"
    )


def save_result(result: SimulationResult, run_dir: Path) -> Path:
    """Write one (DGP, N, R, alpha) cell to a compressed .npz in run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "full_grid": np.asarray(result.full_grid, dtype=float),
        "true_probs": np.asarray(result.true_probs, dtype=float),
    }
    # Double-underscore separators: method names themselves contain single
    # underscores (e.g. 'cp_wald'), so a single '_' would not split cleanly.
    for m, cov in result.theta_coverage_per_point.items():
        arrays[f"theta_coverage_per_point__{m}"] = np.asarray(cov, dtype=float)
    for m, per_quantity in (result.rep_hits or {}).items():
        for q, hits in per_quantity.items():
            arrays[f"rep_hits__{m}__{q}"] = np.asarray(hits, dtype=bool)
    meta = {
        "dgp_name": result.dgp_name,
        "alpha": result.alpha,
        "N": result.N,
        "R": result.R,
        "elapsed_s": result.elapsed_s,
        "coverages": result.coverages,
    }
    arrays["meta"] = np.array(json.dumps(meta))

    path = run_dir / result_filename(result)
    np.savez_compressed(path, **arrays)
    return path


def load_result(path: Path) -> SimulationResult:
    with np.load(path) as data:
        meta = json.loads(data["meta"].item())
        theta_coverage_per_point: dict[str, np.ndarray] = {}
        rep_hits: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            if key.startswith("theta_coverage_per_point__"):
                method = key.split("__", 1)[1]
                theta_coverage_per_point[method] = data[key]
            elif key.startswith("rep_hits__"):
                _, method, quantity = key.split("__")
                rep_hits.setdefault(method, {})[quantity] = data[key]
        return SimulationResult(
            dgp_name=meta["dgp_name"],
            alpha=meta["alpha"],
            N=meta["N"],
            R=meta["R"],
            coverages=meta["coverages"],
            elapsed_s=meta["elapsed_s"],
            full_grid=data["full_grid"],
            true_probs=data["true_probs"],
            theta_coverage_per_point=theta_coverage_per_point,
            rep_hits=rep_hits or None,
        )


def write_manifest(run_dir: Path, params: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )


def load_run(run_dir: Path) -> tuple[dict, list[SimulationResult]]:
    """Load the manifest and every result cell saved in a run directory."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    results = [load_result(p) for p in sorted(run_dir.glob("*.npz"))]
    results.sort(key=lambda r: (r.dgp_name, r.N, r.R, r.alpha))
    return manifest, results


def latest_run_dir(results_root: Path = RESULTS_ROOT) -> Path:
    """Most recent run directory (run_* names sort chronologically)."""
    runs = sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {results_root}")
    return runs[-1]


# ---------------------------------------------------------------------------
# Flat CSV summary
# ---------------------------------------------------------------------------

def write_summary_csv(results: list[SimulationResult], path: Path) -> None:
    """Rewrite the flat per-cell coverage summary from scratch.

    One row per (DGP, alpha, N, R, method, quantity). ``mc_se`` is the Monte
    Carlo standard error of the coverage estimate (std of the per-repetition
    coverage over sqrt(reps)). ``ci95_lo``/``ci95_hi`` bound the 95% CI on
    the coverage estimate itself (``ci_method``: Wilson for the scalar
    functionals, repetition-level percentile bootstrap for theta). All four
    are empty when per-repetition hits were not saved.
    """
    fieldnames = [
        "dgp", "alpha", "N", "R", "method", "quantity",
        "nominal", "coverage", "acd", "rcd", "mc_se",
        "ci95_lo", "ci95_hi", "ci_method", "repetitions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            for method, per_quantity in r.coverages.items():
                for quantity, coverage in per_quantity.items():
                    mc_se = ""
                    reps = ""
                    ci_lo = ci_hi = ci_method = ""
                    hits = (r.rep_hits or {}).get(method, {}).get(quantity)
                    if hits is not None:
                        lo, hi, ci_method = coverage_ci(hits)
                        ci_lo, ci_hi = f"{lo:.6f}", f"{hi:.6f}"
                        per_rep = np.asarray(hits, dtype=float)
                        if per_rep.ndim == 2:  # theta: average over points
                            per_rep = per_rep.mean(axis=1)
                        reps = per_rep.shape[0]
                        mc_se = f"{per_rep.std(ddof=1) / np.sqrt(reps):.6f}"
                    writer.writerow({
                        "dgp": r.dgp_name,
                        "alpha": r.alpha,
                        "N": r.N,
                        "R": r.R,
                        "method": method,
                        "quantity": quantity,
                        "nominal": f"{1 - r.alpha:.4f}",
                        "coverage": f"{coverage:.6f}",
                        "acd": f"{acd(coverage, r.alpha):+.6f}",
                        "rcd": f"{rcd(coverage, r.alpha):+.6f}",
                        "mc_se": mc_se,
                        "ci95_lo": ci_lo,
                        "ci95_hi": ci_hi,
                        "ci_method": ci_method,
                        "repetitions": reps,
                    })
