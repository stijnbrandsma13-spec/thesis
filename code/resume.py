"""Resume an interrupted simulation run, computing only the missing cells.

``main.py`` always starts a fresh ``run_<timestamp>`` directory and recomputes
every (DGP, N, R) block. When a long run is interrupted, this script picks the
existing run back up instead: it replays the *exact* same deterministic seed
chain (``SeedSequence(SEED).spawn(total)`` -> per-block ``.spawn(REPETITIONS)``),
skips any block whose two per-alpha ``.npz`` cells are already on disk, and
computes only what is missing. Because the seeding is position-based, the
resumed blocks are bit-identical to what an uninterrupted run would have
produced.

Usage (from the project root)::

    python code/resume.py                       # resume the latest run
    python code/resume.py output/results/run_20260612_014757   # a specific run
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import main as m
from utils.results_io import (
    alpha_tag,
    latest_run_dir,
    load_run,
    save_result,
    write_summary_csv,
)


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _cell_files(run_dir: Path, dgp_name: str, N: int, R: int) -> list[Path]:
    return [
        run_dir / f"{dgp_name}_N{N}_R{R}_alpha{alpha_tag(a)}.npz"
        for a in m.ALPHAS
    ]


def resume(run_dir: Path) -> None:
    _setup_logging()
    logging.info("Resuming run in %s", run_dir)

    # Identical seed chain to main(): position-based, so block idx p always
    # draws block_seeds[p] regardless of which blocks ran before.
    root_seed = np.random.SeedSequence(m.SEED)
    total = len(m.BETA_DISTRIBUTIONS) * len(m.N_SET) * len(m.R_SET)
    block_seeds = root_seed.spawn(total)

    computed = 0
    skipped = 0
    pos = 0
    with ProcessPoolExecutor(
        max_workers=m.MAX_WORKERS,
        initializer=m._limit_worker_threads,
    ) as executor:
        for beta_dist in m.BETA_DISTRIBUTIONS:
            for N in m.N_SET:
                for R in m.R_SET:
                    idx = pos
                    pos += 1
                    dgp_name = beta_dist.__name__

                    if all(f.exists() for f in _cell_files(run_dir, dgp_name, N, R)):
                        skipped += 1
                        logging.info(
                            "[%d/%d] %-45s  N=%6d  R=%4d  SKIP (on disk)",
                            idx + 1, total, dgp_name, N, R,
                        )
                        continue

                    rep_seeds = block_seeds[idx].spawn(m.REPETITIONS)
                    logging.info(
                        "[%d/%d] %-45s  N=%6d  R=%4d  starting (%d reps)",
                        idx + 1, total, dgp_name, N, R, m.REPETITIONS,
                    )
                    block, estimates = m.run_simulation_block(
                        beta_dist, N, R, rep_seeds, executor,
                        collect_estimates=True,
                    )
                    if estimates is not None:
                        m.plot_estimated_distributions(
                            dgp_name, block[0].full_grid, estimates, N, R,
                        )
                    for r in block:
                        save_result(r, run_dir)
                    computed += 1
                    summary = "  ".join(
                        f"a={r.alpha:.2f}: "
                        + " / ".join(
                            f"{mm}={r.coverages[mm]['theta']:.3f}" for mm in m.METHODS
                        )
                        for r in block
                    )
                    logging.info(
                        "[%d/%d] %-45s  N=%6d  R=%4d  done (%.1fs)  theta: %s",
                        idx + 1, total, dgp_name, N, R, block[0].elapsed_s, summary,
                    )

    logging.info(
        "Resume finished: %d blocks computed, %d skipped (already on disk).",
        computed, skipped,
    )

    # Rebuild every derived artefact from the full set of cells on disk, so the
    # summary CSV / log / tables / figures reflect the complete run.
    manifest, results = load_run(run_dir)
    write_summary_csv(results, run_dir / "coverage_summary.csv")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    m.write_log(results, m.LOGS_DIR / f"simulation_resumed_{ts}.log")

    try:
        from postprocess import generate_figures, generate_tables
        logging.info("Generating tables and figures from %s", run_dir)
        generate_tables(
            results,
            manifest={
                "repetitions": m.REPETITIONS,
                "methods": list(m.METHODS),
                "dgps": [d.__name__ for d in m.BETA_DISTRIBUTIONS],
            },
            tables_dir=m.TABLES_DIR,
        )
        generate_figures(results, figures_dir=m.COVERAGE_FIGURES_DIR)
    except Exception:  # noqa: BLE001 - postprocess is best-effort; cells are safe
        logging.exception("Postprocess (tables/figures) failed; results are saved.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_dir()
    resume(target)
