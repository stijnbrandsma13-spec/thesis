"""Regenerate coverage figures and LaTeX tables from saved simulation results.

Reads the .npz result files written by main.py (see utils/results_io.py) and
produces, without re-running the simulation:

  - per-point coverage maps      -> output/figures/coverage_*.png
  - the paper's coverage tables  -> output/tables/coverage_theta.tex,
                                    coverage_mean_1.tex, coverage_mean_2.tex

The tables are complete LaTeX table environments (caption, label, notes)
matching the format in paper/sections/results.tex, so they can be pulled in
with \\input{../output/tables/coverage_theta} or copied into the section.

Usage (from the repo root):
    python code/postprocess.py                  # latest run in output/results
    python code/postprocess.py output/results/run_20260611_213000
    python code/postprocess.py --tables-only    # skip the (slower) figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.results_io import (
    N_BOOTSTRAP,
    SimulationResult,
    coverage_ci,
    latest_run_dir,
    load_run,
    write_summary_csv,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
# Coverage maps live in their own subfolder, alongside figures/true and
# figures/estimated written by visualization.py and main.py respectively.
COVERAGE_FIGURES_DIR = FIGURES_DIR / "coverage"
TABLES_DIR = OUTPUT_DIR / "tables"

# Display names in the paper's column order (manifest order is the same).
DGP_DISPLAY = {
    "beta_bimodal_support_probs": "Bimodal",
    "beta_tight_normal_support_probs": "Tight normal",
    "beta_wide_normal_support_probs": "Wide normal",
    "beta_strictly_uniform_support_probs": "Strictly uniform",
    "beta_almost_single_type_support_probs": "Almost single type",
    "beta_single_type_support_probs": "Single type",
    "beta_four_types_support_probs": "Four types",
}
METHOD_DISPLAY = {
    "fkrb": "FKRB",
    "cp_wald": "Wald",
    "qlr": "QLR",
}

_NOTES_SHARED = (
    r"FKRB denotes the interval proposed by \citet{FoxKimRyanBajari2011}, "
    r"see \cref{@FKRBEQ@}. Wald is the sieve Wald interval from "
    r"\citet{ChenPouzo2015}, see \cref{eq:CIChenPouzo}, and QLR is the "
    r"(studentized) sieve quasi-likelihood ratio interval, see "
    r"\cref{eq:CIQLR}. The nominal coverage is $1-\alpha$; coverage above "
    r"this level reflects conservativeness."
)

# How the 95% CI on the empirical coverage estimate itself is obtained;
# referenced in the per-table notes below.
_CI_NOTE_BOOTSTRAP = (
    r"Brackets contain a 95\% percentile-bootstrap confidence interval for "
    r"the empirical coverage, obtained by resampling the "
    rf"$@REPS@$ Monte Carlo repetitions with replacement ({N_BOOTSTRAP} "
    r"bootstrap samples); resampling whole repetitions preserves the "
    r"dependence of the hit indicators across the support points that are "
    r"averaged over."
)
_CI_NOTE_WILSON = (
    r"Brackets contain a 95\% Wilson score confidence interval for the "
    r"empirical coverage."
)

TABLE_SPECS = {
    "theta": {
        "filename": "coverage_theta.tex",
        "label": "tab:coverage",
        "caption": (
            r"Empirical coverage of the original FKRB "
            r"\citep{FoxKimRyanBajari2011}, and sieve Wald and sieve QLR "
            r"\citep{ChenPouzo2015} confidence intervals for the "
            r"support-point probabilities $\theta_r$, by data generating "
            r"process, sample size $N$, grid size $R$ and significance "
            r"level $\alpha$."
        ),
        "notes": (
            r"\textit{Notes:} Each cell reports the empirical coverage "
            r"probability of the corresponding confidence interval, computed "
            r"across $@REPS@$ Monte Carlo repetitions of the FKRB estimator "
            r"on data generated from the seven DGPs of \Cref{sec:dgp} and "
            r"averaged over the support points $\theta_r$ on the "
            r"prespecified grid $\mathcal{B}$. "
            + _CI_NOTE_BOOTSTRAP + " "
            + _NOTES_SHARED.replace("@FKRBEQ@", "eq:CIFKRB")
        ),
    },
    "mean_1": {
        "filename": "coverage_mean_1.tex",
        "label": "tab:coverage_Eb1",
        "caption": (
            r"Empirical coverage of the original FKRB "
            r"\citep{FoxKimRyanBajari2011}, and sieve Wald and sieve QLR "
            r"\citep{ChenPouzo2015} confidence intervals for "
            r"$\mathbb{E}[\beta_1]$, by data generating process, sample "
            r"size $N$, grid size $R$ and significance level $\alpha$."
        ),
        "notes": (
            r"\textit{Notes:} Each cell reports the empirical coverage "
            r"probability of the corresponding confidence interval for the "
            r"linear functional $\mathbb{E}[\beta_1] = \sum_r \theta_r "
            r"\beta_1^{(r)}$, computed across $@REPS@$ Monte Carlo "
            r"repetitions of the FKRB estimator on data generated from the "
            r"seven DGPs of \Cref{sec:dgp}. "
            + _CI_NOTE_WILSON + " "
            + _NOTES_SHARED.replace("@FKRBEQ@", "eq:CIfunctional")
        ),
    },
    "mean_2": {
        "filename": "coverage_mean_2.tex",
        "label": "tab:coverage_Eb2",
        "caption": (
            r"Empirical coverage of the original FKRB "
            r"\citep{FoxKimRyanBajari2011}, and sieve Wald and sieve QLR "
            r"\citep{ChenPouzo2015} confidence intervals for "
            r"$\mathbb{E}[\beta_2]$, by data generating process, sample "
            r"size $N$, grid size $R$ and significance level $\alpha$."
        ),
        "notes": (
            r"\textit{Notes:} Each cell reports the empirical coverage "
            r"probability of the corresponding confidence interval for the "
            r"linear functional $\mathbb{E}[\beta_2] = \sum_r \theta_r "
            r"\beta_2^{(r)}$, computed across $@REPS@$ Monte Carlo "
            r"repetitions of the FKRB estimator on data generated from the "
            r"seven DGPs of \Cref{sec:dgp}. "
            + _CI_NOTE_WILSON + " "
            + _NOTES_SHARED.replace("@FKRBEQ@", "eq:CIfunctional")
        ),
    },
}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_coverage_figures(result: SimulationResult, figures_dir: Path) -> int:
    """Save one per-point coverage map per method; returns number written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.visualization import plot_coverage_grid, pretty_dgp_name

    figures_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.alpha:.2f}".replace(".", "p")
    pretty = pretty_dgp_name(result.dgp_name)
    for method, coverage in result.theta_coverage_per_point.items():
        label = METHOD_DISPLAY.get(method, method)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        plot_coverage_grid(
            grid=result.full_grid,
            coverage_per_point=coverage,
            alpha=result.alpha,
            true_probs=result.true_probs,
            title=f"Empirical coverage of {label} CI for {pretty}",
            subtitle=(
                f"$1 - \\alpha = {1 - result.alpha:.2f}$, "
                f"$N = {result.N}$, $R = {result.R}$"
            ),
            ax=ax,
        )
        fname = (
            f"coverage_{result.dgp_name}_alpha{tag}"
            f"_N{result.N}_R{result.R}_{method}.png"
        )
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return len(result.theta_coverage_per_point)


def generate_figures(
    results: list[SimulationResult], figures_dir: Path = COVERAGE_FIGURES_DIR
) -> int:
    n = sum(save_coverage_figures(r, figures_dir) for r in results)
    print(f"Wrote {n} coverage figures to {figures_dir}")
    return n


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _latex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def make_coverage_table(
    results: list[SimulationResult],
    quantity: str,
    manifest: dict,
) -> str:
    """Render one coverage table (full LaTeX table environment) as a string."""
    spec = TABLE_SPECS[quantity]

    methods = manifest.get("methods") or list(results[0].coverages)
    dgps_present = {r.dgp_name for r in results}
    dgps = [d for d in (manifest.get("dgps") or DGP_DISPLAY) if d in dgps_present]
    dgps += sorted(dgps_present - set(dgps))  # any DGP missing from manifest

    cells: dict[tuple, float] = {}
    cells_ci: dict[tuple, tuple[float, float]] = {}
    for r in results:
        for m in methods:
            cells[(r.alpha, r.N, r.R, r.dgp_name, m)] = r.coverages[m][quantity]
            hits = (r.rep_hits or {}).get(m, {}).get(quantity)
            if hits is not None:
                lo, hi, _ = coverage_ci(hits)
                cells_ci[(r.alpha, r.N, r.R, r.dgp_name, m)] = (lo, hi)
    configs = sorted({(r.alpha, r.N, r.R) for r in results})

    reps = manifest.get("repetitions")
    if reps is None and results[0].rep_hits:
        first = next(iter(next(iter(results[0].rep_hits.values())).values()))
        reps = first.shape[0]

    n_cols = 3 + len(dgps) * len(methods)
    n_methods = len(methods)

    dgp_header = " & ".join(
        rf"\multicolumn{{{n_methods}}}{{c}}{{{DGP_DISPLAY.get(d, d)}}}"
        for d in dgps
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{4 + i * n_methods}-{3 + (i + 1) * n_methods}}}"
        for i in range(len(dgps))
    )
    method_header = " & ".join(
        METHOD_DISPLAY.get(m, m) for _ in dgps for m in methods
    )

    def _ci_cell(bounds: tuple[float, float] | None) -> str:
        if bounds is None:
            return ""
        # Drop leading zeros and the comma, and stack the bounds on two
        # lines: a 21-data-column table leaves ~14pt per X column, so even
        # in \tiny the interval does not fit on one line.
        lo, hi = (f"{b:.3f}".replace("0.", ".") for b in bounds)
        return rf"{{\tiny [{lo}\newline {hi}]}}"

    rows: list[str] = []
    prev_alpha = None
    for alpha, N, R in configs:
        if prev_alpha is not None and alpha != prev_alpha:
            rows.append(rf"        \cmidrule(l){{1-{n_cols}}}")
        prev_alpha = alpha
        values = []
        ci_values = []
        for d in dgps:
            for m in methods:
                v = cells.get((alpha, N, R, d, m))
                values.append("--" if v is None else f"{v:.3f}")
                ci_values.append(_ci_cell(cells_ci.get((alpha, N, R, d, m))))
        rows.append(
            f"        {alpha:g} & {_latex_int(N)} & {R} & "
            + " & ".join(values) + r" \\"
        )
        # 95% CI on the coverage estimate, in a tight row under the value.
        if any(ci_values):
            rows.append(
                "        & & & "
                + " & ".join(ci_values)
                + r" \\[0.4ex]"
            )

    notes = spec["notes"].replace("@REPS@", str(reps) if reps else "?")

    return "\n".join([
        r"\begin{table}[htbp]",
        r"    \centering",
        rf"    \caption{{{spec['caption']}}}",
        rf"    \label{{{spec['label']}}}",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{1.5pt}",
        rf"    \begin{{tabularx}}{{\textwidth}}{{ccc *{{{n_cols - 3}}}{{>{{\centering\arraybackslash}}X}}}}",
        r"        \toprule",
        rf"        & & & {dgp_header} \\",
        rf"        {cmidrules}",
        rf"        $\alpha$ & $N$ & $R$ & {method_header} \\",
        r"        \midrule",
        *rows,
        r"        \bottomrule",
        r"    \end{tabularx}",
        r"    \begin{minipage}{\textwidth}",
        r"        \vspace{0.5ex}",
        rf"        \footnotesize {notes}",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])


def generate_tables(
    results: list[SimulationResult],
    manifest: dict,
    tables_dir: Path = TABLES_DIR,
) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    quantities = {q for r in results for m in r.coverages.values() for q in m}
    written = []
    for quantity, spec in TABLE_SPECS.items():
        if quantity not in quantities:
            continue
        path = tables_dir / spec["filename"]
        path.write_text(
            make_coverage_table(results, quantity, manifest), encoding="utf-8"
        )
        written.append(path)
        print(f"Wrote {path}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures and LaTeX tables from saved results."
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Run directory (default: latest run under output/results)",
    )
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    manifest, results = load_run(run_dir)
    if not results:
        raise SystemExit(f"No result files (*.npz) found in {run_dir}")
    print(f"Loaded {len(results)} result cells from {run_dir}")

    if not args.figures_only:
        generate_tables(results, manifest)
        write_summary_csv(results, run_dir / "coverage_summary.csv")
    if not args.tables_only:
        generate_figures(results)


if __name__ == "__main__":
    main()
