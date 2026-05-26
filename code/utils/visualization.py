import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Callable

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
})


def plot_coverage_grid(
    grid: np.ndarray,
    coverage_per_point: np.ndarray,
    alpha: float = 0.05,
    true_probs: np.ndarray | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    resolution: int = 600,
    show_points: bool = False,
) -> plt.Axes:
    """Voronoi-style coverage map: every pixel takes the colour of its
    nearest support point, so there is no white background between points.

    Green encodes the nominal level (1 - alpha) — i.e. exact coverage —
    while red encodes 100% empirical coverage (maximally conservative).
    Pixels whose nearest support point under-covers (< nominal) are shown
    in blue so the failure mode is visible rather than clipped.

    Parameters
    ----------
    grid : ndarray, shape (R, 2)
        Support points in (beta_1, beta_2) space.
    coverage_per_point : ndarray, shape (R,)
        Empirical coverage at each grid point, averaged across MC reps.
    alpha : float
        Nominal CI level; nominal coverage is ``1 - alpha``.
    true_probs : ndarray, shape (R,), optional
        Marker sizes are scaled by the true probability mass.
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
        Existing axes to draw on; a new figure is created otherwise.
    resolution : int
        Pixels per side for the nearest-neighbour fill.
    show_points : bool
        Whether to overlay the support points on top of the fill.

    Returns
    -------
    matplotlib.axes.Axes
    """
    grid = np.asarray(grid, dtype=float)
    cov = np.asarray(coverage_per_point, dtype=float)
    if grid.shape[0] != cov.shape[0]:
        raise ValueError("grid and coverage_per_point must have matching length")
    if grid.shape[1] != 2:
        raise ValueError("plot_coverage_grid requires a 2-D support grid")

    nominal = 1.0 - alpha

    # The colorbar visually allocates the bottom 1/3 to [0, nominal] (the
    # under-coverage region) and the top 2/3 to [nominal, 1] (the
    # over-coverage region), so the over-coverage detail isn't crammed into
    # a sliver. Achieved by pairing fixed colormap stops with a piecewise-
    # linear norm that maps nominal -> 1/3.
    cov_cmap = LinearSegmentedColormap.from_list(
        "purple_blue_green_red",
        [
            (0.0,    "#6a1b9a"),  # purple at 0
            (1 / 6,  "#1f77b4"),  # blue mid-undercover
            (1 / 3,  "#2ca02c"),  # green at nominal
            (2 / 3,  "#f4d03f"),  # yellow midway above nominal
            (1.0,    "#d62728"),  # red at 1
        ],
    )

    class _PiecewiseNorm(Normalize):
        def __init__(self, vmin, vcenter, vmax, center_frac=1 / 3):
            super().__init__(vmin, vmax, clip=False)
            self.vcenter = vcenter
            self.center_frac = center_frac

        def __call__(self, value, clip=None):
            x = np.asarray(value, dtype=float)
            out = np.empty_like(x)
            lo = x <= self.vcenter
            out[lo] = (
                (x[lo] - self.vmin) / max(self.vcenter - self.vmin, 1e-12)
            ) * self.center_frac
            out[~lo] = self.center_frac + (
                (x[~lo] - self.vcenter) / max(self.vmax - self.vcenter, 1e-12)
            ) * (1 - self.center_frac)
            return np.ma.array(np.clip(out, 0.0, 1.0))

    cov_norm = _PiecewiseNorm(vmin=0.0, vcenter=nominal, vmax=1.0, center_frac=1 / 3)

    # Extent: pad slightly so corner Voronoi cells are visible.
    x_min, x_max = grid[:, 0].min(), grid[:, 0].max()
    y_min, y_max = grid[:, 1].min(), grid[:, 1].max()
    pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 0.5
    pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
    x_lo, x_hi = x_min - pad_x, x_max + pad_x
    y_lo, y_hi = y_min - pad_y, y_max + pad_y

    # Nearest-neighbour assignment over a fine pixel grid.
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    XX, YY = np.meshgrid(xs, ys)
    pixels = np.column_stack([XX.ravel(), YY.ravel()])  # (P, 2)
    # Squared distances (P, R) — fine at typical R; KDTree if R becomes huge.
    d2 = (
        (pixels[:, 0, None] - grid[None, :, 0]) ** 2
        + (pixels[:, 1, None] - grid[None, :, 1]) ** 2
    )
    nn = d2.argmin(axis=1)  # (P,)
    pixel_cov = cov[nn].reshape(XX.shape)

    rgba = cov_cmap(cov_norm(pixel_cov))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    ax.imshow(
        rgba,
        extent=(x_lo, x_hi, y_lo, y_hi),
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )

    # Manual colorbar: matplotlib's standard colorbar lays its axis out
    # linearly in DATA coords, so a piecewise norm would visually compress
    # the over-coverage band into a sliver. Here we lay the axis out
    # linearly in COLORMAP coords (so the cmap stops are evenly spaced)
    # and place data-valued ticks via the norm. The bottom 1/3 of the bar
    # spans [0, nominal]; the top 2/3 spans [nominal, 1].
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    bar = np.linspace(0.0, 1.0, 512)[:, None]
    cax.imshow(
        bar, aspect="auto", origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=cov_cmap,
    )
    cax.set_xticks([])
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.grid(False)
    # Tick at fractional positions; labels are the corresponding data values.
    tick_fracs = [0.0, 1 / 6, 1 / 3, 2 / 3, 1.0]
    tick_labels = [
        "0.00",
        f"{nominal / 2:.2f}",
        f"{nominal:.2f}",
        f"{(nominal + 1) / 2:.3f}",
        "1.00",
    ]
    cax.set_yticks(tick_fracs)
    cax.set_yticklabels(tick_labels)
    cax.tick_params(axis="y", which="both", left=False, right=True,
                    labelleft=False, labelright=True)
    cax.axhline(1 / 3, color="black", linewidth=0.8)
    cax.set_ylabel(f"Empirical coverage (nominal = {nominal:.2f})")
    cax.yaxis.set_label_position("right")

    if show_points:
        if true_probs is not None:
            tp = np.asarray(true_probs, dtype=float)
            sizes = 15.0 + 200.0 * (tp / tp.max()) if tp.max() > 0 else np.full_like(tp, 25.0)
        else:
            sizes = np.full(grid.shape[0], 25.0)
        ax.scatter(
            grid[:, 0], grid[:, 1],
            s=sizes, facecolors="none", edgecolors="black", linewidths=0.5,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$\beta_1$")
    ax.set_ylabel(r"$\beta_2$")
    ax.set_title(title or f"Per-point CI coverage  ($1-\\alpha = {nominal:.2f}$)")
    ax.grid(False)

    return ax


def plot_cdf_3D(
    cdf: Callable[[np.ndarray], float],
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    n_grid: int = 50,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the estimated bivariate CDF as a 3-D surface.

    Parameters
    ----------
    cdf : Callable
        Function that accepts a 1-D array ``[b1, b2]`` and returns a scalar
        in [0, 1], as returned by :meth:`FKRBEstimator.get_cdf`.
    x_range : tuple of float, optional
        (min, max) range for the beta_1 axis.
    y_range : tuple of float, optional
        (min, max) range for the beta_2 axis.
    n_grid : int, optional
        Number of evaluation points along each axis.
    save_path : str, optional
        If given, write the figure to this path instead of (or in addition to)
        displaying it.
    show : bool, optional
        Whether to call ``plt.show()``. Set to False when batch-saving.
    """
    x = np.linspace(*x_range, n_grid)
    y = np.linspace(*y_range, n_grid)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = cdf(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', linewidth=0.5, antialiased=True, shade=True)

    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_zlabel('CDF')
    ax.set_xlim(*x_range)
    ax.invert_xaxis()
    ax.set_ylim(*y_range)
    ax.set_zlim(0, 1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pmf_3D(
    support: np.ndarray,
    probs: np.ndarray,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a discrete bivariate PMF as a 3-D bar chart.

    Parameters
    ----------
    support : ndarray, shape (R, 2)
        Support points in (beta_1, beta_2) space.
    probs : ndarray, shape (R,)
        Probability mass at each support point.
    x_range, y_range : tuple of float, optional
        Axis ranges; default to the data extent.
    save_path : str, optional
        If given, write the figure to this path.
    show : bool, optional
        Whether to call ``plt.show()``. Set to False when batch-saving.
    """
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if x_range is None:
        x_range = (support[:, 0].min(), support[:, 0].max())
    if y_range is None:
        y_range = (support[:, 1].min(), support[:, 1].max())

    unique_x = np.unique(support[:, 0])
    unique_y = np.unique(support[:, 1])
    dx = (unique_x[1] - unique_x[0]) * 0.9 if len(unique_x) > 1 else 0.5
    dy = (unique_y[1] - unique_y[0]) * 0.9 if len(unique_y) > 1 else 0.5

    cmap = plt.get_cmap('jet')
    vmax = probs.max() if probs.max() > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    colors = cmap(norm(probs))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        support[:, 0] - dx / 2, support[:, 1] - dy / 2, np.zeros_like(probs),
        dx, dy, probs,
        color=colors, shade=True, edgecolor='k', linewidth=0.2,
    )

    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_zlabel('PMF')
    ax.set_xlim(*x_range)
    ax.invert_xaxis()
    ax.set_ylim(*y_range)
    ax.set_zlim(0, max(vmax * 1.05, 1e-6))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Allow ``python code/utils/visualization.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.dgp import (
        beta_bimodal_support_probs,
        beta_bivariate_normal_support_probs,
        beta_concentrated_spike_support_probs,
        beta_diffuse_support_probs,
        beta_discrete_uniform_support_probs,
    )

    figs_dir = Path(__file__).resolve().parents[2] / 'output' / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    DGPS = {
        'beta_bivariate_normal': beta_bivariate_normal_support_probs,
        'beta_bimodal': beta_bimodal_support_probs,
        'beta_diffuse': beta_diffuse_support_probs,
        'beta_concentrated_spike': beta_concentrated_spike_support_probs,
        'beta_discrete_uniform': beta_discrete_uniform_support_probs,
    }

    R = 225
    grid_range = (-4.5, 3.5)

    def make_discrete_cdf(support: np.ndarray, probs: np.ndarray) -> Callable[[np.ndarray], float]:
        support = np.asarray(support)
        probs = np.asarray(probs)
        def cdf(beta: np.ndarray) -> float:
            mask = (support <= np.asarray(beta)).all(axis=1)
            return float(probs[mask].sum())
        return cdf

    for name, fn in DGPS.items():
        _, support, probs = fn(R)
        pmf_path = figs_dir / f'pmf_{name}_R{R}.png'
        cdf_path = figs_dir / f'cdf_{name}_R{R}.png'
        plot_pmf_3D(
            support, probs,
            x_range=grid_range, y_range=grid_range,
            save_path=str(pmf_path), show=False,
        )
        plot_cdf_3D(
            make_discrete_cdf(support, probs),
            x_range=grid_range, y_range=grid_range,
            save_path=str(cdf_path), show=False,
        )
        print(f'Saved {pmf_path.name} and {cdf_path.name}')
