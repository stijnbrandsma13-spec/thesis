import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree
import pyvista as pv
import vtk
from PIL import Image, ImageChops

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
})


def pretty_dgp_name(name: str) -> str:
    """Turn a DGP identifier into a display name for plot titles.

    Accepts either the bare key (``'beta_tight_normal'``) or the full factory
    name (``'beta_tight_normal_support_probs'``) and returns ``'Tight Normal'``.
    """
    name = name.removesuffix('_support_probs').removeprefix('beta_')
    return name.replace('_', ' ').title()


def plot_coverage_grid(
    grid: np.ndarray,
    coverage_per_point: np.ndarray,
    alpha: float = 0.05,
    true_probs: np.ndarray | None = None,
    title: str | None = None,
    subtitle: str | None = None,
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
        Plot title (rendered larger, above the subtitle).
    subtitle : str, optional
        Smaller second line drawn just below the title (e.g. the parameter
        settings). When given, ``title`` is raised above it.
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
    # KD-tree nearest-neighbour lookup: a brute-force (P, R) distance matrix
    # would be ~650 MB at the default resolution and R = 225.
    nn = cKDTree(grid).query(pixels)[1]  # (P,)
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
    main_title = title or f"Per-point CI coverage  ($1-\\alpha = {nominal:.2f}$)"
    if subtitle:
        # The subtitle sits in the slot directly above the map (set_title); the
        # main title is lifted above it with a pixel offset so the two lines can
        # carry different sizes. annotation_clip=False + a bbox_inches='tight'
        # save keep the raised title from being cropped.
        ax.set_title(subtitle, fontsize=10)
        ax.annotate(
            main_title, xy=(0.5, 1.0), xytext=(0.0, 24.0),
            xycoords="axes fraction", textcoords="offset points",
            ha="center", va="bottom", fontsize=13, annotation_clip=False,
        )
    else:
        ax.set_title(main_title)
    ax.grid(False)

    return ax


# ---------------------------------------------------------------------------
# 3-D bar charts (PMF / CDF)
#
# These are rendered with PyVista/VTK rather than matplotlib's mplot3d.  VTK is
# a true 3-D engine with a real depth buffer, so bars occlude one another
# correctly regardless of viewing angle.  matplotlib's mplot3d sorts whole
# ``Poly3DCollection``s with the painter's algorithm, which mis-orders tall
# bars and produces the colour "bleed-through" we were fighting.  The visual
# style is kept deliberately close to the old mplot3d look: jet colormap, solid
# per-bar colour, thin black edges, serif axis labels, a back-plane grid, and a
# perspective view from roughly elev=25 / azim=-60.
# ---------------------------------------------------------------------------

# Camera direction (in data space) used for every 3-D bar chart.  Viewing from
# the (-x, -y, +z) octant puts the (x_min, y_min) corner nearest the viewer at
# the front-bottom, so the CDF rises away from the viewer (peak at the back) and
# the PMF is seen from the same perspective.  The azimuth is deliberately offset
# from a symmetric 45° (|x| != |y|) so lattice bars stagger and none hide
# directly behind another.
_CAM_DIRECTION = np.array([-1.0, -1.5, 0.82])

# VTK's bundled fonts have no Greek glyphs, so axis text like "β₁" renders
# empty.  STIXGeneral (shipped with matplotlib) is a Times-like serif that
# covers Greek and reads close to the paper's Computer Modern; the italic cut
# is used for the β titles, mirroring LaTeX math italics.
import matplotlib.font_manager as _fm
_FONT_PATH = _fm.findfont('STIXGeneral')
_FONT_ITALIC_PATH = _fm.findfont(
    _fm.FontProperties(family='STIXGeneral', style='italic')
)
_VTK_FONT_FILE = 4  # vtkTextProperty font-family code selecting an external TTF


def _trim_white(path: str, pad: int = 24) -> None:
    """Crop uniform white margins off a saved PNG, leaving a small padding.

    Lets us frame the scene generously (so axis titles are never clipped) and
    then tighten to the actual content, mimicking matplotlib's ``bbox_inches``.
    """
    img = Image.open(path).convert('RGB')
    bg = Image.new('RGB', img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.width, right + pad)
    bottom = min(img.height, bottom + pad)
    img.crop((left, top, right, bottom)).save(path)


def _add_png_title(path: str, title: str, font_size: int = 46, pad: int = 18) -> None:
    """Prepend a white band carrying a centred serif title to a saved PNG.

    Done in pixel space (after ``_trim_white``) rather than as a VTK overlay so
    the title can never overlap the 3-D scene and uses the same STIXGeneral face
    as the axis labels, at a size sitting just above the axis titles.
    """
    from PIL import ImageDraw, ImageFont
    img = Image.open(path).convert('RGB')
    font = ImageFont.truetype(_FONT_PATH, font_size)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    tw, th = right - left, bottom - top
    band = th + 2 * pad
    out = Image.new('RGB', (img.width, img.height + band), (255, 255, 255))
    out.paste(img, (0, band))
    draw = ImageDraw.Draw(out)
    draw.text(((img.width - tw) / 2 - left, pad - top), title,
              fill=(0, 0, 0), font=font)
    out.save(path)


def _add_overlay_title(plotter, title: str, font_size: int = 30) -> None:
    """Overlay a top-centred serif title on the interactive window.

    Only used for the on-screen (``show=True``) view; saved PNGs get their title
    from ``_add_png_title`` instead, added after the screenshot is taken.
    """
    actor = vtk.vtkTextActor()
    actor.SetInput(title)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.5, 0.96)
    tp = actor.GetTextProperty()
    tp.SetFontFamily(_VTK_FONT_FILE)
    tp.SetFontFile(_FONT_PATH)
    tp.SetColor(0.0, 0.0, 0.0)
    tp.SetFontSize(font_size)
    tp.SetJustificationToCentered()
    tp.SetVerticalJustificationToTop()
    plotter.add_actor(actor)


def _add_billboard_text(
    plotter,
    text: str,
    position,
    *,
    font_size: int = 26,
    italic: bool = False,
    justify: str = 'centered',
    vjustify: str = 'centered',
    display_offset: tuple[int, int] = (0, 0),
):
    """Add camera-facing text at a 3-D anchor using a Greek-capable TTF.

    Billboard actors keep a constant on-screen size and always face the camera,
    so every label renders at the same size regardless of its depth in the
    scene (cube-axes labels auto-scale with camera distance, which under a
    parallel projection makes far labels come out *larger* than near ones).
    ``display_offset`` nudges the text in screen pixels relative to the
    projected anchor.
    """
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*position)
    actor.SetDisplayOffset(*display_offset)
    tp = actor.GetTextProperty()
    tp.SetFontFamily(_VTK_FONT_FILE)
    tp.SetFontFile(_FONT_ITALIC_PATH if italic else _FONT_PATH)
    tp.SetColor(0.0, 0.0, 0.0)
    tp.SetFontSize(font_size)
    {
        'left': tp.SetJustificationToLeft,
        'centered': tp.SetJustificationToCentered,
        'right': tp.SetJustificationToRight,
    }[justify]()
    {
        'bottom': tp.SetVerticalJustificationToBottom,
        'centered': tp.SetVerticalJustificationToCentered,
        'top': tp.SetVerticalJustificationToTop,
    }[vjustify]()
    plotter.add_actor(actor)
    return actor


def _add_axis_title(plotter, text: str, position, font_size: int = 44) -> None:
    """Add an axis title; 'β1'/'β2' are typeset as an italic β with a smaller
    pixel-offset digit, since STIXGeneral has no Unicode subscript glyphs."""
    if len(text) == 2 and text[1].isdigit():
        _add_billboard_text(plotter, text[0], position, font_size=font_size,
                            italic=True)
        _add_billboard_text(plotter, text[1], position,
                            font_size=int(0.62 * font_size), justify='left',
                            display_offset=(int(0.30 * font_size),
                                            -int(0.28 * font_size)))
    else:
        _add_billboard_text(plotter, text, position,
                            font_size=int(0.80 * font_size))


def _fmt_ticks(ticks: np.ndarray) -> list[str]:
    """Format ticks with the fewest decimals rendering all values exactly,
    using a true Unicode minus sign."""
    for dec in range(5):
        if np.allclose(np.round(ticks, dec), ticks, atol=1e-12):
            break
    return [f'{t:.{dec}f}'.replace('-', '−') for t in ticks]


def _line_mesh(segments) -> "pv.PolyData":
    """Combine ``[(p0, p1), ...]`` 3-D segments into a single line PolyData."""
    pts = np.asarray([p for seg in segments for p in seg], dtype=float)
    n = len(segments)
    cells = np.column_stack([
        np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2),
    ])
    poly = pv.PolyData(pts)
    poly.lines = cells.ravel()
    return poly


def _bars_to_mesh(bounds: np.ndarray, values: np.ndarray) -> "pv.PolyData | None":
    """Build one combined VTK mesh from axis-aligned bars.

    Parameters
    ----------
    bounds : ndarray, shape (n, 6)
        Per-bar ``(xmin, xmax, ymin, ymax, zmin, zmax)`` extents.
    values : ndarray, shape (n,)
        Scalar attached to every face of each bar; drives the colormap so each
        bar is a single flat colour.

    Returns
    -------
    pyvista.PolyData or None
        ``None`` when there are no bars to draw.
    """
    boxes = []
    for (x0, x1, y0, y1, z0, z1), v in zip(bounds, values):
        box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1), level=0, quads=True)
        box.cell_data['value'] = np.full(box.n_cells, float(v))
        boxes.append(box)
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    return boxes[0].merge(boxes[1:], merge_points=False)


def _render_bar_chart(
    mesh: "pv.PolyData | None",
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_top: float,
    zlabel: str,
    clim: tuple[float, float],
    save_path: str | None,
    show: bool,
    z_bottom: float = 0.0,
    title: str | None = None,
) -> None:
    """Render a bar mesh to screen and/or a PNG with the shared house style.

    ``z_bottom`` lowers the box floor below zero so downward bars (negative
    values, e.g. unconstrained OLS weights) fit inside the box; the z = 0
    plane is then marked with black lines on the two far walls.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    sx, sy = x1 - x0, y1 - y0

    # The data is wide and flat (z spans [0, 1] vs. ~8 units in x/y), so an
    # honest aspect ratio squashes the z-axis to a sliver.  Like matplotlib's
    # ``box_aspect``, we stretch the geometry in z to a comfortable visual
    # height and label the axis with the *true* values via ``axes_ranges``.
    z_visual = 0.55 * max(sx, sy)
    z_span = z_top - z_bottom
    z_scale = z_visual / z_span if z_span > 0 else 1.0
    zb = z_bottom * z_scale  # visual floor height (0 unless z_bottom < 0)
    zt = z_top * z_scale     # visual top height

    plotter = pv.Plotter(off_screen=not show, window_size=(1800, 1400))
    plotter.set_background('white')

    # Box walls are placed at the true bar extent (the mesh's x/y bounds), so
    # the outer faces of the edge bars sit exactly on the walls — bars touch the
    # box without poking through it or leaving a gap.
    bx0, bx1, by0, by1 = x0, x1, y0, y1

    if mesh is not None:
        scaled = mesh.copy()
        pts = scaled.points.copy()
        pts[:, 2] *= z_scale
        scaled.points = pts
        mb = scaled.bounds
        bx0, bx1, by0, by1 = mb[0], mb[1], mb[2], mb[3]
        plotter.add_mesh(
            scaled,
            scalars='value',
            cmap='jet',
            clim=clim,
            preference='cell',
            show_edges=True,
            edge_color='black',
            line_width=1.0,
            show_scalar_bar=False,
            ambient=0.4,
            diffuse=0.66,
            specular=0.05,
        )

    # Axes are drawn by hand (box edges, gridlines, tick stubs, billboard
    # labels) instead of vtkCubeAxesActor.  The cube-axes actor can only label
    # an axis by stretching a value range linearly over the wall span, and the
    # walls extend half a bar beyond the outermost grid points — so labels were
    # systematically offset from the bars (a bar at -4.5 read as ~-4.25).  Here
    # every tick/gridline/label is placed at its *true* data coordinate, and
    # billboard text gives constant on-screen label sizes and full control of
    # the font and of label placement at the box corners.
    x_ticks = np.linspace(x0, x1, 5)
    y_ticks = np.linspace(y0, y1, 5)
    z_ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(z_bottom, z_top)
    eps = 1e-9 * max(z_span, 1e-12)
    z_ticks = z_ticks[(z_ticks > z_bottom + eps) & (z_ticks <= z_top + eps)]
    zv = z_ticks * z_scale  # visual (stretched) z positions

    t = 0.018 * max(sx, sy)  # tick-stub length
    dt = t / np.sqrt(2.0)    # stub length per component on the diagonal z edge

    # Black geometry: outline of the floor and the two far walls (the vertical
    # edge nearest the camera is left open, as mplot3d does), plus tick stubs
    # pointing outward from the three labelled edges.
    edges = [
        ((bx0, by0, zb), (bx1, by0, zb)),
        ((bx1, by0, zb), (bx1, by1, zb)),
        ((bx1, by1, zb), (bx0, by1, zb)),
        ((bx0, by1, zb), (bx0, by0, zb)),
        ((bx0, by1, zb), (bx0, by1, zt)),
        ((bx1, by0, zb), (bx1, by0, zt)),
        ((bx1, by1, zb), (bx1, by1, zt)),
        ((bx0, by1, zt), (bx1, by1, zt)),
        ((bx1, by0, zt), (bx1, by1, zt)),
    ]
    edges += [((x, by0, zb), (x, by0 - t, zb)) for x in x_ticks]
    edges += [((bx0, y, zb), (bx0 - t, y, zb)) for y in y_ticks]
    edges += [((bx0, by1, z), (bx0 - dt, by1 + dt, z)) for z in zv]
    # When the floor sits below zero, mark the z = 0 plane on the two far
    # walls in black so downward (negative) bars read against a baseline.
    has_negative = z_bottom < -eps
    if has_negative:
        edges += [
            ((bx0, by1, 0.0), (bx1, by1, 0.0)),
            ((bx1, by1, 0.0), (bx1, by0, 0.0)),
        ]
    plotter.add_mesh(_line_mesh(edges), color='black', line_width=1.2,
                     lighting=False)

    # Gray gridlines on the floor and the two far walls, at the tick coords.
    grid_lines = []
    for x in x_ticks:
        grid_lines.append(((x, by0, zb), (x, by1, zb)))
        grid_lines.append(((x, by1, zb), (x, by1, zt)))
    for y in y_ticks:
        grid_lines.append(((bx0, y, zb), (bx1, y, zb)))
        grid_lines.append(((bx1, y, zb), (bx1, y, zt)))
    for z in zv:
        if z >= zt - 1e-9:  # the top tick's lines coincide with the rim
            continue
        if has_negative and abs(z) < 1e-9:  # z = 0 already drawn in black
            continue
        grid_lines.append(((bx0, by1, z), (bx1, by1, z)))
        grid_lines.append(((bx1, by1, z), (bx1, by0, z)))
    plotter.add_mesh(_line_mesh(grid_lines), color='#b0b0b0', line_width=1.0,
                     lighting=False)

    # Tick labels.  No label at the z floor: it sits on the corner shared with
    # the β₂ axis and collides with that axis' last tick label.  The x labels
    # are nudged right and the y labels left (in pixels) so the two -4.5
    # labels meeting at the front corner don't touch.
    for x, s in zip(x_ticks, _fmt_ticks(x_ticks)):
        _add_billboard_text(plotter, s, (x, by0 - 1.8 * t, zb),
                            justify='centered', vjustify='top',
                            display_offset=(12, -4))
    for y, s in zip(y_ticks, _fmt_ticks(y_ticks)):
        _add_billboard_text(plotter, s, (bx0 - 1.8 * t, y, zb),
                            justify='right', vjustify='top',
                            display_offset=(-12, -4))
    for z, s in zip(zv, _fmt_ticks(z_ticks)):
        _add_billboard_text(plotter, s, (bx0 - 1.6 * dt, by1 + 1.6 * dt, z),
                            justify='right', vjustify='centered')

    # Axis titles, anchored just outside the relevant box edge (far enough
    # out that they clear the middle tick label).
    _add_axis_title(plotter, 'β1', (0.5 * (bx0 + bx1), by0 - 0.22 * sy, zb))
    _add_axis_title(plotter, 'β2', (bx0 - 0.22 * sx, 0.5 * (by0 + by1), zb))
    _add_axis_title(plotter, zlabel,
                    (bx0 - 0.12 * sx, by1 + 0.12 * sy, zb + 0.55 * z_visual))

    # Orthographic projection reads more like the old mplot3d figure (no
    # perspective fan-out of the bars) and keeps the grid lines parallel.
    plotter.enable_parallel_projection()

    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    cz = zb + 0.45 * z_visual
    span = max(sx, sy, z_visual, 1e-6)
    direction = _CAM_DIRECTION / np.linalg.norm(_CAM_DIRECTION)
    pos = np.array([cx, cy, cz]) + direction * span * 3.0
    plotter.camera_position = [tuple(pos), (cx, cy, cz), (0.0, 0.0, 1.0)]
    # Fit everything (incl. axis titles) into view, then a gentle zoom; the
    # leftover white margin is cropped from the saved PNG by _trim_white.
    plotter.reset_camera()
    plotter.camera.zoom(1.1)

    if save_path is not None:
        plotter.screenshot(save_path)
        _trim_white(save_path)
        if title:
            _add_png_title(save_path, title)
    if show:
        if title:
            _add_overlay_title(plotter, title)
        plotter.show()
    plotter.close()


def plot_cdf_3D(
    support: np.ndarray,
    probs: np.ndarray,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Plot the discrete bivariate CDF as a 3-D step-function bar chart.

    One bar is drawn per cell of the support grid; its height equals the CDF
    evaluated once at the cell's lower-left corner.  No smoothing is applied
    between cells, giving a fully jagged step-function appearance.  Rendered
    with PyVista/VTK so bars occlude correctly from any angle.

    Parameters
    ----------
    support : ndarray, shape (R, 2)
        Support points in (beta_1, beta_2) space.
    probs : ndarray, shape (R,)
        Probability mass at each support point.
    x_range, y_range : tuple of float, optional
        Axis display ranges; default to the data extent plus one cell.
    save_path : str, optional
        If given, write the figure to this path.
    show : bool, optional
        Whether to open an interactive window. Set to False when batch-saving.
    title : str, optional
        Title drawn centred above the chart in the plot's serif font.
    """
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)

    unique_x = np.unique(support[:, 0])
    unique_y = np.unique(support[:, 1])
    nx, ny = len(unique_x), len(unique_y)

    dx = float(unique_x[1] - unique_x[0]) if nx > 1 else 1.0
    dy = float(unique_y[1] - unique_y[0]) if ny > 1 else 1.0

    # Evaluate CDF once per cell at each (xi, yj) lower-left corner.
    xi_grid, yj_grid = np.meshgrid(unique_x, unique_y, indexing='ij')  # (nx, ny)
    cdf_matrix = np.zeros((nx, ny))
    for r in range(nx):
        for c in range(ny):
            mask = (support[:, 0] <= xi_grid[r, c]) & (support[:, 1] <= yj_grid[r, c])
            cdf_matrix[r, c] = probs[mask].sum()

    xi_flat = xi_grid.ravel()
    yj_flat = yj_grid.ravel()
    cdf_flat = cdf_matrix.ravel()

    # The unconstrained OLS weights need not form a valid distribution, so the
    # cumulative sum can climb above 1 or dip below 0.  We therefore let the
    # vertical range follow the data exactly as the PMF does, instead of pinning
    # the top to 1.  Pinning it made an overshooting CDF poke far out of the box;
    # ``_trim_white`` then cropped to that thin spike and the saved PNG came out
    # tall and narrow.  Because ``_render_bar_chart`` stretches whatever range we
    # pass to the SAME fixed visual box height, every PMF/CDF figure keeps an
    # identical aspect ratio no matter how far the CDF overshoots; only the z
    # tick labels change.
    vmax = cdf_flat.max() if cdf_flat.max() > 0 else 1.0
    vmin = min(float(cdf_flat.min()), 0.0)

    # Bars are centred on each grid point (consistent with the PMF) and tile the
    # grid with a negligible gap to avoid z-fighting on shared faces.  Positive
    # cells rise from z = 0; negative cells (possible only for OLS) hang below
    # it.  Near-zero cells keep a 1e-9 stub so the box stays non-degenerate while
    # reading as flat -- a valid CDF legitimately starts at ~0 at the front.
    bar_factor = 0.999
    hx = dx * bar_factor / 2
    hy = dy * bar_factor / 2
    z_lo = np.where(cdf_flat < 0.0, cdf_flat, 0.0)
    z_hi = np.where(cdf_flat < 0.0, 0.0, np.maximum(cdf_flat, 1e-9))
    bounds = np.column_stack([
        xi_flat - hx, xi_flat + hx,
        yj_flat - hy, yj_flat + hy,
        z_lo, z_hi,
    ])
    mesh = _bars_to_mesh(bounds, cdf_flat)

    if x_range is None:
        x_range = (unique_x[0], unique_x[-1])
    if y_range is None:
        y_range = (unique_y[0], unique_y[-1])

    _render_bar_chart(
        mesh,
        x_range=x_range,
        y_range=y_range,
        z_top=max(vmax * 1.05, 1e-6),
        z_bottom=min(vmin * 1.05, 0.0),
        zlabel='CDF',
        clim=(vmin, vmax),
        save_path=save_path,
        show=show,
        title=title,
    )


def plot_pmf_3D(
    support: np.ndarray,
    probs: np.ndarray,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Plot a discrete bivariate PMF as a 3-D bar chart.

    Parameters
    ----------
    support : ndarray, shape (R, 2)
        Support points in (beta_1, beta_2) space.
    probs : ndarray, shape (R,)
        Probability mass at each support point.  Negative values (e.g. from
        an unconstrained OLS fit) are drawn as bars below the z = 0 plane.
    x_range, y_range : tuple of float, optional
        Axis ranges; default to the data extent.
    save_path : str, optional
        If given, write the figure to this path.
    show : bool, optional
        Whether to call ``plt.show()``. Set to False when batch-saving.
    title : str, optional
        Title drawn centred above the chart in the plot's serif font.
    """
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if x_range is None:
        x_range = (support[:, 0].min(), support[:, 0].max())
    if y_range is None:
        y_range = (support[:, 1].min(), support[:, 1].max())

    unique_x = np.unique(support[:, 0])
    unique_y = np.unique(support[:, 1])
    # Lean bars at 40 % of cell spacing to emphasise discreteness.
    dx = (unique_x[1] - unique_x[0]) * 0.4 if len(unique_x) > 1 else 0.2
    dy = (unique_y[1] - unique_y[0]) * 0.4 if len(unique_y) > 1 else 0.2

    vmax = probs.max() if probs.max() > 0 else 1.0
    vmin = min(float(probs.min()), 0.0)

    # Draw a bar at *every* grid point, including zero/near-zero mass, so the
    # full discrete support is visible.  Near-zero bars are clamped to a small
    # floor so they render as short stubs (a true zero-height box is degenerate
    # and would be invisible); colour still reflects the true mass, so they stay
    # at the bottom of the colormap.  Negative weights (e.g. unconstrained OLS
    # estimates) render as bars hanging below the z = 0 plane.
    floor = 0.02 * vmax
    negative = probs <= -floor
    z_lo = np.where(negative, probs, 0.0)
    z_hi = np.where(negative, 0.0, np.maximum(probs, floor))
    cx = support[:, 0]
    cy = support[:, 1]
    bounds = np.column_stack([
        cx - dx / 2, cx + dx / 2,
        cy - dy / 2, cy + dy / 2,
        z_lo, z_hi,
    ])
    mesh = _bars_to_mesh(bounds, probs)

    _render_bar_chart(
        mesh,
        x_range=x_range,
        y_range=y_range,
        z_top=max(vmax * 1.05, 1e-6),
        z_bottom=min(vmin * 1.05, 0.0),
        zlabel='PMF',
        clim=(vmin, vmax),
        save_path=save_path,
        show=show,
        title=title,
    )


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Allow ``python code/utils/visualization.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.dgp import (
        beta_almost_single_type_support_probs,
        beta_bimodal_support_probs,
        beta_four_types_support_probs,
        beta_single_type_support_probs,
        beta_strictly_uniform_support_probs,
        beta_tight_normal_support_probs,
        beta_wide_normal_support_probs,
    )

    figs_dir = Path(__file__).resolve().parents[2] / 'output' / 'figures' / 'true'
    figs_dir.mkdir(parents=True, exist_ok=True)

    DGPS = {
        'beta_bimodal': beta_bimodal_support_probs,
        'beta_tight_normal': beta_tight_normal_support_probs,
        'beta_wide_normal': beta_wide_normal_support_probs,
        'beta_strictly_uniform': beta_strictly_uniform_support_probs,
        'beta_almost_single_type': beta_almost_single_type_support_probs,
        'beta_single_type': beta_single_type_support_probs,
        'beta_four_types': beta_four_types_support_probs,
    }

    R_VALUES = (25, 225)
    grid_range = (-4.5, 3.5)

    for R in R_VALUES:
        for name, fn in DGPS.items():
            _, support, probs = fn(R)
            pretty = pretty_dgp_name(name)
            pmf_path = figs_dir / f'pmf_{name}_R{R}.png'
            cdf_path = figs_dir / f'cdf_{name}_R{R}.png'
            plot_pmf_3D(
                support, probs,
                x_range=grid_range, y_range=grid_range,
                save_path=str(pmf_path), show=False,
                title=f'True PMF of {pretty}',
            )
            plot_cdf_3D(
                support, probs,
                x_range=grid_range, y_range=grid_range,
                save_path=str(cdf_path), show=False,
                title=f'True CDF of {pretty}',
            )
            print(f'Saved {pmf_path.name} and {cdf_path.name}')
