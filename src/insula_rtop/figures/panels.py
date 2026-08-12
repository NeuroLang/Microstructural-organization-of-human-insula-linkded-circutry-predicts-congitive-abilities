"""Individual panels, each matched to one panel of a published figure.

Every function draws into an axes the caller supplies, so
:mod:`insula_rtop.figures.run` can assemble them into the article's two-section
layout without any of them knowing about the others.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from nilearn import plotting as niplot

from insula_rtop.analysis.gradients import patch_frame
from insula_rtop.atlases.fslr import HEMI_LETTERS, group_midthickness
from insula_rtop.figures.style import (
    FIT_COLOR,
    RTOP_CMAP,
    SCATTER_COLOR,
    SUBDIVISION_COLORS,
    panel_title,
    significance_bracket,
)
from insula_rtop.imaging import read_surface

#: Which way anterior points on screen, per hemisphere. Each hemisphere is
#: drawn from its own lateral side, so anterior is to the left for the left
#: hemisphere and to the right for the right -- the convention the article's
#: Figure 3C axis labels ("A + P" vs "P + A") spell out.
ANTERIOR_ON_SCREEN = {"L": np.pi, "R": 0.0}

#: Padding, in millimetres, around the insular patch when zooming a rendering.
ZOOM_PAD_MM = 12.0


def zoom_to_patch(ax, hemi: str, patch: np.ndarray, pad: float = ZOOM_PAD_MM) -> None:
    """Crop a rendered 3-D surface to the insula, as the article's panels do.

    The article shows a small whole-hemisphere view with a box, then an
    enlarged view of what is inside it. Renderings here go straight to the
    enlargement: the axis limits are set from the *inflated* surface's own
    coordinates over the patch, so the crop follows the insula rather than a
    hard-coded window.
    """
    box = render_coords(hemi)[patch]
    for setter, axis in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(box[:, axis].min() - pad, box[:, axis].max() + pad)
    ax.set_box_aspect([max(np.ptp(box[:, i]) + 2 * pad, 1e-6) for i in range(3)])


# ---- brain renderings -----------------------------------------------------


def inflated_surface(hemi: str):
    """S1200 group-average inflated surface, the article's rendering target."""
    from insula_rtop.atlases.fslr import _data_dir

    return read_surface(
        _data_dir() / f"S1200.{hemi}.inflated_MSMAll.32k_fs_LR.surf.gii"
    )


@lru_cache(maxsize=2)
def render_coords(hemi: str) -> np.ndarray:
    """Inflated coordinates in the frame nilearn actually draws in.

    ``plot_surf`` re-centres the mesh on its own centroid before handing it to
    matplotlib, so anything drawn into the same axes afterwards -- axis limits,
    markers, arrows -- has to be shifted by the same amount or it lands tens of
    millimetres away from the vertex it belongs to.
    """
    coords, _ = inflated_surface(hemi)
    return coords - coords.mean(axis=0)


@lru_cache(maxsize=2)
def sulcal_depth(hemi: str) -> np.ndarray:
    """S1200 group sulcal depth, for the grey shading under the overlays.

    Stored as a CIFTI over grayordinates, so it has to be expanded back onto
    the full standard mesh the same way the HCP-MMP labels are.
    """
    import nibabel as nib

    from insula_rtop.atlases.fslr import N_VERTICES_32K, _data_dir
    from insula_rtop.atlases.glasser import _N_GRAY, _load_mmp

    data = np.asarray(
        nib.load(str(_data_dir() / "S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii")).dataobj
    ).ravel()
    _, _, gray = _load_mmp()
    start = 0 if hemi == "L" else _N_GRAY["L"]
    out = np.zeros(N_VERTICES_32K)
    out[gray[hemi]] = data[start : start + _N_GRAY[hemi]]
    return out


def subdivision_surface(
    ax,
    hemi: str,
    labels: np.ndarray,
    names: dict[int, str],
    *,
    view: str = "lateral",
    zoom: bool = False,
):
    """Figure 2A/7A: the subdivisions painted on the inflated surface.

    The insula shows on the lateral view; the ACC is on the medial wall and is
    invisible from the side, so its panels ask for ``view="medial"``.
    """
    coords, faces = inflated_surface(hemi)
    ordered = [names[i] for i in sorted(names)]
    cmap = mcolors.ListedColormap([SUBDIVISION_COLORS[n] for n in ordered])
    # NaN, not 0, outside the parcels: 0 would be painted as a real label and
    # swamp the whole hemisphere in the first colour.
    shown = np.where(labels > 0, labels.astype(float), np.nan)
    niplot.plot_surf_roi(
        (coords, faces),
        roi_map=shown,
        hemi="left" if hemi == "L" else "right",
        view=view,
        bg_map=sulcal_depth(hemi),
        bg_on_data=True,
        alpha=1.0,
        cmap=cmap,
        avg_method="median",
        axes=ax,
        figure=ax.figure,
        colorbar=False,
    )
    if zoom:
        zoom_to_patch(ax, hemi, np.flatnonzero(labels > 0))
    ax.set_facecolor("white")
    return ax


def rtop_surface(
    ax, hemi: str, values: np.ndarray, patch: np.ndarray, vlim, *, zoom: bool = False
):
    """Figure 3A/4: population-mean RTOP on the insula, on the inflated surface."""
    coords, faces = inflated_surface(hemi)
    shown = np.full(len(coords), np.nan)
    shown[patch] = values[patch]
    niplot.plot_surf_stat_map(
        (coords, faces),
        stat_map=shown,
        hemi="left" if hemi == "L" else "right",
        view="lateral",
        cmap=RTOP_CMAP,
        vmin=vlim[0],
        vmax=vlim[1],
        bg_map=sulcal_depth(hemi),
        bg_on_data=True,
        alpha=1.0,
        axes=ax,
        figure=ax.figure,
        colorbar=False,
    )
    if zoom:
        zoom_to_patch(ax, hemi, patch)
    ax.set_facecolor("white")
    return ax


# ---- quantitative panels --------------------------------------------------


def violin_by_subdivision(
    ax, table: pd.DataFrame, atlas: str, seg: str, order, *, letter: str, title: str
):
    """Figure 2B/2E/7B: violins with inner boxes, grouped by hemisphere."""
    subset = table[(table["atlas"] == atlas) & (table["seg"] == seg)].copy()
    subset["Hemisphere"] = subset["hemi"].map({"L": "Left", "R": "Right"})

    sns.violinplot(
        data=subset,
        x="Hemisphere",
        y="rtop",
        hue="subdivision",
        order=["Left", "Right"],
        hue_order=list(order),
        palette={n: SUBDIVISION_COLORS[n] for n in order},
        inner="box",
        linewidth=0.8,
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("Hemisphere")
    ax.set_ylabel("RTOP value [a.u.]")
    panel_title(ax, letter, title)

    # Brackets over the two adjacent contrasts the article marks, then enough
    # headroom above them for the legend to sit clear.
    low, top = subset["rtop"].min(), subset["rtop"].max()
    span = top - low
    for base_x in (0.0, 1.0):
        for i, offset in enumerate((-0.27, 0.0)):
            significance_bracket(
                ax,
                base_x + offset,
                base_x + offset + 0.27,
                top + span * (0.05 + 0.09 * i),
                span * 0.025,
            )
    ax.set_ylim(low - span * 0.06, top + span * 0.52)
    ax.legend(
        title="Insular Subdivision" if seg == "insula" else "ACC Subdivision",
        loc="upper right",
        framealpha=0.95,
    )
    return ax


def stability_panel(ax, curve: pd.DataFrame, *, letter: str, title: str):
    """Figure 2C/2F/7C: detection rate against sample size, one line per contrast."""
    from insula_rtop.figures.style import ACC_PLOT_ORDER, PLOT_ORDER

    order = ACC_PLOT_ORDER if curve["a"].iloc[0].startswith("ACC") else PLOT_ORDER
    rank = {name: i for i, name in enumerate(order)}
    curve = curve.copy()
    # Name each contrast posterior-first, as the article does ("LPI vs. LdAI").
    first = curve.apply(lambda r: min(r["a"], r["b"], key=lambda n: rank[n]), axis=1)
    second = curve.apply(lambda r: max(r["a"], r["b"], key=lambda n: rank[n]), axis=1)
    curve["a"], curve["b"] = first, second

    groups = sorted(
        curve.groupby(["hemi", "a", "b"]),
        key=lambda kv: (kv[0][0], rank[kv[0][1]], rank[kv[0][2]]),
    )
    for (hemi, a, b), group in groups:
        group = group.sort_values("n")
        # The article dashes the hardest contrast (dAI vs vAI) and marks the
        # rest solid; keying on the pair reproduces that without hard-coding.
        hard = {a, b} in ({"dAI", "vAI"}, {"ACC-dAI", "ACC-vAI"})
        ax.plot(
            group["n"],
            100 * group["detection_rate"],
            marker="s" if hemi == "L" else "o",
            markersize=5,
            linewidth=1.6,
            linestyle="--" if hard else "-",
            label=f"{hemi}{a} vs. {hemi}{b}",
        )
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(-4, 104)
    ax.legend(fontsize=7, loc="center right", framealpha=0.9)
    panel_title(ax, letter, title)
    return ax


def polar_directions(
    ax, directions: pd.DataFrame, summary: pd.DataFrame, hemi: str, order
):
    """Figure 3C/3F: rose histograms per subdivision, with the mean direction.

    The angles are plotted raw and the *axis* is mirrored for the left
    hemisphere -- zero at the west, increasing clockwise -- which is what the
    article does. Reflecting the data instead would draw the same picture but
    label it back to front, making the two figures impossible to compare tick
    for tick. Either way each hemisphere reads from its own lateral side:
    anterior at 0, dorsal at pi/2.
    """
    if hemi == "L":
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)

    bins = np.linspace(-np.pi, np.pi, 37)
    # One scale for all three subdivisions. Normalising each to its own peak
    # would draw a diffuse distribution as tall as a tight one, hiding exactly
    # the difference in concentration the panel exists to show.
    histograms = {}
    for name in order:
        angles = directions[
            (directions["hemi"] == hemi) & (directions["subdivision"] == name)
        ]["angle"].dropna()
        if len(angles):
            histograms[name] = np.histogram(angles, bins=bins)[0]
    peak = max((h.max() for h in histograms.values()), default=1) or 1

    for name in order:
        if name not in histograms:
            continue
        counts, edges = histograms[name], bins
        ax.bar(
            edges[:-1],
            counts / peak,
            width=np.diff(edges),
            align="edge",
            color=SUBDIVISION_COLORS[name],
            alpha=0.35,
            edgecolor="none",
        )
        row = summary[(summary["hemi"] == hemi) & (summary["subdivision"] == name)]
        if len(row):
            mean = float(row["mean_direction_rad"].iloc[0])
            # The line's length carries the resultant length, so a weakly
            # oriented subdivision does not draw a full-radius spoke.
            radius = float(row["resultant_length"].iloc[0])
            ax.plot(
                [mean, mean],
                [0, radius],
                color=SUBDIVISION_COLORS[name],
                lw=2.5,
            )

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax.set_xticklabels(
        [
            "0",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$",
            r"$5\pi/4$",
            r"$3\pi/2$",
            r"$7\pi/4$",
        ],
        fontsize=7,
    )
    axis_label = "A  +  P" if hemi == "L" else "P  +  A"
    ax.text(
        0.5,
        -0.13,
        f"D\n{axis_label}\nV",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )
    return ax


def regression_panel(
    ax, x, y, *, r: float, p: float, letter: str, title: str, xlabel="RTOP Measure"
):
    """Figure 8: scatter with a fitted line, CI band, and the r/p annotation."""
    sns.regplot(
        x=np.asarray(x),
        y=np.asarray(y),
        scatter_kws={"s": 14, "color": SCATTER_COLOR, "alpha": 0.85},
        line_kws={"color": FIT_COLOR, "linewidth": 1.8},
        ci=95,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cognitive Control")
    ax.text(
        0.97,
        0.06,
        f"r = {r:.2f}\np {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )
    panel_title(ax, letter, title)
    return ax


# ---- gradient arrows ------------------------------------------------------


def gradient_arrows_on_surface(
    ax,
    hemi: str,
    labels: np.ndarray,
    names: dict[int, str],
    summary: pd.DataFrame,
    *,
    length: float = 13.0,
    lift: float = 3.0,
):
    """Figure 3B/3E: an arrow at each subdivision's centroid, on the rendering.

    The group mean direction is a 3-D vector in world space. It is drawn as its
    component in the plane of the insular patch -- the same plane the polar
    plots beside it report angles in, so the two panels can be read against
    each other -- which for a lateral view is also very nearly the plane of the
    screen.

    **The arrow is scaled by how much of the direction survives that
    projection.** A subdivision whose mean direction points out of the patch
    plane -- which happens where the patch curves enough that its tangential
    gradients accumulate along the cap axis, as the right dAI's does -- draws a
    stub, not a confident full-length arrow pointing somewhere arbitrary.

    The inflated surface supplies only where to put the arrow: the
    subdivision's centroid, lifted outwards so it floats clear of the cortex
    instead of being buried in it.
    """
    inflated = render_coords(hemi)
    # The plane comes from the group midthickness, the geometry the directions
    # were measured on, not from the inflated surface being drawn.
    mid, _ = group_midthickness(hemi)
    _, plane = patch_frame(mid, np.flatnonzero(labels > 0))
    # Axes3D depth-sorts artists and ignores zorder unless told not to, which
    # otherwise hides arrows inside the mesh however far they are lifted.
    ax.computed_zorder = False

    for index, name in sorted(names.items()):
        row = summary[(summary["hemi"] == hemi) & (summary["subdivision"] == name)]
        sub = np.flatnonzero(labels == index)
        if not len(sub) or not len(row):
            continue
        mean = row[["mean_vx", "mean_vy", "mean_vz"]].to_numpy(float)[0]
        if not np.isfinite(mean).all():
            continue
        # Keep only the component lying in the patch plane; what is left of
        # the unit vector after that sets the arrow's length.
        in_plane = (mean @ plane[0]) * plane[0] + (mean @ plane[1]) * plane[1]
        visible = np.linalg.norm(in_plane)
        if visible == 0:
            continue
        arrow = in_plane / visible * (length * visible)

        # The mean of a curved patch's vertices sits *inside* the mesh, so the
        # arrow goes on the subdivision vertex closest to it and is pushed out
        # along the radius -- on an inflated surface that is the outward normal.
        anchor = sub[
            np.argmin(((inflated[sub] - inflated[sub].mean(axis=0)) ** 2).sum(1))
        ]
        outward = inflated[anchor] / np.linalg.norm(inflated[anchor])
        start = inflated[anchor] + lift * outward - arrow / 2
        ax.quiver(
            *start,
            *arrow,
            color="black",
            linewidth=2.2,
            arrow_length_ratio=0.32,
            zorder=10,
        )
    return ax


def subdivision_legend(ax, order, *, ncol: int = 1, loc: str = "upper right"):
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markersize=8,
            color=SUBDIVISION_COLORS[n],
            label=n,
        )
        for n in order
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=9,
        framealpha=0.9,
        ncol=ncol,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )


def load_population_rtop(deriv_root: Path, subjects, hemi: str) -> np.ndarray:
    """Mean RTOP per vertex across participants, for the surface renderings."""
    from insula_rtop.analysis.isocontours import population_mean
    from insula_rtop.imaging import read_surface_scalars
    from insula_rtop.surface.run import surface_paths

    stack = [
        read_surface_scalars(surface_paths(deriv_root, sid)[hemi])
        for sid in subjects
        if surface_paths(deriv_root, sid)[hemi].exists()
    ]
    if not stack:
        raise RuntimeError(f"No surface RTOP found for hemisphere {hemi}")
    return population_mean(np.vstack(stack))


__all__ = [
    "HEMI_LETTERS",
    "gradient_arrows_on_surface",
    "inflated_surface",
    "load_population_rtop",
    "polar_directions",
    "regression_panel",
    "rtop_surface",
    "stability_panel",
    "subdivision_legend",
    "subdivision_surface",
    "violin_by_subdivision",
]
