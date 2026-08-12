"""Figures 2, 3, 4, 7 and 8, laid out as in Menon et al. (2020).

Each composite mirrors the article: two stacked sections, **I. Functional
Parcellation** (Deen et al., 2011) over **II. Multimodal Parcellation** (Glasser
et al., 2016), separated by a dashed rule, with panel letters ``A.``-``F.``.
Everything is read from the tables the analysis step wrote plus the per-subject
surfaces, so figures redraw without recomputing any statistics.

One panel of the article cannot be reproduced here: Figure 4's right-hand
cytoarchitectonic reference (Hb2-L / Hb3-R, with the asg/msg/psg/VENs labels) is
reproduced in the article from separate published histology, not computed from
the HCP data. The RTOP half of that figure is reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from insula_rtop.analysis.extract import read_group_table
from insula_rtop.analysis.run import ANALYSIS_DIR
from insula_rtop.atlases.fslr import HEMI_LETTERS
from insula_rtop.atlases.run import load_segmentation
from insula_rtop.figures import panels
from insula_rtop.figures.style import (
    ACC_PLOT_ORDER,
    PLOT_ORDER,
    RTOP_CMAP,
    SECTION_FUNCTIONAL,
    SECTION_MULTIMODAL,
    apply_theme,
    section_heading,
    section_rule,
)

FIGURES_DIR = "figures"

#: Every figure this module can draw. Kept at module level so the Hydra
#: experiments can be checked against it -- a figure with no experiment cannot
#: be redrawn on its own.
FIGURE_BUILDERS = (
    "figure2_insula_subdivisions",
    "figure3_gradients",
    "figure4_rtop_maps",
    "figure7_acc_subdivisions",
    "figure8_cognitive_control",
)

#: ``(atlas, section heading, panel letters)`` for the two-section composites.
SECTIONS = (
    ("Deen2011", SECTION_FUNCTIONAL, "ABC"),
    ("HCPMMP1", SECTION_MULTIMODAL, "DEF"),
)


def _read(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, sep="\t", **kwargs) if path.exists() else None


def analysed_sample(analysis_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """The group table restricted to the subjects the statistics actually used.

    ``group_rtop.tsv`` is written *before* the 1.5 IQR exclusion, so reading it
    raw would draw violins over a different sample than the ANOVA reports --
    and the excluded subjects are precisely the extreme ones, so they dominate
    the tails. ``outliers.json`` records who was dropped.
    """
    table = read_group_table(analysis_dir)
    outliers_path = analysis_dir / "outliers.json"
    if outliers_path.exists():
        excluded = set(json.loads(outliers_path.read_text())["excluded_subjects"])
        table = table[~table["subject"].astype(str).isin(excluded)]
    return table, sorted(table["subject"].astype(str).unique())


# ---- Figure 2 / Figure 7 ---------------------------------------------------


def figure_subdivisions(
    table: pd.DataFrame,
    analysis_dir: Path,
    deriv_root: Path,
    *,
    seg: str = "insula",
) -> plt.Figure:
    """Figure 2 (insula) or Figure 7 (ACC): ROIs, violins, stability."""
    order = ACC_PLOT_ORDER if seg == "acc" else PLOT_ORDER
    label = "ACC" if seg == "acc" else "Insula"
    # The ACC lives on the medial wall and is invisible from the side.
    view = "medial" if seg == "acc" else "lateral"
    # Only HCP-MMP defines an ACC segmentation, so Figure 7 is a single section
    # and must not leave the other half of the canvas blank.
    sections = [s for s in SECTIONS if not (seg == "acc" and s[0] == "Deen2011")]
    if len(sections) == 1:
        sections = [(sections[0][0], sections[0][1], "ABC")]

    fig = plt.figure(figsize=(16, 4.8 * len(sections)))
    height = 1.0 / len(sections)

    for row, (atlas, heading, letters) in enumerate(sections):
        top = 0.94 - row * height
        section_heading(fig, heading, top)

        labels, names = load_segmentation(deriv_root, atlas, seg)
        for column, hemi in enumerate(HEMI_LETTERS):
            ax = fig.add_axes(
                [0.03 + 0.11 * column, top - 0.72 * height, 0.11, 0.60 * height],
                projection="3d",
            )
            panels.subdivision_surface(ax, hemi, labels[hemi], names, view=view)
            ax.set_title(
                "Left hemisphere" if hemi == "L" else "Right hemisphere",
                fontsize=9,
                y=-0.05,
            )
        fig.text(
            0.03, top - 0.08 * height, f"{letters[0]}. {label} Subdivisions",
            fontsize=13, ha="left", va="top",
        )

        ax_violin = fig.add_axes([0.31, top - 0.76 * height, 0.30, 0.66 * height])
        panels.violin_by_subdivision(
            ax_violin, table, atlas, seg, order,
            letter=letters[1], title=f"RTOP Across {label} Subdivisions",
        )

        curve = _read(analysis_dir / f"stability_{atlas}_{seg}.tsv")
        ax_stab = fig.add_axes([0.68, top - 0.76 * height, 0.29, 0.66 * height])
        if curve is not None:
            panels.stability_panel(
                ax_stab, curve,
                letter=letters[2], title=f"Stability Across {label} Subdivisions",
            )

    if len(sections) > 1:
        section_rule(fig, 0.485)
    return fig


# ---- Figure 3 --------------------------------------------------------------


def figure_gradients(
    analysis_dir: Path, deriv_root: Path, subjects: list[str]
) -> plt.Figure:
    """Figure 3: mean RTOP, gradient directions, and their distribution."""
    fig = plt.figure(figsize=(16, 9))

    for row, (atlas, heading, letters) in enumerate(SECTIONS):
        top = 0.94 - row * 0.5
        section_heading(fig, heading, top)
        labels, names = load_segmentation(deriv_root, atlas, "insula")
        directions = _read(
            analysis_dir / f"gradient_directions_{atlas}.tsv", dtype={"subject": str}
        )
        summary = _read(analysis_dir / f"gradients_{atlas}.tsv")

        means = {h: panels.load_population_rtop(deriv_root, subjects, h)
                 for h in HEMI_LETTERS}
        patches = {h: np.flatnonzero(labels[h] > 0) for h in HEMI_LETTERS}
        pooled = np.concatenate([means[h][patches[h]] for h in HEMI_LETTERS])
        pooled = pooled[np.isfinite(pooled)]
        vlim = tuple(np.percentile(pooled, [2, 98]))

        fig.text(0.03, top - 0.04, f"{letters[0]}. RTOP Averaged Over All Participants",
                 fontsize=13, ha="left", va="top")
        for column, hemi in enumerate(HEMI_LETTERS):
            ax = fig.add_axes(
                [0.01 + 0.13 * column, top - 0.34, 0.13, 0.28], projection="3d"
            )
            panels.rtop_surface(
                ax, hemi, means[hemi], patches[hemi], vlim, zoom=True
            )
        cax = fig.add_axes([0.06, top - 0.375, 0.14, 0.014])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(*vlim), cmap=RTOP_CMAP),
            cax=cax, orientation="horizontal",
        )
        cbar.set_label("RTOP", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

        fig.text(0.28, top - 0.04, f"{letters[1]}. Main Gradient Directions",
                 fontsize=13, ha="left", va="top")
        for column, hemi in enumerate(HEMI_LETTERS):
            ax = fig.add_axes(
                [0.28 + 0.16 * column, top - 0.36, 0.16, 0.30], projection="3d"
            )
            panels.subdivision_surface(ax, hemi, labels[hemi], names, zoom=True)
            if summary is not None:
                panels.gradient_arrows_on_surface(
                    ax, hemi, labels[hemi], names, summary
                )
        # A legend of its own, clear of both renderings.
        legend_ax = fig.add_axes([0.28, top - 0.40, 0.32, 0.03])
        legend_ax.axis("off")
        panels.subdivision_legend(legend_ax, PLOT_ORDER, ncol=3, loc="center")

        fig.text(0.63, top - 0.04,
                 f"{letters[2]}. Distribution of Main Gradient Directions",
                 fontsize=13, ha="left", va="top")
        for column, hemi in enumerate(HEMI_LETTERS):
            ax = fig.add_axes(
                [0.63 + 0.18 * column, top - 0.40, 0.16, 0.33], projection="polar"
            )
            if directions is not None and summary is not None:
                panels.polar_directions(
                    ax, directions, summary, hemi, PLOT_ORDER
                )

    section_rule(fig, 0.485)
    return fig


# ---- Figure 4 --------------------------------------------------------------


def figure_isocontours(deriv_root: Path, subjects: list[str]) -> plt.Figure:
    """Figure 4 (RTOP half): the insular RTOP map, enlarged, per hemisphere."""
    fig = plt.figure(figsize=(14, 9))
    for row, ((atlas, heading, _), letter) in enumerate(
        zip(SECTIONS, "AB", strict=True)
    ):
        top = 0.94 - row * 0.5
        section_heading(fig, heading, top)
        labels, _ = load_segmentation(deriv_root, atlas, "insula")
        fig.text(0.03, top - 0.04,
                 f"{letter}. RTOP vs Cytoarchitectonic Organization",
                 fontsize=13, ha="left", va="top")

        means = {h: panels.load_population_rtop(deriv_root, subjects, h)
                 for h in HEMI_LETTERS}
        patches = {h: np.flatnonzero(labels[h] > 0) for h in HEMI_LETTERS}
        pooled = np.concatenate([means[h][patches[h]] for h in HEMI_LETTERS])
        pooled = pooled[np.isfinite(pooled)]
        vlim = tuple(np.percentile(pooled, [2, 98]))

        for column, hemi in enumerate(HEMI_LETTERS):
            ax = fig.add_axes(
                [0.04 + 0.46 * column, top - 0.38, 0.40, 0.32], projection="3d"
            )
            panels.rtop_surface(
                ax, hemi, means[hemi], patches[hemi], vlim, zoom=True
            )
            fig.text(
                0.24 + 0.46 * column,
                top - 0.40,
                "Left Hemisphere" if hemi == "L" else "Right Hemisphere",
                fontsize=11, ha="center", va="top",
            )
        # Top right, clear of both the section rule below and the brains.
        cax = fig.add_axes([0.78, top - 0.075, 0.16, 0.012])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(*vlim), cmap=RTOP_CMAP),
            cax=cax, orientation="horizontal",
        )
        cbar.set_label("RTOP", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

    section_rule(fig, 0.485)
    fig.text(
        0.5, 0.012,
        "The article pairs each map with a cytoarchitectonic reference "
        "reproduced from separate published histology; only the RTOP maps are "
        "computed from these data.",
        ha="center", fontsize=8, style="italic", color="0.35",
    )
    return fig


# ---- Figure 8 --------------------------------------------------------------


def figure_cca(analysis_dir: Path) -> plt.Figure:
    """Figure 8: canonical correlation and cross-validated prediction."""
    results = json.loads((analysis_dir / "results.json").read_text())
    fig = plt.figure(figsize=(12, 10))

    for row, ((atlas, heading, _), letters) in enumerate(
        zip(SECTIONS, ("AB", "CD"), strict=True)
    ):
        top = 0.94 - row * 0.5
        section_heading(fig, heading, top)
        summary = results.get(f"cca_{atlas}", {})
        scatter = _read(analysis_dir / f"cca_prediction_{atlas}.tsv")
        scores = _read(analysis_dir / f"cca_scores_{atlas}.tsv")

        ax = fig.add_axes([0.09, top - 0.365, 0.36, 0.30])
        if scores is not None:
            panels.regression_panel(
                ax, scores["brain"], scores["behavior"],
                r=summary.get("canonical_r", float("nan")),
                p=summary.get("canonical_p", float("nan")),
                letter=letters[0], title="Canonical Correlation",
            )
        ax = fig.add_axes([0.58, top - 0.365, 0.36, 0.30])
        if scatter is not None:
            panels.regression_panel(
                ax, scatter["predicted"], scatter["observed"],
                r=summary.get("prediction_r", float("nan")),
                p=summary.get("prediction_p", float("nan")),
                letter=letters[1], title="Cross-Validation",
            )
    section_rule(fig, 0.485)
    return fig


# ---- driver ---------------------------------------------------------------


def _save(fig: plt.Figure, stem: Path, formats, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi)
    plt.close(fig)


def run_figures(
    deriv_root: Path,
    *,
    formats=("png", "pdf"),
    dpi: int = 200,
    only: list[str] | None = None,
) -> list[Path]:
    """Redraw figures from the analysis outputs.

    *only* selects a subset by name; ``None`` draws all of them. One Hydra
    experiment per figure sets it -- see ``pipeline/experiment/``.
    """
    apply_theme()
    deriv_root = Path(deriv_root)
    analysis_dir = deriv_root / ANALYSIS_DIR
    out_dir = deriv_root / FIGURES_DIR
    table, subjects = analysed_sample(analysis_dir)

    written: list[Path] = []
    figures = {
        "figure2_insula_subdivisions": lambda: figure_subdivisions(
            table, analysis_dir, deriv_root, seg="insula"
        ),
        "figure3_gradients": lambda: figure_gradients(
            analysis_dir, deriv_root, subjects
        ),
        "figure4_rtop_maps": lambda: figure_isocontours(deriv_root, subjects),
        "figure7_acc_subdivisions": lambda: figure_subdivisions(
            table, analysis_dir, deriv_root, seg="acc"
        ),
        "figure8_cognitive_control": lambda: figure_cca(analysis_dir),
    }
    assert set(figures) == set(FIGURE_BUILDERS)
    if only is not None:
        unknown = sorted(set(only) - set(figures))
        if unknown:
            raise ValueError(
                f"Unknown figure(s): {unknown}. Known: {', '.join(figures)}"
            )
        figures = {k: v for k, v in figures.items() if k in only}

    for stem, build in figures.items():
        _save(build(), out_dir / stem, formats, dpi)
        written.append(out_dir / stem)

    print(f"Wrote {len(written)} figure(s) to {out_dir}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args(argv)
    run_figures(
        args.deriv_root, formats=tuple(args.formats), dpi=args.dpi, only=args.only
    )


if __name__ == "__main__":
    main()


