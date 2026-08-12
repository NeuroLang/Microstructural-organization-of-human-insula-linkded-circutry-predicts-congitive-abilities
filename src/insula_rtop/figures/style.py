"""Shared look for the reproduced figures, matched to the published ones.

The published figures are seaborn's ``darkgrid`` theme with a three-colour
subdivision palette (posterior cyan, dorsal-anterior green, ventral-anterior
red), violins with inner boxes, and regression panels with a translucent CI
band. Every choice here is read off the article's Figures 2, 3, 7 and 8 rather
than invented, so a reader can put the two side by side.

Panels are laid out the same way too: two sections stacked vertically, headed
**I. Functional Parcellation** (Deen et al., 2011) and **II. Multimodal
Parcellation** (Glasser et al., 2016), with panel letters ``A.``-``F.``
left-aligned above each axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

#: Subdivision colours, sampled from the published figures.
SUBDIVISION_COLORS = {
    "PI": "#29B6D8",
    "dAI": "#63C363",
    "vAI": "#EF5B5B",
    "ACC-PI": "#29B6D8",
    "ACC-dAI": "#63C363",
    "ACC-vAI": "#EF5B5B",
}

#: The published panels order subdivisions posterior-first, which is also
#: descending RTOP -- so the ordering is the result, read left to right.
PLOT_ORDER = ("PI", "dAI", "vAI")
ACC_PLOT_ORDER = ("ACC-PI", "ACC-dAI", "ACC-vAI")

#: Section headings, as printed in the article.
SECTION_FUNCTIONAL = "I. Functional Parcellation"
SECTION_MULTIMODAL = "II. Multimodal Parcellation"

#: The RTOP colour scale, matched to the article's Figures 3A and 4: low values
#: green, rising through teal and blue to purple at the top. Defined once so
#: Figures 3 and 4 cannot drift apart -- they show the same quantity.
RTOP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rtop",
    ["#3FBF3F", "#2FBFA0", "#2E86D8", "#2B3FC8", "#5B2D91", "#7B2D6B"],
)

#: seaborn's default blue and orange, used for the Figure 8 scatter and fit.
SCATTER_COLOR = "#4C72B0"
FIT_COLOR = "#DD8452"


def apply_theme() -> None:
    """seaborn ``darkgrid``, the theme the published figures use."""
    sns.set_theme(style="darkgrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 13,
            "axes.titleweight": "normal",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
        }
    )


def panel_title(ax, letter: str, text: str) -> None:
    """``A. Title``, left-aligned above the axes, as in the article."""
    ax.set_title(f"{letter}. {text}", loc="left", pad=10)


def section_heading(fig, text: str, y: float) -> None:
    """A bold centred section heading spanning the figure."""
    fig.text(0.5, y, text, ha="center", va="center", fontsize=16, fontweight="bold")


def section_rule(fig, y: float) -> None:
    """The dashed separator the article draws between the two sections."""
    fig.add_artist(
        plt.Line2D(
            [0.02, 0.98], [y, y], color="0.75", linestyle="--", linewidth=1
        )
    )


def significance_bracket(
    ax, x1: float, x2: float, y: float, height: float, label: str = "*"
) -> None:
    """The ``*`` bracket the article draws over significant contrasts."""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        linewidth=1.2,
        color="0.15",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + height,
        label,
        ha="center",
        va="bottom",
        color="0.15",
        fontsize=13,
    )
