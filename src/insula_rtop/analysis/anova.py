"""Subdivision x hemisphere ANOVA and post-hoc contrasts (Figures 2B and 7B).

    We conducted an ANOVA with factors subdivision and hemisphere [...] post-hoc
    paired t-tests with Bonferroni correction. (Materials and methods)

Both factors are within-subject -- every participant contributes all three
subdivisions in both hemispheres -- so this is a 3 x 2 repeated-measures ANOVA,
run with :class:`statsmodels.stats.anova.AnovaRM`.

The expected result is a monotone ordering ``vAI < dAI < PI`` in both
hemispheres (Figure 2B), mirrored by ``ACC-vAI < ACC-dAI < ACC-PI`` (Figure 7B).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Paired-samples effect size: mean difference over the SD of differences."""
    diff = np.asarray(y) - np.asarray(x)
    return float(diff.mean() / diff.std(ddof=1))


def long_form(table: pd.DataFrame, atlas: str, seg: str) -> pd.DataFrame:
    """Complete cases only: AnovaRM requires a fully balanced design."""
    subset = table[(table["atlas"] == atlas) & (table["seg"] == seg)][
        ["subject", "hemi", "subdivision", "rtop"]
    ].dropna()
    counts = subset.groupby("subject").size()
    n_cells = subset["hemi"].nunique() * subset["subdivision"].nunique()
    complete = counts[counts == n_cells].index
    return subset[subset["subject"].isin(complete)].copy()


def repeated_measures_anova(
    table: pd.DataFrame, atlas: str, seg: str
) -> tuple[pd.DataFrame, int]:
    """3 x 2 repeated-measures ANOVA. Returns ``(anova_table, n_subjects)``."""
    data = long_form(table, atlas, seg)
    n_subjects = data["subject"].nunique()
    if n_subjects < 3:
        raise ValueError(
            f"Only {n_subjects} complete subject(s) for atlas={atlas!r}, "
            f"seg={seg!r}; a repeated-measures ANOVA needs more."
        )
    result = AnovaRM(
        data, depvar="rtop", subject="subject", within=["subdivision", "hemi"]
    ).fit()
    return result.anova_table, n_subjects


def posthoc_pairs(
    table: pd.DataFrame,
    atlas: str,
    seg: str,
    *,
    order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Bonferroni-corrected paired t-tests between subdivisions, per hemisphere.

    The correction spans every test in the family: all subdivision pairs in both
    hemispheres.
    """
    data = long_form(table, atlas, seg)
    subdivisions = list(order) if order else sorted(data["subdivision"].unique())
    hemis = sorted(data["hemi"].unique())
    pairs = list(itertools.combinations(subdivisions, 2))
    n_tests = len(pairs) * len(hemis)

    rows = []
    for hemi in hemis:
        wide = data[data["hemi"] == hemi].pivot(
            index="subject", columns="subdivision", values="rtop"
        )
        for a, b in pairs:
            x, y = wide[a].to_numpy(), wide[b].to_numpy()
            t, p = stats.ttest_rel(x, y)
            rows.append(
                {
                    "hemi": hemi,
                    "a": a,
                    "b": b,
                    "mean_a": float(x.mean()),
                    "mean_b": float(y.mean()),
                    "t": float(t),
                    "p": float(p),
                    "p_bonferroni": float(min(p * n_tests, 1.0)),
                    "cohens_d": cohens_d_paired(x, y),
                    "n": int(len(x)),
                }
            )
    return pd.DataFrame(rows)


def subdivision_ordering(
    table: pd.DataFrame, atlas: str, seg: str
) -> dict[str, list[str]]:
    """Subdivisions sorted by mean RTOP, per hemisphere -- the paper's claim."""
    data = long_form(table, atlas, seg)
    means = data.groupby(["hemi", "subdivision"])["rtop"].mean()
    return {
        hemi: means[hemi].sort_values().index.tolist()
        for hemi in means.index.get_level_values(0).unique()
    }


def summarize(
    table: pd.DataFrame,
    atlas: str,
    seg: str,
    *,
    order: tuple[str, ...] | None = None,
) -> dict:
    """Everything Figure 2B / 7B reports, in one dict."""
    anova_table, n_subjects = repeated_measures_anova(table, atlas, seg)
    data = long_form(table, atlas, seg)
    descriptives = (
        data.groupby(["hemi", "subdivision"])["rtop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return {
        "atlas": atlas,
        "seg": seg,
        "n_subjects": n_subjects,
        "anova": anova_table,
        "descriptives": descriptives,
        "posthoc": posthoc_pairs(table, atlas, seg, order=order),
        "ordering": subdivision_ordering(table, atlas, seg),
    }


def format_summary(summary: dict) -> str:
    """A readable block for the run log and the report."""
    lines = [
        f"=== {summary['atlas']} / {summary['seg']} "
        f"(N = {summary['n_subjects']}) ===",
        "Repeated-measures ANOVA (subdivision x hemisphere):",
        summary["anova"].to_string(),
        "",
        "Means:",
        summary["descriptives"].to_string(index=False),
        "",
        "Post-hoc paired t-tests (Bonferroni-corrected):",
        summary["posthoc"].to_string(index=False),
        "",
        "Ordering by mean RTOP (lowest first):",
    ]
    for hemi, order in summary["ordering"].items():
        lines.append(f"  {hemi}: {' < '.join(order)}")
    return "\n".join(lines)
