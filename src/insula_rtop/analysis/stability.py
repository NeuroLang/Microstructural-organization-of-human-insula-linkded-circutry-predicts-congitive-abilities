"""How many participants it takes to see the subdivision differences (Fig 2C, 7C).

    To evaluate stability of our findings regarding RTOP differences between
    ROIs, we used subsampling procedures and determined minimum sample sizes
    that consistently reproduced findings. [...] A sample size of N = 25 was
    sufficient to achieve a stable differentiation (p<0.01) between PI and vAI
    in both hemispheres, while differentiating the vAI and dAI required a larger
    sample size of N = 100. (Results)

For each candidate sample size the full cohort is subsampled without replacement
many times, the paired t-test is re-run on each subsample, and the fraction of
subsamples reaching ``p < alpha`` is recorded. The reported minimum is the
smallest sample size at which that fraction reaches
:data:`DEFAULT_STABILITY_THRESHOLD`.

**Assumption.** The paper reports "stable differentiation (p<0.01)" without
defining "stable". This uses 95% of subsamples significant at alpha = 0.01;
both numbers are parameters, and the whole detection curve is returned so a
different definition can be read straight off it. See README, "Assumptions".
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from insula_rtop.analysis.anova import long_form

#: Significance level, quoted from the Results.
DEFAULT_ALPHA = 0.01

#: Fraction of subsamples that must reach *alpha* for a sample size to count as
#: stable. This reproduction's definition, not the paper's.
DEFAULT_STABILITY_THRESHOLD = 0.95

#: Sample sizes swept by default, chosen to bracket the paper's N = 25 and 100.
DEFAULT_SAMPLE_SIZES = (10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400)

DEFAULT_N_RESAMPLES = 1000


def detection_curve(
    table: pd.DataFrame,
    atlas: str,
    seg: str,
    *,
    sample_sizes=DEFAULT_SAMPLE_SIZES,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    order: tuple[str, ...] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Fraction of subsamples reaching ``p < alpha``, per pair, hemisphere and N."""
    data = long_form(table, atlas, seg)
    subdivisions = list(order) if order else sorted(data["subdivision"].unique())
    rng = np.random.default_rng(seed)

    rows = []
    for hemi in sorted(data["hemi"].unique()):
        wide = data[data["hemi"] == hemi].pivot(
            index="subject", columns="subdivision", values="rtop"
        )
        n_total = len(wide)
        for a, b in itertools.combinations(subdivisions, 2):
            diff = (wide[b] - wide[a]).to_numpy()
            for n in sample_sizes:
                if n > n_total:
                    continue
                # One (n_resamples, n) index matrix, then a vectorised paired
                # t-test: the test on paired data is a one-sample test on the
                # differences, so no loop over resamples is needed.
                idx = rng.random((n_resamples, n_total)).argsort(axis=1)[:, :n]
                samples = diff[idx]
                _, p = stats.ttest_1samp(samples, popmean=0.0, axis=1)
                rows.append(
                    {
                        "hemi": hemi,
                        "a": a,
                        "b": b,
                        "n": int(n),
                        "detection_rate": float(np.mean(p < alpha)),
                        "n_resamples": int(n_resamples),
                        "alpha": alpha,
                    }
                )
    return pd.DataFrame(rows)


def minimum_sample_sizes(
    curve: pd.DataFrame, *, threshold: float = DEFAULT_STABILITY_THRESHOLD
) -> pd.DataFrame:
    """Smallest swept N whose detection rate reaches *threshold*, per pair."""
    rows = []
    for (hemi, a, b), group in curve.groupby(["hemi", "a", "b"]):
        reached = group[group["detection_rate"] >= threshold]["n"]
        rows.append(
            {
                "hemi": hemi,
                "a": a,
                "b": b,
                "min_n": int(reached.min()) if len(reached) else None,
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    table: pd.DataFrame,
    atlas: str,
    seg: str,
    *,
    sample_sizes=DEFAULT_SAMPLE_SIZES,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
    order: tuple[str, ...] | None = None,
    seed: int = 0,
) -> dict:
    curve = detection_curve(
        table,
        atlas,
        seg,
        sample_sizes=sample_sizes,
        n_resamples=n_resamples,
        alpha=alpha,
        order=order,
        seed=seed,
    )
    return {
        "atlas": atlas,
        "seg": seg,
        "curve": curve,
        "minimum_sample_sizes": minimum_sample_sizes(curve, threshold=threshold),
    }


def format_summary(summary: dict) -> str:
    return "\n".join(
        [
            f"=== stability: {summary['atlas']} / {summary['seg']} ===",
            "Minimum sample size for stable differentiation:",
            summary["minimum_sample_sizes"].to_string(index=False),
        ]
    )
