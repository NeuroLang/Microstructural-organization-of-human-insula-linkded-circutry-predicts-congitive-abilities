"""The paper's second cohort filter: drop RTOP outliers at 1.5 IQR.

    Twenty participants were excluded in the analysis because of outliers of
    RTOP values (defined as 1.5 interquartile ranges), leading to 413 subjects
    in the final sample. (Appendix 1)

**Assumption.** The paper does not say *which* RTOP the interquartile range is
computed on. The default here is the subject's mean cortical RTOP, averaged over
both hemispheres -- the broadest summary of the measurement, and the one that
catches a globally mis-scaled subject (a bad D_vent, a failed fit) rather than a
subject who is merely unusual in the insula. ``statistic="insula"`` computes it
on the mean insular RTOP instead. See README, "Assumptions".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insula_rtop.analysis.extract import CORTEX_ROI

#: Tukey fence multiplier, quoted from Appendix 1.
IQR_MULTIPLIER = 1.5


def subject_statistic(
    table: pd.DataFrame,
    *,
    statistic: str = "cortex",
    atlas: str = "Deen2011",
) -> pd.Series:
    """One RTOP summary per subject, for the outlier rule to act on."""
    if statistic == "cortex":
        subset = table[table["seg"] == CORTEX_ROI]
    elif statistic == "insula":
        subset = table[(table["atlas"] == atlas) & (table["seg"] == "insula")]
    else:
        raise ValueError(
            f"Unknown outlier statistic {statistic!r}; expected 'cortex' or 'insula'."
        )
    if subset.empty:
        raise KeyError(f"No rows to compute the {statistic!r} outlier statistic from")
    return subset.groupby("subject")["rtop"].mean()


def tukey_fences(
    values: pd.Series, *, multiplier: float = IQR_MULTIPLIER
) -> tuple[float, float]:
    """``(lower, upper)`` = Q1 - k*IQR, Q3 + k*IQR."""
    finite = values.dropna()
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def find_outliers(
    table: pd.DataFrame,
    *,
    statistic: str = "cortex",
    atlas: str = "Deen2011",
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[list[str], dict]:
    """Subjects outside the Tukey fences, plus the fences themselves."""
    values = subject_statistic(table, statistic=statistic, atlas=atlas)
    lower, upper = tukey_fences(values, multiplier=multiplier)
    # NaN means the statistic could not be computed at all -- an unusable
    # subject, excluded for the same reason as a numeric outlier.
    outlying = values[~values.between(lower, upper) | values.isna()]
    diagnostics = {
        "statistic": statistic,
        "multiplier": multiplier,
        "lower_fence": lower,
        "upper_fence": upper,
        "n_subjects": int(values.size),
        "n_outliers": int(outlying.size),
    }
    return sorted(outlying.index.astype(str)), diagnostics


def drop_outliers(
    table: pd.DataFrame,
    *,
    statistic: str = "cortex",
    atlas: str = "Deen2011",
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Return the table without the outlying subjects."""
    outliers, diagnostics = find_outliers(
        table, statistic=statistic, atlas=atlas, multiplier=multiplier
    )
    kept = table[~table["subject"].astype(str).isin(outliers)].copy()
    return kept, outliers, diagnostics
