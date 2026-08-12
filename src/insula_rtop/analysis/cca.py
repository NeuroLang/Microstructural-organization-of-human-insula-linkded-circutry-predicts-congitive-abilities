"""Insula microstructure against cognitive control ability (Figure 8).

    Brain measures consisted of RTOP values for each subdivision of the insular
    cortex in each hemisphere (six variables in total). [...] Together, there
    were 11 behavioral measures. The relationship between brain and behavioral
    measures was examined using Canonical Correlation Analysis (CCA) and a
    cross-validation with prediction approach. [...] Prediction analysis was
    performed using leave-one-out cross-validation. Pearson's correlation was
    used to evaluate the correlation between the predicted brain and behavioral
    measures. (Results / Materials and methods)

Reported in the paper: ``r = 0.19, p < 0.001, Cohen's d = 0.39`` on held-out
data.

Two things the paper leaves open, both handled as parameters:

* **Deconfounding.** No covariate adjustment is described, so none is applied by
  default. ``covariates=`` regresses named columns (age, sex, ...) out of both
  blocks if wanted.
* **Family structure.** HCP-YA contains twins and siblings, and leave-one-out
  cross-validation puts a subject's relatives in the training set. That inflates
  held-out prediction. The published analysis is reproduced as described;
  ``group_column="Family_ID"`` switches to leave-one-family-out so the size of
  the effect can be measured. See README, "Assumptions".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from insula_rtop.analysis.extract import pivot_subdivisions
from insula_rtop.constants import CCA_BEHAVIORAL_COLUMNS


def assemble(
    table: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    atlas: str = "Deen2011",
    seg: str = "insula",
    behavioral_columns=CCA_BEHAVIORAL_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the brain and behaviour blocks on complete cases.

    Returns ``(brain, behavior)`` with a shared subject index.
    """
    brain = pivot_subdivisions(table, atlas, seg)
    behavior = participants.set_index(participants["Subject"].astype(str))[
        list(behavioral_columns)
    ]
    shared = brain.index.intersection(behavior.index)
    brain, behavior = brain.loc[shared], behavior.loc[shared]
    keep = brain.notna().all(axis=1) & behavior.notna().all(axis=1)
    return brain[keep], behavior[keep]


def regress_out(block: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Residualise every column of *block* on *covariates* (with an intercept)."""
    design = np.column_stack([np.ones(len(covariates)), covariates.to_numpy(float)])
    coefficients, *_ = np.linalg.lstsq(design, block.to_numpy(float), rcond=None)
    residuals = block.to_numpy(float) - design @ coefficients
    return pd.DataFrame(residuals, index=block.index, columns=block.columns)


def fit_cca(
    brain: pd.DataFrame, behavior: pd.DataFrame, *, n_components: int = 1
) -> dict:
    """Fit CCA on z-scored blocks and report the canonical correlations."""
    x = StandardScaler().fit_transform(brain.to_numpy(float))
    y = StandardScaler().fit_transform(behavior.to_numpy(float))
    model = CCA(n_components=n_components, max_iter=1000).fit(x, y)
    x_scores, y_scores = model.transform(x, y)

    correlations, p_values = [], []
    for i in range(n_components):
        r, p = stats.pearsonr(x_scores[:, i], y_scores[:, i])
        correlations.append(float(r))
        p_values.append(float(p))

    return {
        "n": int(len(brain)),
        "canonical_correlations": correlations,
        "canonical_p_values": p_values,
        "brain_weights": pd.DataFrame(
            model.x_weights_,
            index=brain.columns,
            columns=[f"component{i + 1}" for i in range(n_components)],
        ),
        "behavior_weights": pd.DataFrame(
            model.y_weights_,
            index=behavior.columns,
            columns=[f"component{i + 1}" for i in range(n_components)],
        ),
        "brain_scores": x_scores,
        "behavior_scores": y_scores,
    }


def _fold_sign(model, x_train_scaled: np.ndarray) -> float:
    """Put a fold's canonical axis in a common orientation, using only that fold.

    A canonical variate's sign is arbitrary and is re-drawn independently on
    every fold, so the folds' scores cannot be pooled until they agree on which
    way the axis points. Anchoring on the *mean of the brain variables* -- a
    fixed, data-independent direction -- fixes it from training data alone.

    Anchoring on a full-sample CCA instead, which is the obvious shortcut, would
    let the held-out subject vote on its own orientation. One bit per fold is a
    small leak, but it is still a leak, and it is avoidable.
    """
    scores = x_train_scaled @ model.x_rotations_[:, 0]
    anchor = x_train_scaled.mean(axis=1)
    if scores.std() == 0 or anchor.std() == 0:
        return 1.0
    return float(np.sign(np.corrcoef(scores, anchor)[0, 1])) or 1.0


def cross_validated_prediction(
    brain: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    covariates: pd.DataFrame | None = None,
) -> dict:
    """Leave-one-out prediction of the behavioural canonical variate.

    Every quantity that touches the held-out subject is estimated on the
    training fold alone: the covariate residualisation, both scalers, the CCA
    weights, and the sign convention. The held-out subject is then projected
    with those training weights, which is what makes the correlation between
    predicted and observed genuinely out of sample.

    Residualising covariates before the split -- the natural place to write it
    -- would leak, because the regression coefficients would be fitted on data
    including the test subject. So it happens inside the loop.
    """
    x = brain.to_numpy(float)
    y = behavior.to_numpy(float)
    design = None if covariates is None else covariates.to_numpy(float)
    splitter = LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    split_args = (x, y, groups.to_numpy()) if groups is not None else (x,)

    predicted, observed = [], []
    for train, test in splitter.split(*split_args):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        if design is not None:
            x_train, x_test = _residualise(design, train, test, x_train, x_test)
            y_train, y_test = _residualise(design, train, test, y_train, y_test)

        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train)
        x_train_scaled = x_scaler.transform(x_train)
        model = CCA(n_components=1, max_iter=1000).fit(
            x_train_scaled, y_scaler.transform(y_train)
        )
        sign = _fold_sign(model, x_train_scaled)
        x_scores, y_scores = model.transform(
            x_scaler.transform(x_test), y_scaler.transform(y_test)
        )
        predicted.extend((sign * x_scores[:, 0]).tolist())
        observed.extend((sign * y_scores[:, 0]).tolist())

    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    r, p = stats.pearsonr(predicted, observed)
    return {
        "n": int(len(observed)),
        "r": float(r),
        "p": float(p),
        # Cohen's d from a correlation: d = 2r / sqrt(1 - r^2).
        "cohens_d": float(2 * r / np.sqrt(1 - r**2)),
        "scheme": "leave-one-group-out" if groups is not None else "leave-one-out",
        "predicted": predicted,
        "observed": observed,
    }


def _residualise(design, train, test, block_train, block_test):
    """Regress *design* out of a block, fitting the model on the training fold."""
    d_train = np.column_stack([np.ones(len(train)), design[train]])
    d_test = np.column_stack([np.ones(len(test)), design[test]])
    coefficients, *_ = np.linalg.lstsq(d_train, block_train, rcond=None)
    return block_train - d_train @ coefficients, block_test - d_test @ coefficients


def _covariate_design(
    participants: pd.DataFrame, columns: list[str], subjects: pd.Index
) -> pd.DataFrame:
    """Numeric covariate block for *subjects*, dummy-coded and complete-case."""
    missing = [c for c in columns if c not in participants.columns]
    if missing:
        raise KeyError(
            f"Covariate column(s) {missing} are not in participants.tsv. "
            "Add them to cohort.PARTICIPANT_COLUMNS or OPTIONAL_COLUMNS and "
            "re-run the `cohort` step."
        )
    block = participants.loc[subjects, columns].dropna()
    return pd.get_dummies(block, drop_first=True).astype(float)


def summarize(
    table: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    atlas: str = "Deen2011",
    seg: str = "insula",
    covariate_columns: tuple[str, ...] = (),
    group_column: str | None = None,
) -> dict:
    brain, behavior = assemble(table, participants, atlas=atlas, seg=seg)
    if len(brain) < 10:
        raise ValueError(
            f"Only {len(brain)} complete case(s) for the CCA; expected the cohort."
        )

    indexed = participants.set_index(participants["Subject"].astype(str))
    covariates = None
    if covariate_columns:
        covariates = _covariate_design(indexed, list(covariate_columns), brain.index)
        # A subject with a missing covariate would otherwise turn the whole
        # least-squares fit into NaN, so drop it as an incomplete case -- the
        # same rule `assemble` applies to the brain and behaviour blocks.
        brain = brain.loc[covariates.index]
        behavior = behavior.loc[covariates.index]

    groups = indexed.loc[brain.index, group_column] if group_column else None
    # The in-sample fit is in-sample by definition, so residualising it up front
    # is fine. The cross-validated one gets the raw blocks and does its own
    # residualisation per fold -- see cross_validated_prediction.
    fit_brain = regress_out(brain, covariates) if covariates is not None else brain
    fit_behavior = (
        regress_out(behavior, covariates) if covariates is not None else behavior
    )
    return {
        "atlas": atlas,
        "seg": seg,
        "covariates": list(covariate_columns),
        "fit": fit_cca(fit_brain, fit_behavior),
        "prediction": cross_validated_prediction(
            brain, behavior, groups=groups, covariates=covariates
        ),
    }


def format_summary(summary: dict) -> str:
    fit = summary["fit"]
    prediction = summary["prediction"]
    weights = (
        fit["behavior_weights"]["component1"].sort_values(ascending=False).to_string()
    )
    return "\n".join(
        [
            f"=== CCA: {summary['atlas']} / {summary['seg']} (N = {fit['n']}) ===",
            f"Covariates regressed out: {summary['covariates'] or 'none'}",
            f"Canonical correlation: r = {fit['canonical_correlations'][0]:.3f}, "
            f"p = {fit['canonical_p_values'][0]:.3g}",
            "",
            f"Out-of-sample prediction ({prediction['scheme']}, "
            f"n = {prediction['n']}):",
            f"  r = {prediction['r']:.3f}, p = {prediction['p']:.3g}, "
            f"Cohen's d = {prediction['cohens_d']:.3f}",
            "",
            "Brain weights (component 1):",
            fit["brain_weights"]["component1"].to_string(),
            "",
            "Behavioural weights (component 1, sorted):",
            weights,
        ]
    )
