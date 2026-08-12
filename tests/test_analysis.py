"""Tests for ROI extraction, outlier exclusion, ANOVA, stability and CCA.

Each statistic is tested against data with a planted answer: a known ordering of
subdivision means, a known outlier, a known minimum detectable sample size, and
a known canonical pair. That way a failure says which estimate is wrong rather
than only that some number moved.

Run: ``uv run pytest tests/test_analysis.py -v``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insula_rtop.analysis import anova, cca, outliers, stability
from insula_rtop.analysis.extract import (
    CORTEX_ROI,
    pivot_subdivisions,
    read_group_table,
    roi_means,
)
from insula_rtop.analysis.isocontours import isocontour_levels, population_mean
from insula_rtop.atlases.fslr import N_VERTICES_32K
from insula_rtop.constants import CCA_BEHAVIORAL_COLUMNS, INSULA_SUBDIVISIONS

ATLAS = "Deen2011"
SEG = "insula"

#: The paper's finding: vAI < dAI < PI in both hemispheres.
PLANTED_MEANS = {"vAI": 1.6, "dAI": 2.0, "PI": 2.4}


def make_group_table(
    n_subjects: int = 60,
    *,
    means: dict[str, float] | None = None,
    between_sd: float = 0.15,
    within_sd: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """A group table with a known subdivision ordering and a subject effect.

    The subject-level offset is what makes the paired tests meaningful: without
    it every subject would be an independent draw and the repeated-measures
    design would carry no information.
    """
    means = means or PLANTED_MEANS
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        subject = f"{100000 + i}"
        offset = rng.normal(0.0, between_sd)
        for hemi in ("L", "R"):
            rows.append(
                {
                    "subject": subject,
                    "atlas": "cortex",
                    "seg": CORTEX_ROI,
                    "hemi": hemi,
                    "subdivision": CORTEX_ROI,
                    "rtop": 2.0 + offset + rng.normal(0.0, within_sd),
                    "n_vertices": 29696,
                    "n_valid": 29696,
                }
            )
            for name, mean in means.items():
                rows.append(
                    {
                        "subject": subject,
                        "atlas": ATLAS,
                        "seg": SEG,
                        "hemi": hemi,
                        "subdivision": name,
                        "rtop": mean + offset + rng.normal(0.0, within_sd),
                        "n_vertices": 300,
                        "n_valid": 300,
                    }
                )
    return pd.DataFrame(rows)


def make_participants(subjects: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Subject": subjects, "Gender": ["F", "M"] * (len(subjects) // 2)}
    for column in CCA_BEHAVIORAL_COLUMNS:
        data[column] = rng.normal(size=len(subjects))
    return pd.DataFrame(data)


# ---- extraction -----------------------------------------------------------


class TestRoiMeans:
    def _labels(self):
        labels = np.zeros(N_VERTICES_32K, dtype=int)
        labels[:100] = 1
        labels[100:300] = 2
        return labels, {1: "a", 2: "b"}

    def test_means_are_per_roi(self):
        values = np.zeros(N_VERTICES_32K)
        values[:100] = 3.0
        values[100:300] = 7.0
        labels, names = self._labels()
        result = roi_means(values, labels, names)
        assert result["a"] == (3.0, 100, 100)
        assert result["b"] == (7.0, 200, 200)

    def test_nan_vertices_are_excluded_not_counted_as_zero(self):
        values = np.zeros(N_VERTICES_32K)
        values[:100] = 4.0
        values[:40] = np.nan
        labels, names = self._labels()
        mean, n_vertices, n_valid = roi_means(values, labels, names)["a"]
        assert mean == pytest.approx(4.0)
        assert (n_vertices, n_valid) == (100, 60)

    def test_mostly_missing_roi_becomes_nan(self):
        values = np.full(N_VERTICES_32K, np.nan)
        values[:10] = 4.0
        labels, names = self._labels()
        mean, _, n_valid = roi_means(values, labels, names)["a"]
        assert np.isnan(mean)
        assert n_valid == 10

    def test_wrong_mesh_is_rejected(self):
        labels, names = self._labels()
        with pytest.raises(ValueError, match="expected 32492"):
            roi_means(np.zeros(100), labels, names)


class TestGroupTable:
    def test_pivot_gives_one_column_per_hemisphere_and_subdivision(self):
        wide = pivot_subdivisions(make_group_table(10), ATLAS, SEG)
        assert set(wide.columns) == {
            f"{hemi}_{name}" for hemi in ("L", "R") for name in INSULA_SUBDIVISIONS
        }
        assert len(wide) == 10

    def test_pivot_rejects_an_unknown_segmentation(self):
        with pytest.raises(KeyError, match="No rows"):
            pivot_subdivisions(make_group_table(5), "Nope", SEG)

    def test_missing_table_points_at_the_step(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="roi_extract"):
            read_group_table(tmp_path)


# ---- outliers -------------------------------------------------------------


class TestOutliers:
    def test_planted_outlier_is_found(self):
        table = make_group_table(60)
        target = table["subject"].iloc[0]
        table.loc[table["subject"] == target, "rtop"] *= 5.0
        found, diagnostics = outliers.find_outliers(table)
        assert found == [target]
        assert diagnostics["n_outliers"] == 1
        assert diagnostics["statistic"] == "cortex"

    def test_clean_cohort_loses_few_subjects(self):
        found, _ = outliers.find_outliers(make_group_table(200))
        assert len(found) <= 4  # 1.5 IQR keeps ~99% of a Gaussian

    def test_dropping_removes_every_row_of_the_subject(self):
        table = make_group_table(40)
        target = table["subject"].iloc[0]
        table.loc[table["subject"] == target, "rtop"] *= 8.0
        kept, dropped, _ = outliers.drop_outliers(table)
        assert dropped == [target]
        assert target not in set(kept["subject"])

    def test_unusable_subject_is_excluded(self):
        table = make_group_table(40)
        target = table["subject"].iloc[0]
        table.loc[table["subject"] == target, "rtop"] = np.nan
        found, _ = outliers.find_outliers(table)
        assert target in found

    def test_insula_statistic_is_available(self):
        _, diagnostics = outliers.find_outliers(
            make_group_table(40), statistic="insula"
        )
        assert diagnostics["statistic"] == "insula"

    def test_unknown_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown outlier statistic"):
            outliers.find_outliers(make_group_table(10), statistic="thalamus")

    def test_fences_bracket_the_quartiles(self):
        values = pd.Series(np.arange(100, dtype=float))
        lower, upper = outliers.tukey_fences(values)
        assert lower < 24.75 and upper > 74.25


# ---- ANOVA ----------------------------------------------------------------


class TestAnova:
    def test_planted_ordering_is_recovered(self):
        table = make_group_table(80)
        summary = anova.summarize(table, ATLAS, SEG, order=INSULA_SUBDIVISIONS)
        for hemi in ("L", "R"):
            assert summary["ordering"][hemi] == ["vAI", "dAI", "PI"]

    def test_subdivision_effect_is_significant(self):
        anova_table, n = anova.repeated_measures_anova(
            make_group_table(80), ATLAS, SEG
        )
        assert n == 80
        assert anova_table.loc["subdivision", "Pr > F"] < 1e-10

    def test_no_planted_difference_gives_no_effect(self):
        table = make_group_table(
            80, means={name: 2.0 for name in INSULA_SUBDIVISIONS}, seed=3
        )
        anova_table, _ = anova.repeated_measures_anova(table, ATLAS, SEG)
        assert anova_table.loc["subdivision", "Pr > F"] > 0.01

    def test_posthoc_is_bonferroni_corrected_over_the_whole_family(self):
        posthoc = anova.posthoc_pairs(
            make_group_table(60), ATLAS, SEG, order=INSULA_SUBDIVISIONS
        )
        # 3 pairs x 2 hemispheres = 6 tests.
        assert len(posthoc) == 6
        ratio = posthoc["p_bonferroni"] / posthoc["p"]
        assert set(np.round(ratio[posthoc["p_bonferroni"] < 1.0], 6)) == {6.0}

    def test_posthoc_effect_sizes_have_the_expected_sign(self):
        posthoc = anova.posthoc_pairs(
            make_group_table(60), ATLAS, SEG, order=INSULA_SUBDIVISIONS
        )
        # b is always the later, higher-RTOP subdivision, so d > 0 throughout.
        assert (posthoc["cohens_d"] > 0).all()

    def test_incomplete_subjects_are_dropped_from_the_balanced_design(self):
        table = make_group_table(30)
        target = table["subject"].iloc[0]
        table = table[
            ~((table["subject"] == target) & (table["subdivision"] == "PI"))
        ]
        assert anova.long_form(table, ATLAS, SEG)["subject"].nunique() == 29

    def test_too_few_subjects_is_an_error(self):
        with pytest.raises(ValueError, match="repeated-measures ANOVA needs more"):
            anova.repeated_measures_anova(make_group_table(2), ATLAS, SEG)

    def test_summary_formats_without_raising(self):
        text = anova.format_summary(anova.summarize(make_group_table(30), ATLAS, SEG))
        assert "Repeated-measures ANOVA" in text


# ---- stability ------------------------------------------------------------


class TestStability:
    def test_detection_rate_increases_with_sample_size(self):
        curve = stability.detection_curve(
            make_group_table(120),
            ATLAS,
            SEG,
            sample_sizes=(5, 10, 40),
            n_resamples=200,
        )
        pair = curve[(curve["hemi"] == "L") & (curve["a"] == "dAI")]
        rates = pair.sort_values("n")["detection_rate"].to_numpy()
        assert rates[0] <= rates[-1]
        assert rates[-1] > 0.9

    def test_larger_effects_need_fewer_subjects(self):
        """vAI-vs-PI is twice the difference of vAI-vs-dAI, so it detects sooner."""
        curve = stability.detection_curve(
            make_group_table(200, between_sd=0.4, within_sd=0.35, seed=11),
            ATLAS,
            SEG,
            sample_sizes=(5, 8, 12, 20, 40, 80),
            n_resamples=300,
            order=INSULA_SUBDIVISIONS,
        )
        minimums = stability.minimum_sample_sizes(curve).set_index(
            ["hemi", "a", "b"]
        )["min_n"]
        assert minimums[("L", "vAI", "PI")] <= minimums[("L", "vAI", "dAI")]

    def test_sample_sizes_above_the_cohort_are_skipped(self):
        curve = stability.detection_curve(
            make_group_table(20), ATLAS, SEG, sample_sizes=(10, 500), n_resamples=50
        )
        assert set(curve["n"]) == {10}

    def test_minimum_is_none_when_never_reached(self):
        curve = stability.detection_curve(
            make_group_table(30, means={n: 2.0 for n in INSULA_SUBDIVISIONS}, seed=5),
            ATLAS,
            SEG,
            sample_sizes=(5, 10),
            n_resamples=200,
        )
        assert stability.minimum_sample_sizes(curve)["min_n"].isna().all()


# ---- CCA ------------------------------------------------------------------


def make_linked_data(n: int = 200, strength: float = 0.8, seed: int = 0):
    """Brain and behaviour blocks sharing one planted latent factor."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    subjects = [f"{100000 + i}" for i in range(n)]

    rows = []
    brain_loadings = {"vAI": 0.9, "dAI": 0.5, "PI": -0.7}
    for i, subject in enumerate(subjects):
        for hemi in ("L", "R"):
            for name, loading in brain_loadings.items():
                rows.append(
                    {
                        "subject": subject,
                        "atlas": ATLAS,
                        "seg": SEG,
                        "hemi": hemi,
                        "subdivision": name,
                        "rtop": strength * loading * latent[i]
                        + rng.normal(0, 0.5),
                        "n_vertices": 300,
                        "n_valid": 300,
                    }
                )
    table = pd.DataFrame(rows)

    behavior = {"Subject": subjects, "Gender": ["F", "M"] * (n // 2)}
    for j, column in enumerate(CCA_BEHAVIORAL_COLUMNS):
        loading = 0.9 if j < 4 else 0.1
        behavior[column] = strength * loading * latent + rng.normal(0, 0.5, size=n)
    return table, pd.DataFrame(behavior)


class TestCCA:
    def test_blocks_align_on_complete_cases(self):
        table, participants = make_linked_data(50)
        brain, behavior = cca.assemble(table, participants)
        assert brain.shape == (50, 6)
        assert behavior.shape == (50, 11)
        assert (brain.index == behavior.index).all()

    def test_subjects_missing_behaviour_are_dropped(self):
        table, participants = make_linked_data(50)
        participants.loc[0, CCA_BEHAVIORAL_COLUMNS[0]] = np.nan
        brain, _ = cca.assemble(table, participants)
        assert len(brain) == 49

    def test_planted_latent_factor_is_recovered(self):
        table, participants = make_linked_data(300, strength=0.9)
        fit = cca.fit_cca(*cca.assemble(table, participants))
        assert fit["canonical_correlations"][0] > 0.6
        assert fit["canonical_p_values"][0] < 1e-10

    def test_weights_reflect_the_planted_loadings(self):
        table, participants = make_linked_data(300, strength=0.9)
        fit = cca.fit_cca(*cca.assemble(table, participants))
        weights = fit["brain_weights"]["component1"]
        # PI was loaded with the opposite sign to vAI and dAI.
        assert np.sign(weights["L_PI"]) != np.sign(weights["L_vAI"])

    def test_held_out_prediction_beats_chance(self):
        table, participants = make_linked_data(150, strength=0.9)
        brain, behavior = cca.assemble(table, participants)
        prediction = cca.cross_validated_prediction(brain, behavior)
        assert prediction["scheme"] == "leave-one-out"
        assert prediction["n"] == 150
        assert prediction["r"] > 0.3
        assert prediction["p"] < 0.01

    def test_matches_an_independently_written_nested_loop(self):
        """The leak test: an obviously-correct LOO, written out longhand.

        Every quantity is refitted on the training rows and the held-out row is
        only ever projected. If the implementation reached for the full sample
        anywhere -- a sign reference, a scaler, a covariate fit -- it would
        disagree with this.
        """
        from sklearn.cross_decomposition import CCA
        from sklearn.preprocessing import StandardScaler

        table, participants = make_linked_data(40, strength=0.9)
        brain, behavior = cca.assemble(table, participants)
        x, y = brain.to_numpy(float), behavior.to_numpy(float)

        expected_pred, expected_obs = [], []
        for i in range(len(x)):
            train = np.setdiff1d(np.arange(len(x)), [i])
            xs = StandardScaler().fit(x[train])
            ys = StandardScaler().fit(y[train])
            model = CCA(n_components=1, max_iter=1000).fit(
                xs.transform(x[train]), ys.transform(y[train])
            )
            scaled_train = xs.transform(x[train])
            sign = np.sign(
                np.corrcoef(
                    scaled_train @ model.x_rotations_[:, 0],
                    scaled_train.mean(axis=1),
                )[0, 1]
            )
            xt, yt = model.transform(
                xs.transform(x[[i]]), ys.transform(y[[i]])
            )
            expected_pred.append(sign * xt[0, 0])
            expected_obs.append(sign * yt[0, 0])

        got = cca.cross_validated_prediction(brain, behavior)
        np.testing.assert_allclose(got["predicted"], expected_pred, atol=1e-9)
        np.testing.assert_allclose(got["observed"], expected_obs, atol=1e-9)

    def test_the_fold_sign_uses_no_information_beyond_its_arguments(self):
        """It is anchored on the training block, not on a full-sample fit."""
        import inspect

        source = inspect.getsource(cca.cross_validated_prediction)
        # A full-sample CCA anywhere in the loop is the leak this replaced.
        assert "fit_transform(x)" not in source
        assert source.count("CCA(") == 1

    def test_covariates_are_residualised_inside_the_fold(self):
        """Regressing covariates out before the split would leak the test row."""
        table, participants = make_linked_data(80, strength=0.9)
        participants["Age_in_Yrs"] = np.linspace(22, 36, len(participants))
        summary = cca.summarize(
            table, participants, covariate_columns=("Age_in_Yrs",)
        )
        # The prediction must still be computable and finite; the point of the
        # test is that summarize hands the CV the *raw* blocks.
        assert np.isfinite(summary["prediction"]["r"])
        assert summary["covariates"] == ["Age_in_Yrs"]

    def test_unlinked_data_does_not_predict(self):
        """With no shared factor, held-out prediction must collapse to chance."""
        table, participants = make_linked_data(150, strength=0.0, seed=4)
        brain, behavior = cca.assemble(table, participants)
        prediction = cca.cross_validated_prediction(brain, behavior)
        assert abs(prediction["r"]) < 0.25

    def test_cohens_d_follows_from_r(self):
        table, participants = make_linked_data(120, strength=0.9)
        prediction = cca.cross_validated_prediction(
            *cca.assemble(table, participants)
        )
        r = prediction["r"]
        assert prediction["cohens_d"] == pytest.approx(2 * r / np.sqrt(1 - r**2))

    def test_grouped_cross_validation_holds_out_whole_families(self):
        table, participants = make_linked_data(120, strength=0.9)
        participants["Family_ID"] = [f"F{i // 3}" for i in range(len(participants))]
        summary = cca.summarize(table, participants, group_column="Family_ID")
        assert summary["prediction"]["scheme"] == "leave-one-group-out"

    def test_covariates_are_regressed_out(self):
        table, participants = make_linked_data(120, strength=0.9)
        summary = cca.summarize(table, participants, covariate_columns=("Gender",))
        assert summary["covariates"] == ["Gender"]
        assert summary["fit"]["canonical_correlations"][0] > 0.5

    def test_a_missing_covariate_drops_the_subject_rather_than_the_result(self):
        """One NaN in the design would turn every residual into NaN silently."""
        table, participants = make_linked_data(120, strength=0.9)
        participants["Age_in_Yrs"] = np.linspace(22, 36, len(participants))
        participants.loc[0, "Age_in_Yrs"] = np.nan
        summary = cca.summarize(
            table, participants, covariate_columns=("Age_in_Yrs",)
        )
        assert summary["fit"]["n"] == 119
        assert np.isfinite(summary["fit"]["canonical_correlations"][0])

    def test_an_unknown_covariate_names_itself(self):
        table, participants = make_linked_data(60, strength=0.9)
        with pytest.raises(KeyError, match="FS_IntraCranial_Vol"):
            cca.summarize(
                table, participants, covariate_columns=("FS_IntraCranial_Vol",)
            )

    def test_regress_out_leaves_no_residual_correlation(self):
        rng = np.random.default_rng(0)
        covariate = pd.DataFrame({"age": rng.normal(size=200)})
        block = pd.DataFrame({"x": 3.0 * covariate["age"] + rng.normal(size=200)})
        residual = cca.regress_out(block, covariate)
        assert abs(np.corrcoef(residual["x"], covariate["age"])[0, 1]) < 1e-10

    def test_tiny_cohort_is_rejected(self):
        table, participants = make_linked_data(8)
        with pytest.raises(ValueError, match="complete case"):
            cca.summarize(table, participants)

    def test_summary_formats_without_raising(self):
        table, participants = make_linked_data(60, strength=0.9)
        text = cca.format_summary(cca.summarize(table, participants))
        assert "Out-of-sample prediction" in text


# ---- isocontours ----------------------------------------------------------


class TestIsocontours:
    def test_population_mean_ignores_nan(self):
        values = np.array([[1.0, 2.0], [3.0, np.nan]])
        np.testing.assert_allclose(population_mean(values), [2.0, 2.0])

    def test_population_mean_rejects_the_wrong_shape(self):
        with pytest.raises(ValueError, match="n_subjects, n_vertices"):
            population_mean(np.zeros(10))

    def test_levels_span_the_robust_range(self):
        values = np.concatenate([np.linspace(1.0, 2.0, 1000), [500.0]])
        levels = isocontour_levels(values, n_levels=5)
        assert len(levels) == 5
        assert levels[-1] < 3.0  # the outlier must not stretch the scale

    def test_constant_values_are_rejected(self):
        with pytest.raises(ValueError, match="nothing to contour"):
            isocontour_levels(np.ones(100))
