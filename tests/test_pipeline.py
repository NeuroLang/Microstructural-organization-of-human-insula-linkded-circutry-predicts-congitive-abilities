"""Config completeness and an end-to-end run on synthetic data.

``TestPipelineConfig`` composes the real Hydra config and asserts that every key
the runner reads exists. A missing key is otherwise invisible until the step is
launched on the cluster, and by then Hydra has created the output directory and
written the config snapshot, so the run *looks* like it started.

``TestEndToEnd`` drives every step after ``rtop_volume`` on a synthetic BIDS
tree. It skips the MAPL fit -- that is covered analytically in
``tests/test_rtop.py`` and is far too slow to run per-test -- and instead plants
an RTOP volume with a known spatial gradient, so the surface projection,
extraction, statistics, gradient directions and figures are all exercised
against data whose answer is known.

Run: ``uv run pytest tests/test_pipeline.py -v``
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from _synthetic_hcp import make_behavioral_csv, make_synthetic_hcp_tree
from hydra import compose, initialize_config_module

import insula_rtop.pipeline
from insula_rtop.analysis.run import ANALYSIS_DIR, run_analysis
from insula_rtop.atlases.fslr import N_VERTICES_32K
from insula_rtop.atlases.run import label_path, table_path, write_label_gii
from insula_rtop.figures.run import FIGURES_DIR, run_figures
from insula_rtop.hcp.cohort import run_cohort
from insula_rtop.hcp.to_bids import raw_paths, run_bidsify
from insula_rtop.pipeline.__main__ import STEPS
from insula_rtop.rtop.run import rtop_paths
from insula_rtop.surface.run import run_surface


def load_config(overrides=()):
    with initialize_config_module(
        config_module="insula_rtop.pipeline", version_base=None
    ):
        return compose(config_name="pipeline", overrides=list(overrides))


class TestPipelineConfig:
    def test_default_steps_are_all_known(self):
        cfg = load_config()
        assert set(cfg.steps) <= set(STEPS)

    def test_every_step_the_runner_dispatches_is_configured(self):
        cfg = load_config()
        for step in STEPS:
            if step in ("cohort", "analysis", "figures", "atlas_labels"):
                assert step in cfg, f"{step} has no config block"
            else:
                assert step in cfg or step == "hcp2bids"

    @pytest.mark.parametrize(
        ("block", "keys"),
        [
            ("cohort", ("behavioral_csv", "restricted_csv", "releases",
                        "require_behavior")),
            ("hcp2bids", ("force", "skip_existing", "resume", "slurm_time",
                          "slurm_cpus_per_task", "slurm_mem")),
            ("rtop_volume", ("fit_mask", "ribbon_dilation", "ventricle_erosion",
                             "radial_order", "fit_jobs", "force", "skip_existing",
                             "resume", "slurm_time", "slurm_cpus_per_task",
                             "slurm_mem")),
            ("rtop_surface", ("force", "skip_existing", "resume", "slurm_time",
                              "slurm_cpus_per_task", "slurm_mem")),
            ("atlas_labels", ("cache_dir", "acc_labels", "force")),
            ("analysis", ("outlier_statistic", "n_resamples", "covariates",
                          "group_column", "rebuild_table", "gradient_smoothing")),
            ("figures", ("only", "formats", "dpi")),
            ("slurm", ("use", "n_jobs", "partition", "account", "max_jobs")),
        ],
    )
    def test_required_keys_exist(self, block, keys):
        cfg = load_config()
        for key in keys:
            assert key in cfg[block], f"cfg.{block}.{key} is missing"

    def test_no_key_the_runner_reads_is_missing_from_the_config(self):
        """The list above is hand-written, so derive the same check from source.

        A key added to the runner and forgotten in `pipeline.yaml` is invisible
        until the step is launched -- and by then Hydra has created the output
        directory and written its config snapshot, so the run looks like it
        started. This caught `analysis.gradient_smoothing`.
        """
        import re

        source = Path(insula_rtop.pipeline.__file__).with_name("__main__.py")
        cfg = load_config()
        for block, key in set(
            re.findall(r"cfg\.([a-z_0-9]+)\.([a-z_0-9]+)", source.read_text())
        ):
            assert block in cfg, f"cfg.{block} is missing"
            assert key in cfg[block], f"cfg.{block}.{key} is read but not configured"

    def test_the_default_site_is_portable(self):
        """A fresh clone must not point at anyone's cluster."""
        cfg = load_config()
        for path in (cfg.bids_root, cfg.deriv_root, cfg.site.hcp_data_root):
            assert not str(path).startswith("/"), f"{path} is absolute"
        assert str(cfg.cohort.behavioral_csv).endswith("hcp_behavioral.csv")

    def test_a_cluster_site_overrides_every_path(self):
        cfg = load_config(["site=margaret"])
        assert str(cfg.bids_root).startswith("/data/parietal")
        assert str(cfg.deriv_root).startswith("/data/parietal")

    @pytest.mark.parametrize("site", ["local", "margaret", "example"])
    def test_every_shipped_site_exposes_the_same_keys(self, site):
        """`cp example.yaml mysite.yaml` only works if the key set is fixed."""
        keys = set(load_config([f"site={site}"]).site.keys())
        assert keys == {
            "hcp_data_root",
            "behavioral_csv",
            "restricted_csv",
            "bids_root",
            "deriv_root",
            "slurm_account",
            "slurm_partition",
        }

    def test_individual_site_keys_can_be_overridden(self):
        cfg = load_config(["site.hcp_data_root=/elsewhere/HCP"])
        assert str(cfg.site.hcp_data_root) == "/elsewhere/HCP"

    @pytest.mark.parametrize(
        ("experiment", "stem"),
        [
            ("figure2", "figure2_insula_subdivisions"),
            ("figure3", "figure3_gradients"),
            ("figure4", "figure4_rtop_maps"),
            ("figure7", "figure7_acc_subdivisions"),
            ("figure8", "figure8_cognitive_control"),
        ],
    )
    def test_each_figure_experiment_selects_its_own_figure(self, experiment, stem):
        cfg = load_config([f"+experiment={experiment}"])
        assert list(cfg.steps) == ["figures"]
        assert list(cfg.figures.only) == [stem]
        assert cfg.figures.dpi == 300

    def test_every_figure_has_an_experiment(self):
        """A figure with no experiment cannot be redrawn on its own."""
        from insula_rtop.figures.run import FIGURE_BUILDERS

        experiments = Path(insula_rtop.pipeline.__file__).parent / "experiment"
        selected = set()
        for path in experiments.glob("figure*.yaml"):
            selected |= set(load_config([f"+experiment={path.stem}"]).figures.only)
        assert selected == set(FIGURE_BUILDERS)

    def test_all_figures_experiment_selects_everything(self):
        assert load_config(["+experiment=all_figures"]).figures.only is None

    def test_cohort_releases_match_the_documented_set(self):
        from insula_rtop.constants import Q1_Q6_RELEASES

        assert tuple(load_config().cohort.releases) == Q1_Q6_RELEASES


# ---- end to end -----------------------------------------------------------

N_SUBJECTS = 12
SUBJECTS = [f"{100000 + i}" for i in range(N_SUBJECTS)]


def plant_rtop_volumes(bids_root, deriv_root, subjects, seed=0):
    """Write an RTOP volume per subject with a known anterior-posterior ramp.

    RTOP increases towards -y, so the surface gradient must point posteriorly
    (angle ~ pi in the patch frame) in every subject and subdivision.
    """
    rng = np.random.default_rng(seed)
    for i, subject_id in enumerate(subjects):
        dwi = nib.load(str(raw_paths(bids_root, subject_id)["dwi"]))
        shape = dwi.shape[:3]
        ijk = np.indices(shape).astype(float)
        world_y = dwi.affine[1, 1] * ijk[1] + dwi.affine[1, 3]
        values = 2.0 - 0.05 * world_y + rng.normal(0, 0.01, size=shape)
        values += 0.1 * i / len(subjects)  # a per-subject offset
        out = rtop_paths(deriv_root, subject_id)
        out["normalized"].parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(values.astype(np.float32), dwi.affine),
            str(out["normalized"]),
        )


def plant_segmentations(deriv_root, patch_size=900):
    """Three contiguous fs_LR patches per hemisphere, standing in for the atlases."""
    from insula_rtop.atlases.fslr import group_midthickness

    for atlas, seg, names in (
        ("Deen2011", "insula", ("vAI", "dAI", "PI")),
        ("HCPMMP1", "insula", ("vAI", "dAI", "PI")),
        ("HCPMMP1", "acc", ("ACC-vAI", "ACC-dAI", "ACC-PI")),
    ):
        for hemi in ("L", "R"):
            coords, _ = group_midthickness(hemi)
            # Take a compact patch around a seed so the triangulation inside
            # each subdivision is connected and has faces to compute on.
            seed = coords[1000]
            nearest = np.argsort(np.linalg.norm(coords - seed, axis=1))[
                : 3 * patch_size
            ]
            labels = np.zeros(N_VERTICES_32K, dtype=np.int32)
            for index in range(3):
                chunk = nearest[index * patch_size : (index + 1) * patch_size]
                labels[chunk] = index + 1
            write_label_gii(label_path(deriv_root, atlas, seg, hemi), labels)
        path = table_path(deriv_root, atlas, seg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "index\tname\n"
            + "".join(f"{i + 1}\t{name}\n" for i, name in enumerate(names))
        )


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Everything from `hcp2bids` to `figures`, on synthetic data."""
    root = tmp_path_factory.mktemp("pipeline")
    # The `brain` preset writes real 32k fs_LR geometry: the segmentations the
    # analyses read live on that mesh, so nothing smaller would exercise them.
    hcp = make_synthetic_hcp_tree(root / "hcp", SUBJECTS, preset="brain")
    behavioral = make_behavioral_csv(root / "beh.csv", SUBJECTS)
    bids = root / "bids"
    deriv = root / "derivatives"

    run_cohort(hcp, behavioral, bids)
    run_bidsify(hcp, bids, subjects=SUBJECTS)
    plant_rtop_volumes(bids, deriv, SUBJECTS)
    run_surface(bids, deriv, subjects=SUBJECTS)
    plant_segmentations(deriv)
    results = run_analysis(bids, deriv, subjects=SUBJECTS, n_resamples=50)
    figures = run_figures(deriv, formats=("png",), dpi=60)
    return {
        "root": root,
        "bids": bids,
        "deriv": deriv,
        "results": results,
        "figures": figures,
    }


class TestEndToEnd:
    def test_every_subject_reaches_the_group_table(self, pipeline_run):
        import pandas as pd

        table = pd.read_csv(
            pipeline_run["deriv"] / ANALYSIS_DIR / "group_rtop.tsv",
            sep="\t",
            dtype={"subject": str},
        )
        assert table["subject"].nunique() == N_SUBJECTS
        assert set(table["atlas"]) == {"cortex", "Deen2011", "HCPMMP1"}
        assert table["rtop"].notna().all()

    def test_results_json_carries_the_headline_numbers(self, pipeline_run):
        text = (pipeline_run["deriv"] / ANALYSIS_DIR / "results.json").read_text()
        results = json.loads(text)
        assert results["n_subjects"] + results["n_outliers_excluded"] == N_SUBJECTS
        for key in ("anova_Deen2011_insula", "gradients_Deen2011", "cca_Deen2011"):
            assert key in results

    def test_results_json_is_strict_json(self, pipeline_run):
        """Bare NaN/Infinity literals are not JSON and break most parsers."""
        text = (pipeline_run["deriv"] / ANALYSIS_DIR / "results.json").read_text()
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON: {c}"))

    def test_planted_gradient_direction_is_recovered(self, pipeline_run):
        """The planted posterior-pointing ramp comes back at the right angle.

        Angles are reported in the anatomical plane by default, so a
        posterior-increasing ramp must come back at pi however the insula
        happens to be folded -- the expected value is known in advance rather
        than derived from the pipeline's own frame. The 3-D direction is
        checked too. Together they exercise the whole chain: FEM gradient,
        vertex averaging, tangential projection and circular statistics.
        """
        import pandas as pd

        summary = pd.read_csv(
            pipeline_run["deriv"] / ANALYSIS_DIR / "gradients_Deen2011.tsv", sep="\t"
        )
        pooled = summary[summary["subdivision"] == "pooled"].set_index("hemi")
        assert len(pooled) == 2

        for hemi in ("L", "R"):
            vector = pooled.loc[hemi, ["mean_vx", "mean_vy", "mean_vz"]]
            assert float(vector.iloc[1]) < -0.5

            observed = float(pooled.loc[hemi, "mean_direction_rad"])
            wrapped = np.abs(np.angle(np.exp(1j * (observed - np.pi))))
            assert wrapped < 0.35, f"{hemi}: {observed:.3f} vs pi"

        # A gradient planted identically in every subject must be
        # overwhelmingly non-uniform across the group.
        assert (pooled["rayleigh_p"] < 1e-6).all()
        assert (pooled["resultant_length"] > 0.9).all()

    def test_outlier_report_is_written(self, pipeline_run):
        report = json.loads(
            (pipeline_run["deriv"] / ANALYSIS_DIR / "outliers.json").read_text()
        )
        assert report["statistic"] == "cortex"
        assert report["multiplier"] == 1.5
        assert "excluded_subjects" in report

    def test_analysis_outputs_exist_for_both_parcellations(self, pipeline_run):
        analysis = pipeline_run["deriv"] / ANALYSIS_DIR
        for name in (
            "anova_Deen2011_insula.txt",
            "anova_HCPMMP1_insula.txt",
            "anova_HCPMMP1_acc.txt",
            "posthoc_Deen2011_insula.tsv",
            "stability_Deen2011_insula.tsv",
            "gradient_directions_Deen2011.tsv",
            "cca_Deen2011.txt",
            "cca_prediction_Deen2011.tsv",
        ):
            assert (analysis / name).exists(), name

    def test_figures_are_written(self, pipeline_run):
        figures = pipeline_run["deriv"] / FIGURES_DIR
        written = sorted(p.name for p in figures.glob("*.png"))
        from insula_rtop.figures.run import FIGURE_BUILDERS

        for stem in FIGURE_BUILDERS:
            assert f"{stem}.png" in written, f"{stem} missing from {written}"

    def test_figures_use_the_analysed_sample_not_the_raw_table(self, pipeline_run):
        """group_rtop.tsv predates the outlier exclusion; the figures must not."""
        import pandas as pd

        from insula_rtop.figures.run import analysed_sample

        analysis = pipeline_run["deriv"] / ANALYSIS_DIR
        raw = pd.read_csv(analysis / "group_rtop.tsv", sep="\t", dtype={"subject": str})
        excluded = set(
            json.loads((analysis / "outliers.json").read_text())["excluded_subjects"]
        )
        table, subjects = analysed_sample(analysis)
        assert set(subjects).isdisjoint(excluded)
        assert len(subjects) == raw["subject"].nunique() - len(excluded)

    def test_a_single_figure_can_be_redrawn_alone(self, pipeline_run):
        """What the per-figure Hydra experiments do."""
        written = run_figures(
            pipeline_run["deriv"], formats=("png",), dpi=60,
            only=["figure8_cognitive_control"],
        )
        assert [p.name for p in written] == ["figure8_cognitive_control"]

    def test_an_unknown_figure_name_is_rejected(self, pipeline_run):
        with pytest.raises(ValueError, match="Unknown figure"):
            run_figures(pipeline_run["deriv"], formats=("png",), only=["nope"])

    def test_rerunning_the_analysis_is_reproducible(self, pipeline_run):
        again = run_analysis(
            pipeline_run["bids"],
            pipeline_run["deriv"],
            subjects=SUBJECTS,
            n_resamples=50,
        )
        assert (
            again["cca_Deen2011"]["prediction_r"]
            == pipeline_run["results"]["cca_Deen2011"]["prediction_r"]
        )
