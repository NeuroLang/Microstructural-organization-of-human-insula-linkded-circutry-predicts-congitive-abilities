"""Tests for HCP discovery, cohort selection and the BIDS migration.

No network, no cluster, no real HCP data: everything runs against the synthetic
tree in ``tests/_synthetic_hcp.py``.

Run: ``uv run pytest tests/test_hcp_ingest.py -v``
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from _synthetic_hcp import make_behavioral_csv, make_synthetic_hcp_tree

from insula_rtop.constants import CCA_BEHAVIORAL_COLUMNS, DIFFUSION_TIME_S
from insula_rtop.hcp import to_bids
from insula_rtop.hcp.cohort import (
    load_behavioral,
    read_participants_tsv,
    run_cohort,
    select_cohort,
)
from insula_rtop.hcp.layout import discover_subjects, resolve_subject_paths

SUBJECTS = ["100001", "100002", "100003"]


# ---- discovery ------------------------------------------------------------


class TestDiscovery:
    def test_finds_complete_subjects(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        assert discover_subjects(root) == SUBJECTS

    def test_excludes_subject_without_dwi(self, tmp_path):
        root = make_synthetic_hcp_tree(
            tmp_path / "hcp", SUBJECTS, missing_dwi_for=("100002",)
        )
        assert discover_subjects(root) == ["100001", "100003"]

    def test_excludes_subject_without_surfaces(self, tmp_path):
        root = make_synthetic_hcp_tree(
            tmp_path / "hcp", SUBJECTS, missing_surface_for=("100003",)
        )
        assert discover_subjects(root) == ["100001", "100002"]

    def test_excludes_subject_without_aparc_aseg(self, tmp_path):
        root = make_synthetic_hcp_tree(
            tmp_path / "hcp", SUBJECTS, missing_aparc_for=("100001",)
        )
        assert discover_subjects(root) == ["100002", "100003"]

    def test_excludes_subject_with_zero_byte_inputs(self, tmp_path):
        """An interrupted download leaves 0-byte files that still `exists()`."""
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        paths = resolve_subject_paths(root, "100002")
        for path in (paths.dwi, paths.bval, paths.bvec):
            path.write_bytes(b"")
        assert discover_subjects(root) == ["100001", "100003"]
        assert set(paths.missing()) == {paths.dwi, paths.bval, paths.bvec}

    def test_numeric_subject_id_is_coerced(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        # Hydra parses unquoted numeric CLI list entries as int.
        paths = resolve_subject_paths(root, 100001)
        assert paths.missing() == []


# ---- cohort ---------------------------------------------------------------


class TestCohortSelection:
    def test_release_filter(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(
            tmp_path / "beh.csv",
            SUBJECTS,
            releases={"100001": "Q1", "100002": "S500", "100003": "S1200"},
        )
        cohort, counts = select_cohort(load_behavioral(csv), root)
        assert cohort["Subject"].tolist() == ["100001", "100002"]
        assert counts["release_in_Q1+Q2+Q3+S500"] == 2
        assert counts["complete_hcp_inputs"] == 2

    def test_releases_none_keeps_everything(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(
            tmp_path / "beh.csv", SUBJECTS, releases={"100003": "S1200"}
        )
        cohort, _ = select_cohort(load_behavioral(csv), root, releases=None)
        assert cohort["Subject"].tolist() == SUBJECTS

    def test_incomplete_behavior_dropped_only_when_required(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(tmp_path / "beh.csv", SUBJECTS)
        beh = load_behavioral(csv)
        beh.loc[beh["Subject"] == "100002", CCA_BEHAVIORAL_COLUMNS[0]] = np.nan

        strict, _ = select_cohort(beh, root)
        assert strict["Subject"].tolist() == ["100001", "100003"]

        lenient, _ = select_cohort(beh, root, require_behavior=False)
        assert lenient["Subject"].tolist() == SUBJECTS

    def test_missing_hcp_inputs_drop_subject(self, tmp_path):
        root = make_synthetic_hcp_tree(
            tmp_path / "hcp", SUBJECTS, missing_dwi_for=("100001",)
        )
        csv = make_behavioral_csv(tmp_path / "beh.csv", SUBJECTS)
        cohort, counts = select_cohort(load_behavioral(csv), root)
        assert "100001" not in cohort["Subject"].tolist()
        assert counts["complete_hcp_inputs"] == 2

    def test_explicit_subject_list_wins(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(tmp_path / "beh.csv", SUBJECTS)
        cohort, _ = select_cohort(
            load_behavioral(csv), root, subjects=["100003", "100001"]
        )
        assert cohort["Subject"].tolist() == ["100001", "100003"]

    def test_missing_required_column_is_an_error(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "bad.csv"
        pd.DataFrame({"Subject": SUBJECTS}).to_csv(csv, index=False)
        with pytest.raises(KeyError, match="missing required column"):
            load_behavioral(csv)

    def test_participants_tsv_roundtrip(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(tmp_path / "beh.csv", SUBJECTS)
        bids = tmp_path / "bids"
        run_cohort(root, csv, bids)
        df = read_participants_tsv(bids)
        assert df["participant_id"].tolist() == [f"sub-{s}" for s in SUBJECTS]
        # Subject IDs must survive as strings, never be coerced to int: HCP has
        # leading-zero IDs and every path in the pipeline is built from them.
        assert df["Subject"].tolist() == SUBJECTS
        assert all(isinstance(s, str) for s in df["Subject"])

    def test_empty_cohort_raises(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        csv = make_behavioral_csv(
            tmp_path / "beh.csv", SUBJECTS, releases={s: "S1200" for s in SUBJECTS}
        )
        with pytest.raises(RuntimeError, match="no subjects"):
            run_cohort(root, csv, tmp_path / "bids")


# ---- BIDS migration -------------------------------------------------------


def _bidsify_one(tmp_path, **kwargs):
    root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
    bids = tmp_path / "bids"
    to_bids.process_subject("100001", root, bids, **kwargs)
    return root, bids


class TestBidsify:
    def test_images_are_symlinks_to_the_hcp_tree(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        raw = to_bids.raw_paths(bids, "100001")
        src = resolve_subject_paths(root, "100001")
        assert raw["dwi"].is_symlink()
        assert raw["dwi"].resolve() == src.dwi.resolve()
        assert raw["t1w"].resolve() == src.t1w.resolve()

    def test_generated_files_are_real(self, tmp_path):
        _, bids = _bidsify_one(tmp_path)
        raw = to_bids.raw_paths(bids, "100001")
        assert not raw["json"].is_symlink()
        assert not raw["bvec"].is_symlink()

    def test_bvec_keeps_hcp_orientation_and_values(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        src = np.loadtxt(resolve_subject_paths(root, "100001").bvec)
        out = np.loadtxt(to_bids.raw_paths(bids, "100001")["bvec"])
        assert out.shape == src.shape == (3, 30)
        np.testing.assert_allclose(out, src, atol=1e-6)

    def test_bvec_wrong_shape_is_an_error(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        bvec = resolve_subject_paths(root, "100001").bvec
        np.savetxt(bvec, np.loadtxt(bvec).T)  # write the transposed (N, 3) form
        with pytest.raises(ValueError, match=r"expected a \(3, N\)"):
            to_bids.process_subject("100001", root, tmp_path / "bids")

    def test_sidecar_carries_diffusion_times(self, tmp_path):
        _, bids = _bidsify_one(tmp_path)
        meta = json.loads(to_bids.raw_paths(bids, "100001")["json"].read_text())
        assert meta["LargeDeltaTime"] == pytest.approx(0.0431)
        assert meta["SmallDeltaTime"] == pytest.approx(0.0106)
        assert meta["EffectiveDiffusionTime"] == pytest.approx(DIFFUSION_TIME_S)
        assert meta["NumberOfVolumes"] == 30

    def test_native_space_surfaces_are_linked(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        deriv = to_bids.derivative_paths(bids, "100001")
        src = resolve_subject_paths(root, "100001")
        for hemi in ("L", "R"):
            assert (
                deriv[f"midthickness_native_{hemi}"].resolve()
                == src.midthickness_native[hemi].resolve()
            )

    def test_missing_input_is_reported_with_the_path(self, tmp_path):
        root = make_synthetic_hcp_tree(
            tmp_path / "hcp", SUBJECTS, missing_aparc_for=("100001",)
        )
        with pytest.raises(FileNotFoundError, match="aparc"):
            to_bids.process_subject("100001", root, tmp_path / "bids")


class TestIdempotency:
    def test_rerun_without_flags_raises(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            to_bids.process_subject("100001", root, bids)

    def test_skip_existing_is_a_quiet_noop(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        json_path = to_bids.raw_paths(bids, "100001")["json"]
        json_path.write_text("{}")
        to_bids.process_subject("100001", root, bids, skip_existing=True)
        assert json_path.read_text() == "{}"

    def test_force_rewrites(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        json_path = to_bids.raw_paths(bids, "100001")["json"]
        json_path.write_text("{}")
        to_bids.process_subject("100001", root, bids, force=True)
        assert json.loads(json_path.read_text())["NumberOfVolumes"] == 30

    def test_completeness_keys_on_the_last_written_file(self, tmp_path):
        root, bids = _bidsify_one(tmp_path)
        assert to_bids.is_bidsify_complete(bids, "100001")
        to_bids.derivative_paths(bids, "100001")["dwi_mask"].unlink()
        assert not to_bids.is_bidsify_complete(bids, "100001")

    def test_resume_skips_completed_subjects(self, tmp_path, capsys):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        bids = tmp_path / "bids"
        to_bids.process_subject("100001", root, bids)
        n_ok, n_fail = to_bids.run_bidsify(
            root, bids, subjects=SUBJECTS, resume=True, skip_existing=True
        )
        assert (n_ok, n_fail) == (2, 0)
        assert "1 subject(s) already complete" in capsys.readouterr().out


class TestDatasetDescriptions:
    def test_both_descriptions_written(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        bids = tmp_path / "bids"
        to_bids.write_dataset_descriptions(bids, root)
        raw = json.loads((bids / "dataset_description.json").read_text())
        deriv = json.loads(
            (
                bids / "derivatives" / to_bids.DERIVATIVE_NAME
                / "dataset_description.json"
            ).read_text()
        )
        assert raw["DatasetType"] == "raw"
        assert deriv["DatasetType"] == "derivative"
        assert str(root.resolve()) in raw["SourceDatasets"][0]["URL"]

    def test_never_clobbers_hand_edits(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", SUBJECTS)
        bids = tmp_path / "bids"
        to_bids.write_dataset_descriptions(bids, root)
        path = bids / "dataset_description.json"
        path.write_text('{"Name": "edited by hand"}')
        to_bids.write_dataset_descriptions(bids, root)
        assert json.loads(path.read_text())["Name"] == "edited by hand"
