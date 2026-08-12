"""Tests for the parcellations.

The HCP-MMP half runs entirely offline: the atlas is a data file inside the
``hcp_utils`` dependency, so it is always available. The Deen half is split --
the projection maths is tested against synthetic volumes with known geometry,
while the two tests that actually reach https://bendeen.com are marked ``slow``.

Run: ``uv run pytest tests/test_atlases.py -v`` (add ``-m "not slow"`` offline)
"""

from __future__ import annotations

import numpy as np
import pytest

from insula_rtop.atlases import deen, glasser
from insula_rtop.atlases import run as atlas_run
from insula_rtop.atlases.fslr import N_VERTICES_32K, group_midthickness
from insula_rtop.atlases.run import (
    build_segmentation,
    label_path,
    load_segmentation,
    read_table,
    run_atlases,
    table_path,
    write_label_gii,
)
from insula_rtop.constants import INSULA_SUBDIVISIONS

HEMIS = ("L", "R")


# ---- fs_LR reference geometry ---------------------------------------------


class TestFsLR:
    @pytest.mark.parametrize("hemi", HEMIS)
    def test_group_surface_is_on_the_standard_mesh(self, hemi):
        coords, faces = group_midthickness(hemi)
        assert coords.shape == (N_VERTICES_32K, 3)
        assert faces.shape[1] == 3
        assert faces.max() < N_VERTICES_32K

    def test_group_surface_is_in_mni_space(self):
        """Left hemisphere sits at negative x; a flipped surface would not."""
        left, _ = group_midthickness("L")
        right, _ = group_midthickness("R")
        assert left[:, 0].mean() < 0 < right[:, 0].mean()


# ---- HCP-MMP --------------------------------------------------------------


class TestGlasser:
    @pytest.mark.parametrize("hemi", HEMIS)
    def test_labels_cover_the_standard_mesh(self, hemi):
        labels = glasser.mmp_labels(hemi)
        assert labels.shape == (N_VERTICES_32K,)
        # 180 areas per hemisphere, plus 0 on the medial wall.
        assert len(np.unique(labels)) == 181

    def test_medial_wall_is_unlabelled(self):
        """The grayordinate expansion must leave the medial wall at 0."""
        labels = glasser.mmp_labels("L")
        assert (labels == 0).sum() == N_VERTICES_32K - 29696

    def test_area_lookup_is_hemisphere_specific(self):
        assert glasser.area_index("L", "AVI") != glasser.area_index("R", "AVI")

    def test_unknown_area_is_an_error(self):
        # 'Pol1' is the paper's spelling; the atlas calls it 'PoI1'.
        with pytest.raises(KeyError, match="Pol1"):
            glasser.area_index("L", "Pol1")

    def test_every_area_the_paper_names_exists(self):
        named = {
            *sum(glasser.INSULA_AREA_GROUPS.values(), ()),
            *sum(glasser.ACC_AREA_GROUPS.values(), ()),
        }
        for hemi in HEMIS:
            for area in named:
                assert glasser.area_mask(hemi, [area]).any()

    @pytest.mark.parametrize("hemi", HEMIS)
    def test_insula_groups_are_disjoint_and_nonempty(self, hemi):
        labels, names = glasser.build_segmentation(hemi, glasser.INSULA_AREA_GROUPS)
        assert list(names.values()) == ["vAI", "dAI", "PI"]
        for i in names:
            assert (labels == i).sum() > 0

    def test_overlapping_groups_are_rejected(self):
        with pytest.raises(ValueError, match="overlaps an earlier group"):
            glasser.build_segmentation("L", {"a": ("AVI",), "b": ("AVI", "MI")})

    def test_empty_group_is_rejected(self):
        with pytest.raises(KeyError):
            glasser.build_segmentation("L", {"a": ("NotAnArea",)})

    @pytest.mark.parametrize("hemi", HEMIS)
    def test_insula_subdivisions_are_ordered_ventral_to_posterior(self, hemi):
        """vAI is the most inferior, PI the most posterior -- the paper's axes."""
        coords, _ = group_midthickness(hemi)
        labels, names = glasser.build_segmentation(hemi, glasser.INSULA_AREA_GROUPS)
        centroid = {names[i]: coords[labels == i].mean(axis=0) for i in names}
        assert centroid["vAI"][2] < centroid["dAI"][2]  # vAI below dAI
        assert centroid["PI"][1] < centroid["dAI"][1]  # PI behind dAI

    @pytest.mark.parametrize("hemi", HEMIS)
    def test_acc_subdivisions_run_rostroventral_to_caudodorsal(self, hemi):
        """p24 -> a24pr -> p24pr, the ordering the ACC assignment assumes."""
        coords, _ = group_midthickness(hemi)
        labels, names = glasser.build_segmentation(hemi, glasser.ACC_AREA_GROUPS)
        centroid = {names[i]: coords[labels == i].mean(axis=0) for i in names}
        assert (
            centroid["ACC-vAI"][1] > centroid["ACC-dAI"][1] > centroid["ACC-PI"][1]
        )
        assert (
            centroid["ACC-vAI"][2] < centroid["ACC-dAI"][2] < centroid["ACC-PI"][2]
        )


# ---- Deen 2011 ------------------------------------------------------------


def synthetic_deen_masks(hemi: str, radius: float = 6.0):
    """Three disjoint spheres centred on real vertices of the group surface."""
    import nibabel as nib

    coords, _ = group_midthickness(hemi)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-90.0, -126.0, -72.0]
    shape = (91, 109, 91)
    ijk = np.indices(shape).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, ijk)

    rng = np.random.default_rng(0)
    seeds = coords[rng.choice(len(coords), size=3, replace=False)]
    masks = {}
    for sub, seed in zip(INSULA_SUBDIVISIONS, seeds, strict=True):
        inside = np.linalg.norm(world - seed, axis=1) < radius
        masks[(hemi, sub)] = nib.Nifti1Image(
            inside.reshape(shape).astype(np.uint8), affine
        )
    return masks, seeds


class TestDeenProjection:
    def test_masks_merge_into_one_label_volume(self):
        masks, _ = synthetic_deen_masks("L")
        volume, affine, names = deen.build_label_volume(masks, "L")
        assert names == dict(enumerate(INSULA_SUBDIVISIONS, start=1))
        assert set(np.unique(volume)) == {0, 1, 2, 3}
        assert affine.shape == (4, 4)

    def test_overlapping_masks_are_rejected(self):
        import nibabel as nib

        masks, _ = synthetic_deen_masks("L")
        first = masks[("L", INSULA_SUBDIVISIONS[0])]
        masks[("L", INSULA_SUBDIVISIONS[1])] = nib.Nifti1Image(
            np.asarray(first.dataobj), first.affine
        )
        with pytest.raises(ValueError, match="overlaps an earlier one"):
            deen.build_label_volume(masks, "L")

    def test_projection_labels_vertices_near_the_seed(self):
        masks, seeds = synthetic_deen_masks("L")
        volume, affine, names = deen.build_label_volume(masks, "L")
        labels = deen.project_to_surface(volume, affine, "L")

        coords, _ = group_midthickness("L")
        for i, seed in enumerate(seeds, start=1):
            hit = coords[labels == i]
            assert len(hit) > 0
            # Every labelled vertex must lie within the sphere plus the search
            # radius; nothing may be labelled from across the brain.
            assert np.linalg.norm(hit - seed, axis=1).max() < 6.0 + (
                deen.SEARCH_RADIUS_MM + 2.0
            )

    def test_distant_vertices_stay_unlabelled(self):
        masks, _ = synthetic_deen_masks("L", radius=4.0)
        volume, affine, _ = deen.build_label_volume(masks, "L")
        labels = deen.project_to_surface(volume, affine, "L", search_radius_mm=0.5)
        assert (labels == 0).sum() > N_VERTICES_32K * 0.99

    def test_empty_volume_is_rejected(self):
        with pytest.raises(ValueError, match="is empty"):
            deen.project_to_surface(np.zeros((4, 4, 4), int), np.eye(4), "L")


@pytest.mark.slow
class TestDeenDownload:
    def test_archive_contains_all_six_masks(self, tmp_path):
        masks = deen.extract_masks(deen.download(tmp_path))
        assert set(masks) == set(deen.MASK_FILENAMES)
        for img in masks.values():
            assert img.shape[:3] == (91, 109, 91)  # MNI152 2 mm

    @pytest.mark.parametrize("hemi", HEMIS)
    def test_subdivision_centroids_land_in_the_insula(self, tmp_path, hemi):
        labels, names = deen.build_segmentation(tmp_path, hemi)
        coords, _ = group_midthickness(hemi)
        sign = -1 if hemi == "L" else 1
        for i, name in names.items():
            centroid = coords[labels == i].mean(axis=0)
            assert 28 < sign * centroid[0] < 45, f"{name} is not lateral enough"
            assert -20 < centroid[1] < 25, f"{name} is outside the insular range"


@pytest.mark.slow
class TestAtlasStep:
    def test_builds_every_segmentation_and_reloads_it(self, tmp_path):
        run_atlases(tmp_path)
        for atlas, seg in (("Deen2011", "insula"), ("HCPMMP1", "acc")):
            labels, names = load_segmentation(tmp_path, atlas, seg)
            assert set(labels) == set(HEMIS)
            assert len(names) == 3
            for hemi in HEMIS:
                assert labels[hemi].shape == (N_VERTICES_32K,)
                assert set(np.unique(labels[hemi])) == {0, 1, 2, 3}

    def test_rerun_skips_existing(self, tmp_path, capsys):
        run_atlases(tmp_path)
        capsys.readouterr()
        run_atlases(tmp_path)
        assert capsys.readouterr().out.count("exists, skipping") == 3

    def test_custom_acc_labels_override_the_atlas(self, tmp_path):
        custom = {}
        for hemi in HEMIS:
            path = tmp_path / f"custom_{hemi}.label.gii"
            labels = np.zeros(N_VERTICES_32K, np.int32)
            labels[:30] = 1
            labels[30:60] = 2
            labels[60:90] = 3
            write_label_gii(path, labels)
            custom[hemi] = path

        labels, names = build_segmentation(
            "HCPMMP1", "acc", "L", cache_dir=tmp_path, acc_labels=custom
        )
        assert (labels[:30] == 1).all()
        assert list(names.values()) == ["ACC-vAI", "ACC-dAI", "ACC-PI"]


class TestSegmentationIO:
    def test_unknown_segmentation_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown segmentation"):
            build_segmentation("Nope", "insula", "L", cache_dir=tmp_path)

    def test_missing_segmentation_points_at_the_step(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="atlas_labels"):
            load_segmentation(tmp_path, "Deen2011", "insula")

    def test_label_and_table_roundtrip(self, tmp_path):
        labels = np.zeros(N_VERTICES_32K, np.int32)
        labels[:10] = 2
        write_label_gii(label_path(tmp_path, "X", "y", "L"), labels)
        path = table_path(tmp_path, "X", "y")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("index\tname\n1\ta\n2\tb\n")
        assert read_table(path) == {1: "a", 2: "b"}

    def test_wrong_vertex_count_is_rejected(self, tmp_path):
        table = table_path(tmp_path, "X", "y")
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("index\tname\n1\ta\n")
        for hemi in HEMIS:
            path = label_path(tmp_path, "X", "y", hemi)
            write_label_gii(path, np.zeros(100, np.int32))
        with pytest.raises(ValueError, match="expected 32492"):
            load_segmentation(tmp_path, "X", "y")


class TestSuppliedInsulaSegmentation:
    """A parcellation made elsewhere need not use this pipeline's label values.

    Renumbering is the kind of step that fails silently -- swap two labels and
    every downstream statistic is still computed, just about the wrong regions.
    """

    @staticmethod
    def _write(tmp_path, name, values):
        import nibabel as nib

        data = np.zeros(N_VERTICES_32K, dtype=np.int32)
        for value, start in values.items():
            data[start : start + 10] = value
        path = tmp_path / name
        gii = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(data, intent="NIFTI_INTENT_LABEL")]
        )
        nib.save(gii, str(path))
        return path

    def _spec(self, tmp_path):
        # Their file numbers posterior-to-anterior; ours is the reverse.
        left = self._write(tmp_path, "L.func.gii", {1: 0, 2: 100, 3: 200})
        right = self._write(tmp_path, "R.func.gii", {4: 0, 5: 100, 6: 200})
        return {
            "L": {"file": str(left), "PI": 1, "dAI": 2, "vAI": 3},
            "R": {"file": str(right), "PI": 4, "dAI": 5, "vAI": 6},
        }

    def test_labels_are_renumbered_to_the_pipeline_order(self, tmp_path):
        spec = self._spec(tmp_path)
        labels, names = atlas_run.supplied_segmentation(spec, "L")
        assert names == {1: "vAI", 2: "dAI", 3: "PI"}
        # Their 1 (PI) must land on our 3, their 3 (vAI) on our 1.
        assert set(np.flatnonzero(labels == 3)) == set(range(0, 10))
        assert set(np.flatnonzero(labels == 1)) == set(range(200, 210))

    def test_a_subdivision_with_no_vertices_is_an_error(self, tmp_path):
        spec = self._spec(tmp_path)
        spec["L"]["vAI"] = 99
        with pytest.raises(ValueError, match="matches no vertex"):
            atlas_run.supplied_segmentation(spec, "L")

    def test_a_missing_subdivision_names_itself(self, tmp_path):
        spec = self._spec(tmp_path)
        del spec["L"]["dAI"]
        with pytest.raises(KeyError, match="dAI"):
            atlas_run.supplied_segmentation(spec, "L")
