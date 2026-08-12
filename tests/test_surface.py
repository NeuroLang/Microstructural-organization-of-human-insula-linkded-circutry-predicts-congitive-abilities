"""Tests for the volume-to-surface projection.

Trilinear interpolation is exact for affine intensity fields, so a constant
volume and a linear ramp both have closed-form answers at every vertex. Those
two cases pin the world-to-voxel transform, the interpolation order, and the
index ordering together -- an axis swap or an off-by-one in the affine shows up
immediately in the ramp test.

Run: ``uv run pytest tests/test_surface.py -v``
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
from _synthetic_hcp import make_grid_surface, make_synthetic_hcp_tree

from insula_rtop.hcp import to_bids
from insula_rtop.imaging import (
    read_surface,
    read_surface_scalars,
    write_surface_scalars,
)
from insula_rtop.rtop.run import rtop_paths
from insula_rtop.surface import run as surface_run
from insula_rtop.surface.sample import sample_volume_on_surface

SHAPE = (16, 16, 16)
ZOOM = 1.0


def make_volume(values: np.ndarray) -> nib.Nifti1Image:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([ZOOM] * 3)
    affine[:3, 3] = -0.5 * ZOOM * np.asarray(SHAPE)
    return nib.Nifti1Image(np.asarray(values, np.float32), affine)


def write_patch(path, z: float = 0.0, extent: float = 3.0):
    coords, faces = make_grid_surface(n_side=5, extent=extent, z=z)
    gii = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
            ),
            nib.gifti.GiftiDataArray(
                faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(gii, str(path))
    return coords


class TestSampling:
    def test_constant_volume_gives_a_constant_surface(self, tmp_path):
        vol = make_volume(np.full(SHAPE, 7.5))
        surf = tmp_path / "patch.surf.gii"
        write_patch(surf)
        values = sample_volume_on_surface(vol, surf)
        np.testing.assert_allclose(values, 7.5, atol=1e-5)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_linear_ramp_is_reproduced_exactly(self, tmp_path, axis):
        """Trilinear interpolation is exact on affine fields, per axis."""
        idx = np.indices(SHAPE)[axis].astype(float)
        vol = make_volume(idx)
        surf = tmp_path / "patch.surf.gii"
        coords = write_patch(surf, z=0.5)

        values = sample_volume_on_surface(vol, surf)
        # Voxel index along `axis` for each vertex, from the same affine.
        expected = (coords[:, axis] - vol.affine[axis, 3]) / ZOOM
        np.testing.assert_allclose(values, expected, atol=1e-4)

    def test_vertices_outside_the_field_of_view_are_nan(self, tmp_path):
        vol = make_volume(np.ones(SHAPE))
        surf = tmp_path / "patch.surf.gii"
        # extent 40 mm on a 16 mm volume: the corners fall well outside.
        write_patch(surf, extent=40.0)
        values = sample_volume_on_surface(vol, surf)
        assert np.isnan(values).any()
        assert np.isfinite(values).any()

    def test_unfitted_voxels_become_nan_not_zero(self, tmp_path):
        """A vertex over a skipped voxel must not read as 'RTOP == 0'."""
        vol = make_volume(np.ones(SHAPE))
        valid = np.ones(SHAPE, bool)
        valid[:, :, :8] = False  # everything below world z = 0 is unfitted
        surf = tmp_path / "patch.surf.gii"
        write_patch(surf, z=-2.0)
        values = sample_volume_on_surface(vol, surf, valid_mask=valid)
        assert np.isnan(values).all()

    def test_partial_stencil_is_rejected_not_averaged(self, tmp_path):
        """One invalid stencil corner must NaN the vertex, not bias it low.

        The vertex sits at z = 7.5 in voxel indices, so its stencil spans the
        z = 7 and z = 8 planes. With z = 7 unfitted (and therefore zero in the
        volume) a nearest-voxel check would accept the vertex and return 0.5
        instead of 1.0 -- a quietly halved RTOP that nothing downstream could
        detect.
        """
        data = np.ones(SHAPE)
        data[:, :, :8] = 0.0
        vol = make_volume(data)
        valid = np.ones(SHAPE, bool)
        valid[:, :, :8] = False

        surf = tmp_path / "patch.surf.gii"
        write_patch(surf, z=-0.5)  # world -0.5 -> voxel index 7.5
        assert np.isnan(sample_volume_on_surface(vol, surf, valid_mask=valid)).all()
        # Without the mask the same vertex really does read the averaged 0.5.
        np.testing.assert_allclose(
            sample_volume_on_surface(vol, surf), 0.5, atol=1e-6
        )

    def test_gifti_scalar_roundtrip(self, tmp_path):
        data = np.arange(25, dtype=float)
        path = tmp_path / "x.shape.gii"
        write_surface_scalars(path, data, intent="NIFTI_INTENT_SHAPE")
        np.testing.assert_allclose(read_surface_scalars(path), data)

    def test_non_surface_gifti_is_rejected(self, tmp_path):
        path = tmp_path / "scalars.shape.gii"
        write_surface_scalars(path, np.zeros(5), intent="NIFTI_INTENT_SHAPE")
        with pytest.raises(ValueError, match="not a GIFTI surface"):
            read_surface(path)


class TestSurfaceStep:
    def _prepare(self, tmp_path):
        """A BIDS tree with a hand-made RTOP volume, ready for the surface step."""
        hcp = make_synthetic_hcp_tree(tmp_path / "hcp", ["100001"])
        bids = tmp_path / "bids"
        to_bids.process_subject("100001", hcp, bids)

        deriv = tmp_path / "derivatives"
        out = rtop_paths(deriv, "100001")
        out["normalized"].parent.mkdir(parents=True, exist_ok=True)
        dwi = nib.load(str(to_bids.raw_paths(bids, "100001")["dwi"]))
        values = np.full(dwi.shape[:3], 2.5, np.float32)
        nib.save(nib.Nifti1Image(values, dwi.affine), str(out["normalized"]))
        return bids, deriv

    def test_writes_both_hemispheres_and_a_sidecar(self, tmp_path):
        bids, deriv = self._prepare(tmp_path)
        surface_run.process_subject("100001", bids, deriv)
        paths = surface_run.surface_paths(deriv, "100001")
        for hemi in ("L", "R"):
            values = read_surface_scalars(paths[hemi])
            np.testing.assert_allclose(values, 2.5, atol=1e-5)
        meta = json.loads(paths["json"].read_text())
        assert meta["Coverage"]["L"]["ValidVertices"] == 36
        assert meta["Interpolation"] == "trilinear"

    def test_requires_the_rtop_volume(self, tmp_path):
        hcp = make_synthetic_hcp_tree(tmp_path / "hcp", ["100001"])
        bids = tmp_path / "bids"
        to_bids.process_subject("100001", hcp, bids)
        with pytest.raises(FileNotFoundError, match="rtop_volume"):
            surface_run.process_subject("100001", bids, tmp_path / "derivatives")

    def test_zero_voxels_are_treated_as_unfitted(self, tmp_path):
        bids, deriv = self._prepare(tmp_path)
        path = rtop_paths(deriv, "100001")["normalized"]
        img = nib.load(str(path))
        nib.save(
            nib.Nifti1Image(np.zeros(img.shape, np.float32), img.affine), str(path)
        )
        surface_run.process_subject("100001", bids, deriv)
        sidecar = surface_run.surface_paths(deriv, "100001")["json"]
        assert json.loads(sidecar.read_text())["Coverage"]["L"]["ValidVertices"] == 0

    def test_idempotency_states(self, tmp_path):
        bids, deriv = self._prepare(tmp_path)
        surface_run.process_subject("100001", bids, deriv)
        with pytest.raises(FileExistsError, match="already exists"):
            surface_run.process_subject("100001", bids, deriv)
        surface_run.process_subject("100001", bids, deriv, skip_existing=True)
        surface_run.process_subject("100001", bids, deriv, force=True)
