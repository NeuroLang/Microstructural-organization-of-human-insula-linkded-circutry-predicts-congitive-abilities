"""Build a tiny on-disk HCP tree so the ingestion code can be tested offline.

Two presets, selected with ``preset=``:

``tiny`` (default)
    A 16 mm cube with a 6-vertex-square surface patch. Fast enough to build in
    every test; enough to exercise path resolution, symlinking, the ventricle
    and ribbon masks and the volume-to-surface interpolation.

``brain``
    Real 32k fs_LR geometry -- the HCP S1200 group-average midthickness,
    jittered per subject -- inside a coarse whole-head volume. Slower, but the
    only way to exercise anything that touches a real segmentation, because
    those live on the 32492-vertex standard mesh.

Deliberate choices, mirroring ``spherical_integral_gnn``'s
``tests/_synthetic_wmparc.py``:

* the anatomical and diffusion grids have different resolutions, so any code
  that resamples between them is genuinely exercised rather than accidentally
  passing on identical affines;
* the ``missing_*_for`` arguments parameterise the negative cases, so discovery
  and completeness logic can be tested without hand-building broken trees;
* every mask is defined in *world* millimetres rather than voxel indices, so
  the two presets share one definition.

Underscore-prefixed so pytest does not collect it as a test module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SHELLS = (1000.0, 2000.0, 3000.0)
N_B0 = 3
N_PER_SHELL = 9

#: aparc+aseg value used to paint the ventricle compartment in the fixture.
VENTRICLE_LABEL = 4
#: ribbon.nii.gz values for left/right cortical grey matter (FreeSurfer's
#: convention: 3 = L cortex, 42 = R cortex).
RIBBON_CORTEX_LABELS = (3, 42)


@dataclass(frozen=True)
class Preset:
    """Volume geometry and the world-space extent of each compartment."""

    anat_zoom: float
    dwi_zoom: float
    #: ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` of the volumes, in mm.
    fov: tuple[tuple[float, float], ...]
    #: World-space box painted with VENTRICLE_LABEL, same format as *fov*.
    ventricle_box: tuple[tuple[float, float], ...]
    #: World-space slab painted as cortical ribbon; ignored in the brain preset,
    #: where the ribbon is derived from the surfaces themselves.
    ribbon_box: tuple[tuple[float, float], ...] | None
    fslr_surfaces: bool


PRESETS = {
    "tiny": Preset(
        anat_zoom=1.0,
        dwi_zoom=2.0,
        fov=((-8, 8), (-8, 8), (-8, 8)),
        ventricle_box=((-5, 5), (-5, 5), (-7, -3)),
        ribbon_box=((-6, 6), (-6, 6), (-1, 2)),
        fslr_surfaces=False,
    ),
    "brain": Preset(
        anat_zoom=4.0,
        dwi_zoom=8.0,
        fov=((-88, 88), (-128, 88), (-72, 88)),
        ventricle_box=((-16, 16), (-44, 12), (-4, 24)),
        ribbon_box=None,
        fslr_surfaces=True,
    ),
}


def make_gradient_table(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small multi-shell scheme: ``(bvals, bvecs)`` with bvecs shaped (3, N)."""
    rng = np.random.default_rng(seed)
    bvals = [0.0] * N_B0
    bvecs = [[0.0, 0.0, 0.0]] * N_B0
    for shell in SHELLS:
        directions = rng.normal(size=(N_PER_SHELL, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        bvals.extend([shell] * N_PER_SHELL)
        bvecs.extend(directions.tolist())
    return np.asarray(bvals), np.asarray(bvecs).T


def _grid(preset: Preset, zoom: float) -> tuple[tuple[int, int, int], np.ndarray]:
    """``(shape, affine)`` for a grid of the given voxel size covering the FOV."""
    lows = np.array([low for low, _ in preset.fov], dtype=float)
    highs = np.array([high for _, high in preset.fov], dtype=float)
    shape = np.ceil((highs - lows) / zoom).astype(int)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([zoom] * 3)
    affine[:3, 3] = lows
    return tuple(int(n) for n in shape), affine


def _world_coordinates(shape, affine) -> np.ndarray:
    """``(*shape, 3)`` array of the world coordinate of every voxel centre."""
    ijk = np.indices(shape).reshape(3, -1).T
    return nib.affines.apply_affine(affine, ijk).reshape(*shape, 3)


def _box_mask(world: np.ndarray, box) -> np.ndarray:
    mask = np.ones(world.shape[:3], dtype=bool)
    for axis, (low, high) in enumerate(box):
        mask &= (world[..., axis] >= low) & (world[..., axis] <= high)
    return mask


def _vertex_mask(world_shape, affine, coords: np.ndarray) -> np.ndarray:
    """Voxels containing a surface vertex, dilated by one -- a crude ribbon."""
    ijk = np.rint(
        nib.affines.apply_affine(np.linalg.inv(affine), coords)
    ).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(world_shape)), axis=1)
    mask = np.zeros(world_shape, dtype=bool)
    ijk = ijk[inside]
    mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return ndimage.binary_dilation(mask, iterations=1)


def _write_surface(path: Path, coords: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    nib.save(gii, str(path))


def make_grid_surface(
    n_side: int = 6, extent: float = 4.0, z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A flat triangulated square patch, entirely inside the ``tiny`` volumes."""
    lin = np.linspace(-extent, extent, n_side)
    xx, yy = np.meshgrid(lin, lin, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    faces = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            a = i * n_side + j
            b = a + 1
            c = a + n_side
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return coords, np.asarray(faces)


def _surfaces_for(preset: Preset, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    if not preset.fslr_surfaces:
        return make_grid_surface()
    from insula_rtop.atlases.fslr import group_midthickness

    return group_midthickness(hemi)


def make_synthetic_hcp_tree(
    root: Path,
    subject_ids: list[str],
    *,
    preset: str = "tiny",
    missing_dwi_for: tuple[str, ...] = (),
    missing_surface_for: tuple[str, ...] = (),
    missing_aparc_for: tuple[str, ...] = (),
    seed: int = 0,
) -> Path:
    """Write ``<root>/<subject_id>/{T1w,MNINonLinear}/...`` for each subject."""
    root = Path(root)
    spec = PRESETS[preset]
    bvals, bvecs = make_gradient_table(seed=seed)
    n_vol = bvals.size

    anat_shape, anat_affine = _grid(spec, spec.anat_zoom)
    dwi_shape, dwi_affine = _grid(spec, spec.dwi_zoom)
    anat_world = _world_coordinates(anat_shape, anat_affine)
    rng = np.random.default_rng(seed)

    surfaces = {hemi: _surfaces_for(spec, hemi) for hemi in ("L", "R")}
    ventricle = _box_mask(anat_world, spec.ventricle_box)
    if spec.ribbon_box is not None:
        ribbon = _box_mask(anat_world, spec.ribbon_box)
    else:
        ribbon = np.zeros(anat_shape, dtype=bool)
        for coords, _ in surfaces.values():
            ribbon |= _vertex_mask(anat_shape, anat_affine, coords)

    for sid in subject_ids:
        t1w_dir = root / sid / "T1w"
        diff_dir = t1w_dir / "Diffusion"
        diff_dir.mkdir(parents=True, exist_ok=True)

        if sid not in missing_dwi_for:
            data = rng.random((*dwi_shape, n_vol)).astype(np.float32) + 1.0
            nib.save(nib.Nifti1Image(data, dwi_affine), diff_dir / "data.nii.gz")
        np.savetxt(diff_dir / "bvals", bvals[None, :], fmt="%.1f")
        np.savetxt(diff_dir / "bvecs", bvecs, fmt="%.6f")
        nib.save(
            nib.Nifti1Image(np.ones(dwi_shape, np.uint8), dwi_affine),
            diff_dir / "nodif_brain_mask.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(rng.random(anat_shape).astype(np.float32), anat_affine),
            t1w_dir / "T1w_acpc_dc_restore.nii.gz",
        )

        if sid not in missing_aparc_for:
            aseg = np.where(ventricle, VENTRICLE_LABEL, 0).astype(np.int16)
            nib.save(nib.Nifti1Image(aseg, anat_affine), t1w_dir / "aparc+aseg.nii.gz")
        nib.save(
            nib.Nifti1Image(
                np.where(ribbon, RIBBON_CORTEX_LABELS[0], 0).astype(np.int16),
                anat_affine,
            ),
            t1w_dir / "ribbon.nii.gz",
        )

        if sid not in missing_surface_for:
            jitter = rng.normal(0.0, 0.2, size=3) if spec.fslr_surfaces else 0.0
            for base in (t1w_dir, root / sid / "MNINonLinear"):
                for hemi, (coords, faces) in surfaces.items():
                    _write_surface(
                        base
                        / "fsaverage_LR32k"
                        / f"{sid}.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii",
                        coords + jitter,
                        faces,
                    )
    return root


def make_behavioral_csv(
    path: Path, subject_ids: list[str], *, releases: dict[str, str] | None = None
) -> Path:
    """A minimal stand-in for HCP's ``hcp_behavioral.csv``."""
    import pandas as pd

    from insula_rtop.constants import CCA_BEHAVIORAL_COLUMNS

    rng = np.random.default_rng(1)
    releases = releases or {sid: "Q1" for sid in subject_ids}
    df = pd.DataFrame(
        {
            "Subject": subject_ids,
            "Release": [releases.get(sid, "Q1") for sid in subject_ids],
            "Gender": ["F" if i % 2 else "M" for i in range(len(subject_ids))],
            "Age": ["26-30"] * len(subject_ids),
        }
    )
    for col in CCA_BEHAVIORAL_COLUMNS:
        df[col] = rng.normal(size=len(subject_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
