"""Which voxels the MAPL fit is actually run in.

The paper computed RTOP "at the voxel level on the preprocessed diffusion MRI
images", i.e. everywhere. Nothing downstream ever reads RTOP outside two places:
the midthickness vertices, and the ventricles. Fitting MAP-MRI with per-voxel
GCV over a whole 1.25 mm brain is the single dominant cost of the pipeline --
about 32 core-hours per subject on margaret -- so the default restricts it.

Three strategies:

``surface+ventricles`` (default)
    Exactly the voxels the pipeline reads: the eight voxels of each
    midthickness vertex's trilinear stencil, plus the eroded ventricle mask.
    Every RTOP value the analyses use is bit-identical to a whole-brain fit,
    because MAP-MRI is fitted independently per voxel and trilinear
    interpolation touches nothing else. On margaret this is ~236k voxels
    against 741k for the whole brain: a 3.1x saving for no approximation.

``ribbon+ventricles``
    The dilated FreeSurfer cortical ribbon. Fits a superset of the above
    (~490k voxels), which leaves a usable RTOP map across the grey matter
    rather than only under the mesh.

``brain``
    The whole brain mask. What the paper describes.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from insula_rtop.imaging import (
    binary_mask_from_labels,
    dilate,
    read_surface,
    resample_label_to_grid,
    world_to_voxel,
)
from insula_rtop.rtop.ventricle import DEFAULT_EROSION, ventricle_mask

#: The eight corners of the trilinear interpolation stencil.
_STENCIL = np.array(
    [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
)

#: FreeSurfer ``ribbon.nii.gz`` values for left and right cortical grey matter.
RIBBON_CORTEX_LABELS = (3, 42)

#: Voxels of dilation applied to the ribbon on the diffusion grid. Trilinear
#: interpolation at a vertex reads the 8 surrounding voxel centres, so one voxel
#: covers the stencil; two gives margin for vertices sitting just outside the
#: ribbon after the anatomical-to-diffusion grid change.
DEFAULT_RIBBON_DILATION = 2


def ribbon_mask(
    ribbon: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    *,
    dilation: int = DEFAULT_RIBBON_DILATION,
) -> np.ndarray:
    """Dilated cortical-ribbon mask on *reference*'s grid."""
    resampled = resample_label_to_grid(ribbon, reference)
    return dilate(binary_mask_from_labels(resampled, RIBBON_CORTEX_LABELS), dilation)


def surface_stencil_mask(
    reference: nib.Nifti1Image, surfaces
) -> np.ndarray:
    """Voxels read by trilinear interpolation at the vertices of *surfaces*.

    Trilinear interpolation at a point reads exactly the eight voxel centres
    bracketing it, so marking ``floor(ijk)`` plus the unit cube is not a
    heuristic dilation but the precise read set.
    """
    shape = np.asarray(reference.shape[:3])
    mask = np.zeros(tuple(shape), dtype=bool)
    for surface in surfaces:
        coords, _ = read_surface(surface)
        base = np.floor(world_to_voxel(coords, reference.affine)).astype(int)
        for offset in _STENCIL:
            ijk = base + offset
            inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
            ijk = ijk[inside]
            mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return mask


def build_fit_mask(
    reference: nib.Nifti1Image,
    *,
    ribbon: nib.Nifti1Image,
    aparc_aseg: nib.Nifti1Image,
    brain_mask: nib.Nifti1Image | None = None,
    surfaces=(),
    strategy: str = "surface+ventricles",
    ribbon_dilation: int = DEFAULT_RIBBON_DILATION,
    ventricle_erosion: int = DEFAULT_EROSION,
) -> np.ndarray:
    """Boolean mask of the voxels to fit, on the diffusion grid.

    Parameters
    ----------
    surfaces
        Midthickness surfaces, required by ``"surface+ventricles"``.
    strategy
        ``"surface+ventricles"`` (default), ``"ribbon+ventricles"`` or
        ``"brain"``. See the module docstring.
    """
    shape = reference.shape[:3]
    if strategy == "brain":
        if brain_mask is None:
            return np.ones(shape, dtype=bool)
        return np.asarray(brain_mask.dataobj) > 0

    ventricles = ventricle_mask(aparc_aseg, reference, erosion=ventricle_erosion)
    if strategy == "surface+ventricles":
        if not surfaces:
            raise ValueError(
                "The 'surface+ventricles' fit mask needs the subject's "
                "midthickness surfaces; pass surfaces=[...]."
            )
        mask = surface_stencil_mask(reference, surfaces) | ventricles
    elif strategy == "ribbon+ventricles":
        mask = ribbon_mask(ribbon, reference, dilation=ribbon_dilation) | ventricles
    else:
        raise ValueError(
            f"Unknown fit-mask strategy {strategy!r}; expected "
            "'surface+ventricles', 'ribbon+ventricles' or 'brain'."
        )

    if brain_mask is not None and strategy == "ribbon+ventricles":
        # Never fit outside the brain when the dilation pushed us there. The
        # surface stencil is deliberately left whole: a vertex sitting just
        # outside the brain mask still has to be sampled from somewhere, and
        # clipping it would turn that vertex into a silent NaN.
        mask &= np.asarray(brain_mask.dataobj) > 0
    return mask
