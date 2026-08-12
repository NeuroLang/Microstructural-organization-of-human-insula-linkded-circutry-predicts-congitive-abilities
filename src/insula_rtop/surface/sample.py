"""Project a volume onto the 32k fs_LR cortical surface.

Appendix 1, "Projecting RTOP values to the template cortical surface":

    To obtain the values of RTOP on a template cortical surface we started by
    sampling the voxel-level RTOP on to the participant-specific 32k cortical
    surfaces. We used the provided mid-thickness surfaces. This was done with the
    objective of projecting onto the surface the interpolated voxel RTOP value at
    the mid-point between the pial surface and the grey-white matter interface.
    Finally, to bring the surface-projected RTOP values to the common HCP
    template space we used the correspondence between the subject-specific
    surfaces and the template-registered MSMAll surfaces.

Both halves of that reduce to one interpolation. The HCP structural package
ships ``T1w/fsaverage_LR32k/<id>.<L|R>.midthickness_MSMAll.32k_fs_LR.surf.gii``:
midthickness geometry in the subject's ACPC space -- the space the diffusion
volume is already in -- carrying 32k fs_LR standard-mesh indices under the
MSMAll registration. Sampling the volume at those vertex coordinates therefore
lands directly in template vertex correspondence, with no resampling step and no
``wb_command``.

A vertex may fall outside the volume's field of view or in a voxel the MAPL fit
skipped. Those vertices are returned as NaN rather than 0, so a missing value
can never be mistaken for "RTOP equals the free-water value".
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from insula_rtop.imaging import read_surface, sample_volume_at_points, world_to_voxel

#: The eight corners of the trilinear interpolation stencil.
_STENCIL = np.array(
    [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
)


def sample_volume_on_surface(
    volume_img: nib.Nifti1Image,
    surface_path,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Trilinearly interpolate *volume_img* at the vertices of *surface_path*.

    Parameters
    ----------
    valid_mask
        Boolean volume, on the same grid, marking voxels that hold a real value.
        Pass the RTOP fit mask here so vertices over unfitted tissue are not
        silently read as zeros.

        A vertex is accepted only when **all eight** corners of its trilinear
        stencil are valid, not merely the nearest one. Checking the nearest
        voxel would let a vertex sitting one voxel inside the mask boundary
        average in a zero from just outside it and return a value that is
        quietly biased low -- which is worse than a NaN, because nothing
        downstream could tell.

    Returns
    -------
    (n_vertices,) array of sampled values, NaN where unavailable.
    """
    coords, _ = read_surface(surface_path)
    data = np.asarray(volume_img.dataobj, dtype=float)
    values = sample_volume_at_points(data, volume_img.affine, coords)

    ijk = world_to_voxel(coords, volume_img.affine)
    shape = np.asarray(data.shape[:3])
    valid = np.all((np.rint(ijk) >= 0) & (np.rint(ijk) < shape), axis=1)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, bool)
        base = np.floor(ijk).astype(int)
        for offset in _STENCIL:
            corner = np.clip(base + offset, 0, shape - 1)
            valid &= valid_mask[corner[:, 0], corner[:, 1], corner[:, 2]]

    return np.where(valid, values, np.nan)
