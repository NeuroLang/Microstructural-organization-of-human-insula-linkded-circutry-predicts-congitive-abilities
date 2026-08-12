"""Grid and coordinate helpers shared by the volume and surface steps.

Everything here is pure ``numpy``/``scipy``/``nibabel``. There is deliberately
no dependency on Connectome Workbench, FSL or FreeSurfer: the HCP tree already
provides the diffusion volume and the 32k fs_LR surfaces in the *same* ACPC
space, so all that is ever needed is an affine and an interpolation.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage


def world_to_voxel(coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map world (RAS mm) coordinates to continuous voxel indices.

    Parameters
    ----------
    coords : (N, 3) array
    affine : (4, 4) array
        The image's ``voxel -> world`` affine.

    Returns
    -------
    (N, 3) array of fractional voxel indices.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    homogeneous = np.column_stack([coords, np.ones(len(coords))])
    return (homogeneous @ inv.T)[:, :3]


def sample_volume_at_points(
    volume: np.ndarray,
    affine: np.ndarray,
    coords: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Interpolate a 3-D *volume* at world *coords*.

    ``order=1`` is trilinear, which is exact for affine intensity fields and is
    what Appendix 1 describes ("the interpolated voxel RTOP value at the
    mid-point between the pial surface and the grey-white matter interface").
    """
    ijk = world_to_voxel(coords, affine)
    return ndimage.map_coordinates(
        np.asarray(volume, dtype=float), ijk.T, order=order, mode="nearest"
    )


def resample_label_to_grid(
    label_img: nib.Nifti1Image, target_img: nib.Nifti1Image
) -> np.ndarray:
    """Nearest-neighbour resample a label volume onto *target_img*'s grid.

    Used to carry the FreeSurfer segmentations (0.7 mm anatomical grid) onto the
    1.25 mm diffusion grid. Nearest-neighbour, never linear: interpolating label
    numbers would invent labels that do not exist.
    """
    target_shape = target_img.shape[:3]
    ijk = np.indices(target_shape).reshape(3, -1).T
    world = nib.affines.apply_affine(target_img.affine, ijk)
    src_ijk = world_to_voxel(world, label_img.affine)
    resampled = ndimage.map_coordinates(
        np.asarray(label_img.dataobj), src_ijk.T, order=0, mode="constant", cval=0
    )
    return resampled.reshape(target_shape)


def binary_mask_from_labels(volume: np.ndarray, labels) -> np.ndarray:
    return np.isin(volume, list(labels))


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    return ndimage.binary_dilation(mask, iterations=iterations)


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    return ndimage.binary_erosion(mask, iterations=iterations)


def read_surface(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a GIFTI surface, returning ``(vertices, faces)``."""
    gii = nib.load(str(path))
    coords = gii.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    faces = gii.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if not coords or not faces:
        raise ValueError(f"{path} is not a GIFTI surface (POINTSET + TRIANGLE)")
    return np.asarray(coords[0].data, float), np.asarray(faces[0].data, int)


def write_surface_scalars(path, data: np.ndarray, *, intent: str) -> None:
    """Write a per-vertex scalar array as a GIFTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = nib.gifti.GiftiDataArray(
        np.asarray(data, dtype=np.float32),
        intent=intent,
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(nib.gifti.GiftiImage(darrays=[array]), str(path))


def read_surface_scalars(path) -> np.ndarray:
    gii = nib.load(str(path))
    return np.asarray(gii.darrays[0].data, dtype=float)
