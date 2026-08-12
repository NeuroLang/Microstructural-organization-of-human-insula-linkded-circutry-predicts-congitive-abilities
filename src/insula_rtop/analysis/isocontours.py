"""Population-average RTOP isocontours over the insula (Figure 4).

    Population-average RTOP isocontours aligned to cytoarchitecture and VEN
    expression. (Figure 4A)

The insula is flattened onto its own principal plane so the isolines can be
drawn in two dimensions: the horizontal axis is anterior-posterior, the vertical
axis ventral-dorsal. This is a drawing step only -- the gradient direction
analysis fits no plane (see :mod:`insula_rtop.analysis.gradients`). Contours are
computed on the triangulation itself, not on a resampled raster, so no smoothing
is introduced beyond averaging RTOP across participants.
"""

from __future__ import annotations

import numpy as np

from insula_rtop.analysis.gradients import patch_faces, patch_frame
from insula_rtop.atlases.fslr import group_midthickness


def flatten_patch(
    coords: np.ndarray, patch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """2-D coordinates of the patch vertices in its own PCA frame.

    Returns ``(xy, axes)``: ``xy`` is ``(len(patch), 2)`` with x
    anterior-positive and y dorsal-positive.
    """
    origin, axes = patch_frame(coords, patch)
    return (coords[patch] - origin) @ axes.T, axes


def population_mean(values: np.ndarray) -> np.ndarray:
    """Mean RTOP per vertex across participants, ignoring NaN.

    Parameters
    ----------
    values : (n_subjects, n_vertices) array
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected (n_subjects, n_vertices), got {values.shape}")
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=0)


def patch_triangulation(
    hemi: str, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(xy, triangles, patch)`` for the labelled patch of one hemisphere.

    ``triangles`` indexes into ``xy`` (i.e. into the patch), ready for
    ``matplotlib.tri.Triangulation``.
    """
    coords, faces = group_midthickness(hemi)
    patch = np.flatnonzero(labels > 0)
    if patch.size < 3:
        raise ValueError(f"Patch in hemisphere {hemi} has fewer than 3 vertices")

    xy, _ = flatten_patch(coords, patch)
    local = np.full(len(coords), -1, dtype=int)
    local[patch] = np.arange(patch.size)
    triangles = local[patch_faces(faces, patch)]
    return xy, triangles, patch


def isocontour_levels(
    values: np.ndarray, *, n_levels: int = 8, robust: bool = True
) -> np.ndarray:
    """Evenly spaced contour levels spanning the data.

    ``robust`` clips to the 2nd-98th percentile so a handful of extreme vertices
    do not compress every contour into the middle of the range.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values to contour")
    low, high = (
        np.percentile(finite, [2, 98]) if robust else (finite.min(), finite.max())
    )
    if not high > low:
        raise ValueError("Values are constant; there is nothing to contour")
    return np.linspace(low, high, n_levels)
