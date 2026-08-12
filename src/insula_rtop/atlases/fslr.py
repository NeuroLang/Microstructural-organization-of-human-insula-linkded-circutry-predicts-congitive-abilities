"""The fs_LR 32k reference geometry every parcellation is expressed on.

Two things are needed to turn a volumetric MNI atlas into vertex labels, and to
run the gradient analysis in a frame shared by all subjects: the standard-mesh
vertex count, and a group-average midthickness surface in MNI space.

Both come from the ``hcp_utils`` wheel, which redistributes the HCP S1200
group-average MSMAll surfaces and the HCP-MMP 1.0 parcellation as plain data
files under an MIT licence. Nothing from that package's API is imported -- only
the files -- so the pipeline does not inherit its (unmaintained) code.

Using the *group-average* surface rather than each subject's own MNI-space
surface is deliberate: fs_LR 32k already puts every subject in vertex
correspondence, so a volumetric ROI has to become one shared vertex labelling
for the paper's between-subdivision ANOVA to be comparing the same vertices in
everyone. Projecting per subject would make the ROI itself a random variable.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from insula_rtop.imaging import read_surface

#: Vertices per hemisphere on the 32k fs_LR standard mesh.
N_VERTICES_32K = 32492

HEMI_LETTERS = ("L", "R")


def _data_dir() -> Path:
    return Path(str(resources.files("hcp_utils") / "data"))


def group_midthickness_path(hemi: str) -> Path:
    """HCP S1200 group-average MSMAll midthickness, in MNI152NLin6Asym space."""
    if hemi not in HEMI_LETTERS:
        raise ValueError(f"hemi must be 'L' or 'R', got {hemi!r}")
    return _data_dir() / f"S1200.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii"


@lru_cache(maxsize=4)
def group_midthickness(hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of the group-average midthickness surface."""
    coords, faces = read_surface(group_midthickness_path(hemi))
    if len(coords) != N_VERTICES_32K:
        raise ValueError(
            f"{group_midthickness_path(hemi)} has {len(coords)} vertices, "
            f"expected {N_VERTICES_32K}"
        )
    return coords, faces


def check_vertex_count(values: np.ndarray, what: str = "surface data") -> np.ndarray:
    """Fail loudly on anything that is not on the 32k fs_LR mesh."""
    values = np.asarray(values)
    if values.shape[0] != N_VERTICES_32K:
        raise ValueError(
            f"{what} has {values.shape[0]} vertices, expected {N_VERTICES_32K} "
            "(32k fs_LR standard mesh)"
        )
    return values
