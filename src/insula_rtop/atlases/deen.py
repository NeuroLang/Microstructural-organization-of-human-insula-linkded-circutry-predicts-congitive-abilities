"""The Deen et al. (2011) tripartite insula parcellation, carried to fs_LR 32k.

    Our primary analysis focused on an independent tripartite insula
    parcellation provided by Deen and colleagues. (Results)

The masks are published as six binary volumes in MNI152 2 mm space at
https://bendeen.com/data/ (``InsulaCluster_K3_{L,R}-{dAI,vAI,PI}.nii.gz``), from

    Deen, Pitskel & Pelphrey (2011). Three systems of insular functional
    connectivity identified with cluster analysis. Cerebral Cortex 21:1498-1506.

HCP's ``MNINonLinear`` space is MNI152NLin6Asym -- the same FSL MNI152 the masks
are defined in -- so the group-average midthickness surface can be sampled in
the mask volumes directly, with no registration.

The subdivisions are volumetric and the insula is thin, so a mask can miss a
midthickness vertex that lies just outside it. Sampling is therefore done with a
small search: a vertex takes the label of the nearest labelled voxel within
:data:`SEARCH_RADIUS_MM`, and is left unlabelled beyond that. Without it a
sizeable fraction of insular vertices fall through the gaps between 2 mm voxels
and the folded midthickness sheet.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from insula_rtop.atlases.fslr import HEMI_LETTERS, group_midthickness
from insula_rtop.constants import INSULA_SUBDIVISIONS

DEEN_URL = "http://bendeen.com/data-zip/InsulaClusters.zip"

#: Files inside the archive, keyed by ``(hemisphere, subdivision)``.
MASK_FILENAMES = {
    (hemi, sub): f"InsulaCluster_K3_{hemi}-{sub}.nii.gz"
    for hemi in HEMI_LETTERS
    for sub in INSULA_SUBDIVISIONS
}

#: How far a vertex may sit from the nearest labelled voxel and still inherit
#: its label. Two 2 mm voxels: enough to bridge the gap between the sampled
#: midthickness sheet and a volumetric ROI, small enough not to leak across the
#: circular sulcus into the operculum.
SEARCH_RADIUS_MM = 4.0


def download(cache_dir: Path, *, url: str = DEEN_URL) -> Path:
    """Fetch ``InsulaClusters.zip`` into *cache_dir*, once."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "InsulaClusters.zip"
    if archive.exists():
        return archive

    import requests

    # bendeen.com answers 406 to requests' default User-Agent.
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "insula-rtop/0.1 (+https://doi.org/10.7554/eLife.53470)"},
    )
    response.raise_for_status()
    archive.write_bytes(response.content)
    return archive


def extract_masks(archive: Path) -> dict[tuple[str, str], nib.Nifti1Image]:
    """Read the six binary masks out of the archive, keyed by (hemi, subdivision)."""
    masks: dict[tuple[str, str], nib.Nifti1Image] = {}
    with zipfile.ZipFile(archive) as zf:
        available = set(zf.namelist())
        for key, filename in MASK_FILENAMES.items():
            if filename not in available:
                raise KeyError(
                    f"{filename} missing from {archive}. Expected the "
                    "InsulaClusters.zip published at https://bendeen.com/data/"
                )
            # The archive members are ``.nii.gz``; from_bytes wants plain NIfTI.
            payload = gzip.decompress(zf.read(filename))
            masks[key] = nib.Nifti1Image.from_bytes(payload)
    return masks


def build_label_volume(
    masks: dict[tuple[str, str], nib.Nifti1Image], hemi: str
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Merge one hemisphere's three binary masks into a single label volume."""
    reference = masks[(hemi, INSULA_SUBDIVISIONS[0])]
    volume = np.zeros(reference.shape[:3], dtype=np.int32)
    names: dict[int, str] = {}
    for i, sub in enumerate(INSULA_SUBDIVISIONS, start=1):
        img = masks[(hemi, sub)]
        if img.shape[:3] != reference.shape[:3]:
            raise ValueError(
                f"Deen mask {hemi}-{sub} has shape {img.shape[:3]}, "
                f"expected {reference.shape[:3]}"
            )
        mask = np.asarray(img.dataobj) > 0
        overlap = volume[mask] != 0
        if overlap.any():
            raise ValueError(
                f"Deen subdivision {sub} overlaps an earlier one in hemisphere "
                f"{hemi} at {int(overlap.sum())} voxel(s)"
            )
        volume[mask] = i
        names[i] = sub
    return volume, reference.affine, names


def project_to_surface(
    volume: np.ndarray,
    affine: np.ndarray,
    hemi: str,
    *,
    search_radius_mm: float = SEARCH_RADIUS_MM,
) -> np.ndarray:
    """Label each fs_LR 32k vertex from the nearest labelled voxel.

    Nearest-neighbour within *search_radius_mm*; vertices further than that from
    any labelled voxel stay 0.
    """
    coords, _ = group_midthickness(hemi)
    labelled = np.flatnonzero(volume.ravel())
    if labelled.size == 0:
        raise ValueError(f"Label volume for hemisphere {hemi} is empty")

    ijk = np.column_stack(np.unravel_index(labelled, volume.shape))
    world = nib.affines.apply_affine(affine, ijk)
    distances, nearest = cKDTree(world).query(
        coords, distance_upper_bound=search_radius_mm
    )

    out = np.zeros(len(coords), dtype=np.int32)
    within = np.isfinite(distances)
    out[within] = volume.ravel()[labelled[nearest[within]]]
    return out


def build_segmentation(
    cache_dir: Path, hemi: str, *, search_radius_mm: float = SEARCH_RADIUS_MM
) -> tuple[np.ndarray, dict[int, str]]:
    """Deen vertex labelling for one hemisphere, downloading the masks if needed."""
    masks = extract_masks(download(cache_dir))
    volume, affine, names = build_label_volume(masks, hemi)
    labels = project_to_surface(
        volume, affine, hemi, search_radius_mm=search_radius_mm
    )
    missing = [name for i, name in names.items() if not (labels == i).any()]
    if missing:
        raise ValueError(
            f"Deen subdivision(s) {missing} reached no vertex in hemisphere "
            f"{hemi}; the search radius ({search_radius_mm} mm) may be too small."
        )
    return labels, names
