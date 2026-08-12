"""HCP-MMP 1.0 regions, and the tripartite insula / ACC groupings built on them.

Appendix 1, "Human insula and anterior cingulate cortex ROIs":

    To replicate our findings, we used the multimodal whole-brain atlas derived
    from a large number of HCP participants (Glasser et al., 2016). We
    constructed an equivalent tripartite organization we defined as a homologous
    set of dAI, vAI and PI subdivisions by (i) combining regions AVI+MI+FOP3
    into a dAI subdivision, (ii) using AAIC as the corresponding vAI
    subdivision, and (iii) combining Pol1+Pol2+Ig+FOP2 into a PI subdivision.
    [...] We also examined whether ACC subdivisions defined by their distinct
    functional connectivity with the insular subdivisions, matched distinct ACC
    areas p24, a24pr and p24pr in the HCP multimodal atlas.

Two readings had to be fixed to make that executable:

* "Pol1+Pol2" is HCP-MMP's **PoI1** and **PoI2** (posterior insular areas 1 and
  2). No areas named ``Pol1``/``Pol2`` exist in the atlas; ``PoI1``/``PoI2`` are
  the posterior-insula areas the sentence is grouping, and they sit next to
  ``Ig`` and ``FOP2`` exactly as the grouping implies.
* The paper says the three ACC areas *match* the three functionally-defined
  subdivisions, without stating which goes with which. The assignment used here
  follows the ventral-to-dorsal, rostral-to-caudal ordering of Figure 7 --
  ``p24`` (pregenual, most ventral) for ACC-vAI, ``a24pr`` for ACC-dAI, and
  ``p24pr`` (most posterior) for ACC-PI -- and is declared as data in
  :data:`ACC_AREA_GROUPS` so it can be corrected in one place.

The ACC assignment is an assumption, not a quotation; see README, "Assumptions".
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from insula_rtop.atlases.fslr import HEMI_LETTERS, N_VERTICES_32K

#: Tripartite insula grouping, quoted from Appendix 1 (with Pol -> PoI).
INSULA_AREA_GROUPS: dict[str, tuple[str, ...]] = {
    "vAI": ("AAIC",),
    "dAI": ("AVI", "MI", "FOP3"),
    "PI": ("PoI1", "PoI2", "Ig", "FOP2"),
}

#: ACC grouping. The areas are quoted from Appendix 1; the mapping onto the
#: three functionally-defined subdivisions is this reproduction's assumption.
ACC_AREA_GROUPS: dict[str, tuple[str, ...]] = {
    "ACC-vAI": ("p24",),
    "ACC-dAI": ("a24pr",),
    "ACC-PI": ("p24pr",),
}

#: Grayordinate counts of the CIFTI cortical surface models, used to split
#: ``map_all`` back into hemispheres.
_N_GRAY = {"L": 29696, "R": 29716}


def _data_file(name: str) -> Path:
    return Path(str(resources.files("hcp_utils") / "data" / name))


@lru_cache(maxsize=1)
def _load_mmp() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """``(map_all, labels, gray_indices)`` from the bundled HCP-MMP 1.0 data."""
    mmp = np.load(_data_file("mmp_1.0.npz"), allow_pickle=True)
    vertex_info = np.load(_data_file("fMRI_vertex_info_32k.npz"), allow_pickle=True)
    gray = {"L": vertex_info["grayl"], "R": vertex_info["grayr"]}
    for hemi, idx in gray.items():
        if idx.size != _N_GRAY[hemi]:
            raise ValueError(
                f"hcp_utils grayordinate index for {hemi} has {idx.size} entries, "
                f"expected {_N_GRAY[hemi]}"
            )
    return mmp["map_all"], mmp["labels"], gray


@lru_cache(maxsize=4)
def mmp_labels(hemi: str) -> np.ndarray:
    """HCP-MMP 1.0 area index per vertex, on the 32k fs_LR mesh (0 = unlabelled).

    ``hcp_utils`` stores the parcellation over CIFTI grayordinates, which omit
    the medial wall; this expands it back onto the full standard mesh.
    """
    if hemi not in HEMI_LETTERS:
        raise ValueError(f"hemi must be 'L' or 'R', got {hemi!r}")
    map_all, _, gray = _load_mmp()
    start = 0 if hemi == "L" else _N_GRAY["L"]
    values = map_all[start : start + _N_GRAY[hemi]]
    out = np.zeros(N_VERTICES_32K, dtype=np.int32)
    out[gray[hemi]] = values
    return out


def area_index(hemi: str, area: str) -> int:
    """Index of HCP-MMP area *area* (e.g. ``"AVI"``) in hemisphere *hemi*."""
    _, labels, _ = _load_mmp()
    name = f"{hemi}_{area}"
    matches = np.flatnonzero(labels == name)
    if matches.size != 1:
        raise KeyError(
            f"HCP-MMP area {name!r} not found (or ambiguous) in the bundled "
            "atlas. Available names look like 'L_V1', 'R_AVI'."
        )
    return int(matches[0])


def area_mask(hemi: str, areas) -> np.ndarray:
    """Boolean vertex mask for the union of *areas* in hemisphere *hemi*."""
    labels = mmp_labels(hemi)
    indices = [area_index(hemi, a) for a in areas]
    return np.isin(labels, indices)


def build_segmentation(
    hemi: str, groups: dict[str, tuple[str, ...]]
) -> tuple[np.ndarray, dict[int, str]]:
    """Vertex labelling for a named grouping of HCP-MMP areas.

    Returns
    -------
    (labels, names)
        ``labels`` is 0 outside every group and ``i+1`` inside group *i*;
        ``names`` maps those indices to the group names.
    """
    out = np.zeros(N_VERTICES_32K, dtype=np.int32)
    names: dict[int, str] = {}
    for i, (name, areas) in enumerate(groups.items(), start=1):
        mask = area_mask(hemi, areas)
        if not mask.any():
            raise ValueError(f"HCP-MMP group {name!r} is empty in hemisphere {hemi}")
        overlap = out[mask] != 0
        if overlap.any():
            raise ValueError(
                f"HCP-MMP group {name!r} overlaps an earlier group in "
                f"hemisphere {hemi} at {int(overlap.sum())} vertices"
            )
        out[mask] = i
        names[i] = name
    return out, names
