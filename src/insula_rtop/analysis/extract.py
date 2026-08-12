"""Reduce per-vertex RTOP to one number per subject, ROI and hemisphere.

    We computed mean RTOP values across mesh-vertices in three insular
    subdivisions. (Results, "Insula microstructure across its functionally
    defined subdivisions")

Produces a single long-format table, ``group_rtop.tsv``, that every downstream
analysis reads::

    subject  atlas      seg     hemi  subdivision  rtop     n_vertices  n_valid
    100206   Deen2011   insula  L     vAI          1.83     216         216
    ...

plus a ``cortex`` pseudo-ROI per hemisphere -- the mean over every
non-medial-wall vertex -- which is what the outlier rule is applied to.

Vertices are averaged with ``nanmean``: the surface step writes NaN wherever a
vertex fell outside the field of view or over a voxel the MAPL fit skipped, and
those must not be counted as zeros. ``n_valid`` records how many actually
contributed, so a subject whose ROI was mostly NaN is visible rather than
silently averaged from three vertices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insula_rtop.atlases import glasser
from insula_rtop.atlases.fslr import HEMI_LETTERS, check_vertex_count
from insula_rtop.atlases.run import SEGMENTATIONS, load_segmentation
from insula_rtop.imaging import read_surface_scalars
from insula_rtop.surface.run import surface_paths

GROUP_TABLE = "group_rtop.tsv"

#: Name of the whole-cortex pseudo-ROI added alongside the real subdivisions.
CORTEX_ROI = "cortex"

#: A subject-ROI mean computed from fewer than this fraction of the ROI's
#: vertices is not trustworthy and is dropped to NaN.
MIN_VALID_FRACTION = 0.5


def cortex_mask(hemi: str) -> np.ndarray:
    """Non-medial-wall vertices, taken from the HCP-MMP grayordinate coverage."""
    return glasser.mmp_labels(hemi) > 0


def roi_means(
    values: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    *,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, tuple[float, int, int]]:
    """``{name: (mean, n_vertices, n_valid)}`` over each labelled ROI."""
    check_vertex_count(values, "surface values")
    check_vertex_count(labels, "segmentation")
    out: dict[str, tuple[float, int, int]] = {}
    for index, name in sorted(names.items()):
        mask = labels == index
        roi = values[mask]
        n_vertices = int(mask.sum())
        valid = np.isfinite(roi)
        n_valid = int(valid.sum())
        mean = (
            float(roi[valid].mean())
            if n_vertices and n_valid >= min_valid_fraction * n_vertices
            else float("nan")
        )
        out[name] = (mean, n_vertices, n_valid)
    return out


def load_all_segmentations(deriv_root: Path, segmentations=SEGMENTATIONS) -> dict:
    """Read every segmentation once, keyed by ``(atlas, seg)``.

    Hoisted out of the per-subject loop deliberately: reading six ``label.gii``
    files per subject is a few hundred megabytes of pointless I/O over a cohort.
    """
    return {
        (atlas, seg): load_segmentation(deriv_root, atlas, seg)
        for atlas, seg in segmentations
    }


def extract_subject(
    subject_id: str,
    deriv_root: Path,
    segmentations=SEGMENTATIONS,
    *,
    loaded: dict | None = None,
) -> list[dict]:
    """Rows of the group table for one subject."""
    if loaded is None:
        loaded = load_all_segmentations(deriv_root, segmentations)
    paths = surface_paths(deriv_root, subject_id)
    rows: list[dict] = []
    for hemi in HEMI_LETTERS:
        if not paths[hemi].exists():
            raise FileNotFoundError(
                f"[{subject_id}] {paths[hemi]} not found. Run the "
                "`rtop_surface` step first."
            )
        values = read_surface_scalars(paths[hemi])

        cortex = cortex_mask(hemi)
        finite = np.isfinite(values) & cortex
        rows.append(
            {
                "subject": subject_id,
                "atlas": "cortex",
                "seg": CORTEX_ROI,
                "hemi": hemi,
                "subdivision": CORTEX_ROI,
                "rtop": float(values[finite].mean()) if finite.any() else float("nan"),
                "n_vertices": int(cortex.sum()),
                "n_valid": int(finite.sum()),
            }
        )

        for atlas, seg in segmentations:
            labels, names = loaded[(atlas, seg)]
            for name, (mean, n_vertices, n_valid) in roi_means(
                values, labels[hemi], names
            ).items():
                rows.append(
                    {
                        "subject": subject_id,
                        "atlas": atlas,
                        "seg": seg,
                        "hemi": hemi,
                        "subdivision": name,
                        "rtop": mean,
                        "n_vertices": n_vertices,
                        "n_valid": n_valid,
                    }
                )
    return rows


def build_group_table(
    subjects: list[str],
    deriv_root: Path,
    *,
    segmentations=SEGMENTATIONS,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract every subject and write ``group_rtop.tsv``."""
    deriv_root = Path(deriv_root)
    out_dir = Path(out_dir) if out_dir else deriv_root / "analysis"

    loaded = load_all_segmentations(deriv_root, segmentations)
    rows: list[dict] = []
    missing: list[str] = []
    for subject_id in subjects:
        try:
            rows.extend(
                extract_subject(subject_id, deriv_root, segmentations, loaded=loaded)
            )
        except FileNotFoundError as error:
            missing.append(f"{subject_id}: {error}")
    if missing:
        # Not fatal -- a cluster run can legitimately still be finishing -- but
        # never silent: a table quietly missing a third of the cohort would
        # change every statistic downstream.
        print(f"WARNING: {len(missing)} subject(s) had no surface RTOP:")
        for line in missing[:5]:
            print(f"  {line}")
    if not rows:
        raise RuntimeError(
            "No subject produced surface RTOP. Run the `rtop_surface` step first."
        )

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / GROUP_TABLE, sep="\t", index=False, na_rep="n/a")
    print(
        f"Wrote {len(table)} row(s) for "
        f"{table['subject'].nunique()} subject(s) to {out_dir / GROUP_TABLE}"
    )
    return table


def read_group_table(analysis_dir: Path) -> pd.DataFrame:
    path = Path(analysis_dir) / GROUP_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the `roi_extract` step first."
        )
    return pd.read_csv(path, sep="\t", dtype={"subject": str})


def pivot_subdivisions(table: pd.DataFrame, atlas: str, seg: str) -> pd.DataFrame:
    """Wide view: one row per subject, one column per ``hemi/subdivision``."""
    subset = table[(table["atlas"] == atlas) & (table["seg"] == seg)]
    if subset.empty:
        raise KeyError(f"No rows for atlas={atlas!r}, seg={seg!r}")
    wide = subset.pivot_table(
        index="subject", columns=["hemi", "subdivision"], values="rtop"
    )
    wide.columns = [f"{hemi}_{name}" for hemi, name in wide.columns]
    return wide.sort_index()
