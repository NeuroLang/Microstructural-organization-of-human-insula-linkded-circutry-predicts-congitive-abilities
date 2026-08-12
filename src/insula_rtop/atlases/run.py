"""Build every parcellation used by the analyses, once, on the fs_LR 32k mesh.

Writes a cohort-level derivative (no ``sub-`` directories: fs_LR vertices
correspond across subjects, so one labelling serves everyone)::

    <deriv_root>/atlases/
        dataset_description.json
        atlas-Deen2011_seg-insula_hemi-<L|R>_space-fsLR_den-32k_dseg.label.gii
        atlas-Deen2011_seg-insula_dseg.tsv
        atlas-HCPMMP1_seg-insula_hemi-<L|R>_space-fsLR_den-32k_dseg.label.gii
        atlas-HCPMMP1_seg-insula_dseg.tsv
        atlas-HCPMMP1_seg-acc_hemi-<L|R>_space-fsLR_den-32k_dseg.label.gii
        atlas-HCPMMP1_seg-acc_dseg.tsv

Segmentations built:

===============  ========  ==========================================
atlas            seg       source
===============  ========  ==========================================
``Deen2011``     insula    Deen et al. (2011) MNI masks -> surface
``HCPMMP1``      insula    Glasser et al. (2016) area groupings
``HCPMMP1``      acc       Glasser et al. (2016) areas p24/a24pr/p24pr
===============  ========  ==========================================

The ACC segmentation can be replaced wholesale by pointing ``--acc-labels`` at a
pair of ``label.gii`` files -- the study's own functionally-defined ACC ROIs, if
they are recovered -- without touching the analyses that consume it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from insula_rtop.atlases import deen, glasser
from insula_rtop.atlases.fslr import HEMI_LETTERS, check_vertex_count
from insula_rtop.constants import INSULA_SUBDIVISIONS

PIPELINE_NAME = "atlases"

#: ``(atlas, seg)`` pairs this step produces, in build order.
SEGMENTATIONS = (
    ("Deen2011", "insula"),
    ("HCPMMP1", "insula"),
    ("HCPMMP1", "acc"),
)


def label_path(deriv_root: Path, atlas: str, seg: str, hemi: str) -> Path:
    return (
        Path(deriv_root)
        / PIPELINE_NAME
        / f"atlas-{atlas}_seg-{seg}_hemi-{hemi}_space-fsLR_den-32k_dseg.label.gii"
    )


def table_path(deriv_root: Path, atlas: str, seg: str) -> Path:
    return Path(deriv_root) / PIPELINE_NAME / f"atlas-{atlas}_seg-{seg}_dseg.tsv"


def write_label_gii(path: Path, labels: np.ndarray) -> None:
    """Write per-vertex integer labels as a GIFTI label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = nib.gifti.GiftiDataArray(
        np.asarray(labels, dtype=np.int32),
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
    )
    nib.save(nib.gifti.GiftiImage(darrays=[array]), str(path))


def read_label_gii(path: Path) -> np.ndarray:
    return check_vertex_count(
        np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.int32), str(path)
    )


def write_table(path: Path, names: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["index\tname"] + [f"{i}\t{name}" for i, name in sorted(names.items())]
    path.write_text("\n".join(lines) + "\n")


def read_table(path: Path) -> dict[int, str]:
    rows = path.read_text().splitlines()[1:]
    return {int(i): name for i, name in (row.split("\t") for row in rows if row)}


def supplied_segmentation(spec: dict, hemi: str) -> tuple[np.ndarray, dict[int, str]]:
    """Read a segmentation from files, renumbered to this pipeline's order.

    A file someone else made need not use the same label values, so the spec
    names them explicitly rather than guessing from geometry::

        {"L": {"file": "left.func.gii", "vAI": 3, "dAI": 2, "PI": 1}, "R": {...}}

    Anything not listed becomes 0, so a file carrying extra parcels contributes
    only the three subdivisions asked for.
    """
    entry = dict(spec[hemi])
    raw = read_label_gii(Path(entry.pop("file")))
    names = dict(enumerate(INSULA_SUBDIVISIONS, start=1))
    missing = set(INSULA_SUBDIVISIONS) - set(entry)
    if missing:
        raise KeyError(
            f"insula_labels[{hemi!r}] must give a label value for every "
            f"subdivision; missing {sorted(missing)}"
        )
    out = np.zeros_like(raw)
    for index, name in names.items():
        value = int(entry[name])
        if not (raw == value).any():
            raise ValueError(
                f"insula_labels[{hemi!r}][{name!r}] = {value} matches no vertex"
            )
        out[raw == value] = index
    return out, names


def build_segmentation(
    atlas: str,
    seg: str,
    hemi: str,
    *,
    cache_dir: Path,
    acc_labels: dict | None = None,
    insula_labels: dict | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Dispatch to the right builder for one ``(atlas, seg, hemi)``."""
    if atlas == "Deen2011" and seg == "insula":
        if insula_labels:
            return supplied_segmentation(insula_labels, hemi)
        return deen.build_segmentation(cache_dir, hemi)
    if atlas == "HCPMMP1" and seg == "insula":
        return glasser.build_segmentation(hemi, glasser.INSULA_AREA_GROUPS)
    if atlas == "HCPMMP1" and seg == "acc":
        if acc_labels:
            labels = read_label_gii(Path(acc_labels[hemi]))
            names = dict(enumerate(glasser.ACC_AREA_GROUPS, start=1))
            return labels, names
        return glasser.build_segmentation(hemi, glasser.ACC_AREA_GROUPS)
    raise ValueError(f"Unknown segmentation: atlas={atlas!r}, seg={seg!r}")


def run_atlases(
    deriv_root: Path,
    *,
    cache_dir: Path | None = None,
    acc_labels: dict | None = None,
    insula_labels: dict | None = None,
    force: bool = False,
) -> None:
    deriv_root = Path(deriv_root)
    cache_dir = Path(cache_dir) if cache_dir else deriv_root / PIPELINE_NAME / "cache"
    write_dataset_description(deriv_root)

    for atlas, seg in SEGMENTATIONS:
        table = table_path(deriv_root, atlas, seg)
        paths = [label_path(deriv_root, atlas, seg, h) for h in HEMI_LETTERS]
        if table.exists() and all(p.exists() for p in paths) and not force:
            print(f"atlas-{atlas}_seg-{seg}: exists, skipping.")
            continue

        names: dict[int, str] = {}
        for hemi in HEMI_LETTERS:
            labels, names = build_segmentation(
                atlas,
                seg,
                hemi,
                cache_dir=cache_dir,
                acc_labels=acc_labels,
                insula_labels=insula_labels,
            )
            check_vertex_count(labels, f"atlas-{atlas}_seg-{seg}_hemi-{hemi}")
            write_label_gii(label_path(deriv_root, atlas, seg, hemi), labels)
            sizes = {names[i]: int((labels == i).sum()) for i in sorted(names)}
            print(f"atlas-{atlas}_seg-{seg}_hemi-{hemi}: {sizes}")
        write_table(table, names)


def write_dataset_description(deriv_root: Path) -> None:
    root = Path(deriv_root) / PIPELINE_NAME
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_description.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "Name": PIPELINE_NAME,
                "BIDSVersion": "1.9.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "insula_rtop.atlases",
                        "Description": (
                            "Insula and ACC segmentations on the 32k fs_LR mesh: "
                            "Deen et al. (2011) functional insula clusters "
                            "projected from MNI152 2 mm, and groupings of HCP-MMP "
                            "1.0 areas (Glasser et al., 2016) as specified in "
                            "Menon et al. (2020) eLife 9:e53470, Appendix 1."
                        ),
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


def load_segmentation(
    deriv_root: Path, atlas: str, seg: str
) -> tuple[dict[str, np.ndarray], dict[int, str]]:
    """Read back a segmentation: ``({hemi: labels}, {index: name})``."""
    table = table_path(deriv_root, atlas, seg)
    if not table.exists():
        raise FileNotFoundError(
            f"{table} not found. Run the `atlas_labels` step (or "
            "`build-atlas-labels`) first."
        )
    labels = {
        hemi: read_label_gii(label_path(deriv_root, atlas, seg, hemi))
        for hemi in HEMI_LETTERS
    }
    return labels, read_table(table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--acc-labels",
        nargs=2,
        metavar=("LEFT_LABEL_GII", "RIGHT_LABEL_GII"),
        default=None,
        help="Replace the HCP-MMP ACC segmentation with these fs_LR 32k labels.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    acc_labels = (
        dict(zip(HEMI_LETTERS, args.acc_labels, strict=True))
        if args.acc_labels
        else None
    )
    run_atlases(
        args.deriv_root,
        cache_dir=args.cache_dir,
        acc_labels=acc_labels,
        force=args.force,
    )


if __name__ == "__main__":
    main()


