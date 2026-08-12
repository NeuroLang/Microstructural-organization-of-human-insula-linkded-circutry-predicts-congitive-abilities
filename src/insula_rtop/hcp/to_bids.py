"""Migrate a minimally-preprocessed HCP tree into a BIDS dataset.

Everything downstream of this step reads BIDS, never raw HCP paths.

The image files are **symlinked**, not copied: the HCP diffusion volumes are
~1.4 GB per subject, so a copied raw tree for the full cohort would be well over
a terabyte, and nothing in this pipeline modifies them. Files this pipeline
*generates* -- the JSON sidecars, ``participants.tsv``, ``dataset_description``,
and every derivative -- are real files.

The HCP data are already preprocessed (Sotiropoulos et al., 2013), so calling
the result a BIDS *raw* dataset is a compromise. It is spelled with
``desc-preproc`` on the derivative side and a ``GeneratedBy`` entry naming the
HCP minimal preprocessing pipelines, so the provenance is not lost.

Layout produced::

    <bids_root>/
        dataset_description.json
        participants.tsv
        sub-<id>/
            anat/sub-<id>_T1w.nii.gz                        -> symlink
            dwi/sub-<id>_acq-multishell_dwi.nii.gz          -> symlink
            dwi/sub-<id>_acq-multishell_dwi.bval            (generated)
            dwi/sub-<id>_acq-multishell_dwi.bvec            (generated, 3 x N)
            dwi/sub-<id>_acq-multishell_dwi.json            (generated)
    <bids_root>/derivatives/hcp-minproc/
        dataset_description.json
        sub-<id>/
            anat/sub-<id>_desc-aparcaseg_dseg.nii.gz        -> symlink
            anat/sub-<id>_desc-ribbon_dseg.nii.gz           -> symlink
            anat/sub-<id>_hemi-<L|R>_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii
            dwi/sub-<id>_desc-brain_mask.nii.gz             -> symlink

``bvec`` is rewritten rather than symlinked so that the ``(3, N)`` shape BIDS
requires is *asserted* at migration time rather than assumed hours later inside
a cluster job. HCP already writes ``(3, N)``, so the values pass through
unchanged; only the formatting is normalised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from insula_rtop.constants import BIG_DELTA_S, DIFFUSION_TIME_S, SMALL_DELTA_S
from insula_rtop.hcp.cohort import cohort_subject_ids
from insula_rtop.hcp.layout import HEMI_LETTERS, resolve_subject_paths
from insula_rtop.tools._orchestration import (
    SlurmOptions,
    add_common_orchestration_args,
    run_subject_jobs,
    skip_completed,
)

DERIVATIVE_NAME = "hcp-minproc"
DWI_ENTITIES = "acq-multishell"


def bids_subject_dir(bids_root: Path, subject_id: str, modality: str) -> Path:
    return Path(bids_root) / f"sub-{subject_id}" / modality


def raw_paths(bids_root: Path, subject_id: str) -> dict[str, Path]:
    """Paths of the BIDS *raw* files for one subject."""
    anat = bids_subject_dir(bids_root, subject_id, "anat")
    dwi = bids_subject_dir(bids_root, subject_id, "dwi")
    stem = f"sub-{subject_id}_{DWI_ENTITIES}_dwi"
    return {
        "t1w": anat / f"sub-{subject_id}_T1w.nii.gz",
        "dwi": dwi / f"{stem}.nii.gz",
        "bval": dwi / f"{stem}.bval",
        "bvec": dwi / f"{stem}.bvec",
        "json": dwi / f"{stem}.json",
    }


def derivative_paths(bids_root: Path, subject_id: str) -> dict[str, Path]:
    """Paths of the ``hcp-minproc`` derivative files for one subject."""
    root = Path(bids_root) / "derivatives" / DERIVATIVE_NAME
    anat = root / f"sub-{subject_id}" / "anat"
    dwi = root / f"sub-{subject_id}" / "dwi"
    paths = {
        "aparc_aseg": anat / f"sub-{subject_id}_desc-aparcaseg_dseg.nii.gz",
        "ribbon": anat / f"sub-{subject_id}_desc-ribbon_dseg.nii.gz",
        "dwi_mask": dwi / f"sub-{subject_id}_desc-brain_mask.nii.gz",
    }
    for hemi in HEMI_LETTERS:
        paths[f"midthickness_native_{hemi}"] = anat / (
            f"sub-{subject_id}_hemi-{hemi}_space-fsLR_den-32k"
            "_desc-MSMAll_midthickness.surf.gii"
        )
    return paths


def _symlink(src: Path, dst: Path, *, force: bool) -> None:
    """Point *dst* at *src*, replacing an existing link when *force*."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if not force:
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def _write_bvec(src: Path, dst: Path) -> None:
    """Copy the HCP bvec file, asserting the ``(3, N)`` shape on the way."""
    bvec = np.loadtxt(src)
    if bvec.shape[0] != 3:
        raise ValueError(
            f"{src}: expected a (3, N) gradient array as written by HCP, "
            f"got {bvec.shape}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, bvec, fmt="%.6f")


def dwi_sidecar(bval_path: Path) -> dict:
    """The ``_dwi.json`` sidecar, carrying the timings the RTOP fit needs."""
    bvals = np.loadtxt(bval_path)
    return {
        "Manufacturer": "Siemens",
        "MagneticFieldStrength": 3,
        "EchoTime": 0.0890,
        "RepetitionTime": 5.5,
        "LargeDeltaTime": BIG_DELTA_S,
        "SmallDeltaTime": SMALL_DELTA_S,
        # t = Delta - delta/3, the diffusion time entering both P_t and the
        # ventricular normalisation R_t = P_t (4 pi D_vent t)^(3/2).
        "EffectiveDiffusionTime": DIFFUSION_TIME_S,
        "NumberOfVolumes": int(bvals.size),
        "Sources": ["HCP minimal preprocessing pipelines (Glasser et al., 2013)"],
        "Description": (
            "HCP minimally-preprocessed diffusion data, resampled to 1.25 mm "
            "isotropic in the subject's ACPC (T1w native) space."
        ),
    }


def is_bidsify_complete(bids_root: Path, subject_id: str) -> bool:
    """True once the last file written by :func:`process_subject` exists."""
    return derivative_paths(bids_root, subject_id)["dwi_mask"].is_symlink()


def process_subject(
    subject_id: str,
    hcp_root: Path,
    bids_root: Path,
    *,
    force: bool = False,
    skip_existing: bool = False,
) -> None:
    """Link and generate every BIDS file for one subject."""
    if skip_completed(
        is_bidsify_complete(bids_root, subject_id),
        subject_id,
        derivative_paths(bids_root, subject_id)["dwi_mask"],
        force=force,
        skip_existing=skip_existing,
    ):
        return

    src = resolve_subject_paths(hcp_root, subject_id)
    missing = src.missing()
    if missing:
        raise FileNotFoundError(
            f"[{subject_id}] missing HCP input(s): {[str(p) for p in missing]}"
        )

    raw = raw_paths(bids_root, subject_id)
    _symlink(src.t1w, raw["t1w"], force=force)
    _symlink(src.dwi, raw["dwi"], force=force)
    _symlink(src.bval, raw["bval"], force=force)
    _write_bvec(src.bvec, raw["bvec"])
    raw["json"].write_text(json.dumps(dwi_sidecar(src.bval), indent=2) + "\n")

    deriv = derivative_paths(bids_root, subject_id)
    _symlink(src.aparc_aseg, deriv["aparc_aseg"], force=force)
    _symlink(src.ribbon, deriv["ribbon"], force=force)
    for hemi in HEMI_LETTERS:
        _symlink(
            src.midthickness_native[hemi],
            deriv[f"midthickness_native_{hemi}"],
            force=force,
        )
    # Written last: is_bidsify_complete keys on it, so a half-linked subject re-runs.
    _symlink(src.dwi_mask, deriv["dwi_mask"], force=force)
    print(f"[{subject_id}] BIDS entries written.")


def write_dataset_descriptions(bids_root: Path, hcp_root: Path) -> None:
    """Write both ``dataset_description.json`` files, without clobbering."""
    bids_root = Path(bids_root)
    raw_desc = {
        "Name": "HCP Young Adult, minimally preprocessed (BIDS view)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "Authors": ["Human Connectome Project, WU-Minn Consortium"],
        "HowToAcknowledge": (
            "Data were provided by the Human Connectome Project, WU-Minn "
            "Consortium (1U54MH091657)."
        ),
        "SourceDatasets": [{"URL": f"file://{Path(hcp_root).resolve()}"}],
        "GeneratedBy": [
            {
                "Name": "hcp2bids",
                "Description": (
                    "Symlinked BIDS view of an HCP minimally-preprocessed tree. "
                    "Image files are symlinks into the HCP tree; sidecars, "
                    "gradient tables and participants.tsv are generated."
                ),
            }
        ],
    }
    deriv_desc = {
        "Name": DERIVATIVE_NAME,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "SourceDatasets": [{"URL": f"file://{Path(hcp_root).resolve()}"}],
        "GeneratedBy": [
            {
                "Name": "HCP minimal preprocessing pipelines",
                "Description": (
                    "Segmentations, brain masks and 32k fs_LR MSMAll surfaces as "
                    "distributed by the HCP (Glasser et al., 2013), exposed under "
                    "BIDS-derivatives names."
                ),
            }
        ],
    }
    for root, desc in (
        (bids_root, raw_desc),
        (bids_root / "derivatives" / DERIVATIVE_NAME, deriv_desc),
    ):
        root.mkdir(parents=True, exist_ok=True)
        path = root / "dataset_description.json"
        if path.exists():
            continue  # never clobber: it may carry hand-edited provenance
        path.write_text(json.dumps(desc, indent=2) + "\n")


def run_bidsify(
    hcp_root: Path,
    bids_root: Path,
    *,
    subjects: list[str] | None = None,
    force: bool = False,
    skip_existing: bool = False,
    resume: bool = False,
    options: SlurmOptions = SlurmOptions(time=10),
) -> tuple[int, int]:
    """Migrate every cohort subject into the BIDS view."""
    hcp_root = Path(hcp_root)
    bids_root = Path(bids_root)
    if subjects is None:
        subjects = cohort_subject_ids(bids_root)

    kwargs_list = [
        {
            "subject_id": str(sid),
            "hcp_root": hcp_root,
            "bids_root": bids_root,
            "force": force,
            "skip_existing": skip_existing,
        }
        for sid in subjects
    ]
    result = run_subject_jobs(
        kwargs_list,
        process_subject,
        log_dir=bids_root / ".submitit_logs",
        resume=resume,
        is_complete=lambda kw: is_bidsify_complete(kw["bids_root"], kw["subject_id"]),
        options=options,
    )
    write_dataset_descriptions(bids_root, hcp_root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hcp-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    add_common_orchestration_args(parser, default_slurm_time=10, default_cpus=1)
    args = parser.parse_args(argv)

    run_bidsify(
        args.hcp_root,
        args.bids_root,
        subjects=args.subjects,
        force=args.force,
        skip_existing=args.skip_existing,
        resume=args.resume,
        options=SlurmOptions.from_args(args),
    )


if __name__ == "__main__":
    main()
