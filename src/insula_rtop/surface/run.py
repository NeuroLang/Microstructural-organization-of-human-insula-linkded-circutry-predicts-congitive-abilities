"""Per-subject surface projection of the normalised RTOP volume.

Reads the ``rtop`` derivative and writes::

    <deriv_root>/rtop-surface/
        dataset_description.json
        sub-<id>/dwi/sub-<id>_hemi-<L|R>_space-fsLR_den-32k_desc-MSMAll_rtop.shape.gii
        sub-<id>/dwi/sub-<id>_desc-MSMAll_rtop.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from insula_rtop.hcp.cohort import cohort_subject_ids
from insula_rtop.hcp.layout import HEMI_LETTERS
from insula_rtop.hcp.to_bids import derivative_paths
from insula_rtop.imaging import write_surface_scalars
from insula_rtop.rtop.run import rtop_paths
from insula_rtop.surface.sample import sample_volume_on_surface
from insula_rtop.tools._orchestration import (
    SlurmOptions,
    add_common_orchestration_args,
    run_subject_jobs,
    skip_completed,
)

PIPELINE_NAME = "rtop-surface"


def surface_paths(deriv_root: Path, subject_id: str) -> dict[str, Path]:
    dwi = Path(deriv_root) / PIPELINE_NAME / f"sub-{subject_id}" / "dwi"
    paths = {
        hemi: dwi
        / (
            f"sub-{subject_id}_hemi-{hemi}_space-fsLR_den-32k"
            "_desc-MSMAll_rtop.shape.gii"
        )
        for hemi in HEMI_LETTERS
    }
    paths["json"] = dwi / f"sub-{subject_id}_desc-MSMAll_rtop.json"
    return paths


def is_surface_complete(deriv_root: Path, subject_id: str) -> bool:
    return surface_paths(deriv_root, subject_id)["json"].exists()


def process_subject(
    subject_id: str,
    bids_root: Path,
    deriv_root: Path,
    *,
    force: bool = False,
    skip_existing: bool = False,
) -> None:
    out = surface_paths(deriv_root, subject_id)
    if skip_completed(
        is_surface_complete(deriv_root, subject_id),
        subject_id,
        out["json"],
        force=force,
        skip_existing=skip_existing,
    ):
        return

    rtop_file = rtop_paths(deriv_root, subject_id)["normalized"]
    if not rtop_file.exists():
        raise FileNotFoundError(
            f"[{subject_id}] {rtop_file} not found. Run the `rtop_volume` step first."
        )
    volume = nib.load(str(rtop_file))
    # The fit mask is implicit in the volume: unfitted voxels are exactly 0,
    # since fit_rtop initialises the output array to zeros, and a voxel whose
    # fit raised is NaN. A genuine RTOP is finite and strictly positive, so this
    # recovers the mask without having to store it alongside.
    volume_data = np.asarray(volume.dataobj)
    valid = np.isfinite(volume_data) & (volume_data != 0)

    deriv = derivative_paths(bids_root, subject_id)
    coverage = {}
    for hemi in HEMI_LETTERS:
        surface = deriv[f"midthickness_native_{hemi}"]
        if not surface.exists():
            raise FileNotFoundError(f"[{subject_id}] {surface} not found.")
        values = sample_volume_on_surface(volume, surface, valid_mask=valid)
        write_surface_scalars(out[hemi], values, intent="NIFTI_INTENT_SHAPE")
        coverage[hemi] = {
            "Vertices": int(values.size),
            "ValidVertices": int(np.isfinite(values).sum()),
            "MedianRTOP": (
                float(np.nanmedian(values)) if np.isfinite(values).any() else None
            ),
        }

    out["json"].write_text(
        json.dumps(
            {
                "Description": (
                    "Normalised RTOP trilinearly interpolated at the vertices of "
                    "the subject's midthickness surface, which carries 32k fs_LR "
                    "MSMAll vertex correspondence. NaN marks vertices outside the "
                    "field of view or over voxels the MAPL fit skipped."
                ),
                "SourceVolume": rtop_file.name,
                "Interpolation": "trilinear",
                "Surface": "midthickness_MSMAll.32k_fs_LR",
                "Coverage": coverage,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[{subject_id}] surfaces written "
        f"(L {coverage['L']['ValidVertices']}/{coverage['L']['Vertices']} valid, "
        f"R {coverage['R']['ValidVertices']}/{coverage['R']['Vertices']} valid)."
    )


def write_dataset_description(deriv_root: Path, bids_root: Path) -> None:
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
                "SourceDatasets": [{"URL": f"file://{Path(bids_root).resolve()}"}],
                "GeneratedBy": [
                    {
                        "Name": "insula_rtop.surface",
                        "Description": (
                            "Normalised RTOP sampled at midthickness vertices of "
                            "the 32k fs_LR MSMAll surfaces. Reproduces Menon et "
                            "al. (2020) eLife 9:e53470, Appendix 1, 'Projecting "
                            "RTOP values to the template cortical surface'."
                        ),
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


def run_surface(
    bids_root: Path,
    deriv_root: Path,
    *,
    subjects: list[str] | None = None,
    force: bool = False,
    skip_existing: bool = False,
    resume: bool = False,
    options: SlurmOptions = SlurmOptions(time=20),
) -> tuple[int, int]:
    """Project every subject's RTOP volume onto the fs_LR 32k surfaces."""
    bids_root = Path(bids_root)
    deriv_root = Path(deriv_root)
    if subjects is None:
        subjects = cohort_subject_ids(bids_root)

    write_dataset_description(deriv_root, bids_root)
    kwargs_list = [
        {
            "subject_id": str(sid),
            "bids_root": bids_root,
            "deriv_root": deriv_root,
            "force": force,
            "skip_existing": skip_existing,
        }
        for sid in subjects
    ]
    return run_subject_jobs(
        kwargs_list,
        process_subject,
        log_dir=deriv_root / PIPELINE_NAME / ".submitit_logs",
        resume=resume,
        is_complete=lambda kw: is_surface_complete(kw["deriv_root"], kw["subject_id"]),
        options=options,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--deriv-root", type=Path, required=True)
    add_common_orchestration_args(parser, default_slurm_time=20, default_cpus=1)
    args = parser.parse_args(argv)

    run_surface(
        args.bids_root,
        args.deriv_root,
        subjects=args.subjects,
        force=args.force,
        skip_existing=args.skip_existing,
        resume=args.resume,
        options=SlurmOptions.from_args(args),
    )


if __name__ == "__main__":
    main()
