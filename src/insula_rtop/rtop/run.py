"""Per-subject RTOP volume: MAPL fit, ventricular normalisation, BIDS output.

Reads the BIDS dataset produced by :mod:`insula_rtop.hcp.to_bids` and writes::

    <deriv_root>/rtop/
        dataset_description.json
        sub-<id>/dwi/sub-<id>_space-T1w_desc-rtop_rtop.nii.gz      unnormalised P_t
        sub-<id>/dwi/sub-<id>_space-T1w_desc-rtopnorm_rtop.nii.gz  normalised R_t
        sub-<id>/dwi/sub-<id>_space-T1w_desc-rtopnorm_rtop.json    D_vent + diagnostics

Both volumes are kept: the normalised one is what every analysis uses, and the
unnormalised one plus ``D_vent`` in the sidecar is what makes the normalisation
auditable after the fact without refitting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from insula_rtop.constants import DIFFUSION_TIME_S
from insula_rtop.hcp.cohort import cohort_subject_ids
from insula_rtop.hcp.layout import HEMI_LETTERS
from insula_rtop.hcp.to_bids import derivative_paths, raw_paths
from insula_rtop.rtop.mapl import (
    RADIAL_ORDER,
    build_gradient_table,
    fit_rtop,
    load_gradients,
)
from insula_rtop.rtop.masks import DEFAULT_RIBBON_DILATION, build_fit_mask
from insula_rtop.rtop.normalize import (
    MAX_PLAUSIBLE_RTOP,
    flag_implausible,
    free_water_rtop,
    measured_ventricular_rtop,
    normalize_rtop,
)
from insula_rtop.rtop.ventricle import (
    DEFAULT_EROSION,
    ventricle_mask,
    ventricular_diffusivity,
)
from insula_rtop.tools._orchestration import (
    SlurmOptions,
    add_common_orchestration_args,
    run_subject_jobs,
    skip_completed,
)

PIPELINE_NAME = "rtop"


def rtop_paths(deriv_root: Path, subject_id: str) -> dict[str, Path]:
    dwi = Path(deriv_root) / PIPELINE_NAME / f"sub-{subject_id}" / "dwi"
    stem = f"sub-{subject_id}_space-T1w"
    return {
        "raw": dwi / f"{stem}_desc-rtop_rtop.nii.gz",
        "normalized": dwi / f"{stem}_desc-rtopnorm_rtop.nii.gz",
        "json": dwi / f"{stem}_desc-rtopnorm_rtop.json",
    }


def is_rtop_complete(deriv_root: Path, subject_id: str) -> bool:
    """True once the last file written by :func:`process_subject` exists."""
    return rtop_paths(deriv_root, subject_id)["json"].exists()


@dataclass(frozen=True)
class SubjectInputs:
    """Everything :func:`process_subject` reads off disk, loaded once."""

    dwi: nib.Nifti1Image
    data: np.ndarray
    gtab: object
    aseg: nib.Nifti1Image
    fit_mask: np.ndarray
    ventricles: np.ndarray


def load_subject_inputs(
    subject_id: str,
    bids_root: Path,
    *,
    fit_mask: str,
    ribbon_dilation: int,
    ventricle_erosion: int,
) -> SubjectInputs:
    """Read the BIDS inputs and build the two masks the fit needs."""
    raw = raw_paths(bids_root, subject_id)
    deriv = derivative_paths(bids_root, subject_id)
    for path in (raw["dwi"], raw["bval"], raw["bvec"], deriv["aparc_aseg"]):
        if not path.exists():
            raise FileNotFoundError(
                f"[{subject_id}] {path} not found. Run the `hcp2bids` step first."
            )

    dwi_img = nib.load(str(raw["dwi"]))
    aseg_img = nib.load(str(deriv["aparc_aseg"]))
    bvals, bvecs = load_gradients(raw["bval"], raw["bvec"])
    brain_mask = (
        nib.load(str(deriv["dwi_mask"])) if deriv["dwi_mask"].exists() else None
    )
    return SubjectInputs(
        dwi=dwi_img,
        data=np.asarray(dwi_img.dataobj, dtype=np.float32),
        gtab=build_gradient_table(bvals, bvecs),
        aseg=aseg_img,
        fit_mask=build_fit_mask(
            dwi_img,
            ribbon=nib.load(str(deriv["ribbon"])),
            aparc_aseg=aseg_img,
            brain_mask=brain_mask,
            surfaces=[deriv[f"midthickness_native_{h}"] for h in HEMI_LETTERS],
            strategy=fit_mask,
            ribbon_dilation=ribbon_dilation,
            ventricle_erosion=ventricle_erosion,
        ),
        ventricles=ventricle_mask(aseg_img, dwi_img, erosion=ventricle_erosion),
    )


def build_sidecar(
    normalized: np.ndarray,
    mask: np.ndarray,
    *,
    raw_name: str,
    divisor: float,
    d_vent: float,
    diffusion_time: float,
    radial_order: int,
    fit_mask: str,
    ribbon_dilation: int,
    n_jobs: int,
    n_failed: int,
    n_implausible: int,
    diagnostics: dict,
) -> dict:
    """The provenance record written beside the volumes."""
    finite = normalized[mask]
    finite = finite[np.isfinite(finite)]
    return {
        "Description": (
            "Return-to-origin probability from MAP-MRI with Laplacian "
            "regularization (MAPL), divided by the RTOP measured in this "
            "subject's ventricles. Dimensionless; 1 in ventricular CSF."
        ),
        "RawRTOPFile": raw_name,
        "RadialOrder": radial_order,
        "LaplacianWeighting": "GCV",
        "EffectiveDiffusionTime": diffusion_time,
        "Normalizer": "measured ventricular RTOP",
        "NormalizerValue": divisor,
        # The analytic alternative from Appendix 1's equation. Kept so the
        # other convention is a per-subject rescale rather than a refit.
        "FreeWaterRTOP": free_water_rtop(d_vent, diffusion_time=diffusion_time),
        "FitMaskStrategy": fit_mask,
        "FitMaskVoxels": int(mask.sum()),
        "RibbonDilationVoxels": ribbon_dilation,
        "FitJobs": n_jobs,
        "FailedVoxels": int(n_failed),
        "ImplausibleVoxels": int(n_implausible),
        "MaxPlausibleRTOP": MAX_PLAUSIBLE_RTOP,
        "NormalizedRTOPMedian": float(np.median(finite)) if finite.size else None,
        **diagnostics,
    }


def process_subject(
    subject_id: str,
    bids_root: Path,
    deriv_root: Path,
    *,
    fit_mask: str = "surface+ventricles",
    ribbon_dilation: int = DEFAULT_RIBBON_DILATION,
    ventricle_erosion: int = DEFAULT_EROSION,
    radial_order: int = RADIAL_ORDER,
    diffusion_time: float = DIFFUSION_TIME_S,
    n_jobs: int = 1,
    force: bool = False,
    skip_existing: bool = False,
    progress: bool = False,
) -> None:
    """Fit, normalise and write one subject's RTOP volumes."""
    out = rtop_paths(deriv_root, subject_id)
    if skip_completed(
        is_rtop_complete(deriv_root, subject_id),
        subject_id,
        out["json"],
        force=force,
        skip_existing=skip_existing,
    ):
        return

    inputs = load_subject_inputs(
        subject_id,
        bids_root,
        fit_mask=fit_mask,
        ribbon_dilation=ribbon_dilation,
        ventricle_erosion=ventricle_erosion,
    )
    print(
        f"[{subject_id}] fitting MAPL in {int(inputs.fit_mask.sum())} voxel(s) "
        f"on {n_jobs} core(s)..."
    )

    d_vent, diagnostics = ventricular_diffusivity(
        inputs.data, inputs.gtab, inputs.aseg, inputs.dwi, erosion=ventricle_erosion
    )
    rtop, n_failed = fit_rtop(
        inputs.data,
        inputs.gtab,
        inputs.fit_mask,
        radial_order=radial_order,
        progress=progress,
        n_jobs=n_jobs,
    )

    # The paper's prose normaliser: the RTOP actually measured in this
    # subject's ventricles. The analytic (4 pi D_vent t)^(-3/2) goes into the
    # sidecar as FreeWaterRTOP, so the other convention is a pure rescale.
    divisor, vent_diagnostics = measured_ventricular_rtop(rtop, inputs.ventricles)
    diagnostics.update(vent_diagnostics)

    normalized = normalize_rtop(rtop, divisor)
    normalized, n_implausible = flag_implausible(normalized, inputs.fit_mask)
    # Keep the two volumes consistent: a voxel rejected in one is rejected in
    # both, so the unnormalised map cannot be used to resurrect a failed fit.
    rtop = np.where(np.isnan(normalized), np.float32("nan"), rtop)

    out["raw"].parent.mkdir(parents=True, exist_ok=True)
    header, affine = inputs.dwi.header, inputs.dwi.affine
    nib.save(nib.Nifti1Image(rtop, affine, header), str(out["raw"]))
    nib.save(nib.Nifti1Image(normalized, affine, header), str(out["normalized"]))

    sidecar = build_sidecar(
        normalized,
        inputs.fit_mask,
        raw_name=out["raw"].name,
        divisor=divisor,
        d_vent=d_vent,
        diffusion_time=diffusion_time,
        radial_order=radial_order,
        fit_mask=fit_mask,
        ribbon_dilation=ribbon_dilation,
        n_jobs=n_jobs,
        n_failed=n_failed,
        n_implausible=n_implausible,
        diagnostics=diagnostics,
    )
    # Written last: is_rtop_complete keys on it, so a half-written subject re-runs.
    out["json"].write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"[{subject_id}] D_vent={d_vent:.3e} mm^2/s, "
        f"median normalised RTOP={sidecar['NormalizedRTOPMedian']}, "
        f"{n_failed} failed and {n_implausible} implausible voxel(s)."
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
                        "Name": "insula_rtop.rtop",
                        "Description": (
                            "Voxelwise return-to-origin probability from MAPL "
                            "(dipy), normalised by the subject's ventricular "
                            "free-water RTOP. Reproduces Menon et al. (2020) "
                            "eLife 9:e53470, Appendix 1."
                        ),
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


def run_rtop(
    bids_root: Path,
    deriv_root: Path,
    *,
    subjects: list[str] | None = None,
    fit_mask: str = "surface+ventricles",
    ribbon_dilation: int = DEFAULT_RIBBON_DILATION,
    ventricle_erosion: int = DEFAULT_EROSION,
    radial_order: int = RADIAL_ORDER,
    fit_jobs: int = 1,
    force: bool = False,
    skip_existing: bool = False,
    resume: bool = False,
    options: SlurmOptions = SlurmOptions(time=600),
) -> tuple[int, int]:
    """Fit every subject in the cohort."""
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
            "fit_mask": fit_mask,
            "ribbon_dilation": ribbon_dilation,
            "ventricle_erosion": ventricle_erosion,
            "radial_order": radial_order,
            "n_jobs": fit_jobs,
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
        is_complete=lambda kw: is_rtop_complete(kw["deriv_root"], kw["subject_id"]),
        options=options,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument(
        "--fit-mask",
        choices=("surface+ventricles", "ribbon+ventricles", "brain"),
        default="surface+ventricles",
    )
    parser.add_argument(
        "--fit-jobs",
        type=int,
        default=1,
        help="Processes used for the MAPL fit within one subject.",
    )
    parser.add_argument("--ribbon-dilation", type=int, default=DEFAULT_RIBBON_DILATION)
    parser.add_argument("--ventricle-erosion", type=int, default=DEFAULT_EROSION)
    parser.add_argument("--radial-order", type=int, default=RADIAL_ORDER)
    add_common_orchestration_args(parser, default_slurm_time=600, default_cpus=1)
    args = parser.parse_args(argv)

    run_rtop(
        args.bids_root,
        args.deriv_root,
        subjects=args.subjects,
        fit_mask=args.fit_mask,
        ribbon_dilation=args.ribbon_dilation,
        ventricle_erosion=args.ventricle_erosion,
        radial_order=args.radial_order,
        fit_jobs=args.fit_jobs,
        force=args.force,
        skip_existing=args.skip_existing,
        resume=args.resume,
        options=SlurmOptions.from_args(args),
    )


if __name__ == "__main__":
    main()
