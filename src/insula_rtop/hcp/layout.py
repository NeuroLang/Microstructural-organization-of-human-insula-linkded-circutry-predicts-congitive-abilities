"""Path resolution and subject discovery for a minimally-preprocessed HCP tree.

The layout assumed here is the one the HCP "Structural Preprocessed" and
"Diffusion Preprocessed" packages unpack to, as found on margaret under
``/data/parietal/store4/data/HCP``::

    <root>/<subject_id>/
        T1w/
            T1w_acpc_dc_restore.nii.gz
            aparc+aseg.nii.gz
            ribbon.nii.gz
            Diffusion/{data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz}
            fsaverage_LR32k/<id>.<L|R>.midthickness_MSMAll.32k_fs_LR.surf.gii
        MNINonLinear/
            fsaverage_LR32k/<id>.<L|R>.midthickness_MSMAll.32k_fs_LR.surf.gii

Two facts about this layout carry the whole surface-projection design:

1. ``T1w/Diffusion/data.nii.gz`` is already resampled into the subject's ACPC
   ("T1w native") space, which is the space the ``T1w/fsaverage_LR32k``
   surfaces live in. No registration step is needed between them.
2. Those surfaces are on the 32k fs_LR *standard mesh* with MSMAll vertex
   correspondence, so vertex *i* means the same cortical location in every
   subject. Sampling a volume at their coordinates yields template-space
   surface data directly -- which is exactly the two-step procedure described
   in Appendix 1 ("Projecting RTOP values to the template cortical surface").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HEMI_LETTERS = ("L", "R")


@dataclass(frozen=True)
class SubjectPaths:
    """Every HCP input this pipeline reads, for one subject."""

    subject_id: str
    dwi: Path
    bval: Path
    bvec: Path
    dwi_mask: Path
    t1w: Path
    aparc_aseg: Path
    ribbon: Path
    #: Native (ACPC) space midthickness, keyed by hemisphere. Sampled to get
    #: RTOP. The MNI-space copies under ``MNINonLinear/`` are deliberately not
    #: read: volumetric MNI atlases are carried onto the mesh once, on the S1200
    #: group-average surface, rather than per subject (see
    #: :mod:`insula_rtop.atlases.fslr`).
    midthickness_native: dict[str, Path]

    def required(self) -> list[Path]:
        """Inputs that must exist for the subject to be usable."""
        return [
            self.dwi,
            self.bval,
            self.bvec,
            self.aparc_aseg,
            self.ribbon,
            *self.midthickness_native.values(),
        ]

    def missing(self) -> list[Path]:
        """Required inputs that are absent *or empty*.

        Zero-byte files are the signature of an interrupted download, and the
        HCP tree on margaret contains several (subjects 930449, 937160, 959574
        and 978578 all have 0-byte ``data.nii.gz``, ``bvals`` and ``bvecs``).
        Existence alone would let them into the cohort and fail hours later.
        """
        return [p for p in self.required() if not p.exists() or p.stat().st_size == 0]


def subject_dir(hcp_root: Path, subject_id: str) -> Path:
    return Path(hcp_root) / str(subject_id)


def resolve_subject_paths(hcp_root: Path, subject_id: str) -> SubjectPaths:
    """Build the :class:`SubjectPaths` for *subject_id*. Does not touch disk."""
    # Hydra/OmegaConf parses unquoted numeric CLI list entries (e.g.
    # subjects=[100206]) as int, which breaks Path.__truediv__ below.
    subject_id = str(subject_id)
    sub = subject_dir(hcp_root, subject_id)
    t1w_dir = sub / "T1w"
    diff_dir = t1w_dir / "Diffusion"

    def surf(base: Path, hemi: str) -> Path:
        return (
            base
            / "fsaverage_LR32k"
            / f"{subject_id}.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii"
        )

    return SubjectPaths(
        subject_id=subject_id,
        dwi=diff_dir / "data.nii.gz",
        bval=diff_dir / "bvals",
        bvec=diff_dir / "bvecs",
        dwi_mask=diff_dir / "nodif_brain_mask.nii.gz",
        t1w=t1w_dir / "T1w_acpc_dc_restore.nii.gz",
        aparc_aseg=t1w_dir / "aparc+aseg.nii.gz",
        ribbon=t1w_dir / "ribbon.nii.gz",
        midthickness_native={h: surf(t1w_dir, h) for h in HEMI_LETTERS},
    )


def discover_subjects(hcp_root: Path) -> list[str]:
    """Subject IDs under *hcp_root* that have every input this pipeline needs.

    Discovery globs for the diffusion volume -- the one file no subject can be
    processed without -- and then verifies the rest, so a subject with a partial
    download is excluded rather than failing hours into a cluster run.
    """
    hcp_root = Path(hcp_root)
    candidates = sorted(
        p.parent.parent.parent.name
        for p in hcp_root.glob("*/T1w/Diffusion/data.nii.gz")
    )
    return [
        sid for sid in candidates if not resolve_subject_paths(hcp_root, sid).missing()
    ]
