"""Cohort selection: which HCP subjects enter the reproduction, and why.

The paper used "minimal preprocessed diffusion MRI data from 433 participants
[...] from the HCP Q1-Q6 Data Release", of which 20 were dropped as RTOP
outliers, leaving 413. **The subject IDs are not published** -- the eLife
article's only supplementary file holds the von Mises-Fisher and CCA tables,
and the Zenodo code record (10.5281/zenodo.3759708) contains nothing but a
README. So the exact 433 cannot be recovered.

What *is* reproducible is the cohort definition. In the S1200 behavioural
release the historical Q1-Q6 releases appear as ``Release in {Q1, Q2, Q3,
S500}`` (Q4-Q6 were folded into the S500 release label). Applying that filter
plus on-disk completeness gives 482 subjects on margaret, of which 467 also
have all 11 CCA behavioural measures -- a superset of the paper's 433.

The second half of the paper's selection, dropping RTOP outliers at 1.5 IQR,
happens later in :mod:`insula_rtop.analysis.outliers`, because it needs RTOP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from insula_rtop.constants import CCA_BEHAVIORAL_COLUMNS, Q1_Q6_RELEASES
from insula_rtop.hcp.layout import discover_subjects

SUBJECT_COL = "Subject"
RELEASE_COL = "Release"

#: Columns carried from the HCP tables into ``participants.tsv``. Everything
#: the downstream analyses read must be listed here.
PARTICIPANT_COLUMNS = (
    "Subject",
    "Release",
    "Gender",
    "Age",
    *CCA_BEHAVIORAL_COLUMNS,
)

#: Columns carried when the unrestricted table has them, and skipped when it
#: does not. The FreeSurfer volumes are absent from some HCP exports, and no
#: published analysis needs them -- they are here so head size and brain size
#: can be regressed out of the CCA as a deconfounding check
#: (``analysis.covariates``). See README, "Assumptions".
OPTIONAL_COLUMNS = ("FS_IntraCranial_Vol", "FS_BrainSeg_Vol")

#: Columns taken from the *restricted* table when it is available. Family_ID is
#: not used by any published analysis (the paper's CCA uses leave-one-out CV,
#: which ignores the twin/sibling structure of HCP-YA) but is carried through so
#: that caveat can be quantified.
RESTRICTED_COLUMNS = ("Age_in_Yrs", "Family_ID", "Handedness")


def load_behavioral(
    behavioral_csv: Path, restricted_csv: Path | None = None
) -> pd.DataFrame:
    """Load the HCP unrestricted table, optionally merged with the restricted one."""
    df = pd.read_csv(behavioral_csv)
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)

    keep = [c for c in PARTICIPANT_COLUMNS if c in df.columns]
    missing = [c for c in PARTICIPANT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{behavioral_csv} is missing required column(s): {missing}. "
            "Expected the HCP S1200 unrestricted behavioural table."
        )
    df = df[keep + [c for c in OPTIONAL_COLUMNS if c in df.columns]]

    if restricted_csv is not None and Path(restricted_csv).exists():
        rdf = pd.read_csv(restricted_csv)
        rdf[SUBJECT_COL] = rdf[SUBJECT_COL].astype(str)
        rkeep = [SUBJECT_COL] + [c for c in RESTRICTED_COLUMNS if c in rdf.columns]
        df = df.merge(rdf[rkeep], on=SUBJECT_COL, how="left")

    return df


def select_cohort(
    behavioral: pd.DataFrame,
    hcp_root: Path,
    *,
    releases: tuple[str, ...] | None = Q1_Q6_RELEASES,
    require_behavior: bool = True,
    subjects: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the cohort filters, returning the table and a provenance counter.

    Parameters
    ----------
    releases
        Keep only these values of the ``Release`` column. ``None`` keeps every
        release (~1051 subjects with complete diffusion data on margaret).
    require_behavior
        Drop subjects with any missing value among the 11 CCA measures. The
        paper's CCA needs complete cases; setting this False keeps subjects that
        can still contribute to the RTOP analyses.
    subjects
        Explicit subject list, applied before every other filter. Use this to
        pin a cohort for exact re-runs.

    Returns
    -------
    (table, counts)
        ``counts`` records the surviving N after each filter, in order.
    """
    counts: dict[str, int] = {}
    df = behavioral.copy()
    counts["behavioral_table"] = len(df)

    if subjects is not None:
        wanted = {str(s) for s in subjects}
        df = df[df[SUBJECT_COL].isin(wanted)]
        counts["explicit_subject_list"] = len(df)

    if releases is not None:
        df = df[df[RELEASE_COL].isin(list(releases))]
        counts[f"release_in_{'+'.join(releases)}"] = len(df)

    on_disk = set(discover_subjects(hcp_root))
    df = df[df[SUBJECT_COL].isin(on_disk)]
    counts["complete_hcp_inputs"] = len(df)

    if require_behavior:
        df = df.dropna(subset=list(CCA_BEHAVIORAL_COLUMNS))
        counts["complete_cca_measures"] = len(df)

    df = df.sort_values(SUBJECT_COL).reset_index(drop=True)
    return df, counts


def write_participants_tsv(cohort: pd.DataFrame, bids_root: Path) -> Path:
    """Write ``participants.tsv`` with a BIDS ``participant_id`` column."""
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    out = cohort.copy()
    out.insert(0, "participant_id", "sub-" + out[SUBJECT_COL].astype(str))
    path = bids_root / "participants.tsv"
    out.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


def read_participants_tsv(bids_root: Path) -> pd.DataFrame:
    """Read back the cohort table written by :func:`write_participants_tsv`."""
    path = Path(bids_root) / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the `cohort` step (or `hcp-cohort`) first."
        )
    df = pd.read_csv(path, sep="\t", dtype={SUBJECT_COL: str})
    return df


def cohort_subject_ids(bids_root: Path) -> list[str]:
    return read_participants_tsv(bids_root)[SUBJECT_COL].astype(str).tolist()


def run_cohort(
    hcp_root: Path,
    behavioral_csv: Path,
    bids_root: Path,
    *,
    restricted_csv: Path | None = None,
    releases: tuple[str, ...] | None = Q1_Q6_RELEASES,
    require_behavior: bool = True,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    behavioral = load_behavioral(behavioral_csv, restricted_csv)
    cohort, counts = select_cohort(
        behavioral,
        hcp_root,
        releases=releases,
        require_behavior=require_behavior,
        subjects=subjects,
    )
    if cohort.empty:
        raise RuntimeError(
            "Cohort selection produced no subjects. Filter counts: "
            f"{counts}. Check hcp_root and the Release filter."
        )
    path = write_participants_tsv(cohort, bids_root)
    print("Cohort selection:")
    for name, n in counts.items():
        print(f"  {name:<28s} {n}")
    print(f"Wrote {len(cohort)} participant(s) to {path}")
    return cohort


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hcp-root", type=Path, required=True)
    parser.add_argument("--behavioral-csv", type=Path, required=True)
    parser.add_argument("--restricted-csv", type=Path, default=None)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument(
        "--releases",
        nargs="+",
        default=list(Q1_Q6_RELEASES),
        help="Release values to keep; pass 'all' to disable the filter.",
    )
    parser.add_argument("--no-require-behavior", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=None)
    args = parser.parse_args(argv)

    releases = None if args.releases == ["all"] else tuple(args.releases)
    run_cohort(
        args.hcp_root,
        args.behavioral_csv,
        args.bids_root,
        restricted_csv=args.restricted_csv,
        releases=releases,
        require_behavior=not args.no_require_behavior,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
