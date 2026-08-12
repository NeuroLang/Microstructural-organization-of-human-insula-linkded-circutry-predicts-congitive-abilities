"""Run every statistical analysis and write the results.

Reads ``group_rtop.tsv`` plus the per-subject surfaces, and writes into
``<deriv_root>/analysis/``::

    group_rtop.tsv                 (from the roi_extract step)
    outliers.json                  excluded subjects and the Tukey fences
    gradient_directions.tsv        one angle per subject, hemisphere, subdivision
    anova_<atlas>_<seg>.txt        ANOVA + post-hoc contrasts
    posthoc_<atlas>_<seg>.tsv
    stability_<atlas>_<seg>.tsv    detection curve and minimum sample sizes
    gradients_<atlas>.tsv          circular statistics
    cca_<atlas>.txt                canonical correlation and prediction
    results.json                   the headline numbers, for the report

Every analysis is run twice, once on the Deen et al. (2011) parcellation (the
paper's primary analysis) and once on the HCP-MMP grouping (its replication).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from insula_rtop.analysis import anova, cca, gradients, outliers, stability
from insula_rtop.analysis.extract import build_group_table, read_group_table
from insula_rtop.atlases.fslr import HEMI_LETTERS
from insula_rtop.atlases.run import load_segmentation
from insula_rtop.constants import ACC_SUBDIVISIONS, INSULA_SUBDIVISIONS
from insula_rtop.hcp.cohort import cohort_subject_ids, read_participants_tsv
from insula_rtop.hcp.to_bids import derivative_paths
from insula_rtop.imaging import read_surface, read_surface_scalars
from insula_rtop.surface.run import surface_paths

ANALYSIS_DIR = "analysis"

#: ``(atlas, seg, subdivision order)`` for every analysis that is run twice.
ANALYSES = (
    ("Deen2011", "insula", INSULA_SUBDIVISIONS),
    ("HCPMMP1", "insula", INSULA_SUBDIVISIONS),
    ("HCPMMP1", "acc", ACC_SUBDIVISIONS),
)


def collect_gradient_directions(
    subjects: list[str],
    bids_root: Path,
    deriv_root: Path,
    *,
    atlas: str = "Deen2011",
    seg: str = "insula",
    smoothing_iterations: int = 0,
    frame: str = gradients.DEFAULT_FRAME,
) -> pd.DataFrame:
    """One gradient direction per subject, hemisphere and subdivision.

    Geometry comes from each participant's own midthickness surface, as the
    paper specifies ("gradient field along insular surface of each
    participant"); the labels are the shared fs_LR segmentation.
    """
    labels, names = load_segmentation(deriv_root, atlas, seg)
    rows = []
    for subject_id in subjects:
        paths = surface_paths(deriv_root, subject_id)
        deriv = derivative_paths(bids_root, subject_id)
        for hemi in HEMI_LETTERS:
            if not paths[hemi].exists():
                continue
            values = read_surface_scalars(paths[hemi])
            coords, faces = read_surface(deriv[f"midthickness_native_{hemi}"])
            angles, vectors = gradients.subject_directions(
                values, coords, faces, labels[hemi], names,
                smoothing_iterations=smoothing_iterations,
                frame=frame,
            )
            for subdivision, angle in angles.items():
                vx, vy, vz = vectors[subdivision]
                rows.append(
                    {
                        "subject": subject_id,
                        "hemi": hemi,
                        "subdivision": subdivision,
                        "angle": angle,
                        "vx": vx,
                        "vy": vy,
                        "vz": vz,
                    }
                )
    if not rows:
        raise RuntimeError("No subject had surface RTOP for the gradient analysis.")
    return pd.DataFrame(rows)


def _jsonable(value):
    """Convert numpy/pandas values into something ``json.dumps`` can emit.

    NaN and infinity become ``null``. Python's ``json`` happily writes bare
    ``NaN`` and ``Infinity`` literals, which are not JSON and are rejected by
    most parsers -- and these tables do produce them: ``min_n`` is NaN when a
    contrast never reaches the stability threshold, and the circular dispersion
    is infinite for a perfectly uniform set of directions.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [
            {k: _jsonable(v) for k, v in row.items()}
            for row in value.to_dict(orient="records")
        ]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _group_comparisons(
    table: pd.DataFrame, out_dir: Path, *, n_resamples: int
) -> dict:
    """Figures 2B/2C and 7B/7C: the ANOVA and the subsampling curves."""
    results: dict = {}
    for atlas, seg, order in ANALYSES:
        key = f"{atlas}_{seg}"

        summary = anova.summarize(table, atlas, seg, order=order)
        (out_dir / f"anova_{key}.txt").write_text(anova.format_summary(summary) + "\n")
        summary["posthoc"].to_csv(
            out_dir / f"posthoc_{key}.tsv", sep="\t", index=False
        )
        results[f"anova_{key}"] = {
            "n_subjects": summary["n_subjects"],
            "ordering": summary["ordering"],
            "posthoc": _jsonable(summary["posthoc"]),
        }
        print(anova.format_summary(summary))

        stab = stability.summarize(
            table, atlas, seg, n_resamples=n_resamples, order=order
        )
        stab["curve"].to_csv(out_dir / f"stability_{key}.tsv", sep="\t", index=False)
        results[f"stability_{key}"] = _jsonable(stab["minimum_sample_sizes"])
        print(stability.format_summary(stab))
    return results


def _gradient_analyses(
    subjects: list[str],
    bids_root: Path,
    deriv_root: Path,
    out_dir: Path,
    *,
    smoothing_iterations: int = 0,
    frame: str = gradients.DEFAULT_FRAME,
) -> dict:
    """Figure 3: per-subject gradient directions and their circular statistics."""
    results: dict = {}
    for atlas, seg, _ in ANALYSES:
        if seg != "insula":
            continue
        directions = collect_gradient_directions(
            subjects, bids_root, deriv_root, atlas=atlas, seg=seg,
            smoothing_iterations=smoothing_iterations, frame=frame,
        )
        directions.to_csv(
            out_dir / f"gradient_directions_{atlas}.tsv", sep="\t", index=False
        )
        summary = gradients.summarize(directions)
        summary.to_csv(out_dir / f"gradients_{atlas}.tsv", sep="\t", index=False)
        results[f"gradients_{atlas}"] = _jsonable(summary)
        print(gradients.format_summary(summary))
    return results


def _cca_analyses(
    table: pd.DataFrame,
    participants: pd.DataFrame,
    out_dir: Path,
    *,
    covariate_columns: tuple[str, ...],
    group_column: str | None,
) -> dict:
    """Figure 8: canonical correlation with cognitive control, and its CV."""
    results: dict = {}
    for atlas, seg, _ in ANALYSES:
        if seg != "insula":
            continue
        summary = cca.summarize(
            table,
            participants,
            atlas=atlas,
            seg=seg,
            covariate_columns=covariate_columns,
            group_column=group_column,
        )
        (out_dir / f"cca_{atlas}.txt").write_text(cca.format_summary(summary) + "\n")
        pd.DataFrame(
            {
                "predicted": summary["prediction"]["predicted"],
                "observed": summary["prediction"]["observed"],
            }
        ).to_csv(out_dir / f"cca_prediction_{atlas}.tsv", sep="\t", index=False)
        summary["fit"]["brain_weights"].to_csv(
            out_dir / f"cca_brain_weights_{atlas}.tsv", sep="\t"
        )
        summary["fit"]["behavior_weights"].to_csv(
            out_dir / f"cca_behavior_weights_{atlas}.tsv", sep="\t"
        )
        # The in-sample canonical variates, for Figure 8A's scatter.
        pd.DataFrame(
            {
                "brain": summary["fit"]["brain_scores"][:, 0],
                "behavior": summary["fit"]["behavior_scores"][:, 0],
            }
        ).to_csv(out_dir / f"cca_scores_{atlas}.tsv", sep="\t", index=False)
        results[f"cca_{atlas}"] = {
            "n": summary["fit"]["n"],
            "canonical_r": summary["fit"]["canonical_correlations"][0],
            "canonical_p": summary["fit"]["canonical_p_values"][0],
            "prediction_r": summary["prediction"]["r"],
            "prediction_p": summary["prediction"]["p"],
            "prediction_cohens_d": summary["prediction"]["cohens_d"],
            "prediction_scheme": summary["prediction"]["scheme"],
        }
        print(cca.format_summary(summary))
    return results


def _exclude_outliers(
    table: pd.DataFrame, out_dir: Path, *, statistic: str
) -> tuple[pd.DataFrame, dict]:
    """Apply the paper's 1.5 IQR rule and record who it dropped."""
    table, excluded, fences = outliers.drop_outliers(table, statistic=statistic)
    (out_dir / "outliers.json").write_text(
        json.dumps({**fences, "excluded_subjects": excluded}, indent=2) + "\n"
    )
    # The exact cohort every statistic in this run was computed on. The paper
    # published none, which is why its 433 cannot be recovered; this one is a
    # plain list so a re-run can be pinned to it with `subjects=@<file>`.
    kept_ids = sorted(table["subject"].astype(str).unique())
    (out_dir / "analysed_subjects.txt").write_text(
        "\n".join(kept_ids) + "\n"
    )
    print(
        f"Outlier exclusion ({statistic}, {fences['multiplier']} IQR): "
        f"dropped {len(excluded)}, kept {table['subject'].nunique()}."
    )
    return table, {
        "n_subjects_before_outlier_exclusion": fences["n_subjects"],
        "n_outliers_excluded": len(excluded),
        "n_subjects": int(table["subject"].nunique()),
        "outlier_fences": fences,
    }


def run_analysis(
    bids_root: Path,
    deriv_root: Path,
    *,
    subjects: list[str] | None = None,
    rebuild_table: bool = True,
    outlier_statistic: str = "cortex",
    n_resamples: int = stability.DEFAULT_N_RESAMPLES,
    covariate_columns: tuple[str, ...] = (),
    group_column: str | None = None,
    gradient_smoothing: int = 0,
    gradient_frame: str = gradients.DEFAULT_FRAME,
) -> dict:
    """Run every statistical analysis and write the results."""
    bids_root = Path(bids_root)
    deriv_root = Path(deriv_root)
    out_dir = deriv_root / ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = cohort_subject_ids(bids_root)
    table = (
        build_group_table(subjects, deriv_root, out_dir=out_dir)
        if rebuild_table
        else read_group_table(out_dir)
    )

    table, results = _exclude_outliers(
        table, out_dir, statistic=outlier_statistic
    )
    kept = sorted(table["subject"].astype(str).unique())

    results |= _group_comparisons(table, out_dir, n_resamples=n_resamples)
    results |= _gradient_analyses(
        kept, bids_root, deriv_root, out_dir,
        smoothing_iterations=gradient_smoothing, frame=gradient_frame,
    )
    results |= _cca_analyses(
        table,
        read_participants_tsv(bids_root),
        out_dir,
        covariate_columns=covariate_columns,
        group_column=group_column,
    )

    (out_dir / "results.json").write_text(
        json.dumps(_jsonable(results), indent=2, allow_nan=False) + "\n"
    )
    print(f"\nWrote results to {out_dir}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--no-rebuild-table", action="store_true")
    parser.add_argument(
        "--outlier-statistic", choices=("cortex", "insula"), default="cortex"
    )
    parser.add_argument(
        "--n-resamples", type=int, default=stability.DEFAULT_N_RESAMPLES
    )
    parser.add_argument("--covariates", nargs="*", default=())
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--gradient-smoothing", type=int, default=0)
    parser.add_argument(
        "--gradient-frame", choices=("anatomical", "patch"),
        default=gradients.DEFAULT_FRAME,
    )
    args = parser.parse_args(argv)

    run_analysis(
        args.bids_root,
        args.deriv_root,
        subjects=args.subjects,
        rebuild_table=not args.no_rebuild_table,
        outlier_statistic=args.outlier_statistic,
        n_resamples=args.n_resamples,
        covariate_columns=tuple(args.covariates),
        group_column=args.group_column,
        gradient_smoothing=args.gradient_smoothing,
        gradient_frame=args.gradient_frame,
    )


if __name__ == "__main__":
    main()
