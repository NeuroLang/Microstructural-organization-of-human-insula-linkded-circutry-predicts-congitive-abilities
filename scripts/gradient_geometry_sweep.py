"""Compute the insula RTOP gradient on every available surface, and compare.

The gradient direction depends on the surface it is measured on: a folded
midthickness, an inflated one and a flat map are different metrics, and the
article does not say which it used. This sweeps all of them against the
directions read off Figure 3C, and is the evidence behind README, "Mean
directions, and how close they come".

    uv run python scripts/gradient_geometry_sweep.py \
        --deriv-root /path/menon_insula/derivatives --subjects 120

It reads whichever insula segmentation is in that derivatives tree, so to sweep
a different parcellation, build it there first with
``atlas_labels.insula_labels`` (see README) and re-run.

The answer, over 451 subjects: twelve geometries put left PI between +0.798pi
and +0.898pi against a published +0.61pi, and the closest of them moves vAI
0.41pi the wrong way. No surface reconciles all three subdivisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from insula_rtop.analysis import gradients
from insula_rtop.analysis.run import ANALYSIS_DIR
from insula_rtop.atlases.fslr import _data_dir
from insula_rtop.atlases.run import load_segmentation
from insula_rtop.imaging import read_surface, read_surface_scalars
from insula_rtop.surface.run import surface_paths

#: Left-hemisphere mean directions measured off the published Figure 3C, in
#: units of pi. Measured by eye off a printed polar plot -- see README for why
#: that measurement is itself the leading suspect.
ARTICLE_LEFT = {"PI": 0.61, "dAI": 0.53, "vAI": -0.22}

#: Every surface HCP ships per subject, and S1200 ships for the group.
KINDS = (
    "midthickness_MSMAll",
    "pial_MSMAll",
    "white_MSMAll",
    "inflated_MSMAll",
    "very_inflated_MSMAll",
)
#: Group-only surfaces. A flat map has no anatomical axes of its own, so its
#: frame is recovered by regressing its coordinates on the midthickness ones.
GROUP_ONLY = ("flat", "sphere")


def native_surface(hcp_root: Path, subject: str, hemi: str, kind: str) -> Path:
    return (
        hcp_root
        / subject
        / "T1w"
        / "fsaverage_LR32k"
        / f"{subject}.{hemi}.{kind}.32k_fs_LR.surf.gii"
    )


def anatomical_frame(
    coords: np.ndarray, midthickness: np.ndarray, patch: np.ndarray
) -> np.ndarray:
    """A frame for a surface whose coordinates carry no anatomy of their own.

    Regresses the patch's coordinates on its anatomical (y, z) so that angle 0
    still means "increasing anteriorly" on a flat map or a sphere.
    """
    _, plane = gradients.patch_frame(coords, patch)
    flat = (coords[patch] - coords[patch].mean(axis=0)) @ plane.T
    anat = midthickness[patch][:, 1:] - midthickness[patch][:, 1:].mean(axis=0)
    fit, *_ = np.linalg.lstsq(anat, flat, rcond=None)
    first = fit[0] / np.linalg.norm(fit[0])
    second = fit[1] - (fit[1] @ first) * first
    return np.array([first @ plane, second / np.linalg.norm(second) @ plane])


def subdivision_directions(coords, faces, values, labels, names, frame):
    """Mean direction per subdivision, as an angle in *frame*."""
    patch = np.flatnonzero(labels > 0)
    gradient, area = gradients.vertex_gradients(coords, faces, values, patch)
    magnitude = np.linalg.norm(gradient, axis=1)
    out = {}
    for index, name in sorted(names.items()):
        usable = (
            (labels == index)
            & np.isfinite(magnitude)
            & (magnitude > 0)
            & (area > 0)
        )
        if not usable.any():
            continue
        weights = area[usable] * magnitude[usable]
        mean = (
            (gradient[usable] / magnitude[usable, None]) * weights[:, None]
        ).sum(axis=0)
        mean /= np.linalg.norm(mean)
        out[name] = float(np.arctan2(mean @ frame[1], mean @ frame[0]))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument(
        "--hcp-root", type=Path, default=Path("/data/parietal/store4/data/HCP")
    )
    parser.add_argument("--hemi", default="L")
    parser.add_argument("--subjects", type=int, default=120)
    args = parser.parse_args(argv)

    deriv, hemi = args.deriv_root, args.hemi
    subjects = (deriv / ANALYSIS_DIR / "analysed_subjects.txt").read_text().split()
    subjects = subjects[: args.subjects]
    labels, names = load_segmentation(deriv, "Deen2011", "insula")
    labels = labels[hemi]
    patch = np.flatnonzero(labels > 0)

    group = {
        f"group_{k}": read_surface(_data_dir() / f"S1200.{hemi}.{k}.32k_fs_LR.surf.gii")
        for k in KINDS + GROUP_ONLY
    }
    midthickness = group[f"group_{KINDS[0]}"][0]
    frames = {
        name: (
            anatomical_frame(coords, midthickness, patch)
            if name.rsplit("_", 1)[-1] in GROUP_ONLY
            else gradients.patch_frame(coords, patch)[1]
        )
        for name, (coords, _) in group.items()
    }

    rows = []
    for subject in subjects:
        values = read_surface_scalars(surface_paths(deriv, subject)[hemi])
        geometries = {
            f"native_{k}": read_surface(native_surface(args.hcp_root, subject, hemi, k))
            for k in KINDS
        }
        geometries.update(group)
        for name, (coords, faces) in geometries.items():
            frame = frames[name] if name in frames else (
                gradients.patch_frame(coords, patch)[1]
            )
            angles = subdivision_directions(
                coords, faces, values, labels, names, frame
            )
            rows += [
                {"geometry": name, "subject": subject, "subdivision": k, "angle": v}
                for k, v in angles.items()
            ]

    table = pd.DataFrame(rows)
    summary = []
    for name, group_rows in table.groupby("geometry"):
        row, gaps = {"geometry": name}, []
        for subdivision, published in ARTICLE_LEFT.items():
            angles = group_rows[group_rows.subdivision == subdivision]["angle"]
            mean = gradients.circular_mean(angles.to_numpy()) / np.pi
            row[subdivision] = round(mean, 3)
            gaps.append(
                abs(np.angle(np.exp(1j * np.pi * (mean - published)))) / np.pi
            )
        row["rms_gap_pi"] = round(float(np.sqrt(np.mean(np.square(gaps)))), 3)
        summary.append(row)

    article = ", ".join(f"{k} {v:+.2f}" for k, v in ARTICLE_LEFT.items())
    print(f"{hemi} hemisphere, {len(subjects)} subjects. Article: {article} (pi)")
    print(pd.DataFrame(summary).sort_values("rms_gap_pi").to_string(index=False))


if __name__ == "__main__":
    main()
