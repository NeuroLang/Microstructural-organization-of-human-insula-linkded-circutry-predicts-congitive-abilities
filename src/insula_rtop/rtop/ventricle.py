"""Ventricular CSF diffusivity, the per-subject scale for the RTOP normalisation.

Appendix 1, "Return to Origin Probability (RTOP) Density of Water Molecules":

    To render the unnormalized RTOP measurement, P_t, comparable across
    participants we use its normalized version [...] where D_vent is obtained
    from the ventricles in each participant, by computing the average
    ventricular mean diffusivity. RTOP, R_t, is now a dimensionless quantity
    reflecting the relative enhancement of the unnormalized RTOP density, P_t,
    with respect to free water diffusion.

So D_vent is the *mean diffusivity of a diffusion-tensor fit*, averaged over the
ventricular voxels -- not an RTOP itself. The ventricles are taken from the
FreeSurfer ``aparc+aseg`` segmentation that ships with the HCP structural
package (labels 4, 43, 14, 15: left/right lateral, third and fourth ventricle),
resampled to the diffusion grid.

The mask is eroded before averaging. Ventricle voxels adjacent to the boundary
are heavily partial-volumed with periventricular white matter, whose MD is about
a third of free water's; including them biases D_vent down, and since the
normalisation is ``(4 pi D_vent t)^(3/2)`` a downward bias in D_vent scales every
RTOP value in that subject down by the 3/2 power. This is a per-subject
multiplicative confound in exactly the quantity the paper compares across
subjects, so the erosion matters more than its one line of code suggests.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from dipy.reconst.dti import TensorModel

from insula_rtop.constants import VENTRICLE_LABELS
from insula_rtop.imaging import (
    binary_mask_from_labels,
    erode,
    resample_label_to_grid,
)
from insula_rtop.rtop.mapl import restrict_gradient_table

#: Voxels of erosion applied to the ventricle mask on the diffusion grid.
DEFAULT_EROSION = 1

#: Highest b-value used to estimate D_vent, in s/mm^2.
#:
#: **A forced deviation from the paper**, which says only "the average
#: ventricular mean diffusivity". Fitting a tensor to all three HCP shells
#: returns D_vent = 1.33e-3 mm^2/s, less than half of free water's ~3.0e-3 --
#: because CSF has no signal left above b ~ 1000 and the fit is reading the
#: Rician noise floor. Measured in subject 100307's ventricles:
#:
#:     shell        measured S/S0     free water predicts
#:     b = 1000     0.062             0.050
#:     b = 2000     0.044             0.0025
#:     b = 3000     0.044             0.0001
#:
#: Flat from b = 2000 on: that is the noise floor, not diffusion. Eroding the
#: mask further does not move it (2.85e-3 at erosion 2), confirming it is not
#: partial volume. Restricting the fit to b <= 1000 gives 2.84e-3 mm^2/s, which
#: is free water. Since the normalisation is (4 pi D_vent t)^(3/2), the biased
#: estimate would scale every RTOP value in the study by (1.33/2.84)^1.5 = 0.32.
DVENT_MAX_BVAL = 1100.0

#: A subject with fewer surviving ventricle voxels than this has no usable
#: D_vent estimate; better to fail loudly than to normalise by noise.
MIN_VENTRICLE_VOXELS = 20


def ventricle_mask(
    aparc_aseg: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    *,
    erosion: int = DEFAULT_EROSION,
    labels=VENTRICLE_LABELS,
) -> np.ndarray:
    """Eroded ventricular CSF mask, on *reference*'s grid."""
    resampled = resample_label_to_grid(aparc_aseg, reference)
    mask = binary_mask_from_labels(resampled, labels)
    eroded = erode(mask, erosion)
    # Erosion can wipe out the third and fourth ventricles entirely at 1.25 mm.
    # The lateral ventricles alone are enough, but an all-empty result is not.
    return eroded if eroded.any() else mask


def mean_diffusivity(
    data: np.ndarray, gtab, mask: np.ndarray, *, max_bval: float = DVENT_MAX_BVAL
) -> np.ndarray:
    """DTI mean diffusivity (mm^2/s), one value per voxel inside *mask*.

    Only shells at or below *max_bval* enter the fit; see :data:`DVENT_MAX_BVAL`
    for why that restriction is not optional on CSF.

    The masked signals are extracted before the fit rather than handing dipy the
    4-D volume with ``mask=``: the volume is 145 x 174 x 145 x 288, so upcasting
    it to float64 would cost 8.4 GB to fit a few hundred ventricle voxels.
    """
    restricted, keep = restrict_gradient_table(gtab, max_bval)
    signals = np.asarray(np.asarray(data)[np.asarray(mask, bool)], dtype=np.float64)
    return TensorModel(restricted).fit(signals[:, keep]).md


def ventricular_diffusivity(
    data: np.ndarray,
    gtab,
    aparc_aseg: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    *,
    erosion: int = DEFAULT_EROSION,
    max_bval: float = DVENT_MAX_BVAL,
) -> tuple[float, dict]:
    """``D_vent`` in mm^2/s, plus the diagnostics worth keeping in the sidecar."""
    mask = ventricle_mask(aparc_aseg, reference, erosion=erosion)
    n_voxels = int(mask.sum())
    if n_voxels < MIN_VENTRICLE_VOXELS:
        raise RuntimeError(
            f"Only {n_voxels} ventricle voxel(s) survived erosion "
            f"(minimum {MIN_VENTRICLE_VOXELS}). Check the aparc+aseg "
            "segmentation and its alignment with the diffusion grid."
        )

    values = mean_diffusivity(data, gtab, mask, max_bval=max_bval)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < MIN_VENTRICLE_VOXELS:
        raise RuntimeError(
            f"Only {values.size} ventricle voxel(s) yielded a finite positive "
            "mean diffusivity."
        )

    d_vent = float(values.mean())
    diagnostics = {
        "VentricleVoxels": n_voxels,
        "VentricleVoxelsUsed": int(values.size),
        "VentricleErosionVoxels": erosion,
        "VentricularMaxBval": max_bval,
        "VentricularMeanDiffusivity": d_vent,
        "VentricularMeanDiffusivityMedian": float(np.median(values)),
        "VentricularMeanDiffusivitySD": float(values.std()),
    }
    return d_vent, diagnostics
