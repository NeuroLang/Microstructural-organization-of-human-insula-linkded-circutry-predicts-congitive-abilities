"""The ventricular normalisation that makes RTOP comparable across subjects.

Appendix 1 describes the normaliser two ways, and they are not the same number.

The prose:

    To render the Return-To-Origin probability comparable across subjects, we
    normalized it by the average ventricular Return-To-Origin probability of
    each subject's cortico-spinal fluid in the ventricles.

and the equation, for free diffusion as in the ventricles,

    P_free(t) = (4 pi D_vent t)^(-3/2),  R_t = P_t / P_free(t)

with ``D_vent`` "estimated by computing the average mean diffusivity within the
ventricle". The first divides by an RTOP *measured* in the ventricles; the
second by the RTOP free water *would* have at the measured ventricular
diffusivity. On real HCP data they differ by a factor of about 2.3, because
MAPL fitted across all three shells reads the Rician noise floor in CSF (see
:data:`insula_rtop.rtop.ventricle.DVENT_MAX_BVAL`) and returns an inflated
ventricular RTOP.

**This pipeline uses the measured version** -- the paper's prose --
:func:`measured_ventricular_rtop`. The analytic value is still computed and
recorded in every sidecar as ``FreeWaterRTOP``, so the other convention is a
pure per-subject rescale away.

Either way the divisor is one scalar per subject, so the choice cannot affect
any within-subject result: gradient directions are scale-invariant, and each
subject's vAI/dAI/PI ordering is untouched. It does affect the between-subject
analyses -- the 1.5 IQR outlier rule and the CCA -- which read absolute level.
"""

from __future__ import annotations

import numpy as np

from insula_rtop.constants import DIFFUSION_TIME_S


def free_water_rtop(
    d_vent: float, *, diffusion_time: float = DIFFUSION_TIME_S
) -> float:
    """``P_free(t) = (4 pi D t)^(-3/2)``, in mm^-3 for D in mm^2/s.

    The analytic normaliser implied by Appendix 1's equation. Recorded as a
    diagnostic; :func:`measured_ventricular_rtop` is what the pipeline divides
    by.
    """
    if d_vent <= 0:
        raise ValueError(f"d_vent must be positive, got {d_vent}")
    return float((4.0 * np.pi * d_vent * diffusion_time) ** -1.5)


#: A ventricle voxel further than this factor from the ventricular median RTOP
#: is a fit failure, not CSF. Needed because the divisor is a *mean* over a few
#: hundred voxels: MAPL occasionally returns 1e15, and one such voxel would
#: otherwise set a subject's entire normalisation to nearly zero.
VENTRICLE_OUTLIER_FACTOR = 10.0


def measured_ventricular_rtop(
    rtop: np.ndarray,
    ventricles: np.ndarray,
    *,
    outlier_factor: float = VENTRICLE_OUTLIER_FACTOR,
) -> tuple[float, dict]:
    """Mean unnormalised RTOP over the ventricles: the paper's prose normaliser.

    Voxels that are non-finite, non-positive, or off the ventricular median by
    more than *outlier_factor* are excluded before averaging.

    Returns
    -------
    (value, diagnostics)
    """
    values = np.asarray(rtop, dtype=np.float64)[np.asarray(ventricles, dtype=bool)]
    usable = values[np.isfinite(values) & (values > 0)]
    if usable.size == 0:
        raise RuntimeError(
            "No ventricle voxel yielded a finite positive RTOP; the "
            "normalisation has nothing to divide by."
        )

    median = float(np.median(usable))
    keep = (usable > median / outlier_factor) & (usable < median * outlier_factor)
    kept = usable[keep]
    if kept.size == 0:
        raise RuntimeError("Every ventricle voxel was rejected as an outlier.")

    return float(kept.mean()), {
        "MeasuredVentricularRTOP": float(kept.mean()),
        "MeasuredVentricularRTOPMedian": median,
        "MeasuredVentricularRTOPSD": float(kept.std()),
        "VentricleVoxelsAveraged": int(kept.size),
        "VentricleVoxelsRejected": int(usable.size - kept.size),
    }


def normalize_rtop(rtop: np.ndarray, divisor: float) -> np.ndarray:
    """Divide unnormalised RTOP (mm^-3) by a normaliser in the same units."""
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return np.asarray(rtop, dtype=np.float32) / np.float32(divisor)


#: Largest normalised RTOP treated as a real measurement. Expressed as an
#: enhancement over the ventricular RTOP, so it moves with the normaliser.
#:
#: MAPL is an unconstrained least-squares fit -- ``positivity_constraint=False``
#: is what the paper specifies -- so nothing stops it returning coefficients
#: whose propagator is not a probability density. In practice about 0.1% of
#: voxels come back pathological: subject 100307 spans -9.0e4 to 1.2e15 while
#: the healthy distribution runs p1 = 3.2, p50 = 10.2, p99 = 28.8.
#:
#: The bound is physical, not a percentile. R is the enhancement over free
#: water, and a compartment restricting water to a length L gives roughly
#: R ~ (D_vent t / L^2)^(3/2). With D_vent t = 1.12e-4 mm^2, R = 1000 already
#: implies L below one micron -- smaller than any structure water can be
#: trapped in over a 40 ms diffusion time. So this discards fit failures, not
#: unusual tissue: it is a factor of ten above p99.9 and touches 0.014% of
#: voxels.
#:
#: Handling these is not optional decoration. The paper's own 1.5 IQR outlier
#: rule acts on a mean RTOP, and one voxel at 1e15 puts a subject's cortical
#: mean at 1e10 -- as it did before this was added.
MAX_PLAUSIBLE_RTOP = 1000.0


def flag_implausible(
    normalized: np.ndarray,
    mask: np.ndarray,
    *,
    max_rtop: float = MAX_PLAUSIBLE_RTOP,
) -> tuple[np.ndarray, int]:
    """NaN out fit failures inside *mask*, leaving unfitted voxels at 0.

    A normalised RTOP that is not finite, not positive, or above *max_rtop* is a
    failed fit rather than a measurement. Unfitted voxels keep their exact 0, so
    the "0 means never fitted, NaN means fitted and rejected" convention the
    surface step relies on survives.

    Returns
    -------
    (values, n_flagged)
    """
    values = np.array(normalized, dtype=np.float32, copy=True)
    mask = np.asarray(mask, dtype=bool)
    with np.errstate(invalid="ignore"):
        bad = mask & (
            ~np.isfinite(values) | (values <= 0) | (values > np.float32(max_rtop))
        )
    values[bad] = np.nan
    return values, int(bad.sum())
