"""Return-to-origin probability from MAP-MRI with Laplacian regularization.

Appendix 1, "Computing RTOP from diffusion MRI data":

    At each voxel, we computed RTOP by evaluating a regularized representation
    of the signal based on the Mean Average Propagator formalism with Laplacian
    regularization (Fick et al., 2016) (MAPL) included in the dipy open-source
    software package. The regularization parameter was selected through
    generalized cross validation and RTOP was computed analytically from the
    fitted MAPL parameters.

That maps onto dipy's :class:`dipy.reconst.mapmri.MapmriModel` with
``laplacian_regularization=True`` and ``laplacian_weighting="GCV"``, followed by
``MapmriFit.rtop()``.

**Units.** dipy sets ``tau = big_delta - small_delta / 3`` when the gradient
table carries the pulse timings, which is exactly the paper's ``t``; with
b-values in s/mm^2 and timings in seconds, ``rtop()`` comes out in mm^-3. That
matters because the normalisation in :mod:`insula_rtop.rtop.normalize` multiplies
it by ``(4 pi D_vent t)^(3/2)`` in mm^3. Passing the timings is therefore not
optional -- without them dipy silently falls back to ``tau = 1/(4 pi^2)`` and the
normalised RTOP is wrong by a constant factor of roughly 4.6.
"""

from __future__ import annotations

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel

from insula_rtop.constants import B0_THRESHOLD, BIG_DELTA_S, SMALL_DELTA_S

#: MAP-MRI truncation order. dipy's default, and the one MAPL was validated at
#: in Fick et al. (2016); the paper does not state a different value.
RADIAL_ORDER = 6


def load_gradients(bval_path, bvec_path) -> tuple[np.ndarray, np.ndarray]:
    """Read a BIDS ``.bval``/``.bvec`` pair as ``(bvals, bvecs)``.

    BIDS and HCP both store bvecs as ``(3, N)``; dipy wants ``(N, 3)``.
    """
    bvals = np.atleast_1d(np.loadtxt(bval_path).ravel())
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] == bvals.size:
        bvecs = bvecs.T
    if bvecs.shape != (bvals.size, 3):
        raise ValueError(
            f"bvec shape {bvecs.shape} is incompatible with {bvals.size} b-values"
        )
    return bvals, bvecs


def build_gradient_table(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    big_delta: float = BIG_DELTA_S,
    small_delta: float = SMALL_DELTA_S,
    b0_threshold: float = B0_THRESHOLD,
):
    """Gradient table carrying the pulse timings (see the module docstring)."""
    return gradient_table(
        bvals,
        bvecs=bvecs,
        big_delta=big_delta,
        small_delta=small_delta,
        b0_threshold=b0_threshold,
    )


def restrict_gradient_table(gtab, max_bval: float):
    """A gradient table keeping only ``bvals <= max_bval``, timings intact.

    Returns ``(gtab, keep)`` so the same selection can be applied to signals.
    """
    keep = np.asarray(gtab.bvals) <= max_bval
    restricted = gradient_table(
        np.asarray(gtab.bvals)[keep],
        bvecs=np.asarray(gtab.bvecs)[keep],
        big_delta=gtab.big_delta,
        small_delta=gtab.small_delta,
        b0_threshold=B0_THRESHOLD,
    )
    return restricted, keep


def build_mapl_model(gtab, *, radial_order: int = RADIAL_ORDER) -> MapmriModel:
    """MAPL as specified in Appendix 1: Laplacian regularization, weight by GCV."""
    return MapmriModel(
        gtab,
        radial_order=radial_order,
        laplacian_regularization=True,
        laplacian_weighting="GCV",
        positivity_constraint=False,
        anisotropic_scaling=True,
    )


def _progress_chunks(signals: np.ndarray, size: int = 500):
    """Yield blocks of voxels behind a progress bar, for interactive runs."""
    from tqdm import tqdm

    blocks = np.array_split(signals, max(len(signals) // size, 1))
    return tqdm(blocks, desc="MAPL", unit="blk")


def _fit_chunk(
    signals: np.ndarray, gtab, radial_order: int
) -> tuple[np.ndarray, int]:
    """Fit one block of voxels. Module-level so it can be sent to a worker."""
    model = build_mapl_model(gtab, radial_order=radial_order)
    values = np.empty(len(signals), dtype=np.float32)
    n_failed = 0
    for i, signal in enumerate(signals):
        try:
            values[i] = model.fit(signal).rtop()
        except Exception:
            values[i] = np.nan
            n_failed += 1
    return values, n_failed


def fit_rtop(
    data: np.ndarray,
    gtab,
    mask: np.ndarray | None = None,
    *,
    radial_order: int = RADIAL_ORDER,
    max_failure_fraction: float = 0.01,
    progress: bool = False,
    n_jobs: int = 1,
) -> tuple[np.ndarray, int]:
    """Unnormalised RTOP (``P_t``, in mm^-3) for every voxel inside *mask*.

    The MAP-MRI fit is scale-invariant -- dipy renormalises the coefficients so
    the propagator integrates to one, and GCV's optimum is itself scale-free --
    so the raw HCP signal is passed through without b0 normalisation.

    Voxels outside the mask stay at 0; voxels whose fit raises are set to NaN and
    counted. A handful of degenerate voxels (all-zero signal at a mask edge) must
    not kill a multi-hour cluster job, but a *systematically* failing fit must
    not pass silently either, so more than *max_failure_fraction* failures is an
    error rather than a warning.

    ``n_jobs > 1`` splits the voxels across processes. The fit is independent
    per voxel, so this changes nothing but wall clock: on margaret one subject
    is about 10 core-hours, which only becomes a sane cluster job when spread
    over the cores the allocation already has.

    Returns
    -------
    (rtop, n_failed)
    """
    data = np.asarray(data)
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    # Upcast the masked signals, never the whole 4-D volume: an HCP diffusion
    # series is 145 x 174 x 145 x 288, so a float64 copy of it is 8.4 GB
    # against 545 MB for the ~236k voxels actually fitted.
    signals = np.asarray(data[mask], dtype=np.float64)

    # MAP-MRI's design matrices are tiny (a few hundred coefficients), so BLAS
    # threading is pure overhead on them -- and actively harmful under joblib,
    # where every worker spawns its own pool. Measured on margaret: 29 ms/voxel
    # with one thread, 155 ms/voxel with BLAS left free, and a 16-worker run
    # that burned 128 core-hours of CPU on 1.9 core-hours of work. Pinning to
    # one thread per worker is what makes the cohort a feasible cluster job.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        if n_jobs == 1:
            iterator = [signals] if not progress else _progress_chunks(signals)
            results = [_fit_chunk(chunk, gtab, radial_order) for chunk in iterator]
        else:
            from joblib import Parallel, delayed, parallel_config

            chunks = [c for c in np.array_split(signals, n_jobs * 4) if len(c)]
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_fit_chunk)(chunk, gtab, radial_order) for chunk in chunks
                )

    values = np.concatenate([v for v, _ in results]) if results else np.empty(0)
    n_failed = sum(f for _, f in results)

    n_voxels = max(len(signals), 1)
    if n_failed / n_voxels > max_failure_fraction:
        raise RuntimeError(
            f"MAPL fit failed in {n_failed}/{n_voxels} voxels "
            f"({n_failed / n_voxels:.1%}), above the {max_failure_fraction:.1%} "
            "tolerance. Check the gradient table and the fit mask."
        )

    out = np.zeros(data.shape[:3], dtype=np.float32)
    out[mask] = values
    return out, n_failed
