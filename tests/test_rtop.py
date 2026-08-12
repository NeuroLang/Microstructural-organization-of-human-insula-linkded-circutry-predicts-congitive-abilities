"""Tests for the RTOP volume step.

The load-bearing test is :meth:`TestFreeWaterIdentity.test_normalized_rtop_is_one`.
The paper's normalisation is defined so that freely diffusing water has
``R_t = 1``; if the diffusion time, the b-value units, or dipy's ``tau``
convention were wrong anywhere in the chain, that identity would break. It
therefore pins the whole units story end to end -- gradient table, MAPL fit,
D_vent estimation, and normalisation -- with a value known in closed form.

Run: ``uv run pytest tests/test_rtop.py -v``
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from _synthetic_hcp import make_gradient_table, make_synthetic_hcp_tree

from insula_rtop.constants import DIFFUSION_TIME_S, VENTRICLE_LABELS
from insula_rtop.hcp.layout import resolve_subject_paths
from insula_rtop.imaging import resample_label_to_grid
from insula_rtop.rtop.mapl import build_gradient_table, fit_rtop, load_gradients
from insula_rtop.rtop.masks import build_fit_mask
from insula_rtop.rtop.normalize import (
    MAX_PLAUSIBLE_RTOP,
    flag_implausible,
    free_water_rtop,
    measured_ventricular_rtop,
    normalize_rtop,
)
from insula_rtop.rtop.ventricle import ventricle_mask, ventricular_diffusivity
from insula_rtop.surface.sample import sample_volume_on_surface

#: Free-water diffusivity at body temperature, mm^2/s. The value the ventricles
#: should return, and the one the paper's normalisation divides out.
D_FREE = 3.0e-3


def dense_gradient_table(n_per_shell: int = 60, seed: int = 0):
    """A gradient scheme dense enough for a well-posed MAP-MRI fit."""
    rng = np.random.default_rng(seed)
    bvals = [0.0] * 6
    bvecs = [[0.0, 0.0, 0.0]] * 6
    for shell in (1000.0, 2000.0, 3000.0):
        directions = rng.normal(size=(n_per_shell, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        bvals.extend([shell] * n_per_shell)
        bvecs.extend(directions.tolist())
    bvals = np.asarray(bvals)
    bvecs = np.asarray(bvecs)
    return bvals, bvecs, build_gradient_table(bvals, bvecs)


def isotropic_signal(bvals: np.ndarray, d: float, s0: float = 1.0) -> np.ndarray:
    """Mono-exponential isotropic attenuation: the free-diffusion signal."""
    return s0 * np.exp(-bvals * d)


# ---- analytic ground truth ------------------------------------------------


class TestFreeWaterIdentity:
    @pytest.mark.parametrize("d", [3.0e-3, 2.0e-3, 1.0e-3])
    def test_mapl_recovers_the_analytic_rtop(self, d):
        bvals, _, gtab = dense_gradient_table()
        signal = isotropic_signal(bvals, d)
        rtop, n_failed = fit_rtop(signal[None, None, None, :], gtab)
        assert n_failed == 0
        expected = (4 * np.pi * d * DIFFUSION_TIME_S) ** -1.5
        assert rtop[0, 0, 0] == pytest.approx(expected, rel=0.02)

    def test_normalized_rtop_is_one_in_free_water(self):
        bvals, _, gtab = dense_gradient_table()
        rtop, _ = fit_rtop(isotropic_signal(bvals, D_FREE)[None, None, None, :], gtab)
        normalized = normalize_rtop(rtop, free_water_rtop(D_FREE))
        assert normalized[0, 0, 0] == pytest.approx(1.0, rel=0.02)

    def test_restricted_tissue_has_rtop_above_one(self):
        """Slower diffusion than free water must enhance RTOP, not reduce it."""
        bvals, _, gtab = dense_gradient_table()
        rtop, _ = fit_rtop(isotropic_signal(bvals, 0.7e-3)[None, None, None, :], gtab)
        assert normalize_rtop(rtop, free_water_rtop(D_FREE))[0, 0, 0] > 1.0

    def test_signal_scale_does_not_change_rtop(self):
        """MAP-MRI renormalises its coefficients, so S0 must cancel exactly."""
        bvals, _, gtab = dense_gradient_table()
        a, _ = fit_rtop(isotropic_signal(bvals, D_FREE, 1.0)[None, None, None, :], gtab)
        b, _ = fit_rtop(
            isotropic_signal(bvals, D_FREE, 1500.0)[None, None, None, :], gtab
        )
        assert a[0, 0, 0] == pytest.approx(b[0, 0, 0], rel=1e-4)


class TestNormalization:
    def test_free_water_rtop_matches_closed_form(self):
        assert free_water_rtop(D_FREE) == pytest.approx(
            (4 * np.pi * D_FREE * DIFFUSION_TIME_S) ** -1.5
        )

    def test_normalization_is_a_pure_scaling(self):
        rtop = np.array([[[1.0e4, 2.0e4]]], dtype=np.float32)
        out = normalize_rtop(rtop, free_water_rtop(D_FREE))
        assert out[0, 0, 1] / out[0, 0, 0] == pytest.approx(2.0)

    def test_nonpositive_diffusivity_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            free_water_rtop(0.0)

    def test_nonpositive_divisor_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_rtop(np.ones((1, 1, 1)), 0.0)


class TestMeasuredVentricularNormalizer:
    """The paper's prose normaliser: divide by RTOP measured in the ventricles."""

    def test_ventricles_normalise_to_one_by_construction(self):
        rtop = np.zeros((1, 1, 4), np.float32)
        vent = np.zeros(rtop.shape, bool)
        vent[0, 0, :3] = True
        rtop[0, 0, :3] = [40000.0, 44000.0, 48000.0]
        rtop[0, 0, 3] = 400000.0  # tissue

        divisor, diag = measured_ventricular_rtop(rtop, vent)
        assert divisor == pytest.approx(44000.0)
        assert diag["VentricleVoxelsAveraged"] == 3
        out = normalize_rtop(rtop, divisor)
        assert out[0, 0, :3].mean() == pytest.approx(1.0, rel=1e-5)
        assert out[0, 0, 3] == pytest.approx(400000.0 / 44000.0)

    def test_a_single_pathological_voxel_cannot_set_the_divisor(self):
        """Without the guard, one 1e15 voxel would zero the whole subject."""
        rtop = np.full((1, 1, 200), 45000.0, np.float32)
        rtop[0, 0, 0] = 1e15
        vent = np.ones(rtop.shape, bool)
        divisor, diag = measured_ventricular_rtop(rtop, vent)
        assert divisor == pytest.approx(45000.0)
        assert diag["VentricleVoxelsRejected"] == 1

    def test_the_guard_is_inert_on_clean_data(self):
        """Measured on 40 real subjects: guarded vs plain mean differs <2%."""
        rng = np.random.default_rng(0)
        rtop = rng.normal(46000, 4000, size=(1, 1, 800)).astype(np.float32)
        vent = np.ones(rtop.shape, bool)
        divisor, diag = measured_ventricular_rtop(rtop, vent)
        assert divisor == pytest.approx(float(rtop.mean()), rel=1e-3)
        assert diag["VentricleVoxelsRejected"] == 0

    def test_nonfinite_and_negative_voxels_are_excluded(self):
        rtop = np.array([[[45000.0, np.nan, -3.0, 47000.0]]], np.float32)
        vent = np.ones(rtop.shape, bool)
        divisor, diag = measured_ventricular_rtop(rtop, vent)
        assert divisor == pytest.approx(46000.0)
        assert diag["VentricleVoxelsAveraged"] == 2

    def test_empty_ventricles_are_an_error(self):
        with pytest.raises(RuntimeError, match="nothing to divide by"):
            measured_ventricular_rtop(
                np.zeros((1, 1, 3), np.float32), np.ones((1, 1, 3), bool)
            )


# ---- fitting behaviour ----------------------------------------------------


class TestPlausibility:
    """MAPL is unconstrained, so ~0.1% of real voxels come back non-physical."""

    def _volume(self):
        values = np.array([[[10.0, -5.0, 1e15, np.nan, 0.0, 500.0]]], np.float32)
        mask = np.ones(values.shape, bool)
        mask[0, 0, 4] = False  # the exact 0 marks a voxel that was never fitted
        return values, mask

    def test_negative_and_huge_values_are_rejected(self):
        values, mask = self._volume()
        out, n = flag_implausible(values, mask)
        assert n == 3  # -5, 1e15, NaN
        assert np.isnan(out[0, 0, 1]) and np.isnan(out[0, 0, 2])
        assert out[0, 0, 0] == pytest.approx(10.0)
        assert out[0, 0, 5] == pytest.approx(500.0)  # below the bound, kept

    def test_unfitted_voxels_keep_their_zero(self):
        """0 must keep meaning 'never fitted', not become 'fitted and rejected'."""
        values, mask = self._volume()
        out, _ = flag_implausible(values, mask)
        assert out[0, 0, 4] == 0.0

    def test_the_input_is_not_modified(self):
        values, mask = self._volume()
        flag_implausible(values, mask)
        assert values[0, 0, 1] == pytest.approx(-5.0)

    def test_one_bad_voxel_would_otherwise_destroy_a_mean(self):
        """The reason this exists: the paper's IQR rule acts on a mean."""
        values = np.full((1, 1, 100), 10.0, np.float32)
        values[0, 0, 0] = 1e15
        mask = np.ones(values.shape, bool)
        assert values.mean() > 1e12
        out, _ = flag_implausible(values, mask)
        assert np.nanmean(out) == pytest.approx(10.0)

    def test_the_bound_is_far_above_real_tissue(self):
        """p99.9 of a real subject is ~97; the bound must not touch tissue."""
        assert MAX_PLAUSIBLE_RTOP > 10 * 97


class TestFitRtop:
    def test_only_masked_voxels_are_fitted(self):
        bvals, _, gtab = dense_gradient_table()
        data = np.tile(isotropic_signal(bvals, D_FREE), (2, 2, 2, 1))
        mask = np.zeros((2, 2, 2), bool)
        mask[0, 0, 0] = True
        rtop, _ = fit_rtop(data, gtab, mask)
        assert rtop[0, 0, 0] > 0
        assert rtop[mask == 0].sum() == 0

    def test_degenerate_voxels_are_counted_not_fatal(self):
        bvals, _, gtab = dense_gradient_table()
        data = np.zeros((1, 1, 2, bvals.size))
        data[0, 0, 0] = isotropic_signal(bvals, D_FREE)
        data[0, 0, 1] = np.nan
        rtop, n_failed = fit_rtop(data, gtab, max_failure_fraction=0.5)
        assert n_failed == 1
        assert np.isnan(rtop[0, 0, 1])
        assert rtop[0, 0, 0] > 0

    def test_systematic_failure_raises(self):
        bvals, _, gtab = dense_gradient_table()
        data = np.full((1, 1, 2, bvals.size), np.nan)
        with pytest.raises(RuntimeError, match="MAPL fit failed"):
            fit_rtop(data, gtab)

    def test_parallel_fit_matches_the_serial_one(self):
        """n_jobs is pure wall clock: the fit is independent per voxel."""
        bvals, _, gtab = dense_gradient_table()
        data = np.stack(
            [isotropic_signal(bvals, d) for d in (3.0e-3, 1.5e-3, 0.7e-3, 1.0e-3)]
        ).reshape(2, 2, 1, bvals.size)
        serial, _ = fit_rtop(data, gtab)
        parallel, _ = fit_rtop(data, gtab, n_jobs=2)
        np.testing.assert_allclose(serial, parallel, rtol=1e-6)


class TestGradientLoading:
    def test_bvec_is_transposed_to_dipy_convention(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", ["100001"])
        paths = resolve_subject_paths(root, "100001")
        bvals, bvecs = load_gradients(paths.bval, paths.bvec)
        assert bvecs.shape == (bvals.size, 3)

    def test_mismatched_lengths_are_rejected(self, tmp_path):
        bval = tmp_path / "x.bval"
        bvec = tmp_path / "x.bvec"
        np.savetxt(bval, np.zeros((1, 10)))
        np.savetxt(bvec, np.zeros((3, 7)))
        with pytest.raises(ValueError, match="incompatible"):
            load_gradients(bval, bvec)

    def test_gradient_table_carries_the_paper_timings(self):
        _, _, gtab = dense_gradient_table()
        assert gtab.big_delta - gtab.small_delta / 3 == pytest.approx(
            DIFFUSION_TIME_S
        )


# ---- ventricles and masks -------------------------------------------------


class TestVentricles:
    def _fixture(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", ["100001"])
        paths = resolve_subject_paths(root, "100001")
        return nib.load(str(paths.aparc_aseg)), nib.load(str(paths.dwi))

    def test_mask_lands_on_the_diffusion_grid(self, tmp_path):
        aseg, dwi = self._fixture(tmp_path)
        mask = ventricle_mask(aseg, dwi, erosion=0)
        assert mask.shape == dwi.shape[:3]
        assert mask.any()

    def test_erosion_never_empties_the_mask(self, tmp_path):
        aseg, dwi = self._fixture(tmp_path)
        # 4 anatomical voxels across becomes ~2 diffusion voxels; eroding by 3
        # would wipe it out, and the fallback must keep the uneroded mask.
        assert ventricle_mask(aseg, dwi, erosion=3).any()

    def test_only_ventricle_labels_are_selected(self, tmp_path):
        aseg, dwi = self._fixture(tmp_path)
        resampled = resample_label_to_grid(aseg, dwi)
        mask = ventricle_mask(aseg, dwi, erosion=0)
        assert set(np.unique(resampled[mask])) <= set(VENTRICLE_LABELS)

    def test_diffusivity_recovers_the_simulated_value(self, tmp_path):
        aseg, dwi = self._fixture(tmp_path)
        bvals, bvecs = make_gradient_table()
        gtab = build_gradient_table(bvals, bvecs.T)
        mask = ventricle_mask(aseg, dwi, erosion=0)
        data = np.zeros((*dwi.shape[:3], bvals.size))
        data[mask] = isotropic_signal(bvals, D_FREE)
        data[~mask] = isotropic_signal(bvals, 0.7e-3)

        d_vent, diagnostics = ventricular_diffusivity(data, gtab, aseg, dwi, erosion=0)
        assert d_vent == pytest.approx(D_FREE, rel=1e-3)
        assert diagnostics["VentricleVoxels"] == int(mask.sum())

    def test_noise_floor_at_high_b_does_not_bias_the_estimate(self, tmp_path):
        """The reason D_vent is fitted on b <= 1000 only.

        CSF at b = 2000 and b = 3000 carries no signal: the HCP ventricles
        measure a flat S/S0 ~ 0.044 in both shells where free water predicts
        0.0025 and 0.0001. Simulating exactly that -- a true free-water decay
        clamped at a noise floor -- a fit over all shells lands near half the
        real diffusivity, while the restricted fit recovers it.
        """
        aseg, dwi = self._fixture(tmp_path)
        bvals, bvecs = make_gradient_table()
        gtab = build_gradient_table(bvals, bvecs.T)
        mask = ventricle_mask(aseg, dwi, erosion=0)

        noise_floor = 0.044
        signal = np.maximum(isotropic_signal(bvals, D_FREE), noise_floor)
        data = np.zeros((*dwi.shape[:3], bvals.size))
        data[mask] = signal

        restricted, _ = ventricular_diffusivity(data, gtab, aseg, dwi, erosion=0)
        assert restricted == pytest.approx(D_FREE, rel=0.15)

        biased, _ = ventricular_diffusivity(
            data, gtab, aseg, dwi, erosion=0, max_bval=1e9
        )
        assert biased < 0.6 * D_FREE

    def test_too_few_ventricle_voxels_is_an_error(self, tmp_path):
        aseg, dwi = self._fixture(tmp_path)
        empty = nib.Nifti1Image(
            np.zeros(aseg.shape, np.int16), aseg.affine, aseg.header
        )
        bvals, bvecs = make_gradient_table()
        gtab = build_gradient_table(bvals, bvecs.T)
        data = np.ones((*dwi.shape[:3], bvals.size))
        with pytest.raises(RuntimeError, match="ventricle voxel"):
            ventricular_diffusivity(data, gtab, empty, dwi)


class TestFitMask:
    def _images(self, tmp_path):
        root = make_synthetic_hcp_tree(tmp_path / "hcp", ["100001"])
        p = resolve_subject_paths(root, "100001")
        return (
            nib.load(str(p.dwi)),
            nib.load(str(p.ribbon)),
            nib.load(str(p.aparc_aseg)),
            nib.load(str(p.dwi_mask)),
            list(p.midthickness_native.values()),
        )

    def test_ribbon_plus_ventricles_covers_both(self, tmp_path):
        dwi, ribbon, aseg, brain, _ = self._images(tmp_path)
        mask = build_fit_mask(
            dwi,
            ribbon=ribbon,
            aparc_aseg=aseg,
            brain_mask=brain,
            strategy="ribbon+ventricles",
        )
        assert mask.shape == dwi.shape[:3]
        assert mask[ventricle_mask(aseg, dwi)].all()
        assert mask.sum() < mask.size  # strictly cheaper than a whole-brain fit

    def test_surface_strategy_covers_the_interpolation_stencil(self, tmp_path):
        """Sampling inside the mask must reproduce sampling on the full volume."""
        dwi, ribbon, aseg, brain, surfaces = self._images(tmp_path)
        mask = build_fit_mask(
            dwi, ribbon=ribbon, aparc_aseg=aseg, brain_mask=brain, surfaces=surfaces
        )
        assert mask[ventricle_mask(aseg, dwi)].all()

        # A linear field sampled through the mask must match the exact answer at
        # every vertex: if the stencil were incomplete some vertex would read a
        # zero from an unfitted voxel.
        idx = np.indices(dwi.shape[:3]).astype(float)
        field = np.where(mask, idx[1], 0.0)
        full = idx[1]
        for surface in surfaces:
            image = nib.Nifti1Image(field, dwi.affine)
            reference = nib.Nifti1Image(full, dwi.affine)
            np.testing.assert_allclose(
                sample_volume_on_surface(image, surface),
                sample_volume_on_surface(reference, surface),
                atol=1e-6,
            )

    def test_surface_strategy_is_smaller_than_the_ribbon(self, tmp_path):
        dwi, ribbon, aseg, brain, surfaces = self._images(tmp_path)
        common = {"ribbon": ribbon, "aparc_aseg": aseg, "brain_mask": brain}
        stencil = build_fit_mask(dwi, surfaces=surfaces, **common)
        ribbon_mask_ = build_fit_mask(dwi, strategy="ribbon+ventricles", **common)
        assert stencil.sum() <= ribbon_mask_.sum()

    def test_surface_strategy_needs_surfaces(self, tmp_path):
        dwi, ribbon, aseg, _, _ = self._images(tmp_path)
        with pytest.raises(ValueError, match="needs the subject's"):
            build_fit_mask(dwi, ribbon=ribbon, aparc_aseg=aseg)

    def test_brain_strategy_uses_the_brain_mask(self, tmp_path):
        dwi, ribbon, aseg, brain, _ = self._images(tmp_path)
        mask = build_fit_mask(
            dwi, ribbon=ribbon, aparc_aseg=aseg, brain_mask=brain, strategy="brain"
        )
        np.testing.assert_array_equal(mask, np.asarray(brain.dataobj) > 0)

    def test_unknown_strategy_is_rejected(self, tmp_path):
        dwi, ribbon, aseg, _, _ = self._images(tmp_path)
        with pytest.raises(ValueError, match="Unknown fit-mask strategy"):
            build_fit_mask(dwi, ribbon=ribbon, aparc_aseg=aseg, strategy="cortex")
