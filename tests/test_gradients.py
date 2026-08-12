"""Tests for the gradient and circular-statistics machinery.

Every piece here has a closed-form answer:

* the linear-FEM gradient is *exact* for affine fields, so planting a known
  linear ramp on a triangulated patch must recover its direction and magnitude
  to machine precision -- including on a tilted patch, which is what catches a
  frame that silently assumes the surface is axis-aligned;
* the circular statistics are checked against ``scipy.stats.circmean`` and
  against a Rayleigh test of a known von Mises sample.

Run: ``uv run pytest tests/test_gradients.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic_hcp import make_grid_surface
from scipy import stats

from insula_rtop.analysis.gradients import (
    circular_dispersion,
    circular_mean,
    confidence_interval,
    face_gradients,
    group_statistics,
    patch_faces,
    patch_frame,
    rayleigh_test,
    resultant_length,
    smooth_on_surface,
    spherical_rayleigh,
    subject_directions,
    summarize,
    vertex_gradients,
    von_mises_kappa,
)


def yz_patch(n_side: int = 12, extent: float = 20.0, aspect: float = 2.0):
    """A flat patch in the y-z plane: an idealised insula, x constant.

    Stretched along y by *aspect* so that PC1 is unambiguously the
    anterior-posterior axis. On a square patch the two in-plane principal
    directions are degenerate and PCA returns an arbitrary rotation of them,
    which is a property of the test fixture, not of the code under test.
    """
    coords, faces = make_grid_surface(n_side=n_side, extent=extent)
    # make_grid_surface builds the patch in x-y; rotate it into y-z so that the
    # anterior (+y) and dorsal (+z) axes are the ones that span it.
    coords = np.column_stack([coords[:, 2], coords[:, 0] * aspect, coords[:, 1]])
    return coords, faces


# ---- the FEM gradient -----------------------------------------------------


class TestFaceGradients:
    def test_linear_field_is_recovered_exactly(self):
        coords, faces = yz_patch()
        direction = np.array([0.0, 0.3, -0.7])
        values = coords @ direction
        grad, areas = face_gradients(coords, faces, values)
        # The patch has no x extent, so only the in-plane part is recoverable --
        # which is all of it here, since the ramp has no x component either.
        expected = np.tile([0.0, 0.3, -0.7], (len(faces), 1))
        np.testing.assert_allclose(grad, expected, atol=1e-9)
        assert np.all(areas > 0)

    def test_constant_field_has_zero_gradient(self):
        coords, faces = yz_patch()
        grad, _ = face_gradients(coords, faces, np.full(len(coords), 4.2))
        np.testing.assert_allclose(grad, 0.0, atol=1e-9)

    def test_areas_sum_to_the_patch_area(self):
        coords, faces = yz_patch(n_side=5, extent=10.0, aspect=2.0)
        _, areas = face_gradients(coords, faces, np.zeros(len(coords)))
        assert areas.sum() == pytest.approx(40.0 * 20.0)

    def test_degenerate_triangles_become_nan(self):
        coords = np.array([[0.0, 0, 0], [0, 1, 0], [0, 2, 0]])  # collinear
        faces = np.array([[0, 1, 2]])
        grad, _ = face_gradients(coords, faces, np.array([0.0, 1.0, 2.0]))
        assert np.isnan(grad).all()


class TestPatchFrame:
    def test_axes_are_orthonormal_and_oriented(self):
        coords, _ = yz_patch()
        _, axes = patch_frame(coords, np.arange(len(coords)))
        np.testing.assert_allclose(axes @ axes.T, np.eye(2), atol=1e-9)
        assert axes[0, 1] > 0  # PC1 points anteriorly
        assert axes[1, 2] > 0  # PC2 points dorsally

    def test_frame_follows_a_tilted_patch(self):
        """The axes must lie in the patch, not in the world's y-z plane."""
        coords, _ = yz_patch()
        angle = np.deg2rad(30)
        rotation = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        tilted = coords @ rotation.T
        _, axes = patch_frame(tilted, np.arange(len(tilted)))
        normal = np.cross(axes[0], axes[1])
        assert abs(normal @ rotation @ np.array([1.0, 0, 0])) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_faces_outside_the_patch_are_dropped(self):
        _, faces = yz_patch(n_side=5)
        subset = np.arange(10)
        kept = patch_faces(faces, subset)
        assert len(kept) < len(faces)
        assert np.isin(kept, subset).all()


class TestAngleRecovery:
    @pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 150.0, -60.0])
    def test_planted_direction_is_recovered(self, degrees):
        """A ramp at a known angle in the patch frame comes back at that angle."""
        coords, faces = yz_patch()
        patch = np.arange(len(coords))
        _, axes = patch_frame(coords, patch)

        theta = np.deg2rad(degrees)
        direction = np.cos(theta) * axes[0] + np.sin(theta) * axes[1]
        values = coords @ direction

        grad, areas = face_gradients(coords, faces, values)
        planar = grad @ axes.T
        angles = np.arctan2(planar[:, 1], planar[:, 0])
        recovered = circular_mean(angles, areas * np.linalg.norm(planar, axis=1))
        assert recovered == pytest.approx(theta, abs=1e-6)

    def test_subject_directions_are_per_subdivision(self):
        """Two subdivisions with opposite ramps must come back opposed."""
        coords, faces = yz_patch(n_side=13, extent=24.0)

        labels = np.zeros(len(coords), dtype=int)
        labels[coords[:, 1] < 0] = 1
        labels[coords[:, 1] > 0] = 2
        # Ramps along +y, which is anterior: angle 0 and angle pi.
        anterior = np.array([0.0, 1.0, 0.0])
        values = np.where(labels == 1, coords @ anterior, -(coords @ anterior))

        angles, vectors = subject_directions(
            values, coords, faces, labels, {1: "front", 2: "back"}
        )
        assert angles["front"] == pytest.approx(0.0, abs=1e-2)
        assert abs(angles["back"]) == pytest.approx(np.pi, abs=1e-2)
        # The 3-D vectors must be opposed too, and be unit length.
        assert vectors["front"] @ vectors["back"] == pytest.approx(-1.0, abs=1e-6)
        assert np.linalg.norm(vectors["front"]) == pytest.approx(1.0)

    def test_a_subdivision_of_one_vertex_still_has_a_direction(self):
        """Faces come from the whole patch, so a boundary vertex keeps a stencil.

        Under the previous per-subdivision-triangle rule a one-vertex
        subdivision spanned no triangle and was dropped; a vertex gradient is
        the mean of the triangles meeting at it, which exist.
        """
        coords, faces = yz_patch(n_side=6)
        labels = np.zeros(len(coords), dtype=int)
        labels[:20] = 1
        labels[20] = 2
        values = coords @ np.array([0, 1.0, 0])
        angles, vectors = subject_directions(
            values, coords, faces, labels, {1: "a", 2: "b"}
        )
        # The one-vertex subdivision gets the same direction as its neighbours,
        # which is the planted ramp -- checked in world space, where the answer
        # does not depend on how the patch's own plane happens to be oriented.
        assert np.allclose(vectors["b"], [0.0, 1.0, 0.0], atol=1e-6)
        assert angles["b"] == pytest.approx(angles["a"], abs=1e-6)

    def test_a_subdivision_with_no_vertices_is_nan(self):
        coords, faces = yz_patch(n_side=6)
        labels = np.zeros(len(coords), dtype=int)
        labels[:20] = 1
        values = coords @ np.array([0, 1.0, 0])
        angles, vectors = subject_directions(
            values, coords, faces, labels, {1: "a", 2: "absent"}
        )
        assert np.isfinite(angles["a"])
        assert np.isnan(angles["absent"])
        assert np.isnan(vectors["absent"]).all()

    def test_tiny_patch_is_rejected(self):
        coords, faces = yz_patch(n_side=6)
        labels = np.zeros(len(coords), dtype=int)
        labels[0] = 1
        with pytest.raises(ValueError, match="fewer than 3 vertices"):
            subject_directions(np.zeros(len(coords)), coords, faces, labels, {1: "a"})


class TestSurfaceSmoothing:
    """Evidence suggests the original analysis smoothed; this is the knob."""

    def test_zero_iterations_is_the_identity(self):
        coords, faces = yz_patch()
        values = coords @ np.array([0.0, 0.3, -0.7])
        np.testing.assert_array_equal(smooth_on_surface(values, faces, 0), values)

    def test_smoothing_does_not_rotate_a_linear_gradient(self):
        """A plane stays a plane: the direction must survive, only noise dies.

        Only the direction is asserted, not the vector. Umbrella smoothing on a
        triangulated grid has an asymmetric neighbourhood, so it translates the
        field slightly at the boundary; that changes magnitudes near the edge
        but must not turn the gradient.
        """
        coords, faces = yz_patch(n_side=20, extent=20.0)
        direction = np.array([0.0, 0.3, -0.7])
        values = coords @ direction
        smoothed = smooth_on_surface(values, faces, 4)

        interior = np.flatnonzero(
            (np.abs(coords[:, 1]) < 24) & (np.abs(coords[:, 2]) < 12)
        )
        keep = patch_faces(faces, interior)
        after, _ = face_gradients(coords, keep, smoothed)
        unit = after / np.linalg.norm(after, axis=1, keepdims=True)
        expected = direction / np.linalg.norm(direction)
        assert np.abs(unit @ expected).min() > 0.999

    def test_smoothing_suppresses_noise(self):
        """Which is the point: it is the noise that inflates |grad|."""
        coords, faces = yz_patch(n_side=14, extent=20.0)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1.0, size=len(coords))
        rough, _ = face_gradients(coords, faces, noise)
        smooth, _ = face_gradients(coords, faces, smooth_on_surface(noise, faces, 20))
        assert np.nanmean(np.linalg.norm(smooth, axis=1)) < 0.2 * np.nanmean(
            np.linalg.norm(rough, axis=1)
        )

    def test_nan_vertices_stay_nan(self):
        coords, faces = yz_patch(n_side=6)
        values = np.zeros(len(coords))
        values[0] = np.nan
        out = smooth_on_surface(values, faces, 5)
        assert np.isnan(out[0])
        assert np.isfinite(out[1:]).all()


# ---- circular statistics --------------------------------------------------


class TestCircularStatistics:
    def test_mean_matches_scipy(self):
        rng = np.random.default_rng(0)
        angles = rng.vonmises(0.7, 4.0, size=500)
        assert circular_mean(angles) == pytest.approx(
            stats.circmean(angles, high=np.pi, low=-np.pi), abs=1e-9
        )

    def test_mean_wraps_across_pi(self):
        """The whole point of a circular mean: 170 deg and -170 deg average to 180."""
        angles = np.deg2rad([170.0, -170.0])
        assert abs(circular_mean(angles)) == pytest.approx(np.pi, abs=1e-9)

    def test_weights_shift_the_mean(self):
        angles = np.array([0.0, np.pi / 2])
        assert circular_mean(angles, [1.0, 1.0]) == pytest.approx(np.pi / 4)
        assert circular_mean(angles, [3.0, 1.0]) < np.pi / 4

    def test_resultant_length_spans_the_unit_interval(self):
        assert resultant_length(np.zeros(10)) == pytest.approx(1.0)
        assert resultant_length(np.linspace(-np.pi, np.pi, 361)[:-1]) < 1e-10

    def test_concentrated_sample_is_significant(self):
        rng = np.random.default_rng(1)
        angles = rng.vonmises(0.3, 5.0, size=200)
        z, p, rbar = rayleigh_test(angles)
        assert rbar > 0.8
        assert z == pytest.approx(200 * rbar**2)
        assert p < 1e-10

    def test_uniform_sample_is_not_significant(self):
        rng = np.random.default_rng(2)
        _, p, _ = rayleigh_test(rng.uniform(-np.pi, np.pi, size=300))
        assert p > 0.05

    def test_rayleigh_p_is_calibrated_against_the_null(self):
        """The reported p must be the actual tail probability under uniformity."""
        rng = np.random.default_rng(3)
        n = 50
        null_z = np.array(
            [rayleigh_test(rng.uniform(-np.pi, np.pi, size=n))[0] for _ in range(4000)]
        )
        for target in (0.20, 0.05, 0.01):
            critical = np.quantile(null_z, 1 - target)
            # Invert: a sample whose z sits at that quantile should be reported
            # with a p-value of about `target`.
            rbar = np.sqrt(critical / n)
            reported = np.exp(
                np.sqrt(1 + 4 * n + 4 * (n**2 - (n * rbar) ** 2)) - (1 + 2 * n)
            )
            assert reported == pytest.approx(target, rel=0.15)

    def test_kappa_grows_with_concentration(self):
        assert von_mises_kappa(0.1) < von_mises_kappa(0.6) < von_mises_kappa(0.95)
        assert von_mises_kappa(0.0) == 0.0

    def test_kappa_recovers_the_simulated_concentration(self):
        rng = np.random.default_rng(4)
        for kappa in (1.0, 4.0, 10.0):
            angles = rng.vonmises(0.0, kappa, size=20000)
            assert von_mises_kappa(resultant_length(angles)) == pytest.approx(
                kappa, rel=0.1
            )

    def test_confidence_interval_shrinks_with_n(self):
        rng = np.random.default_rng(5)
        wide = confidence_interval(rng.vonmises(0.0, 4.0, size=30))
        narrow = confidence_interval(rng.vonmises(0.0, 4.0, size=3000))
        assert narrow < wide

    def test_dispersion_is_zero_for_identical_angles(self):
        assert circular_dispersion(resultant_length(np.zeros(20))) == pytest.approx(
            0.0, abs=1e-6
        )


class TestSphericalRayleigh:
    """The paper's test: S = 3 n rbar^2 against chi^2 with 3 df."""

    def test_uniform_directions_have_a_near_zero_resultant(self):
        """A single draw's p-value is Uniform[0, 1] under the null, so assert
        the resultant length -- calibration is covered below, over many draws."""
        rng = np.random.default_rng(0)
        v = rng.normal(size=(500, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        _, _, rbar = spherical_rayleigh(v)
        assert rbar < 0.15

    def test_the_false_positive_rate_is_five_percent(self):
        rng = np.random.default_rng(7)
        hits = 0
        for _ in range(600):
            v = rng.normal(size=(60, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            hits += spherical_rayleigh(v)[1] < 0.05
        assert 0.02 < hits / 600 < 0.09

    def test_concentrated_directions_are_significant(self):
        rng = np.random.default_rng(1)
        v = np.array([0.0, 0.0, 1.0]) + 0.3 * rng.normal(size=(400, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        s, p, rbar = spherical_rayleigh(v)
        assert rbar > 0.9
        assert s == pytest.approx(3 * 400 * rbar**2)
        assert p < 1e-20

    def test_statistic_is_calibrated_against_its_null(self):
        """The 5% point of the null must sit at chi2(3)'s 5% point."""
        rng = np.random.default_rng(2)
        null = []
        for _ in range(3000):
            v = rng.normal(size=(40, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            null.append(spherical_rayleigh(v)[0])
        assert np.quantile(null, 0.95) == pytest.approx(
            stats.chi2.ppf(0.95, df=3), rel=0.1
        )

    def test_statistic_may_exceed_n(self):
        """Which is why the published 724 at N = 413 cannot be a planar test."""
        v = np.tile([0.0, 0.0, 1.0], (413, 1))
        s, _, _ = spherical_rayleigh(v)
        assert s == pytest.approx(3 * 413)
        assert s > 413

    def test_empty_input_is_nan(self):
        s, p, rbar = spherical_rayleigh(np.zeros((0, 3)))
        assert np.isnan(s) and np.isnan(p) and np.isnan(rbar)


class TestGroupSummary:
    def _directions(self, mean_angle: float, n: int = 120):
        import pandas as pd

        rng = np.random.default_rng(7)
        rows = []
        for hemi in ("L", "R"):
            for subdivision in ("vAI", "dAI", "PI"):
                for i, angle in enumerate(rng.vonmises(mean_angle, 6.0, size=n)):
                    rows.append(
                        {
                            "subject": f"{i:04d}",
                            "hemi": hemi,
                            "subdivision": subdivision,
                            "angle": angle,
                            "vx": np.cos(angle),
                            "vy": np.sin(angle),
                            "vz": 0.0,
                        }
                    )
        return pd.DataFrame(rows)

    def test_summary_covers_every_cell_plus_a_pooled_row(self):
        summary = summarize(self._directions(0.4))
        assert len(summary) == 2 * 3 + 2
        assert set(summary["subdivision"]) == {"vAI", "dAI", "PI", "pooled"}

    def test_pooled_row_uses_every_direction(self):
        summary = summarize(self._directions(0.4))
        pooled = summary[summary["subdivision"] == "pooled"]
        assert (pooled["n"] == 360).all()

    def test_recovers_the_simulated_mean_direction(self):
        summary = summarize(self._directions(0.4))
        assert summary["mean_direction_rad"].to_numpy() == pytest.approx(0.4, abs=0.1)
        assert (summary["rayleigh_p"] < 1e-10).all()

    def test_empty_input_is_all_nan(self):
        result = group_statistics(np.array([np.nan, np.nan]))
        assert result["n"] == 0
        assert np.isnan(result["rayleigh_p"])

    def test_summary_carries_the_spherical_test(self):
        summary = summarize(self._directions(0.4))
        assert (summary["spherical_p"] < 1e-10).all()
        # Planted in a plane, so the spherical and planar rbar must agree.
        assert summary["spherical_rbar"].to_numpy() == pytest.approx(
            summary["resultant_length"].to_numpy(), abs=1e-9
        )


class TestVertexGradientsAreTangential:
    """A gradient with respect to the surface cannot point out of it."""

    @staticmethod
    def _cylinder(n_around=24, n_along=9, radius=20.0, length=40.0):
        """A curved strip, so face normals genuinely differ across the patch."""
        theta = np.linspace(-0.9, 0.9, n_around)
        x = np.linspace(-length / 2, length / 2, n_along)
        tt, xx = np.meshgrid(theta, x, indexing="ij")
        coords = np.column_stack(
            [xx.ravel(), radius * np.sin(tt).ravel(), radius * np.cos(tt).ravel()]
        )
        faces = []
        for i in range(n_around - 1):
            for j in range(n_along - 1):
                a = i * n_along + j
                faces += [
                    [a, a + 1, a + n_along],
                    [a + 1, a + n_along + 1, a + n_along],
                ]
        return coords, np.array(faces)

    def test_the_mean_has_no_normal_component(self):
        coords, faces = self._cylinder()
        patch = np.arange(len(coords))
        values = coords @ np.array([1.0, 0.0, 0.0])
        grad, _ = vertex_gradients(coords, faces, values, patch)

        radial = coords / np.linalg.norm(coords[:, 1:], axis=1)[:, None]
        radial[:, 0] = 0.0
        finite = np.isfinite(grad).all(axis=1)
        normal_part = np.abs((grad[finite] * radial[finite]).sum(axis=1))
        assert normal_part.max() < 1e-8

    def test_a_tangential_ramp_is_still_recovered_exactly(self):
        """Removing the normal component must not disturb the in-plane part."""
        coords, faces = self._cylinder()
        patch = np.arange(len(coords))
        along = np.array([1.0, 0.0, 0.0])
        grad, _ = vertex_gradients(coords, faces, values := coords @ along, patch)
        del values
        finite = np.isfinite(grad).all(axis=1)
        np.testing.assert_allclose(
            grad[finite], np.tile(along, (finite.sum(), 1)), atol=1e-8
        )


class TestReportingFrame:
    """The plane angles are reported in is a choice, and it must be explicit.

    `anatomical` is what reproduces the article's published mean directions;
    `patch` rotates every angle by the tilt of the insular sheet, about 20
    degrees. Neither touches the 3-D directions, so no Rayleigh statistic moves.
    """

    def _run(self, frame):
        coords, faces = yz_patch(n_side=13, extent=24.0)
        # Rotate the patch within its own plane, so its PCA axes are turned 25
        # degrees from the anatomical ones while the surface itself is not.
        angle = np.deg2rad(25.0)
        rot = np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
        coords = coords @ rot.T
        labels = np.ones(len(coords), dtype=int)
        values = coords @ np.array([0.0, 1.0, 0.0])
        return subject_directions(
            values, coords, faces, labels, {1: "a"}, frame=frame
        )

    def test_anatomical_reports_the_anatomical_angle(self):
        angles, vectors = self._run("anatomical")
        # The field increases along +y, which is angle 0 anatomically, and the
        # patch lies in the y-z plane so the surface gradient is exactly +y.
        assert angles["a"] == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(vectors["a"], [0.0, 1.0, 0.0], atol=1e-6)

    def test_the_patch_frame_rotates_the_angle_by_the_patch_tilt(self):
        angles, vectors = self._run("patch")
        assert abs(angles["a"]) == pytest.approx(np.deg2rad(25.0), abs=1e-3)

    def test_the_frame_never_changes_the_3d_direction(self):
        _, anatomical = self._run("anatomical")
        _, patch = self._run("patch")
        np.testing.assert_allclose(anatomical["a"], patch["a"], atol=1e-12)

    def test_an_unknown_frame_is_rejected(self):
        with pytest.raises(ValueError, match="anatomical"):
            self._run("insula")
