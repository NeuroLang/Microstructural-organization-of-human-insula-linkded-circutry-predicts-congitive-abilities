"""Panel geometry, on a synthetic mesh so no group surfaces are needed.

The two things tested here are the ones that fail *silently*: an overlay drawn
in the wrong coordinate frame still renders, and an arrow reconstructed in the
wrong patch frame still points somewhere plausible.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from insula_rtop.figures import panels  # noqa: E402

#: A flat 11x11 sheet in the x = 40 plane, offset far from the origin so a
#: missing re-centring shows up as a large error rather than rounding noise.
CENTROID = np.array([40.0, 100.0, 60.0])


def _sheet():
    u, v = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))
    coords = np.column_stack([np.zeros(u.size), u.ravel(), v.ravel()]) + CENTROID
    return coords, np.zeros((1, 3), dtype=int)


@pytest.fixture
def sheet(monkeypatch):
    coords, faces = _sheet()
    panels.render_coords.cache_clear()
    monkeypatch.setattr(panels, "inflated_surface", lambda hemi: (coords, faces))
    monkeypatch.setattr(panels, "group_midthickness", lambda hemi: (coords, faces))
    yield coords
    panels.render_coords.cache_clear()


def test_render_coords_matches_the_frame_nilearn_draws_in(sheet):
    """plot_surf re-centres the mesh; overlays must follow or they land off it."""
    centred = panels.render_coords("L")
    assert np.allclose(centred.mean(axis=0), 0.0)
    assert np.allclose(centred, sheet - sheet.mean(axis=0))


def test_zoom_crops_around_the_patch_in_the_rendered_frame(sheet):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    patch = np.flatnonzero(sheet[:, 1] > sheet[:, 1].mean())
    panels.zoom_to_patch(ax, "L", patch, pad=2.0)

    box = panels.render_coords("L")[patch]
    assert ax.get_ylim() == pytest.approx(
        (box[:, 1].min() - 2.0, box[:, 1].max() + 2.0)
    )
    # The un-centred coordinates would put this window a hundred millimetres
    # away, so the patch would not be inside it at all.
    assert abs(np.mean(ax.get_ylim())) < 20.0
    plt.close(fig)


class TestGradientArrows:
    """The group mean direction is a world-space vector, so it is drawn as-is.

    Nothing here reconstructs it from an angle: the analysis never leaves world
    space, and a figure that re-derived the direction from a frame of its own
    could disagree with the polar plots without anyone noticing.
    """

    def _draw(self, sheet, vector):
        labels = np.where(sheet[:, 1] > sheet[:, 1].mean(), 1, 0)
        summary = pd.DataFrame(
            [
                {
                    "hemi": "L",
                    "subdivision": "PI",
                    "mean_vx": vector[0],
                    "mean_vy": vector[1],
                    "mean_vz": vector[2],
                }
            ]
        )
        drawn = []
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.quiver = lambda *a, **k: drawn.append(np.asarray(a, dtype=float))
        panels.gradient_arrows_on_surface(
            ax, "L", labels, {1: "PI"}, summary, length=4.0, lift=1.0
        )
        plt.close(fig)
        return (drawn[0][:3], drawn[0][3:]) if drawn else (None, None)

    @pytest.mark.parametrize(
        "vector", [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, -0.6, 0.8)]
    )
    def test_a_direction_lying_in_the_patch_is_drawn_unchanged(self, sheet, vector):
        """The sheet spans y-z, so these survive the projection intact."""
        _, drawn = self._draw(sheet, vector)
        assert np.allclose(drawn / np.linalg.norm(drawn), vector, atol=1e-9)

    def test_a_direction_out_of_the_patch_draws_a_shorter_arrow(self, sheet):
        """The out-of-plane part cannot be shown, so it shortens the arrow."""
        _, flat = self._draw(sheet, (0.0, 1.0, 0.0))
        _, tilted = self._draw(sheet, (0.8, 0.6, 0.0))
        assert np.linalg.norm(tilted) < 0.7 * np.linalg.norm(flat)
        # What is drawn is the in-plane part, so it still points anteriorly.
        assert np.allclose(tilted / np.linalg.norm(tilted), (0.0, 1.0, 0.0), atol=1e-9)

    def test_the_arrow_is_centred_on_a_vertex_of_its_own_subdivision(self, sheet):
        start, vector = self._draw(sheet, (0.0, 1.0, 0.0))
        middle = start + vector / 2
        centred = panels.render_coords("L")
        labels = np.where(sheet[:, 1] > sheet[:, 1].mean(), 1, 0)
        own = centred[labels == 1]
        # Within the lift plus a vertex spacing of the nearest own-subdivision
        # vertex, and nowhere near the un-centred coordinates.
        assert np.min(np.linalg.norm(own - middle, axis=1)) < 3.0

    def test_a_subdivision_with_no_direction_draws_nothing(self, sheet):
        start, _ = self._draw(sheet, (np.nan, np.nan, np.nan))
        assert start is None
