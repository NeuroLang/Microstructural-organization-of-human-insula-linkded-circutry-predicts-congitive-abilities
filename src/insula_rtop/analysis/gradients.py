"""RTOP gradients on the insular surface, and their circular statistics (Fig 3).

    We conducted a detailed profile analysis and used directional statistics to
    characterize gradients in RTOP along its anterior-posterior and
    ventral-dorsal axes. [...] We then assessed the significance of this finding
    using the Rayleigh test, against the null hypothesis of a uniform
    directional distribution. (Results / Materials and methods)

and, per participant:

    (1) Computed RTOP gradient field along insular surface of each participant
    (2) Extracted main gradient directions from each participant
    (3) Computed mean direction and 95% confidence interval across participants
    (4) Applied Rayleigh test

**This is a reconstruction.** The paper states the pipeline but not how a
gradient on a folded 3-D surface becomes a single direction. The procedure is:

1. **A gradient per triangle.** The standard linear-FEM gradient of a scalar
   defined at vertices: constant within each triangle, exact for any function
   that is linear across it. Computed over the whole insular patch, so a vertex
   on a subdivision boundary still sees a complete stencil.
2. **A gradient per vertex** -- the area-weighted mean of the gradients of the
   triangles meeting at it. This is the quantity the analysis is built on: the
   gradient of RTOP with respect to the mesh surface, sampled where the data
   itself lives.
3. **One direction per participant and subdivision**: the weighted mean of
   those per-vertex gradients as 3-D unit vectors, weighted by vertex area
   times gradient magnitude, so well-sampled vertices with strong gradients
   dominate. It is a direction in world space; **no plane is fitted and no
   projection is taken** to define it.
4. **Group statistics.** The Rayleigh test on the sphere (below), and the mean
   direction, resultant length, von Mises concentration and 95% CI.

**Angles are a projection for reporting, not the analysis.** Figure 3C's polar
plots and the paper's dispersions are in radians, which needs a plane. Two are
available and ``frame=`` selects between them:

``anatomical`` (the default)
    The sagittal plane: 0 is increasing anteriorly (+y), pi/2 increasing
    dorsally (+z). These are the axes the article's own polar plots are
    labelled with (``A``/``P``, ``D``/``V``), and together with the study's own
    parcellation they reproduce its published mean directions -- left PI at
    +0.621 pi against +0.61 pi. The HCP surfaces are ACPC-aligned, so the axes
    mean the same thing in every subject.

``patch``
    The plane of the subject's own insular patch, from PCA of its vertex
    coordinates. The insula is a sheet tilted about 20 degrees from the
    anatomical axes, so this rotates every angle by roughly that much.

The projection is applied *after* the direction is determined and changes
nothing about it: the 3-D directions, and therefore the spherical Rayleigh
test, are identical either way.

**The paper's test is spherical, not planar.** Its "Rayleigh statistic = 724
[...] N = 413, df = 3" cannot be a circular test: a planar Rayleigh statistic
cannot exceed *n*, and a circular test has 1 or 2 degrees of freedom, not 3.
``df = 3`` is the signature of the Rayleigh test of uniformity *on the sphere*
(Mardia & Jupp, *Directional Statistics*, sec. 10.4.1),

    S = 3 n rbar^2,   S ~ chi^2 with 3 df under uniformity,

applied to the gradient directions as 3-D unit vectors. Computing it that way
puts both hemispheres in the published range, where projecting into the 2-D
frame first does not -- the projection discards the out-of-plane component, and
on the folded insula that is a large part of the direction.

So both are reported. :func:`spherical_rayleigh` is the paper's test, computed
on the 3-D directions and the one to compare against 724 / 270; the circular
statistics describe the same directions seen in the anatomical plane, which is
what Figure 3C's polar plots and the quoted dispersion of 0.36 pi show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Reference axes used to sign the patch frame: +y anterior, +z dorsal, in the
#: RAS convention the HCP surfaces use.
ANTERIOR = np.array([0.0, 1.0, 0.0])
DORSAL = np.array([0.0, 0.0, 1.0])


#: Iterations of neighbour averaging applied to the surface RTOP before the
#: gradient is taken. Zero -- no smoothing -- is the default and is what the
#: Methods describe.
#:
#: Supplementary Table 1 reports a *within-subject* 95% spread of gradient
#: directions of 0.21-0.25 pi. Per-vertex gradients here give 1.83-1.88 pi
#: unsmoothed and 1.20-1.49 pi at 20 iterations, so smoothing does not explain
#: the difference: an individual vertex gradient in an unsmoothed field is
#: mostly noise, and only the average over a subdivision is well determined.
#:
#: A spread that tight is reachable when the directions are projected into the
#: insula's own principal plane, which conditions the angle strongly -- which
#: makes Supplementary Table 1 evidence about the original analysis rather than
#: about smoothing. See README, "Was the surface RTOP smoothed?".
DEFAULT_SMOOTHING_ITERATIONS = 0

#: Plane that 3-D directions are projected into to be reported as angles.
#: ``anatomical`` reproduces the article's published mean directions; see the
#: module docstring and README, "Mean directions".
DEFAULT_FRAME = "anatomical"


def smooth_on_surface(
    values: np.ndarray, faces: np.ndarray, iterations: int
) -> np.ndarray:
    """Average each vertex with its mesh neighbours, *iterations* times.

    Umbrella (uniform Laplacian) smoothing: cheap, geometry-free, and enough to
    test how much of a directional statistic is carried by the coherent trend
    rather than by vertex noise.
    """
    if iterations <= 0:
        return values
    from scipy import sparse

    n = len(values)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency = adjacency + sparse.eye(n)  # keep the vertex's own value
    operator = (
        sparse.diags(1.0 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    )

    out = np.asarray(values, dtype=float)
    finite = np.isfinite(out)
    out = np.where(finite, out, np.nanmean(out) if finite.any() else 0.0)
    for _ in range(iterations):
        out = operator @ out
    return np.where(finite, out, np.nan)


# ---- geometry -------------------------------------------------------------


def patch_frame(coords: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D frame spanning the surface patch: ``(origin, axes)``.

    ``axes`` is ``(2, 3)``: row 0 is the patch's long axis, row 1 its short one.
    Both are unit vectors, signed so that positive means anterior and dorsal
    respectively.

    This is a **flattening** helper: Figure 4 draws its isocontours in this
    plane, and a direction is projected into it to be reported as an angle. No
    direction is *computed* in it -- see the module docstring, step 3.
    """
    points = coords[patch]
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    axes = vh[:2].copy()
    if axes[0] @ ANTERIOR < 0:
        axes[0] *= -1
    if axes[1] @ DORSAL < 0:
        axes[1] *= -1
    return origin, axes


def patch_faces(faces: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Triangles with all three vertices inside the patch."""
    inside = np.zeros(faces.max() + 1, dtype=bool)
    inside[patch] = True
    return faces[inside[faces].all(axis=1)]


def face_gradients(
    coords: np.ndarray, faces: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle gradient of a vertex-wise scalar, plus triangle areas.

    The linear-FEM gradient: for a triangle with vertices ``p_i`` and values
    ``f_i``, ``grad f = (1 / 2A) * sum_i f_i * (n x e_i)`` where ``e_i`` is the
    edge opposite vertex ``i`` and ``n`` the unit normal. Exact for any field
    that is affine over the triangle, which is the test in
    ``tests/test_gradients.py``.

    Returns
    -------
    (gradients, areas)
        ``gradients`` is ``(n_faces, 3)`` in world coordinates; degenerate or
        non-finite triangles come back as NaN.
    """
    p0, p1, p2 = (coords[faces[:, i]] for i in range(3))
    f0, f1, f2 = (values[faces[:, i]] for i in range(3))

    normals = np.cross(p1 - p0, p2 - p0)
    double_area = np.linalg.norm(normals, axis=1)
    areas = 0.5 * double_area

    with np.errstate(invalid="ignore", divide="ignore"):
        unit_normal = normals / double_area[:, None]
        # Edge opposite each vertex, oriented so the cross products add up.
        contribution = (
            f0[:, None] * np.cross(unit_normal, p2 - p1)
            + f1[:, None] * np.cross(unit_normal, p0 - p2)
            + f2[:, None] * np.cross(unit_normal, p1 - p0)
        )
        gradients = contribution / double_area[:, None]

    bad = (double_area <= 0) | ~np.isfinite(gradients).all(axis=1)
    gradients[bad] = np.nan
    return gradients, areas


def vertex_gradients(
    coords: np.ndarray, faces: np.ndarray, values: np.ndarray, patch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex gradient of RTOP with respect to the mesh surface.

    The linear-FEM gradient is constant per triangle, so the value *at a vertex*
    is the area-weighted mean of the triangles meeting there. Triangles are
    taken over the whole insular patch rather than per subdivision, so a vertex
    on a subdivision boundary still sees a complete stencil.

    The result is then projected back onto the vertex's tangent plane. Each
    triangle's gradient is tangential to *that triangle*, and the insula is
    folded, so averaging across triangles with different normals tilts the mean
    out of the surface -- on the right insula far enough that the mean points
    almost along the surface normal, which is not a direction the field can
    vary in. A gradient with respect to the surface is tangential by
    definition, so the normal component is removed.

    Returns
    -------
    (gradients, vertex_areas)
        Both indexed by vertex over the full mesh. Vertices with no usable
        incident triangle are NaN with zero area. ``vertex_areas`` is the
        barycentric area, one third of the incident triangle area, so it sums
        to the patch area.
    """
    triangles = patch_faces(faces, patch)
    face_grad, areas = face_gradients(coords, triangles, values)
    usable = np.isfinite(face_grad).all(axis=1) & (areas > 0)
    triangles, face_grad, areas = triangles[usable], face_grad[usable], areas[usable]

    p0, p1, p2 = (coords[triangles[:, i]] for i in range(3))
    face_normal = np.cross(p1 - p0, p2 - p0)
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]

    total = np.zeros((len(coords), 3))
    normal = np.zeros((len(coords), 3))
    weight = np.zeros(len(coords))
    for corner in range(3):
        np.add.at(total, triangles[:, corner], face_grad * areas[:, None])
        np.add.at(normal, triangles[:, corner], face_normal * areas[:, None])
        np.add.at(weight, triangles[:, corner], areas)

    out = np.full_like(total, np.nan)
    touched = weight > 0
    out[touched] = total[touched] / weight[touched, None]

    unit_normal = np.zeros_like(normal)
    length = np.linalg.norm(normal, axis=1)
    oriented = touched & (length > 0)
    unit_normal[oriented] = normal[oriented] / length[oriented, None]
    out[oriented] -= (out[oriented] * unit_normal[oriented]).sum(axis=1)[
        :, None
    ] * unit_normal[oriented]
    return out, weight / 3.0


# ---- circular statistics --------------------------------------------------


def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean direction, in radians on ``(-pi, pi]``."""
    angles = np.asarray(angles, dtype=float)
    weights = np.ones_like(angles) if weights is None else np.asarray(weights, float)
    finite = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    c = np.sum(weights[finite] * np.cos(angles[finite]))
    s = np.sum(weights[finite] * np.sin(angles[finite]))
    return float(np.arctan2(s, c))


def resultant_length(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean resultant length ``rbar`` in ``[0, 1]``: 0 uniform, 1 concentrated."""
    angles = np.asarray(angles, dtype=float)
    weights = np.ones_like(angles) if weights is None else np.asarray(weights, float)
    finite = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    w = weights[finite]
    c = np.sum(w * np.cos(angles[finite]))
    s = np.sum(w * np.sin(angles[finite]))
    return float(np.hypot(c, s) / w.sum())


def spherical_rayleigh(vectors: np.ndarray) -> tuple[float, float, float]:
    """Rayleigh test of uniformity on the sphere. Returns ``(S, p, rbar)``.

    ``S = 3 n rbar^2`` against ``chi^2`` with 3 df (Mardia & Jupp, sec.
    10.4.1). This is the paper's test -- see the module docstring for why its
    reported "df = 3" and its statistic exceeding *n* both point here.

    Parameters
    ----------
    vectors : (n, 3) array
        One direction per participant; normalised internally.
    """
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 0] / norms[norms > 0, None]
    n = len(vectors)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rbar = float(np.linalg.norm(vectors.mean(axis=0)))
    statistic = 3.0 * n * rbar**2
    return statistic, float(stats.chi2.sf(statistic, df=3)), rbar


def rayleigh_test(angles: np.ndarray) -> tuple[float, float, float]:
    """Rayleigh test of uniformity. Returns ``(z, p, rbar)``.

    ``z = n * rbar**2`` and the p-value uses the standard small-sample
    approximation (Zar, *Biostatistical Analysis*, eq. 27.4), which is accurate
    from about n = 10 upward:

        p = exp(sqrt(1 + 4n + 4(n^2 - R^2)) - (1 + 2n)),  R = n * rbar
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = angles.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rbar = resultant_length(angles)
    resultant = n * rbar
    z = n * rbar**2
    p = float(
        np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - resultant**2)) - (1 + 2 * n))
    )
    return float(z), min(p, 1.0), float(rbar)


def von_mises_kappa(rbar: float) -> float:
    """Maximum-likelihood concentration from ``rbar`` (Fisher's approximation)."""
    if not np.isfinite(rbar) or rbar <= 0:
        return 0.0
    if rbar < 0.53:
        return float(2 * rbar + rbar**3 + 5 * rbar**5 / 6)
    if rbar < 0.85:
        return float(-0.4 + 1.39 * rbar + 0.43 / (1 - rbar))
    return float(1 / (rbar**3 - 4 * rbar**2 + 3 * rbar))


def confidence_interval(angles: np.ndarray, *, confidence: float = 0.95) -> float:
    """Half-width of the CI on the mean direction, in radians.

    The large-sample normal approximation, ``arcsin(z * sqrt(1 / (2 n rbar^2)))``
    (Zar eq. 26.24), valid when the directions are reasonably concentrated.
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = angles.size
    rbar = resultant_length(angles)
    if n < 2 or not np.isfinite(rbar) or rbar <= 0:
        return float("nan")
    z = stats.norm.ppf(0.5 + confidence / 2)
    argument = z * np.sqrt(1.0 / (2 * n * rbar**2))
    return float(np.arcsin(min(argument, 1.0)))


def circular_dispersion(rbar: float) -> float:
    """Circular standard deviation ``sqrt(-2 ln rbar)``, in radians."""
    if not np.isfinite(rbar) or rbar <= 0:
        return float("inf")
    return float(np.sqrt(-2 * np.log(min(rbar, 1.0))))


# ---- per-subject direction extraction -------------------------------------


def reporting_frame(
    frame: str, coords: np.ndarray, patch: np.ndarray
) -> np.ndarray:
    """The 2-D plane angles are reported in. See the module docstring."""
    if frame == "anatomical":
        return np.array([ANTERIOR, DORSAL])
    if frame == "patch":
        return patch_frame(coords, patch)[1]
    raise ValueError(f"frame must be 'anatomical' or 'patch', not {frame!r}")


def subject_directions(
    values: np.ndarray,
    coords: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    *,
    smoothing_iterations: int = DEFAULT_SMOOTHING_ITERATIONS,
    frame: str = DEFAULT_FRAME,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Gradient direction per subdivision, for one subject and hemisphere.

    The subdivision's direction is the weighted mean of its **per-vertex**
    gradients as 3-D unit vectors, weighted by vertex area times gradient
    magnitude. No plane is fitted anywhere.

    Returns
    -------
    (angles, vectors)
        ``vectors`` are the directions, as 3-D unit vectors in world space --
        the analysis quantity, and what the spherical Rayleigh test consumes.
        ``angles`` are those same directions projected into the reporting
        plane: 0 = increasing anteriorly, pi/2 = increasing dorsally. A
        subdivision with no usable vertex gives NaN in both.
    """
    patch = np.flatnonzero(labels > 0)
    if patch.size < 3:
        raise ValueError("Insular patch has fewer than 3 vertices")
    values = smooth_on_surface(values, faces, smoothing_iterations)
    plane = reporting_frame(frame, coords, patch)
    gradients, vertex_areas = vertex_gradients(coords, faces, values, patch)
    magnitude = np.linalg.norm(gradients, axis=1)

    angles_out: dict[str, float] = {}
    vectors_out: dict[str, np.ndarray] = {}
    for index, name in sorted(names.items()):
        usable = (
            (labels == index)
            & np.isfinite(magnitude)
            & (magnitude > 0)
            & (vertex_areas > 0)
        )
        mean = np.full(3, np.nan)
        if usable.any():
            unit = gradients[usable] / magnitude[usable, None]
            weights = vertex_areas[usable] * magnitude[usable]
            resultant = (unit * weights[:, None]).sum(axis=0)
            norm = np.linalg.norm(resultant)
            if norm > 0:
                mean = resultant / norm
        vectors_out[name] = mean
        angles_out[name] = (
            float(np.arctan2(mean @ plane[1], mean @ plane[0]))
            if np.isfinite(mean).all()
            else float("nan")
        )
    return angles_out, vectors_out


def group_statistics(
    directions: np.ndarray, vectors: np.ndarray | None = None
) -> dict:
    """Group statistics over participants, both spherical and planar.

    The spherical columns are the paper's test; the planar ones describe the
    distribution within the anterior-posterior / ventral-dorsal frame.
    """
    directions = np.asarray(directions, dtype=float)
    directions = directions[np.isfinite(directions)]
    z, p, rbar = rayleigh_test(directions)
    spherical: dict = {
        "spherical_n": 0,
        "spherical_rbar": float("nan"),
        "spherical_S": float("nan"),
        "spherical_p": float("nan"),
        # The group mean direction in world space. Figure 3B/3E draws this
        # directly, so the rendering needs no frame of its own either.
        "mean_vx": float("nan"),
        "mean_vy": float("nan"),
        "mean_vz": float("nan"),
    }
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=float)
        finite = vectors[np.isfinite(vectors).all(axis=1)]
        statistic, p_sphere, rbar_sphere = spherical_rayleigh(finite)
        mean_vector = np.full(3, np.nan)
        if len(finite):
            resultant = finite.mean(axis=0)
            norm = np.linalg.norm(resultant)
            if norm > 0:
                mean_vector = resultant / norm
        spherical = {
            "spherical_n": int(len(finite)),
            "spherical_rbar": rbar_sphere,
            "spherical_S": statistic,
            "spherical_p": p_sphere,
            "mean_vx": float(mean_vector[0]),
            "mean_vy": float(mean_vector[1]),
            "mean_vz": float(mean_vector[2]),
        }
    return {
        **spherical,
        "n": int(directions.size),
        "mean_direction_rad": circular_mean(directions),
        "mean_direction_pi": circular_mean(directions) / np.pi,
        "resultant_length": rbar,
        "rayleigh_z": z,
        "rayleigh_R": float(directions.size * rbar),
        "rayleigh_p": p,
        "kappa": von_mises_kappa(rbar),
        "dispersion_rad": circular_dispersion(rbar),
        "dispersion_pi": circular_dispersion(rbar) / np.pi,
        "ci95_halfwidth_rad": confidence_interval(directions),
        "ci95_halfwidth_pi": confidence_interval(directions) / np.pi,
    }


VECTOR_COLUMNS = ("vx", "vy", "vz")


def _vectors_of(group: pd.DataFrame) -> np.ndarray | None:
    if not all(c in group.columns for c in VECTOR_COLUMNS):
        return None
    return group[list(VECTOR_COLUMNS)].to_numpy(dtype=float)


def summarize(directions: pd.DataFrame) -> pd.DataFrame:
    """Group statistics for every ``(hemi, subdivision)``, plus a pooled row.

    Parameters
    ----------
    directions
        Long table with columns ``subject``, ``hemi``, ``subdivision``,
        ``angle``, and optionally ``vx``/``vy``/``vz`` for the spherical test.
    """
    rows = []
    for (hemi, subdivision), group in directions.groupby(["hemi", "subdivision"]):
        rows.append(
            {
                "hemi": hemi,
                "subdivision": subdivision,
                **group_statistics(group["angle"].to_numpy(), _vectors_of(group)),
            }
        )
    # Pooled over the three subdivisions. Note this necessarily *lowers* the
    # resultant length when the subdivisions point in different directions, as
    # they do -- so the per-subdivision rows, not this one, are what the
    # paper's per-hemisphere figures are most comparable to.
    for hemi, group in directions.groupby("hemi"):
        rows.append(
            {
                "hemi": hemi,
                "subdivision": "pooled",
                **group_statistics(group["angle"].to_numpy(), _vectors_of(group)),
            }
        )
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> str:
    columns = [
        "hemi",
        "subdivision",
        "n",
        "spherical_S",
        "spherical_p",
        "spherical_rbar",
        "mean_direction_pi",
        "dispersion_pi",
        "ci95_halfwidth_pi",
        "rayleigh_z",
    ]
    return "\n".join(
        [
            "=== gradient directions ===",
            "spherical_S is the paper's test (3 n rbar^2, chi2 df=3); the "
            "planar columns are angles in units of pi, 0 = anterior, "
            "0.5 = dorsal.",
            summary[columns].to_string(index=False),
        ]
    )
