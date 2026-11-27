"""Real 2-D landscapes of BBOB and MA-BBOB functions (via ``ioh``) as PNG
data URIs in the video palette: dark where the function is high, glowing
violet, pink and amber towards the optimum."""

from __future__ import annotations

import base64
import io

import ioh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CMAP = LinearSegmentedColormap.from_list(
    "blade",
    ["#ffe2a8", "#ffb547", "#ff7ab6", "#7c5cff", "#2a1f5c", "#11121b"],
)


def grid(problem, n=220, lo=-5.0, hi=5.0):
    xs = np.linspace(lo, hi, n)
    z = np.array([[problem([x, y]) for x in xs] for y in xs])
    z = z - problem.optimum.y
    z = np.log10(np.maximum(z, 0) + 1e-8 * (1 + np.abs(z).max()))
    z = (z - z.min()) / (z.max() - z.min())
    # spread the colours evenly over the landscape (rank transform)
    ranks = z.ravel().argsort().argsort().reshape(z.shape)
    return (ranks / ranks.max()) ** 0.55


def to_uri(z, px=600, contours=14, xopt=None) -> str:
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        z, cmap=CMAP, origin="lower", extent=(-5, 5, -5, 5), interpolation="bicubic"
    )
    if contours:
        ax.contour(
            np.linspace(-5, 5, z.shape[0]),
            np.linspace(-5, 5, z.shape[0]),
            z,
            levels=contours,
            colors="white",
            linewidths=0.7,
            alpha=0.28,
        )
    if xopt is not None:
        ax.scatter(
            [xopt[0]], [xopt[1]], s=px * 0.5, c="#ffffff", alpha=0.25, linewidths=0
        )
        ax.scatter([xopt[0]], [xopt[1]], s=px * 0.09, c="#ffffff", linewidths=0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def bbob(fid, instance=1, px=600, n=220, contours=14, mark=True):
    p = ioh.get_problem(fid, instance, 2)
    z = grid(p, n)
    xopt = p.optimum.x if mark and all(abs(v) <= 5 for v in p.optimum.x) else None
    return to_uri(z, px, contours, xopt)


def ma_bbob(instance, px=600, n=220, contours=16):
    p = ioh.problem.ManyAffine(instance, 2)
    z = grid(p, n)
    return to_uri(z, px, contours, p.optimum.x)


def bbob_name(fid):
    return ioh.get_problem(fid, 1, 2).meta_data.name
