# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-render the expensive figures for the Quarto book.

Run once from the repository root with the project virtualenv:

    .venv/bin/python examples/cfd/mesh_transformer/book/figures.py

Cheap figures (bar charts and line plots from the checked-in results JSONs)
are generated at render time inside the book's executable chunks; this script
only builds figures that need model checkpoints, /tmp training histories, or
the benchmark generator modules.

Numbers discipline: every hard-coded fallback constant in this file is copied
from ``results/learned_bie_2026-07-02.json`` (checked in) or from the research
session log; provenance is noted next to each constant.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BOOK_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BOOK_DIR.parent
FIG_DIR = BOOK_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

for _subdirectory in ("", "models", "problems", "studies", "datasets"):
    _entry = str(EXAMPLE_DIR / _subdirectory) if _subdirectory else str(EXAMPLE_DIR)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# ---------------------------------------------------------------------------
# Palette (colorblind-validated categorical order; see the dataviz reference).
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DIVERGING = LinearSegmentedColormap.from_list(
    "book_diverging", [BLUE, "#f0efec", RED]
)

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------------
# Figure 1: learned kernel versus the analytic double-layer kernel.
# ---------------------------------------------------------------------------
def fig_kernel() -> None:
    """Learned 3-parameter kernel member versus the analytic double layer."""
    checkpoint = Path(
        "/tmp/h4_prune/final_2param_seed17/harmonic_panel_bie_2param_reference.pt"
    )
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = state["state_dict"]
        c1 = float(sd["singular_coefficient"])
        d0 = float(sd["regular_coefficients"][0])
        alpha = float(sd["relaxation"])
        source = "checkpoint /tmp/h4_prune/final_2param_seed17 (session artifact)"
    else:
        # Fallback: iteration_4 shared_alpha_3p entry of the checked-in
        # results/learned_bie_2026-07-02.json.
        c1, d0, alpha = -0.15828, -0.00151, 1.17983
        source = "results/learned_bie_2026-07-02.json (checked in)"
    oracle = -1.0 / (2.0 * math.pi)

    results = json.load(open(EXAMPLE_DIR / "results" / "learned_bie_2026-07-02.json"))
    c1_by_seed = results["iteration_4_pruning"]["confirmation_3seed_reference_bank"][
        "learned_c1_by_seed"
    ]

    # Stacked vertically (one panel per row) so the rendered page scrolls
    # down rather than sideways.
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.8))

    # (a) angular profile at |r| = 1: kappa = c1 cos(psi) + d0.
    psi = np.linspace(0.0, np.pi, 200)
    ax = axes[0]
    ax.plot(
        psi, c1 * np.cos(psi) + d0, color=BLUE, lw=2, label="learned (3 params)"
    )
    ax.plot(
        psi,
        oracle * np.cos(psi),
        color=INK,
        lw=1.4,
        ls="--",
        label=r"analytic $-\cos\psi/2\pi$",
    )
    ax.set_xlabel(r"angle $\psi$ between $n_y$ and $x-y$  (rad)")
    ax.set_ylabel(r"$\kappa$ at $|x-y|=1$  (dimensionless)")
    ax.set_title("Angular profile", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # (b) radial decay along psi = 0: kappa = c1/|r| + d0.
    r = np.linspace(0.25, 4.0, 200)
    ax = axes[1]
    ax.plot(r, c1 / r + d0, color=BLUE, lw=2, label="learned")
    ax.plot(r, oracle / r, color=INK, lw=1.4, ls="--", label="analytic")
    ax.set_xlabel(r"$|x-y|/L$  (dimensionless)")
    ax.set_ylabel(r"$\kappa$ at $\psi=0$")
    ax.set_title("Radial decay", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    # (c) learned c1 per seed against the oracle (3-seed reference bank).
    ax = axes[2]
    seeds = ["17", "29", "43"]
    ax.axhline(oracle, color=INK, ls="--", lw=1.4)
    ax.text(2.45, oracle, r"$-1/2\pi$", va="bottom", ha="right", fontsize=9, color=INK)
    ax.plot(range(3), c1_by_seed, "o", color=BLUE, ms=8, zorder=3)
    for i, v in enumerate(c1_by_seed):
        ax.annotate(
            f"{v:.5f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_xticks(range(3), [f"seed {s}" for s in seeds])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.1605, -0.1568)
    ax.set_ylabel(r"learned $c_1$")
    ax.set_title("3 seeds, 3000 steps", fontsize=10)

    fig.suptitle(
        f"Learned kernel vs analytic interior double layer\n({source})",
        fontsize=9,
        color=MUTED,
        y=1.005,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kernel_learned_vs_analytic.png", bbox_inches="tight")
    plt.close(fig)
    print(f"kernel figure: c1={c1:.5f}, d0={d0:.5f}, alpha={alpha:.5f}  [{source}]")


# ---------------------------------------------------------------------------
# Figure 2: training trajectories (session artifacts under /tmp).
# ---------------------------------------------------------------------------
def _history(path: Path) -> tuple[list[int], list[float]] | None:
    if not path.exists():
        return None
    report = json.load(open(path))
    entries = [e for e in report["history"] if "validation_relative_l2" in e]
    return [e["step"] for e in entries], [e["validation_relative_l2"] for e in entries]


def fig_convergence() -> None:
    runs = {
        "MLP pair kernel (19,008 p)": (
            Path(
                "/tmp/laplace_study_d008/early/simple_invariant_pair/seed-17/"
                "pair_kernel_reference.json"
            ),
            YELLOW,
        ),
        "harmonic-panel BIE (13 p)": (
            Path(
                "/tmp/h1_learned_bie/harmonic_panel_bie_3000_seed17/"
                "harmonic_panel_bie_reference.json"
            ),
            AQUA,
        ),
        "harmonic-panel BIE, pruned (3 p)": (
            Path(
                "/tmp/h4_prune/final_2param_seed17/"
                "harmonic_panel_bie_2param_reference.json"
            ),
            BLUE,
        ),
    }
    neumann = Path(
        "/tmp/h2_neumann/full10k_seed17/neumann_harmonic_panel_bie_reference.json"
    )

    # Stacked vertically (one panel per row) so the rendered page scrolls
    # down rather than sideways.
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6))

    ax = axes[0]
    missing = []
    for label, (path, color) in runs.items():
        data = _history(path)
        if data is None:
            missing.append(label)
            continue
        steps, values = data
        ax.plot(steps, values, color=color, lw=2)
        ax.annotate(
            label,
            (steps[-1], values[-1]),
            textcoords="offset points",
            xytext=(4, 4 if "pruned" not in label else -12),
            fontsize=8,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 4400)
    # Leave headroom below the lowest curve so the below-the-line endpoint
    # label ("pruned") stays inside the axes.
    ax.set_ylim(bottom=3e-3)
    ax.set_xlabel("online training step")
    ax.set_ylabel(r"validation relative $L^2$")
    ax.set_title("Dirichlet (seed 17)", fontsize=10)

    ax = axes[1]
    data = _history(neumann)
    if data is not None:
        steps, values = data
        ax.plot(steps, values, color=BLUE, lw=2)
        peak = int(np.argmax(values[:8]))
        ax.annotate(
            "co-adaptation barrier",
            (steps[peak], values[peak]),
            textcoords="offset points",
            xytext=(10, -2),
            fontsize=8,
            color=INK2,
        )
        ax.annotate(
            "Neumann harmonic-panel BIE (14 p)",
            (steps[-1], values[-1]),
            textcoords="offset points",
            xytext=(-6, 8),
            ha="right",
            fontsize=8,
            color=BLUE,
        )
    ax.set_yscale("log")
    ax.set_xlabel("online training step")
    ax.set_ylabel(r"validation relative $L^2$")
    ax.set_title("Neumann (seed 17, 10,000 steps)", fontsize=10)

    fig.suptitle(
        "Validation trajectories (single-machine session artifacts, not checked in)",
        fontsize=9,
        color=MUTED,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_trajectories.png", bbox_inches="tight")
    plt.close(fig)
    if missing:
        print(f"convergence figure: MISSING histories for {missing}")
    else:
        print("convergence figure: all four histories found")


# ---------------------------------------------------------------------------
# Figure 3: 3D geometry/BC gallery from the laplace3d generator.
# ---------------------------------------------------------------------------
def _star_boundaries(seed: int, subdivisions: int) -> dict:
    """Star-tier boundary surfaces + exact Dirichlet data, generator-faithful.

    ``build_laplace3d_sample(tier="star")`` currently hangs: its
    interior-query rejection bound uses ``max_star = sum(|amp|) * 35`` (a
    ~4x-loose harmonic bound), which drives ``hi_factor = 1 - 0.08 -
    max_star`` negative so no query is ever accepted.  The gallery needs only
    the *boundary* surface and its exact Dirichlet trace, so this helper
    replays the generator's own sampling and components
    (``_surface_mesh``, ``_potential_and_gradient``, identical
    distributions), skipping the interior-query stage.  The bug is recorded
    in the book's engineering-review chapter.
    """
    from laplace3d import (
        _FOUR_PI,
        _STAR_MODES,
        _potential_and_gradient,
        _surface_mesh,
    )

    generator = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float, shape=()):
        return torch.empty(shape, dtype=torch.float64).uniform_(
            low, high, generator=generator
        )

    radius = float(uniform(0.7, 1.5))
    center = uniform(-1.0, 1.0, (3,))
    axis = torch.nn.functional.normalize(uniform(-1.0, 1.0, (3,)), dim=0)
    angle = float(uniform(0.0, 2.0 * math.pi))
    skew = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    rotation = (
        torch.eye(3, dtype=torch.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    star = []
    max_star = 0.0
    for mode in _STAR_MODES:
        amplitude = float(uniform(-0.06, 0.06))
        star.append((mode, amplitude))
        max_star += abs(amplitude) * 35.0
    mesh = _surface_mesh(
        radius,
        center,
        rotation,
        star,
        subdivisions=subdivisions,
        inward=False,
        dtype=torch.float64,
    )
    directions = torch.nn.functional.normalize(uniform(-1.0, 1.0, (6, 3)), dim=-1)
    positions = center + radius * (1.0 + max_star + 0.5) * directions
    charges = uniform(-1.0, 1.0, (6,)) * (_FOUR_PI * radius)
    values, _ = _potential_and_gradient(mesh.cell_centroids, charges, positions)
    return {"outer": mesh.with_data(cell_data={"boundary_value": values})}


def fig_gallery3d() -> None:
    from laplace3d import build_laplace3d_sample

    tiers = ["sphere", "star", "shell"]
    seeds = [3, 5, 11]
    elev, azim = 20.0, -60.0
    camera = np.array(
        [
            math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
            math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
            math.sin(math.radians(elev)),
        ]
    )

    fig = plt.figure(figsize=(9.2, 8.0))
    grid = fig.add_gridspec(
        3, 3, left=0.07, right=0.86, top=1.03, bottom=-0.03, wspace=-0.08, hspace=-0.16
    )

    for row, tier in enumerate(tiers):
        for col, seed in enumerate(seeds):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            ax.view_init(elev=elev, azim=azim)
            if tier == "star":
                boundaries = _star_boundaries(1000 * (row + 1) + seed, 3)
            else:
                sample = build_laplace3d_sample(
                    1000 * (row + 1) + seed,
                    tier=tier,
                    bc_regime="dirichlet",
                    subdivisions=3,
                    n_query=8,
                )
                boundaries = sample.domain.boundaries
            all_points = torch.cat([m.points for m in boundaries.values()])
            center = all_points.mean(dim=0).numpy()
            all_values = torch.cat(
                [m.cell_data["boundary_value"] for m in boundaries.values()]
            )
            # Diverging about the per-sample boundary mean: for Laplace the
            # deviation from the mean is the informative part of the trace.
            vmid = float(all_values.mean())
            amp = float((all_values - vmid).abs().max())
            norm = Normalize(vmin=vmid - amp, vmax=vmid + amp)
            for name, mesh in boundaries.items():
                triangles = mesh.points[mesh.cells].numpy()
                values = mesh.cell_data["boundary_value"].numpy()
                if tier == "shell" and name == "outer":
                    # Cut the half facing the camera so the inner boundary of
                    # the multiply connected domain is visible.
                    centroids = triangles.mean(axis=1)
                    keep = (centroids - center) @ camera < 0.0
                    triangles, values = triangles[keep], values[keep]
                collection = Poly3DCollection(
                    triangles,
                    facecolors=DIVERGING(norm(values)),
                    edgecolors=(0, 0, 0, 0.08),
                    linewidths=0.2,
                )
                ax.add_collection3d(collection)
            lo = all_points.min(dim=0).values.numpy()
            hi = all_points.max(dim=0).values.numpy()
            mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.92
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            if col == 0:
                label = {
                    "sphere": "T1 sphere",
                    "star": "T2 star $(Y_{lm})$",
                    "shell": "T3 shell (cutaway)",
                }[tier]
                ax.text2D(
                    -0.02,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    color=INK,
                )

    cax = fig.add_axes([0.89, 0.36, 0.018, 0.28])
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=Normalize(-1, 1))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["min", "boundary\nmean", "max"], fontsize=8)
    cbar.set_label("Dirichlet data $g$ (per sample)", fontsize=9)
    fig.savefig(FIG_DIR / "gallery_3d.png", bbox_inches="tight")
    plt.close(fig)
    print("3D gallery written")


# ---------------------------------------------------------------------------
# Figure 4: 2D conformal-domain gallery across the evaluation splits.
# ---------------------------------------------------------------------------
def fig_gallery2d() -> None:
    from conformal_laplace import build_domain_sample, sample_drive, sample_geometry

    splits = [
        ("training family\nmodes $\\{2,3\\}$\n$\\kappa\\in[0.05,0.35]$", (2, 3), (0.05, 0.35)),
        ("unseen geometry\nmodes $\\{4,5\\}$", (4, 5), (0.05, 0.35)),
        ("stronger deformation\n$\\kappa\\in[0.45,0.65]$", (2, 3), (0.45, 0.65)),
    ]
    n_cols = 4
    fig = plt.figure(figsize=(9.2, 6.6))
    grid = fig.add_gridspec(
        3,
        n_cols,
        left=0.14,
        right=0.86,
        top=0.99,
        bottom=0.01,
        wspace=0.05,
        hspace=0.05,
    )

    for row, (title, modes, deformation) in enumerate(splits):
        for col in range(n_cols):
            ax = fig.add_subplot(grid[row, col])
            seed = 100 * (row + 1) + 7 * col
            geometry = sample_geometry(
                seed, modes=modes, deformation_range=deformation
            )
            drive = sample_drive(seed + 1, modes=(1, 2, 3, 4))
            sample = build_domain_sample(geometry, drive, n_boundary=192, n_query=8)
            boundary = sample.domain.boundaries["dirichlet"]
            segments = boundary.points[boundary.cells].numpy()
            values = boundary.cell_data["boundary_value"].numpy()
            vmid = float(values.mean())
            amp = float(np.abs(values - vmid).max())
            collection = LineCollection(
                segments,
                colors=DIVERGING(Normalize(vmid - amp, vmid + amp)(values)),
                linewidths=2.6,
            )
            ax.add_collection(collection)
            lo = segments.reshape(-1, 2).min(axis=0)
            hi = segments.reshape(-1, 2).max(axis=0)
            mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.12
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.text(
                    -0.24,
                    0.5,
                    title,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=INK,
                )

    cax = fig.add_axes([0.89, 0.36, 0.018, 0.28])
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=Normalize(-1, 1))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["min", "boundary\nmean", "max"], fontsize=8)
    cbar.set_label("Dirichlet trace $g$ (per sample)", fontsize=9)
    fig.savefig(FIG_DIR / "gallery_2d.png", bbox_inches="tight")
    plt.close(fig)
    print("2D gallery written")


if __name__ == "__main__":
    fig_kernel()
    fig_convergence()
    fig_gallery3d()
    fig_gallery2d()
    print(f"figures written to {FIG_DIR}")
