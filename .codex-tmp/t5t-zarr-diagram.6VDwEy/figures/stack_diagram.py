"""Storage-stack slide diagram: "Mesh vs zarr is a category error".

One figure, ``storage_stack()``, in the deck's house style (slides/_common.py):
a three-layer architecture stack (front end / serialization contract / storage
backends) with the flattened-zarr store drawn OFF the stack entirely — the
Slack-thread 1/2/3 taxonomy (research/13, thread 2) made visual.

Semantic colors (stable across the deck):

- front end (Mesh/DomainMesh): NVIDIA green (talk-wide "us" hue); tag ① dark green
- interchangeable backends + tag ②:     SECONDARY #0072B2
- the flattening trap + tag ③:          #CC3311 (house SERIES red, matches
  the loss color in diagrams.py)

Run ``python stack_diagram.py`` to regenerate the preview PNG next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slides"))

import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import (
    BACKGROUND,
    INK,
    MUTED,
    NVIDIA_GREEN,
    NVIDIA_GREEN_DARK,
    SECONDARY,
    presentation_figure,
)

TRAP = "#CC3311"
MONO = "DejaVu Sans Mono"

_HALO = [path_effects.withStroke(linewidth=4, foreground=BACKGROUND)]


def _text(ax, x, y, s, *, color=INK, size=13.5, weight=400, ha="left",
          va="center", mono=False, halo=True, **kw):
    """Direct label with the house halo."""
    return ax.text(
        x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
        family=MONO if mono else "sans-serif",
        path_effects=_HALO if halo else None, **kw,
    )


def _round_box(ax, x, y, w, h, *, edge=MUTED, face="none", lw=1.6, ls="-",
               radius=0.12, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        edgecolor=edge, facecolor=face, linewidth=lw, linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, p0, p1, *, color=INK, lw=2.4, scale=16, style="-|>", zorder=3):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=scale, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(a)
    return a


def _badge(ax, x, y, glyph, color):
    """Circled-digit taxonomy tag sitting on a box corner."""
    return _text(ax, x, y, glyph, color=color, size=21, weight=700,
                 ha="center", va="center", zorder=6)


def storage_stack():
    """Front end / contract / backends stack; flattened zarr off to the side."""
    fig, ax = presentation_figure(11.0, 5.0)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.set_axis_off()
    ax.set_aspect("auto")
    ax.set_position((0.01, 0.01, 0.98, 0.98))

    # Main stack occupies x in [0.30, 7.20]; the flattened store sits right of the divider.
    SL, SR = 0.30, 7.20
    SC = 0.5 * (SL + SR)

    # --- satellite consumers of the front end -------------------------------
    for cx, lab in ((1.15, "mesh ops"),
                    (3.75, "models (GLOBE · mesh attention)"),
                    (6.40, "datapipes")):
        _text(ax, cx, 5.16, lab, color=MUTED, size=13, weight=600, ha="center")
        _arrow(ax, (cx, 4.72), (cx, 4.98), color=MUTED, lw=1.6, scale=11)

    # --- TOP: the front end (accent) -----------------------------------------
    _round_box(ax, SL, 3.72, SR - SL, 0.98, edge=NVIDIA_GREEN_DARK, lw=2.4,
               face=NVIDIA_GREEN + "1A", radius=0.14)
    _text(ax, SL + 0.26, 4.42, "Mesh / DomainMesh", color=NVIDIA_GREEN_DARK, size=16,
          weight=700, mono=True)
    _text(ax, SR - 0.26, 4.42, "the front end", color=NVIDIA_GREEN_DARK, size=13.5,
          weight=600, ha="right")
    _text(ax, SL + 0.26, 4.00,
          "in-memory schema: points · cells · rank-typed fields · "
          "named boundaries · global data",
          size=13, color=INK)

    # --- MIDDLE: the serialization contract ----------------------------------
    _round_box(ax, SL + 0.55, 2.72, SR - SL - 1.10, 0.58, edge=INK, lw=1.8)
    _text(ax, SC, 3.01, "TensorDict — serialization contract", size=14,
          weight=700, ha="center", mono=True)
    _arrow(ax, (SC, 3.32), (SC, 3.70), style="<|-|>", lw=2.4)

    # --- BOTTOM: three interchangeable backends -------------------------------
    bw, gap = 2.16, 0.21
    backends = [
        ("memmap", ".pdmsh directory", "the default", "①", NVIDIA_GREEN_DARK),
        ("HDF5", 'backend="h5"', "inherited", "②", SECONDARY),
        ("zarr", "optional backend", "#1894 · merged", "②", SECONDARY),
    ]
    for i, (title, sub1, sub2, glyph, gcol) in enumerate(backends):
        x0 = SL + i * (bw + gap)
        cx = x0 + 0.5 * bw
        _round_box(ax, x0, 1.28, bw, 1.00, edge=MUTED, lw=1.6)
        _text(ax, cx, 2.02, title, size=14, weight=700, ha="center")
        _text(ax, cx, 1.68, sub1, size=13, color=MUTED, ha="center", mono=True)
        _text(ax, cx, 1.42, sub2, size=13, color=MUTED, ha="center")
        _arrow(ax, (cx, 2.30), (cx, 2.70), style="<|-|>", lw=2.4)  # identical
        _badge(ax, x0 + 0.03, 2.28, glyph, gcol)
    _text(ax, 0.5 * (SL + 0.5 * bw + 3.75), 2.56, "save / load", size=13,
          color=MUTED, ha="center")

    # (taxonomy tags removed: the caption carries the ①/②/③ comparison)

    # --- OFF THE STACK: the flattened store (a different concept) ------------------------
    ax.plot([7.62, 7.62], [0.75, 5.05], color=MUTED, lw=1.2, ls=(0, (1, 3)),
            zorder=1)
    TX, TW = 8.05, 2.70
    _round_box(ax, TX, 1.28, TW, 1.62, edge=TRAP, lw=1.8, ls=(0, (6, 4)),
               face=TRAP + "0D")
    tc = TX + 0.5 * TW
    _text(ax, tc, 3.18, "✗ not the same object", size=13.5, weight=700,
          color=TRAP, ha="center")
    _text(ax, tc, 2.62, "flattened zarr store", size=14.5, weight=700,
          color=TRAP, ha="center")
    _text(ax, tc, 2.26, "arrays without schema", size=13, color=TRAP,
          ha="center")
    _text(ax, tc, 1.92, "boundaries, ranks,", size=13, color=TRAP,
          ha="center")
    _text(ax, tc, 1.60, "names — gone", size=13, color=TRAP, ha="center")
    _badge(ax, TX + 0.03, 2.90, "③", TRAP)
    _text(ax, tc, 0.96, "③ — a different concept", size=13, weight=600, color=TRAP,
          ha="center")
    # nothing connects it to the stack — that IS the diagram

    # --- caption ---------------------------------------------------------------
    _text(ax, 5.5, 0.34,
          "① and ② are functionally comparable; ③ is a different concept, not a backend.",
          size=14.5, weight=600, ha="center")
    return fig


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    fig = storage_stack()
    out = here / "storage_stack.png"
    fig.savefig(out, dpi=110)
    print(out)
