"""Shared visual language for a Quarto Reveal.js presentation.

Keep concept-to-color assignments stable throughout a deck.  The theme chrome
is deliberately quiet; saturated colors belong to data and semantic emphasis.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401 -- convenient re-export in notebook cells
from cycler import cycler
from matplotlib.patches import FancyArrowPatch

BACKGROUND = "#FAFAFA"
INK = "#23373B"
PRIMARY = "#225555"
SECONDARY = "#0072B2"
MUTED = "#687076"
GRID = "#DCE1E4"
PALE_BLUE = "#CCEEFF"
AMBER = "#DDAA33"
AMBER_INK = "#7A5200"

# Talk-wide semantic assignments (one hue per concept, per quarto-books rule):
#   NVIDIA_GREEN       -> PhysicsNeMo-Mesh / GLOBE / "us" (fat marks & fills only;
#                         2.3:1 vs BACKGROUND — never thin lines or body text)
#   NVIDIA_GREEN_DARK  -> "us" for text labels and thin lines (4.3:1 contrast)
#   SECONDARY (blue)   -> highlighted competitor / slip-family BCs
#   MUTED (gray)       -> baseline clusters, context
#   "#CC3311" (red)    -> discarded information, no_slip BC, failure paths
#   "#332288" (indigo) -> inlet/outlet/freestream BCs (green is reserved for "us")
NVIDIA_GREEN = "#76B900"
NVIDIA_GREEN_DARK = "#538300"

# Projection-safe line colors derived from Paul Tol's schemes. Every member
# exceeds 3:1 against BACKGROUND. Assign slots semantically per deck; use direct
# labels, marker shapes, or line styles as redundant cues.
SERIES = [
    "#4477AA",
    "#228833",
    "#AA3377",
    "#CC3311",
    "#7A5200",
    "#332288",
    "#687076",
]

mpl.rcParams.update(
    {
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.prop_cycle": cycler(color=SERIES),
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 23,
        "axes.titleweight": 600,
        "axes.titlelocation": "left",
        "axes.labelsize": 19,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.formatter.useoffset": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "text.color": INK,
        "font.size": 18,
        "font.family": "sans-serif",
        "lines.linewidth": 3.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        # Match fig-width/fig-height in _quarto.yml; the theme scales figures
        # to the full content width, so the aspect ratio is what matters.
        "figure.figsize": (11.0, 4.2),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        # Convert text to vector paths for browser-independent font metrics.
        # This trades selectable SVG text for reliable Chrome/Firefox rendering.
        "svg.fonttype": "path",
        "mathtext.fontset": "dejavusans",
    }
)


def footnote(fig, text: str, *, fontsize: float = 10.5, chars_per_inch: float = 11.5):
    """Wrapped provenance footnote pinned to the figure's lower-left.

    Always use this instead of a raw fig.text for footnotes: a single long
    line extends the tight bounding box horizontally, which compresses the
    figure's actual content when the canvas is laid out (defect class seen
    three times in this deck).
    """
    import textwrap

    width = max(40, int(fig.get_size_inches()[0] * chars_per_inch))
    return fig.text(
        0.01, 0.012, textwrap.fill(text, width=width),
        ha="left", va="bottom", fontsize=fontsize, color=MUTED, linespacing=1.3,
    )


def presentation_figure(width: float = 11.0, height: float = 4.2, **kwargs):
    """Return a slide-sized figure and axes using the house background."""

    return plt.subplots(figsize=(width, height), facecolor=BACKGROUND, **kwargs)


def label_line(ax, x, y, text, color, *, dx=0.0, dy=0.0, **kwargs):
    """Direct-label a series with a quiet halo instead of a legend box."""

    return ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        ha="left",
        va="center",
        fontweight=600,
        path_effects=[path_effects.withStroke(linewidth=4, foreground=BACKGROUND)],
        **kwargs,
    )


def force_arrow(ax, start, vector, label, *, color=PRIMARY, scale=1.0, **kwargs):
    """Draw and label a force vector for a reproducible free-body diagram.

    `vector * scale` controls drawn length. State or annotate the scale whenever
    arrow lengths communicate magnitude; otherwise label the vector magnitude.
    """

    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float)
    end = start + scale * vector
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.5,
        color=color,
        **kwargs,
    )
    ax.add_patch(arrow)
    midpoint = 0.5 * (start + end)
    ax.text(
        midpoint[0],
        midpoint[1],
        label,
        color=color,
        fontweight=600,
        ha="center",
        va="bottom",
        path_effects=[path_effects.withStroke(linewidth=4, foreground=BACKGROUND)],
    )
    return arrow


def finish_axes(ax, *, despine: bool = True):
    """Apply the final low-ink cleanup after plotting."""

    if despine:
        ax.spines[["top", "right"]].set_visible(False)
    return ax


__all__ = [
    "AMBER",
    "NVIDIA_GREEN",
    "NVIDIA_GREEN_DARK",
    "AMBER_INK",
    "BACKGROUND",
    "GRID",
    "INK",
    "MUTED",
    "PALE_BLUE",
    "PRIMARY",
    "SECONDARY",
    "SERIES",
    "finish_axes",
    "footnote",
    "force_arrow",
    "label_line",
    "np",
    "plt",
    "presentation_figure",
]
