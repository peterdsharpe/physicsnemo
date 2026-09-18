"""One visual identity for every figure in the book.

Import this at the top of every ``{python}`` cell that draws:

    from palette import *          # colours, STYLE, house rcParams, helpers

Unlike ``figures.py`` (the offline figure generator), this module never
switches the matplotlib backend, so it is safe inside Quarto's inline cells.

Colour is semantic and fixed for the whole book. An architecture keeps its
hue in every chapter; a *variant* of an architecture is the same hue at a
lighter tint with an open marker; an *ablation* (a configuration measured
only to price a design choice and never adopted) is the same hue, hatched.
Colour is never the only cue: marker shape and line style carry the same
distinction, so every figure survives greyscale and colour-vision deficiency.

    ISLA ................ blue,   square,   solid
    GeoTransolver ....... red,    circle,   solid
    Transolver .......... amber,  triangle, dashed
    context / chrome .... greys (never carries a data distinction)
    field values ........ BLUE -> paper -> RED diverging map (signed fields)
                          viridis (unsigned fields)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as _pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Architecture hues -------------------------------------------------------
BLUE = "#2a78d6"        # ISLA (reference configuration)
BLUE_LIGHT = "#8fbbea"  # ISLA variants (constant gauge, similarity gauge)
BLUE_PALE = "#c9def5"   # ISLA ablations / diagnostics (hatched)
RED = "#e34948"         # GeoTransolver (surface and volume)
RED_LIGHT = "#f2a3a2"   # GeoTransolver research variants
AMBER = "#c98a00"       # Transolver
AMBER_LIGHT = "#e8c46a" # Transolver research variants

# --- Concept colours (architecture figures, never a data series) -------------
VIOLET = "#4a3aa7"      # slices / anchors
AQUA = "#1baf7a"        # queries / interior points
GREEN = "#008300"       # satisfied contract, PASS
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
PAPER = "#f4f3ee"       # data boxes in flow diagrams
WHITE = "#ffffff"

DIVERGING = LinearSegmentedColormap.from_list("book_diverging", [BLUE, "#f0efec", RED])
SEQUENTIAL = "viridis"

# --- Series styles ------------------------------------------------------------
# key -> dict(label, color, marker, ls, mfc). ``mfc`` "full" fills the marker.
# Use STYLE[key] wherever that configuration appears so the legend text is the
# same in every chapter. Labels are the book's canonical names.
STYLE = {
    "isla_ref": dict(label="ISLA, reference configuration", color=BLUE, marker="s", ls="-", mfc="full", hatch=""),
    "isla_cg": dict(label="ISLA, constant-gauge variant", color=BLUE_LIGHT, marker="s", ls=":", mfc="none", hatch=""),
    "isla_sg": dict(label="ISLA, similarity-gauge variant", color=BLUE_LIGHT, marker="D", ls="--", mfc="none", hatch=""),
    "isla_abl": dict(label="ISLA, ablation (not adopted)", color=BLUE_PALE, marker="s", ls=":", mfc="none", hatch="///"),
    "gt": dict(label="GeoTransolver", color=RED, marker="o", ls="-", mfc="full", hatch=""),
    "gt_var": dict(label="GeoTransolver, research variant", color=RED_LIGHT, marker="o", ls=":", mfc="none", hatch=""),
    "t": dict(label="Transolver", color=AMBER, marker="^", ls="--", mfc="full", hatch=""),
    "t_var": dict(label="Transolver, research variant", color=AMBER_LIGHT, marker="^", ls=":", mfc="none", hatch=""),
}


def series_kwargs(key: str, **override):
    """matplotlib ``plot`` keyword arguments for a configuration key."""
    s = STYLE[key]
    kw = dict(color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"],
              mfc=(s["color"] if s["mfc"] == "full" else "white"), mec=s["color"], mew=1.8, ms=7, lw=2.0)
    kw.update(override)
    return kw


def bar_kwargs(key: str, **override):
    """matplotlib ``bar``/``barh`` keyword arguments for a configuration key."""
    s = STYLE[key]
    kw = dict(color=s["color"], hatch=s["hatch"], edgecolor="white" if not s["hatch"] else BLUE, linewidth=0.8)
    kw.update(override)
    return kw


# --- House style ---------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlesize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.formatter.useoffset": False,
    "lines.solid_capstyle": "round",
})


# --- Helpers ---------------------------------------------------------------------
def halo(lw: float = 3.0, fg: str = WHITE):
    """Path effect that gives text a quiet halo so labels stay legible over data."""
    return [_pe.withStroke(linewidth=lw, foreground=fg)]


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left", va="center", fontsize=9, **kw):
    """Direct-label a series at a point instead of using a legend."""
    return ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color, ha=ha, va=va,
                       fontsize=fontsize, path_effects=halo(), **kw)


def note(ax, text, loc="lower right", fontsize=8.5):
    """A small boxed reading note inside the axes (the ratios, the caveat)."""
    pos = {"lower right": (0.985, 0.03, "right", "bottom"), "lower left": (0.015, 0.03, "left", "bottom"),
           "upper right": (0.985, 0.97, "right", "top"), "upper left": (0.015, 0.97, "left", "top")}[loc]
    return ax.text(pos[0], pos[1], text, transform=ax.transAxes, ha=pos[2], va=pos[3], fontsize=fontsize,
                   color=INK2, linespacing=1.4, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BASELINE))


def clean_axes(ax, keep_grid_axis="y"):
    """Remove ticks-only chrome from an axes used for a schematic."""
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


def joukowski(alpha_deg: float = 6.0, n_cells: int = 60, c=complex(-0.08, 0.08), frac=None):
    """The book's running example: a Joukowski airfoil with the Kutta condition.

    Returns a dict with the boundary polyline ``xy`` (n_cells+1, 2), the cell
    midpoints ``mid``, unit outward normals ``nrm``, cell lengths ``w`` (the
    1-D measure), surface pressure coefficient ``cp_s`` at the midpoints, and a
    callable ``cp_field(z)`` giving the pressure coefficient at complex points.

    ``frac`` (optional, increasing from 0 to 1) places the cell boundaries at
    those fractions of the total arclength, so a non-uniform "mesher" can be
    imitated; the default is equal arclength.
    """
    a = abs(1.0 - c)
    alpha = np.deg2rad(alpha_deg)
    beta = -np.angle(1.0 - c)
    Gamma = 4 * np.pi * a * np.sin(alpha + beta)

    def velocity(zeta):
        dW = (np.exp(-1j * alpha) - a**2 * np.exp(1j * alpha) / (zeta - c) ** 2
              + 1j * Gamma / (2 * np.pi * (zeta - c)))
        return np.conj(dW / (1 - 1 / zeta**2))

    tf = np.linspace(0, 2 * np.pi, 4001)
    zf = c + a * np.exp(1j * tf); zf = zf + 1 / zf
    s = np.concatenate([[0], np.cumsum(np.abs(np.diff(zf)))])
    s_cells = np.linspace(0, s[-1], n_cells + 1) if frac is None else np.asarray(frac) * s[-1]
    xb = np.interp(s_cells, s, zf.real); yb = np.interp(s_cells, s, zf.imag)
    xy = np.stack([xb, yb], axis=1)
    t_mid = np.interp((s_cells[:-1] + s_cells[1:]) / 2, s, tf)
    cp_s = 1 - np.abs(velocity(c + a * 1.002 * np.exp(1j * t_mid))) ** 2
    mid = (xy[:-1] + xy[1:]) / 2
    tang = xy[1:] - xy[:-1]
    w = np.linalg.norm(tang, axis=1)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1) / w[:, None]
    nrm *= np.sign(np.sum((mid - xy.mean(axis=0)) * nrm, axis=1))[:, None]

    def cp_field(z):
        # invert z = zeta + 1/zeta, taking the root outside the circle
        zeta = (z + np.sqrt(z * z - 4 + 0j)) / 2
        inside = np.abs(zeta - c) < a
        zeta = np.where(inside, (z - np.sqrt(z * z - 4 + 0j)) / 2, zeta)
        return 1 - np.abs(velocity(zeta)) ** 2

    return dict(xy=xy, mid=mid, nrm=nrm, w=w, cp_s=cp_s, cp_field=cp_field, alpha=alpha,
                g_hat=np.array([np.cos(alpha), np.sin(alpha)]))


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("annotations", "mpl", "np", "plt")] + ["mpl", "np", "plt"]
