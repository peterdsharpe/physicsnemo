"""H-SAT: at production coverage, is the boundary integral SIGNAL or NOISE?

Pure numerics, no learning.  Exact double-layer (solid-angle) kernel on a
sphere.  In trace configuration the query sits ON a panel, so:

    u(x) = [own-panel jump term]  +  [sum over all OTHER panels]

Random subsampling always retains the query's own panel (trace mode declares
it) but keeps only a fraction p of the rest.  Question: at the coverage the
DrivAerML protocol actually uses (0.11%), how well is the off-panel sum --
the actual global elliptic coupling -- estimated?
"""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

torch.set_default_dtype(torch.float64)
mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
P = mesh.points[mesh.cells]                       # (N,3,3)
N = P.shape[0]
centroids = mesh.cell_centroids
areas = mesh.cell_areas

def solid_angle(x, P):
    """van Oosterom-Strackee signed solid angle of triangles P seen from x."""
    a = P[:, 0] - x; b = P[:, 1] - x; c = P[:, 2] - x
    na = a.norm(dim=-1); nb = b.norm(dim=-1); nc = c.norm(dim=-1)
    num = (a * torch.linalg.cross(b, c)).sum(-1)
    den = na*nb*nc + (a*b).sum(-1)*nc + (a*c).sum(-1)*nb + (b*c).sum(-1)*na
    return 2.0 * torch.atan2(num, den)

### Validate the kernel: total solid angle from an interior point = 4*pi.
x_in = torch.tensor([0.0, 0.0, 0.0])
tot = solid_angle(x_in, P).sum()
print(f"kernel check: total solid angle at centre = {float(tot):.10f} "
      f"(4*pi = {4*math.pi:.10f})\n")

### A smooth, non-trivial density so the integral is not a trivial constant.
sigma = 1.0 + 0.5 * centroids[:, 0] + 0.3 * centroids[:, 1] * centroids[:, 2]

def analyse(x, own, label):
    om = solid_angle(x, P)
    contrib = om * sigma
    if own is not None:
        mask = torch.ones(N, dtype=torch.bool); mask[own] = False
        own_term = float(contrib[own])
    else:
        mask = torch.ones(N, dtype=torch.bool); own_term = float("nan")
    rest_true = float(contrib[mask].sum())
    idx_rest = torch.nonzero(mask).flatten()
    M = 200
    print(f"--- {label} ---")
    if own is not None:
        print(f"own-panel term = {own_term:+.6f}   true off-panel sum = {rest_true:+.6f}")
    else:
        print(f"true total = {rest_true:+.6f}")
    print(f"{'coverage':>10} {'n kept':>8} {'est mean':>12} {'est std':>11} "
          f"{'std/|true|':>11} {'std/|own|':>11}")
    for p in (0.0011, 0.01, 0.05, 0.25, 0.75):
        n = max(1, int(round(p * idx_rest.numel())))
        g = torch.Generator().manual_seed(0)
        ests = []
        for _ in range(M):
            sel = idx_rest[torch.randperm(idx_rest.numel(), generator=g)[:n]]
            ests.append(float(contrib[sel].sum()) * (idx_rest.numel() / n))  # HT
        t = torch.tensor(ests)
        rel_t = float(t.std()) / abs(rest_true)
        rel_o = float(t.std()) / abs(own_term) if own is not None else float("nan")
        print(f"{100*p:>9.2f}% {n:>8} {float(t.mean()):>12.4f} {float(t.std()):>11.4f} "
              f"{rel_t:>10.1%} {rel_o:>10.1%}")
    print()

### Trace configuration: query ON a panel (its centroid), nudged to the
### exterior side so the kernel is finite.
own = 1234
x_s = centroids[own] * 1.0000001
analyse(x_s, own, "TRACE query (on the surface) -- the DrivAerML surface task")

### Deep interior, for contrast.
analyse(torch.tensor([0.0, 0.0, 0.0]), None, "INTERIOR query at the centre")
