"""INDEPENDENT verification of the forked branch's singularity-subtraction claim.

Own implementation (van Oosterom-Strackee), own sphere, own sampling.  Checks:
  (1) the solid-angle kernel is exact (sums to -4pi interior, 0 exterior);
  (2) the trace-query cancellation: own-panel jump vs PV over the rest;
  (3) whether subtraction reduces the VARIANCE of a subsampled estimator,
      and by how much, versus the claimed ~6.4x std at 0.11% coverage.
"""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

torch.set_default_dtype(torch.float64)

def solid_angles(verts, x):
    """Signed solid angle of each triangle as seen from x (van Oosterom-Strackee)."""
    a = verts[:, 0] - x; b = verts[:, 1] - x; c = verts[:, 2] - x
    na, nb, nc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = (a * torch.linalg.cross(b, c)).sum(-1)
    den = na*nb*nc + (a*b).sum(-1)*nc + (a*c).sum(-1)*nb + (b*c).sum(-1)*na
    return 2.0 * torch.atan2(num, den)

mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
verts = mesh.points[mesh.cells]
N = mesh.n_cells
print(f"sphere: {N} panels\n")

### (1) kernel validation
interior = torch.tensor([0.03, -0.02, 0.05])
exterior = torch.tensor([3.0, 1.0, -2.0])
si, se = solid_angles(verts, interior).sum(), solid_angles(verts, exterior).sum()
print(f"(1) KERNEL  sum interior = {si:+.10f}  (exact -4pi = {-4*math.pi:+.10f})")
print(f"            sum exterior = {se:+.3e}  (exact 0)")

### (2) the trace-query split
cent = mesh.cell_centroids
q = 0
x = cent[q]
eps = 1e-9 * mesh.points.new_tensor([1.0, 0.0, 0.0])
Om = solid_angles(verts, x + 0.0 * eps)
own, others = Om[q], Om.torch if False else Om.clone()
others[q] = 0.0
print(f"\n(2) TRACE SPLIT at a panel centroid")
print(f"    own-panel term                 = {own:+.4f}")
print(f"    PV over all other panels       = {others.sum():+.4f}")
print(f"    total                          = {Om.sum():+.4f}")
print(f"    ratio |own| / |total|          = {abs(float(own/Om.sum())):.1f}x")

### (3) variance of a subsampled estimator, plain vs subtracted
def sigma(p, kind="linear"):
    if kind == "linear":    return p[:, 2]
    if kind == "quadratic": return p[:, 2]**2 - p[:, 0]*p[:, 1]
    if kind == "osc6":      return torch.cos(6*p[:, 2]) * torch.sin(6*p[:, 0])
    if kind == "white":     return torch.randn(p.shape[0], generator=torch.Generator().manual_seed(5))
    raise ValueError(kind)

print(f"\n(3) SUBSAMPLED ESTIMATOR  (600 random subsamples per cell)")
print(f"{'density':>12} {'coverage':>9} {'plain std':>12} {'subtr std':>12} {'reduction':>10}")
g = torch.Generator().manual_seed(0)
for kind in ("linear", "quadratic", "osc6", "white"):
    s = sigma(cent, kind); s_x = s[q]
    exact = float((Om * s).sum())
    for cov in (0.0011, 0.35):
        n = max(4, int(round(cov * N)))
        plain, subtr = [], []
        for _ in range(600):
            idx = torch.randperm(N, generator=g)[:n]
            scale = N / n
            plain.append(scale * float((Om[idx] * s[idx]).sum()))
            # sum_j Om_j (s_j - s_x) + s_x * sum_all Om_j  (second term known exactly)
            subtr.append(scale * float((Om[idx] * (s[idx] - s_x)).sum())
                         + s_x * float(Om.sum()))
        ps = torch.tensor(plain).std(); ss = torch.tensor(subtr).std()
        if kind == "linear" or cov == 0.0011:
            print(f"{kind:>12} {100*cov:>8.2f}% {ps:>12.4f} {ss:>12.4f} "
                  f"{float(ps/ss):>9.2f}x")
