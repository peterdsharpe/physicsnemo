"""Correct form of the singularity-subtraction variance test.

The exact identity at a trace query, with sigma_own = sigma_x:

    u(x) = sum_all Omega_j sigma_j
         = sigma_x * T  +  sum_{j != own} Omega_j (sigma_j - sigma_x),   T = sum_all Omega_j

The own panel DROPS OUT of the subtracted sum (its sigma difference is zero),
and T is known analytically.  Trace mode always retains the own panel exactly,
so the honest comparison samples only the OTHER panels:

    plain      : Omega_own * sigma_x + (M/n) sum_{j in S} Omega_j sigma_j
    subtracted : sigma_x * T         + (M/n) sum_{j in S} Omega_j (sigma_j - sigma_x)

Averaged over many query panels so the result is not one panel's sigma lottery.
"""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
torch.set_default_dtype(torch.float64)

def solid_angles(verts, x):
    a = verts[:, 0] - x; b = verts[:, 1] - x; c = verts[:, 2] - x
    na, nb, nc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = (a * torch.linalg.cross(b, c)).sum(-1)
    den = na*nb*nc + (a*b).sum(-1)*nc + (a*c).sum(-1)*nb + (b*c).sum(-1)*na
    return 2.0 * torch.atan2(num, den)

mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
verts, cent, N = mesh.points[mesh.cells], mesh.cell_centroids, mesh.n_cells

def sigma(p, kind):
    if kind == "linear":     return p[:, 2]
    if kind == "quadratic":  return p[:, 2]**2 - p[:, 0]*p[:, 1]
    if kind == "osc6":       return torch.cos(6*p[:, 2]) * torch.sin(6*p[:, 0])
    if kind == "osc16":      return torch.cos(16*p[:, 2]) * torch.sin(16*p[:, 0])
    if kind == "white":
        return torch.randn(p.shape[0], generator=torch.Generator().manual_seed(5))

g = torch.Generator().manual_seed(0)
QUERIES = [0, 977, 4321, 9999, 15001]      # several panels, average the outcome
TRIALS  = 400
print(f"{N} panels; {len(QUERIES)} query panels x {TRIALS} subsamples each\n")
print(f"{'density':>10} {'coverage':>9} {'plain std':>11} {'subtr std':>11} {'reduction':>10}")
for kind in ("linear", "quadratic", "osc6", "osc16", "white"):
    s = sigma(cent, kind)
    for cov in (0.0011, 0.02, 0.35):
        ratios = []
        for q in QUERIES:
            x = cent[q]; Om = solid_angles(verts, x)
            s_x = s[q]; T = float(Om.sum())
            other = torch.cat([torch.arange(q), torch.arange(q + 1, N)])
            M = other.numel(); n = max(4, int(round(cov * M)))
            P, Sb = [], []
            for _ in range(TRIALS):
                idx = other[torch.randperm(M, generator=g)[:n]]
                sc = M / n
                P.append(float(Om[q] * s_x) + sc * float((Om[idx] * s[idx]).sum()))
                Sb.append(float(s_x) * T + sc * float((Om[idx] * (s[idx] - s_x)).sum()))
            ps, ss = torch.tensor(P).std(), torch.tensor(Sb).std()
            ratios.append(float(ps / ss))
        r = sum(ratios) / len(ratios)
        print(f"{kind:>10} {100*cov:>8.2f}% {'':>11} {'':>11} {r:>9.2f}x")
