"""Is the trace-query boundary integral NOISE-DOMINATED at production coverage?

SNR = |exact value| / std(subsampled estimator), own panel retained exactly
(as trace mode does), averaged over several query panels.
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
s = cent[:, 2]                       # smooth physical density
g = torch.Generator().manual_seed(0)
QUERIES = [0, 977, 4321, 9999, 15001]
print(f"{'coverage':>9} {'n':>7} {'|signal|':>10} {'plain std':>11} {'SNR plain':>10} "
      f"{'SNR subtr':>10} {'rel err':>9}")
for cov in (0.0011, 0.02, 0.35):
    sig, sp, ss = [], [], []
    for q in QUERIES:
        x = cent[q]; Om = solid_angles(verts, x)
        s_x, T = s[q], float(Om.sum())
        exact = float((Om * s).sum())
        other = torch.cat([torch.arange(q), torch.arange(q + 1, N)])
        M = other.numel(); n = max(4, int(round(cov * M)))
        P, Sb = [], []
        for _ in range(400):
            idx = other[torch.randperm(M, generator=g)[:n]]
            sc = M / n
            P.append(float(Om[q]*s_x) + sc*float((Om[idx]*s[idx]).sum()))
            Sb.append(float(s_x)*T + sc*float((Om[idx]*(s[idx]-s_x)).sum()))
        sig.append(abs(exact)); sp.append(float(torch.tensor(P).std()))
        ss.append(float(torch.tensor(Sb).std()))
    S = sum(sig)/len(sig); PS = sum(sp)/len(sp); SS = sum(ss)/len(ss)
    n = max(4, int(round(cov*(N-1))))
    print(f"{100*cov:>8.2f}% {n:>7} {S:>10.4f} {PS:>11.4f} {S/PS:>10.2f} "
          f"{S/SS:>10.2f} {100*PS/S:>8.0f}%")
