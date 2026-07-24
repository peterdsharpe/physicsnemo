"""H-SAT, sharpened: the trace value is a near-CANCELLATION, so subsampling
noise is measured against a tiny signal.

u_trace = own-panel jump term + principal-value integral over all others.
These are large and nearly equal-and-opposite (second-kind Fredholm).  The
physical answer is their small difference.  Monte-Carlo subsampling adds
noise proportional to the LARGE terms, but the signal is the SMALL one.
"""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
torch.set_default_dtype(torch.float64)

mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
P = mesh.points[mesh.cells]; N = P.shape[0]
centroids = mesh.cell_centroids

def solid_angle(x, P):
    a = P[:,0]-x; b = P[:,1]-x; c = P[:,2]-x
    na,nb,nc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = (a*torch.linalg.cross(b,c)).sum(-1)
    den = na*nb*nc + (a*b).sum(-1)*nc + (a*c).sum(-1)*nb + (b*c).sum(-1)*na
    return 2.0*torch.atan2(num, den)

sigma = 1.0 + 0.5*centroids[:,0] + 0.3*centroids[:,1]*centroids[:,2]

print(f"{'coverage':>9} {'n':>7} | {'|own|':>8} {'|PV int|':>9} {'SIGNAL u':>9} "
      f"| {'noise std':>10} {'SNR':>8} {'rel err of u':>13}")
print("-"*88)
M = 300
for p in (0.0011, 0.005, 0.02, 0.10, 0.35, 0.80):
    snrs, rels, rows = [], [], []
    for own in (1234, 5000, 9999, 15111):     # average over query panels
        x = centroids[own]*1.0000001
        contrib = solid_angle(x, P)*sigma
        own_term = float(contrib[own])
        mask = torch.ones(N, dtype=torch.bool); mask[own]=False
        idx = torch.nonzero(mask).flatten()
        pv_true = float(contrib[mask].sum())
        u_true = own_term + pv_true
        n = max(1, int(round(p*idx.numel())))
        g = torch.Generator().manual_seed(own)
        est = []
        for _ in range(M):
            sel = idx[torch.randperm(idx.numel(), generator=g)[:n]]
            est.append(own_term + float(contrib[sel].sum())*(idx.numel()/n))
        t = torch.tensor(est)
        snrs.append(abs(u_true)/float(t.std()))
        rels.append(float((t-u_true).abs().mean())/abs(u_true))
        rows.append((abs(own_term), abs(pv_true), u_true, float(t.std())))
    a = [sum(r[i] for r in rows)/len(rows) for i in range(4)]
    print(f"{100*p:>8.2f}% {n:>7} | {a[0]:>8.3f} {a[1]:>9.3f} {a[2]:>9.4f} "
          f"| {a[3]:>10.4f} {sum(snrs)/len(snrs):>8.3f} {sum(rels)/len(rels):>12.1%}")

print("\nSNR = |true u| / std(estimate).  SNR < 1 means the sampled boundary")
print("integral carries less signal than noise.")
print("\nDrivAerML production coverage = 10,000 / 8,800,000 = 0.11%.")
