"""The constructive half: singularity subtraction as a CONTROL VARIATE.

For a closed surface, the total solid angle from a point on the exterior
side is ZERO, exactly and analytically.  So

    u = sum_j O_j s_j
      = s(x) * [sum_j O_j]  +  sum_j O_j (s_j - s(x))
      =        s(x) * 0     +  sum_j O_j (s_j - s(x)).

The rewritten integrand is small EVERYWHERE: near the query O_j is huge but
(s_j - s(x)) -> 0; far away (s_j - s(x)) is O(1) but O_j is tiny.  The
near-cancellation is removed ANALYTICALLY instead of numerically.

This is classical BEM singularity subtraction.  What is new here is the
claim that it is a VARIANCE-REDUCTION device for a subsampled operator --
and that trace mode already supplies the one thing it needs, s(x), as the
own-cell readout.
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
M = 300
print(f"{'coverage':>9} | {'PLAIN: SNR':>11} {'rel err':>9} | "
      f"{'SUBTRACTED: SNR':>16} {'rel err':>9} | {'noise reduction':>16}")
print("-"*84)
for p in (0.0011, 0.005, 0.02, 0.10, 0.35):
    A, B, C, D, R = [], [], [], [], []
    for own in (1234, 5000, 9999, 15111):
        x = centroids[own]*1.0000001
        O = solid_angle(x, P)
        s_x = float(sigma[own])
        u_true = float((O*sigma).sum())
        idx = torch.arange(N)
        n = max(1, int(round(p*N)))
        g = torch.Generator().manual_seed(own)
        plain, sub = [], []
        for _ in range(M):
            sel = idx[torch.randperm(N, generator=g)[:n]]
            scale = N/n
            plain.append(float((O[sel]*sigma[sel]).sum())*scale)
            ### Control variate: the exact term contributes s(x)*0 = 0.
            sub.append(float((O[sel]*(sigma[sel]-s_x)).sum())*scale)
        tp, ts = torch.tensor(plain), torch.tensor(sub)
        A.append(abs(u_true)/float(tp.std())); B.append(float((tp-u_true).abs().mean())/abs(u_true))
        C.append(abs(u_true)/float(ts.std())); D.append(float((ts-u_true).abs().mean())/abs(u_true))
        R.append(float(tp.std())/float(ts.std()))
    m = lambda v: sum(v)/len(v)
    print(f"{100*p:>8.2f}% | {m(A):>11.2f} {m(B):>8.0%} | {m(C):>16.2f} {m(D):>8.0%} | "
          f"{m(R):>15.1f}x")
print("\nSame kernel, same samples, same coverage -- only the algebraic")
print("arrangement of the integral differs.")
