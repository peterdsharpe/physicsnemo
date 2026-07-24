"""Honest falsifier: the control variate works because (s_j - s(x)) is small
near the query.  That relies on s being SMOOTH.  How fast does the gain
decay as the density gets rougher?  A learned coefficient field need not be
as smooth as an analytic one.
"""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
torch.set_default_dtype(torch.float64)
mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
P = mesh.points[mesh.cells]; N = P.shape[0]; C = mesh.cell_centroids

def solid_angle(x,P):
    a=P[:,0]-x; b=P[:,1]-x; c=P[:,2]-x
    na,nb,nc=a.norm(dim=-1),b.norm(dim=-1),c.norm(dim=-1)
    num=(a*torch.linalg.cross(b,c)).sum(-1)
    den=na*nb*nc+(a*b).sum(-1)*nc+(a*c).sum(-1)*nb+(b*c).sum(-1)*na
    return 2.0*torch.atan2(num,den)

g0 = torch.Generator().manual_seed(1)
white = torch.randn(N, generator=g0)
### Densities from very smooth (degree-1) to pure white noise.
densities = {
    "linear (very smooth)":      1.0 + 0.5*C[:,0],
    "quadratic":                 1.0 + 0.5*C[:,0] + 0.3*C[:,1]*C[:,2],
    "degree-6 oscillatory":      1.0 + 0.5*torch.cos(6*torch.atan2(C[:,1],C[:,0]))*(1-C[:,2]**2),
    "degree-16 oscillatory":     1.0 + 0.5*torch.cos(16*torch.atan2(C[:,1],C[:,0]))*(1-C[:,2]**2),
    "smooth + 20% white noise":  1.0 + 0.5*C[:,0] + 0.2*white,
    "pure white noise":          1.0 + white,
}
M, p = 300, 0.0011      # production coverage
print(f"at DrivAerML production coverage ({100*p:.2f}%), trace configuration\n")
print(f"{'density':>26} | {'plain SNR':>10} {'sub SNR':>8} {'variance red.':>14}")
print("-"*66)
for name, sig in densities.items():
    A,Bb,R = [],[],[]
    for own in (1234, 5000, 9999, 15111, 20000):
        x = C[own]*1.0000001; O = solid_angle(x,P); s_x=float(sig[own])
        u_true = float((O*sig).sum())
        rest = torch.nonzero(torch.arange(N)!=own).flatten()
        n = max(1,int(round(p*rest.numel())))
        gg = torch.Generator().manual_seed(own)
        pl,su = [],[]
        for _ in range(M):
            sel = rest[torch.randperm(rest.numel(), generator=gg)[:n]]
            sc = rest.numel()/n
            pl.append(float(O[own]*sig[own]) + float((O[sel]*sig[sel]).sum())*sc)
            su.append(0.0                    + float((O[sel]*(sig[sel]-s_x)).sum())*sc)
        tp,ts = torch.tensor(pl), torch.tensor(su)
        A.append(abs(u_true)/float(tp.std())); Bb.append(abs(u_true)/float(ts.std()))
        R.append(float(tp.std())/float(ts.std()))
    m=lambda v: sum(v)/len(v)
    print(f"{name:>26} | {m(A):>10.2f} {m(Bb):>8.2f} {m(R):>13.1f}x")
