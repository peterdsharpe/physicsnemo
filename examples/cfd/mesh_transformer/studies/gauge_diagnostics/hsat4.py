"""Apples-to-apples in the ACTUAL trace configuration: own panel always
retained (that is what trace_of declares), both arms, same samples."""
import math, torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
torch.set_default_dtype(torch.float64)
mesh = sphere_icosahedral.load(radius=1.0, subdivisions=5)
P = mesh.points[mesh.cells]; N = P.shape[0]
centroids = mesh.cell_centroids

def solid_angle(x, P):
    a=P[:,0]-x; b=P[:,1]-x; c=P[:,2]-x
    na,nb,nc=a.norm(dim=-1),b.norm(dim=-1),c.norm(dim=-1)
    num=(a*torch.linalg.cross(b,c)).sum(-1)
    den=na*nb*nc+(a*b).sum(-1)*nc+(a*c).sum(-1)*nb+(b*c).sum(-1)*na
    return 2.0*torch.atan2(num,den)

sigma = 1.0 + 0.5*centroids[:,0] + 0.3*centroids[:,1]*centroids[:,2]
M = 400
print("TRACE configuration: own panel always retained (as trace_of declares).")
print(f"{'coverage':>9} {'n':>7} | {'plain SNR':>10} {'plain err':>10} | "
      f"{'subtracted SNR':>15} {'sub err':>9} | {'variance red.':>14}")
print("-"*86)
for p in (0.0011, 0.005, 0.02, 0.10, 0.35):
    A,B,C,D,R = [],[],[],[],[]
    for own in (1234, 5000, 9999, 15111, 20000):
        x = centroids[own]*1.0000001
        O = solid_angle(x,P); s_x = float(sigma[own])
        u_true = float((O*sigma).sum())
        own_plain = float(O[own]*sigma[own])
        own_sub   = float(O[own]*(sigma[own]-s_x))      # == 0 by construction
        rest = torch.nonzero(torch.arange(N)!=own).flatten()
        n = max(1,int(round(p*rest.numel())))
        g = torch.Generator().manual_seed(own)
        pl, su = [], []
        for _ in range(M):
            sel = rest[torch.randperm(rest.numel(), generator=g)[:n]]
            sc = rest.numel()/n
            pl.append(own_plain + float((O[sel]*sigma[sel]).sum())*sc)
            su.append(own_sub   + float((O[sel]*(sigma[sel]-s_x)).sum())*sc)
        tp,ts = torch.tensor(pl), torch.tensor(su)
        A.append(abs(u_true)/float(tp.std())); B.append(float((tp-u_true).abs().mean())/abs(u_true))
        C.append(abs(u_true)/float(ts.std())); D.append(float((ts-u_true).abs().mean())/abs(u_true))
        R.append(float(tp.std())/float(ts.std()))
    m=lambda v: sum(v)/len(v)
    print(f"{100*p:>8.2f}% {n:>7} | {m(A):>10.2f} {m(B):>9.0%} | {m(C):>15.2f} {m(D):>8.0%} | {m(R):>13.1f}x")
print("\nBoth arms: identical kernel, identical panel samples, identical coverage.")
print("The only difference is subtracting s(x) * (total solid angle = 0).")
