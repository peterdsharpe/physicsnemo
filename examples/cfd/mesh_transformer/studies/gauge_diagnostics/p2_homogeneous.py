"""P2 prototype: norm-factored homogeneous nonlinearity, f(x) = |x| g(x/|x|).

Claim: it recovers LayerNorm's bounded internals WITHOUT breaking zero
preservation, because the norm is multiplied back rather than discarded --
giving exact degree-1 amplification with a fully nonlinear dependence on the
drive's SHAPE.  Measured against the three existing modes on the same ladder.
"""
import math, torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

D = 24
g_mlp = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.SiLU(),
                            torch.nn.Linear(64, 64), torch.nn.SiLU(),
                            torch.nn.Linear(64, D)).to(torch.float64)

def multiplicative(x, depth=3):
    """Caricature of zero_preserving_nonlinear: drive enters multiplicatively."""
    h = x
    for _ in range(depth):
        h = h * (1.0 + 0.5 * (h @ torch.eye(D, dtype=torch.float64)))
    return h

def layernorm_style(x):
    """GeoT-style: normalize and DISCARD the norm."""
    return g_mlp(x / (x.norm(dim=-1, keepdim=True) + 1e-12))

def arcsinh_globe(x, C=1.0):
    """GLOBE: radial arcsinh with a FIXED constant scale."""
    r = x.norm(dim=-1, keepdim=True) + 1e-12
    return g_mlp(x * (C * torch.arcsinh(r / C) / r))

def norm_factored(x):
    """P2: divide the norm out, act nonlinearly on direction, multiply back."""
    r = x.norm(dim=-1, keepdim=True)
    return r * g_mlp(x / (r + 1e-12))

xs = torch.randn(256, D, dtype=torch.float64)
print(f"{'scheme':>22} {'f(0)=0':>8} {'degree':>8} {'amp @2x':>9} {'amp @10x':>10} "
      f"{'|internal| @10x':>16}")
for name, fn in (("multiplicative (current)", multiplicative),
                 ("LayerNorm (GeoT-style)", layernorm_style),
                 ("GLOBE arcsinh (C=1)", lambda x: arcsinh_globe(x, 1.0)),
                 ("norm-factored (P2)", norm_factored)):
    with torch.no_grad():
        z = float(fn(torch.zeros(1, D, dtype=torch.float64)).abs().max())
        v1 = float(fn(xs).norm(dim=-1).mean())
        v2 = float(fn(2 * xs).norm(dim=-1).mean())
        v10 = float(fn(10 * xs).norm(dim=-1).mean())
        # internal magnitude the MLP actually sees at 10x input
        if name.startswith("norm-factored"):
            internal = float((10 * xs / (10 * xs).norm(dim=-1, keepdim=True)).norm(dim=-1).mean())
        elif name.startswith("LayerNorm"):
            internal = float((10 * xs / (10 * xs).norm(dim=-1, keepdim=True)).norm(dim=-1).mean())
        elif name.startswith("GLOBE"):
            r = (10 * xs).norm(dim=-1, keepdim=True)
            internal = float((10 * xs * (torch.arcsinh(r) / r)).norm(dim=-1).mean())
        else:
            internal = float((10 * xs).norm(dim=-1).mean())
        deg = math.log(v10 / v1) / math.log(10.0)
    print(f"{name:>22} {'yes' if z < 1e-12 else 'NO':>8} {deg:>8.2f} "
          f"{v2/v1:>9.2f} {v10/v1:>10.3g} {internal:>16.3g}")

print("\nnonlinearity check for P2 (must be nonlinear in SHAPE, not just a scaling):")
with torch.no_grad():
    a, b = xs[:1], xs[1:2]
    lhs = norm_factored(a + b); rhs = norm_factored(a) + norm_factored(b)
    print(f"  additivity  |f(a+b) - f(a) - f(b)| / |f(a+b)| = "
          f"{float((lhs-rhs).norm()/lhs.norm()):.3f}   (0 => merely linear)")
    lam = 3.7
    print(f"  homogeneity |f(k x) - k f(x)| / |k f(x)|      = "
          f"{float((norm_factored(lam*a) - lam*norm_factored(a)).norm()/(lam*norm_factored(a)).norm()):.2e}")
