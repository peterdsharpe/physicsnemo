"""P2 on the REAL typed machinery: does f(F) = |F| g(F/|F|) keep every
contract when F is an actual ScalarVectorState?

Validates the design before wiring it into block.py:
  - equivariance (the norm is an O(D) invariant; g acts on a typed state)
  - zero preservation, EXACT and independent of g's biases
  - degree exactly 1 in the drive
  - genuine nonlinearity (not secretly linear)
"""
import math, torch
from physicsnemo.experimental.nn.mesh_attention import ScalarVectorState, TypedProjection
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

CS, CV, D, N = 8, 4, 3, 32
g = TypedProjection(scalar_in=CS, vector_in=CV, scalar_out=CS, vector_out=CV, scalar_bias=True).to(torch.float64)

def state_norm(s):
    """O(D)-invariant magnitude of a typed state, per entity."""
    sq = s.scalars.square().sum(-1) + s.vectors.square().sum((-1, -2))
    return sq.clamp_min(0).sqrt()

def homogeneous(s, eps=1e-30):
    r = state_norm(s)
    inv = 1.0 / (r + eps)
    unit = ScalarVectorState(scalars=s.scalars * inv[:, None],
                             vectors=s.vectors * inv[:, None, None])
    out = g(unit)
    return ScalarVectorState(scalars=out.scalars * r[:, None],
                             vectors=out.vectors * r[:, None, None])

def rand_state(scale=1.0):
    return ScalarVectorState(scalars=scale * torch.randn(N, CS, dtype=torch.float64),
                             vectors=scale * torch.randn(N, CV, D, dtype=torch.float64))

F = rand_state()
with torch.no_grad():
    # 1. zero preservation, exact, despite g having biases
    z = homogeneous(ScalarVectorState(scalars=torch.zeros(N, CS, dtype=torch.float64),
                                      vectors=torch.zeros(N, CV, D, dtype=torch.float64)))
    zero_err = max(float(z.scalars.abs().max()), float(z.vectors.abs().max()))

    # 2. degree exactly 1
    lam = 7.3
    Fl = ScalarVectorState(scalars=lam * F.scalars, vectors=lam * F.vectors)
    a, b = homogeneous(Fl), homogeneous(F)
    hom_err = float((a.scalars - lam * b.scalars).abs().max()
                    / (lam * b.scalars).abs().max())

    # 3. O(D) equivariance: rotate vectors, scalars invariant
    th = 0.9
    R = torch.tensor([[math.cos(th), -math.sin(th), 0],
                      [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=torch.float64)
    Fr = ScalarVectorState(scalars=F.scalars, vectors=F.vectors @ R.T)
    c = homogeneous(Fr)
    eq_s = float((c.scalars - b.scalars).abs().max() / b.scalars.abs().max())
    eq_v = float((c.vectors - b.vectors @ R.T).abs().max() / b.vectors.abs().max())

    # 4. genuinely nonlinear
    G = rand_state()
    s1 = homogeneous(ScalarVectorState(scalars=F.scalars + G.scalars,
                                       vectors=F.vectors + G.vectors))
    s2 = homogeneous(G)
    add_err = float((s1.scalars - b.scalars - s2.scalars).abs().max()
                    / s1.scalars.abs().max())

print(f"{'contract':>34} {'measured':>14}   verdict")
print(f"{'zero preservation f(0)=0':>34} {zero_err:>14.2e}   "
      f"{'EXACT' if zero_err == 0 else 'nonzero'}")
print(f"{'degree-1 homogeneity':>34} {hom_err:>14.2e}   "
      f"{'holds' if hom_err < 1e-12 else 'FAILS'}")
print(f"{'O(D) equivariance (scalars)':>34} {eq_s:>14.2e}   "
      f"{'holds' if eq_s < 1e-12 else 'FAILS'}")
print(f"{'O(D) equivariance (vectors)':>34} {eq_v:>14.2e}   "
      f"{'holds' if eq_v < 1e-12 else 'FAILS'}")
print(f"{'nonlinearity (additivity gap)':>34} {add_err:>14.3f}   "
      f"{'nonlinear' if add_err > 1e-3 else 'SECRETLY LINEAR'}")
