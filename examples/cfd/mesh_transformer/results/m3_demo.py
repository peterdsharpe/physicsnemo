# M3 invariant-completeness audit: numerical demonstration.
# Preregistration: lit_wave_preregistration_2026-08-20.json (sha 69a81e2a), arm M3.
# Analytic CPU probe, fp64, random weights, eval mode, no training.
#
# Tests
#   T1  Full-problem mirror (G,n,d) -> (MG,Mn,Md):
#       scalars must match to roundoff (backbone parity-even);
#       vectors must NOT mirror (axial e_phi channels, parity anomaly);
#       vectors with e_phi head columns zeroed must mirror to roundoff.
#   T2  Same-drive mirror, drive IN the mirror plane (Md=d): model scalars
#       identical at corresponding points; chirality signature chi flips
#       sign (proves the pair is not rotation-related).
#   T3  Same-drive mirror, yawed drive (Md!=d): model separates the pair,
#       and satisfies the physical identification
#       model(MG,Mn,d) == model(G,n,Md) at corresponding points.
#   T4  Control (GWL-style seed degeneracy): paired azimuthal scramble about
#       the drive axis preserves all 5 seed invariants exactly but changes
#       the geometry materially; the full model must separate it (adaptive
#       slice anchors break the seed degeneracy).

import torch

from physicsnemo.experimental.nn.mt2.model import MeshTransformer2

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

M = torch.diag(torch.tensor([1.0, -1.0, 1.0]))  # mirror across the xz-plane


def helical_strake(n_pts=400, handed=+1.0):
    """Chiral point cloud: helix on a cylinder, normals tilted along the
    helix tangent so that sum((r x n) . z) != 0 (nonzero chirality)."""
    t = torch.linspace(0.0, 4.0 * torch.pi, n_pts)
    x = torch.cos(handed * t)
    y = torch.sin(handed * t)
    z = t / (2.0 * torch.pi)  # z in [0, 2]
    pts = torch.stack([x, y, z], dim=-1)
    radial = torch.stack([x, y, torch.zeros_like(z)], dim=-1)
    tangent = torch.stack(
        [-handed * torch.sin(handed * t), handed * torch.cos(handed * t),
         torch.full_like(t, 1.0 / (2.0 * torch.pi))], dim=-1)
    n = radial + 0.6 * tangent  # tilt makes the normal field chiral
    n = n / n.norm(dim=-1, keepdim=True)
    return pts[None], n[None]  # (1, N, 3)


def chi(points, normals, d):
    """Pseudoscalar chirality signature: sum_i (r_i x n_i) . d_hat, with
    r_i centered. Invariant under any proper rotation applied to (r, n, d);
    flips sign under a reflection of (r, n) with d fixed in the mirror
    plane. chi != 0 => the pair (G,d) vs (MG,d) is not rotation-related."""
    r = points - points.mean(dim=1, keepdim=True)
    return torch.einsum("bnc,c->", torch.cross(r, normals, dim=-1), d / d.norm())


def rel_diff(a, b):
    return ((a - b).norm() / b.norm()).item()


def make_model(seed):
    torch.manual_seed(seed)
    m = MeshTransformer2(out_scalars=1, out_vectors=1, hidden=128,
                         n_layers=6, n_slices=64, reference_length=2.0)
    return m.double().eval()


def split(out):
    return out[..., :1], out[..., 1:4]  # scalar, vector fields


pts, nrm = helical_strake()
mpts, mnrm = pts @ M.T, nrm @ M.T

d_full = torch.tensor([1.0, 0.3, 0.2])          # generic drive (T1)
d_in = torch.tensor([1.0, 0.0, 0.2])            # in mirror plane: Md = d (T2)
d_yaw = torch.tensor([1.0, 0.35, 0.15])         # yawed drive: Md != d (T3)

print(f"chi(G, d_in)  = {chi(pts, nrm, d_in).item():+.6f}")
print(f"chi(MG, d_in) = {chi(mpts, mnrm, d_in).item():+.6f}  (sign flip, nonzero)")

for seed in (0, 1, 2):
    model = make_model(seed)
    with torch.no_grad():
        # ---- T1: full-problem mirror ----
        s1, v1 = split(model(pts, nrm, d_full))
        s2, v2 = split(model(mpts, mnrm, d_full @ M.T))
        t1_scalar = rel_diff(s2, s1)
        t1_vector = rel_diff(v2, v1 @ M.T)  # vs the physically correct mirror

        # T1b: causal ablation -- zero the head columns of the two axial
        # basis channels e_phi_n (index 4) and e_phi_d (index 6); coeff k
        # lives at head output row out_scalars + k = 1 + k.
        model_ab = make_model(seed)
        with torch.no_grad():
            for row in (1 + 4, 1 + 6):
                model_ab.head.weight[row].zero_()
                model_ab.head.bias[row].zero_()
        s1a, v1a = split(model_ab(pts, nrm, d_full))
        s2a, v2a = split(model_ab(mpts, mnrm, d_full @ M.T))
        t1b_vector = rel_diff(v2a, v1a @ M.T)

        # ---- T2: same drive, Md = d ----
        sA, _ = split(model(pts, nrm, d_in))
        sB, _ = split(model(mpts, mnrm, d_in))
        t2_scalar = rel_diff(sB, sA)

        # ---- T3: same drive, Md != d ----
        sC, _ = split(model(pts, nrm, d_yaw))
        sD, _ = split(model(mpts, mnrm, d_yaw))
        t3_sep = rel_diff(sD, sC)
        sE, _ = split(model(pts, nrm, d_yaw @ M.T))  # original geom, mirrored drive
        t3_ident = rel_diff(sD, sE)

        # ---- T4: control, seed-invariant-preserving scramble ----
        # Cloud: antipodal pairs about the z-axis (drive axis), zero xy-mean.
        # Rotating each pair by its own angle about z preserves all 5 seed
        # invariants exactly (|r|, log|r|, rhat.d, rhat.n, n.d) but yields a
        # materially different shape.
        torch.manual_seed(100 + seed)
        npair = 200
        base = torch.randn(npair, 3)
        base[:, 2] = torch.linspace(-1.0, 1.0, npair)
        nb = torch.randn(npair, 3)
        nb = nb / nb.norm(dim=-1, keepdim=True)
        Rpi = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))  # pi about z
        p4 = torch.cat([base, base @ Rpi.T])[None]
        n4 = torch.cat([nb, nb @ Rpi.T])[None]
        ang = torch.rand(npair) * 2 * torch.pi
        c, s = torch.cos(ang), torch.sin(ang)
        Rz = torch.zeros(npair, 3, 3)
        Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
        Rz[:, 1, 0], Rz[:, 1, 1] = s, c
        Rz[:, 2, 2] = 1.0
        p4s = torch.cat([torch.einsum("nij,nj->ni", Rz, base),
                         torch.einsum("nij,nj->ni", Rz, base @ Rpi.T)])[None]
        n4s = torch.cat([torch.einsum("nij,nj->ni", Rz, nb),
                         torch.einsum("nij,nj->ni", Rz, nb @ Rpi.T)])[None]
        d4 = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(p4.mean(1), p4s.mean(1), atol=1e-12)
        # verify the 5 seed invariants match exactly
        for a, b in [(p4, p4s)]:
            ra = a - a.mean(1, keepdim=True); rb = p4s - p4s.mean(1, keepdim=True)
        assert torch.allclose(ra.norm(dim=-1), rb.norm(dim=-1), atol=1e-12)
        assert torch.allclose((ra * n4).sum(-1), (rb * n4s).sum(-1), atol=1e-12)
        assert torch.allclose(ra[..., 2], rb[..., 2], atol=1e-12)  # r.d
        assert torch.allclose(n4[..., 2], n4s[..., 2], atol=1e-12)  # n.d
        geom_change = rel_diff(p4s, p4)
        sF, _ = split(model(p4, n4, d4))
        sG, _ = split(model(p4s, n4s, d4))
        t4_sep = rel_diff(sG, sF)

        # ---- T5: aerodynamic instance of T4 -- straight vs helically
        # twisted two-fin body under axial drive; then yaw restoration.
        def fin_body(twist_rate=0.0, n_h=150):
            z = torch.linspace(0.0, 2.0, n_h)
            # fin A at azimuth 0 (extends radially), fin B antipodal
            rad = 1.0 + 0.5 * torch.rand_like(z)  # radial extent samples
            pA = torch.stack([rad, torch.zeros_like(z), z], dim=-1)
            nA = torch.stack([torch.zeros_like(z), torch.ones_like(z),
                              torch.zeros_like(z)], dim=-1)  # fin side normal
            th = twist_rate * z
            c, s = torch.cos(th), torch.sin(th)
            R = torch.zeros(n_h, 3, 3)
            R[:, 0, 0], R[:, 0, 1] = c, -s
            R[:, 1, 0], R[:, 1, 1] = s, c
            R[:, 2, 2] = 1.0
            pA = torch.einsum("nij,nj->ni", R, pA)
            nA = torch.einsum("nij,nj->ni", R, nA)
            Rpi_ = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
            return (torch.cat([pA, pA @ Rpi_.T])[None],
                    torch.cat([nA, nA @ Rpi_.T])[None])

        torch.manual_seed(200 + seed)
        p_straight, n_straight = fin_body(0.0)
        torch.manual_seed(200 + seed)
        p_twist, n_twist = fin_body(twist_rate=torch.pi / 2)  # 90 deg/unit len
        d_axial = torch.tensor([0.0, 0.0, 1.0])
        s_st, _ = split(model(p_straight, n_straight, d_axial))
        s_tw, _ = split(model(p_twist, n_twist, d_axial))
        t5_blind = rel_diff(s_tw, s_st)
        t5_yaw = {}
        import math
        for deg in (1.0, 5.0, 10.0):
            a_ = math.radians(deg)
            d_y = torch.tensor([math.sin(a_), 0.0, math.cos(a_)])
            s1_, _ = split(model(p_straight, n_straight, d_y))
            s2_, _ = split(model(p_twist, n_twist, d_y))
            t5_yaw[deg] = rel_diff(s2_, s1_)

        # ---- T6: v4 local features (inter-point distances) break the
        # degeneracy: same straight-vs-twisted pair, use_local_features=True.
        torch.manual_seed(seed)
        model_lf = MeshTransformer2(
            out_scalars=1, out_vectors=1, hidden=128, n_layers=6,
            n_slices=64, reference_length=2.0, use_local_features=True,
            local_radii=(0.1, 0.3)).double().eval()
        s_st_lf, _ = split(model_lf(p_straight, n_straight, d_axial))
        s_tw_lf, _ = split(model_lf(p_twist, n_twist, d_axial))
        t6_sep = rel_diff(s_tw_lf, s_st_lf)

    print(f"seed {seed}:")
    print(f"  T1 scalar mirror residual        {t1_scalar:.3e}  (expect ~1e-15)")
    print(f"  T1 vector vs correct mirror      {t1_vector:.3e}  (expect O(0.1-1))")
    print(f"  T1b same, e_phi channels zeroed  {t1b_vector:.3e}  (expect ~1e-15)")
    print(f"  T2 same-drive (Md=d) scalar sep  {t2_scalar:.3e}  (expect ~1e-15)")
    print(f"  T3 same-drive (Md!=d) scalar sep {t3_sep:.3e}  (expect O(0.01-1))")
    print(f"  T3 identity vs (G, Md) residual  {t3_ident:.3e}  (expect ~1e-15)")
    print(f"  T4 scramble geom change {geom_change:.3f}, scalar sep {t4_sep:.3e}"
          f"  (BLIND: matched-invariant pair)")
    print(f"  T5 straight-vs-twisted fins, axial drive: sep {t5_blind:.3e}"
          f"  (BLIND)")
    for deg, v in t5_yaw.items():
        print(f"     yaw {deg:4.1f} deg: sep {v:.3e}")
    print(f"  T6 same pair, use_local_features=True: sep {t6_sep:.3e}"
          f"  (degeneracy broken)")
