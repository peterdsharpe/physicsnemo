# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the hierarchical, equivariant MeshAttention layer and block.

Key correctness properties exercised here:

- **Convergence oracle**: the dual-tree forward at ``theta = 0`` reproduces the
  dense brute-force reference to floating-point precision (every interaction is
  near/exact), validating the entire four-phase hierarchical machinery.
- **Equivariance**: the (exact, dense) layer is O(D)-equivariant - scalar
  outputs are invariant and vector outputs co-rotate/reflect.
- **Discretization-invariance**: as an area-weighted quadrature of an integral
  operator, the output is invariant to refining the source discretization.
- **Gradients**: autograd through the tree matches the dense reference at
  ``theta = 0`` (the tree only *selects* near/far pairs; the math is exact).
"""

import pytest
import torch

from physicsnemo.experimental.nn import MeshAttention, MeshTransformerBlock
from physicsnemo.mesh.spatial import ClusterTree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inputs(n, scalar_dim, vector_dim, n_dims, device, dtype, seed=0):
    """Random (scalars, vectors, positions, areas) on a given device/dtype."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    scalars = torch.randn(n, scalar_dim, generator=g, dtype=dtype).to(device)
    vectors = (
        torch.randn(n, vector_dim, n_dims, generator=g, dtype=dtype).to(device)
        if vector_dim > 0
        else None
    )
    positions = torch.randn(n, n_dims, generator=g, dtype=dtype).to(device)
    areas = (torch.rand(n, generator=g, dtype=dtype) + 0.5).to(device)
    return scalars, vectors, positions, areas


def _orthogonal(n_dims, reflection, dtype, device, seed=1):
    """A random orthogonal matrix with det +1 (rotation) or -1 (reflection)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n_dims, n_dims, generator=g, dtype=dtype))
    det = torch.det(q)
    want_negative = reflection
    if (det < 0) != want_negative:
        q[:, 0] = -q[:, 0]
    return q.to(device)


# ---------------------------------------------------------------------------
# Forward / shapes / configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vector_dim", [0, 2])
def test_forward_shapes_and_finiteness(device, vector_dim):
    """Forward returns correct shapes and finite values; vectors optional."""
    torch.manual_seed(0)
    layer = MeshAttention(scalar_dim=12, vector_dim=vector_dim, heads=4, dim_head=8).to(
        device
    )
    s, v, p, a = _inputs(150, 12, vector_dim, 3, device, torch.float32)
    out_s, out_v = layer(s, p, v, a, theta=1.0)
    assert out_s.shape == (150, 12)
    assert torch.isfinite(out_s).all()
    if vector_dim > 0:
        assert out_v.shape == (150, vector_dim, 3)
        assert torch.isfinite(out_v).all()
    else:
        assert out_v is None


def test_public_attributes():
    """Public constructor attributes are recorded (lightweight contract test)."""
    layer = MeshAttention(
        scalar_dim=8,
        vector_dim=3,
        heads=2,
        dim_head=16,
        qk_norm="cosine",
        far_field="m0",
        mass_normalize=True,
    )
    assert layer.scalar_dim == 8
    assert layer.vector_dim == 3
    assert layer.heads == 2
    assert layer.dim_head == 16
    assert layer.qk_norm == "cosine"
    assert layer.far_field == "m0"
    assert layer.mass_normalize is True
    assert layer.vector_invariants is True


def test_invalid_configurations():
    """Constructor rejects inconsistent or unsupported options."""
    with pytest.raises(ValueError, match="out_vector_dim must be 0"):
        MeshAttention(scalar_dim=8, vector_dim=0, out_vector_dim=2)
    with pytest.raises(ValueError, match="qk_norm"):
        MeshAttention(scalar_dim=8, qk_norm="softmax")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="m0\\+m1"):
        MeshAttention(scalar_dim=8, vector_dim=2, far_field="m0+m1")


# ---------------------------------------------------------------------------
# Convergence oracle: hierarchical(theta=0) == dense reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qk_norm", ["layernorm", "cosine", "none"])
@pytest.mark.parametrize("mass_normalize", [False, True])
@pytest.mark.parametrize("vector_invariants", [False, True])
def test_convergence_to_dense_at_theta_zero(
    device, qk_norm, mass_normalize, vector_invariants
):
    """At theta=0 the dual-tree forward equals the dense O(N^2) reference."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10,
            vector_dim=2,
            heads=4,
            dim_head=8,
            qk_norm=qk_norm,
            mass_normalize=mass_normalize,
            vector_invariants=vector_invariants,
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(80, 10, 2, 3, device, torch.float64)
    ref_s, ref_v = layer.forward_reference(s, p, v, a)
    hier_s, hier_v = layer(s, p, v, a, theta=0.0)
    assert (ref_s - hier_s).abs().max() < 1e-8
    assert (ref_v - hier_v).abs().max() < 1e-8


def test_hierarchical_accuracy_improves_as_theta_decreases(device):
    """Approximation error decreases monotonically toward the dense result."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(120, 10, 2, 3, device, torch.float64)
    ref_s, _ = layer.forward_reference(s, p, v, a)

    def rel_err(theta):
        out_s, _ = layer(s, p, v, a, theta=theta)
        return ((out_s - ref_s).norm() / ref_s.norm()).item()

    errs = [rel_err(t) for t in (2.0, 1.0, 0.5, 0.1)]
    assert all(torch.isfinite(torch.tensor(e)) for e in errs)
    # Smaller theta -> more exact near pairs -> smaller error. (The absolute
    # error floor here reflects the M0 far-field on spatially-incoherent random
    # data; exactness is only guaranteed at theta=0, covered by the convergence
    # test above. On smooth fields the monopole error is far smaller.)
    assert errs[-1] < errs[0]
    assert errs[-1] < 0.15


@pytest.mark.parametrize("mass_normalize", [False, True])
def test_far_field_phases_exact_with_constant_envelope(device, mass_normalize):
    """Validate the nf/fn/far broadcast phases, which theta=0 never exercises.

    At theta=0 every interaction is near, so the convergence test above only
    covers the near phase. Here we make the cluster monopole *exact* by using a
    constant envelope (a huge length scale => g == 1 everywhere) and dropping
    the content term (content_gain_init=0, the only thing the far field omits).
    The hierarchical forward must then equal the dense reference at theta>0,
    which routes through - and so validates - the (near,far), (far,near), and
    (far,far) broadcasts plus the source-coverage / no-double-count property.
    With ``mass_normalize=True`` the appended unit mass column rides the same
    far-field monopoles, so this also validates the hierarchical envelope-mass
    accumulation (which theta=0 never routes through the far phases).
    """
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10,
            vector_dim=2,
            heads=4,
            dim_head=8,
            content_gain_init=0.0,  # a = 0 -> far field drops nothing
            baseline_gain_init=1.0,
            lengthscale_init=1e11,  # g == 1 exactly in fp64 -> monopole exact
            mass_normalize=mass_normalize,
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(160, 10, 2, 3, device, torch.float64)
    ref_s, ref_v = layer.forward_reference(s, p, v, a)

    tree = ClusterTree.from_points(p, areas=a)
    saw_far = False
    for theta in (0.5, 1.0, 2.0):
        plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=theta)
        saw_far = saw_far or (plan.n_far_nodes + plan.n_nf + plan.n_fn) > 0
        out_s, out_v = layer(s, p, v, a, source_tree=tree, target_tree=tree, plan=plan)
        assert (out_s - ref_s).abs().max() < 1e-9, theta
        assert (out_v - ref_v).abs().max() < 1e-9, theta
    assert saw_far, "no far-field phase was exercised; increase N or theta"


def test_far_field_with_expand_far_targets_plan(device):
    """A plan built with ``expand_far_targets=True`` is consumed correctly.

    Same constant-envelope construction as above, but the (far, far) node
    pairs are expanded into (near, far) target-point/source-node pairs at plan
    time, exercising MeshAttention's nf phase as the sole far-field carrier.
    """
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10,
            vector_dim=2,
            heads=4,
            dim_head=8,
            content_gain_init=0.0,
            baseline_gain_init=1.0,
            lengthscale_init=1e11,
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(160, 10, 2, 3, device, torch.float64)
    ref_s, ref_v = layer.forward_reference(s, p, v, a)

    tree = ClusterTree.from_points(p, areas=a)
    plan = tree.find_dual_interaction_pairs(
        target_tree=tree, theta=1.0, expand_far_targets=True
    )
    assert plan.n_far_nodes == 0  # all (far,far) entries were expanded
    assert plan.n_nf > 0
    out_s, out_v = layer(s, p, v, a, source_tree=tree, target_tree=tree, plan=plan)
    assert (out_s - ref_s).abs().max() < 1e-9
    assert (out_v - ref_v).abs().max() < 1e-9


def test_cross_attention_hierarchical_matches_dense_at_theta_zero(device):
    """The hierarchical path with distinct source/target trees is exact at theta=0.

    The other convergence tests only exercise the self-attention tree path;
    this one routes through the cross-attention plumbing (separate target tree
    built without areas, target-centroid aggregation, distinct n_tgt != n_src).
    """
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(90, 10, 2, 3, device, torch.float64, seed=0)
    qs, qv, qp, _ = _inputs(37, 10, 2, 3, device, torch.float64, seed=7)

    ref_s, ref_v = layer.forward_reference(
        s, p, v, a, query_scalars=qs, query_vectors=qv, query_positions=qp
    )
    hier_s, hier_v = layer(
        s,
        p,
        v,
        a,
        query_scalars=qs,
        query_vectors=qv,
        query_positions=qp,
        theta=0.0,
    )
    assert (ref_s - hier_s).abs().max() < 1e-8
    assert (ref_v - hier_v).abs().max() < 1e-8


def test_query_positions_without_query_scalars_raises(device):
    """Cross-attention without query_scalars must fail loudly, not silently.

    Without this guard the call is treated as self-attention: the tree is
    built over the *source* positions while target indices are applied to the
    given query positions - silently wrong when the lengths happen to match.
    """
    torch.manual_seed(0)
    layer = MeshAttention(scalar_dim=10, vector_dim=2, heads=2, dim_head=8).to(device)
    s, v, p, a = _inputs(50, 10, 2, 3, device, torch.float32)
    qp = torch.randn(50, 3, device=device)  # same length: the dangerous case
    with pytest.raises(ValueError, match="query_scalars"):
        layer(s, p, v, a, query_positions=qp)
    with pytest.raises(ValueError, match="query_scalars"):
        layer.forward_reference(s, p, v, a, query_positions=qp)
    with pytest.raises(ValueError, match="query_scalars"):
        layer(s, p, v, a, query_vectors=v)


def test_pure_bf16_forward(device):
    """A plain ``.bfloat16()`` module (no autocast) runs the hierarchical path.

    The hierarchical accumulators are promoted to >= fp32 internally; the
    result must be cast back so the half-precision output projections accept
    it.
    """
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=2, dim_head=8)
        .to(device)
        .bfloat16()
    )
    s, v, p, a = _inputs(80, 10, 2, 3, device, torch.float32)
    s, v, p, a = s.bfloat16(), v.bfloat16(), p.bfloat16(), a.bfloat16()
    out_s, out_v = layer(s, p, v, a, theta=1.0)
    assert out_s.dtype == torch.bfloat16 and torch.isfinite(out_s.float()).all()
    assert out_v.dtype == torch.bfloat16 and torch.isfinite(out_v.float()).all()
    ref_s, ref_v = layer.forward_reference(s, p, v, a)
    assert ref_s.dtype == torch.bfloat16


def test_precomputed_source_aggregates_roundtrip(device):
    """compute_source_aggregates reproduces the on-the-fly forward exactly."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10, vector_dim=2, heads=4, dim_head=8, mass_normalize=True
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(120, 10, 2, 3, device, torch.float64)
    tree = ClusterTree.from_points(p, areas=a)
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=1.0)

    out0_s, out0_v = layer(s, p, v, a, source_tree=tree, target_tree=tree, plan=plan)
    agg = layer.compute_source_aggregates(s, p, v, a, source_tree=tree)
    out1_s, out1_v = layer(
        s,
        p,
        v,
        a,
        source_tree=tree,
        target_tree=tree,
        plan=plan,
        source_aggregates=agg,
    )
    assert (out0_s - out1_s).abs().max() == 0.0
    assert (out0_v - out1_v).abs().max() == 0.0

    # A mismatched aggregate (wrong layer settings -> wrong value width) is
    # rejected instead of silently producing garbage.
    other = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )  # mass_normalize=False -> no mass column
    bad_agg = other.compute_source_aggregates(s, p, v, a, source_tree=tree)
    with pytest.raises(ValueError, match="source_aggregates"):
        layer(
            s,
            p,
            v,
            a,
            source_tree=tree,
            target_tree=tree,
            plan=plan,
            source_aggregates=bad_agg,
        )


def test_source_tree_built_with_different_areas_raises(device):
    """A precomputed tree whose build-time areas mismatch is rejected.

    The far-field monopole is scaled by the tree's build-time total area, so
    this mismatch silently mis-scales the entire far field if allowed through.
    """
    torch.manual_seed(0)
    layer = MeshAttention(scalar_dim=10, vector_dim=2, heads=2, dim_head=8).to(device)
    s, v, p, a = _inputs(60, 10, 2, 3, device, torch.float32)
    tree = ClusterTree.from_points(p)  # built with default areas == ones
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=1.0)
    with pytest.raises(ValueError, match="different areas"):
        layer(s, p, v, 2.0 * a, source_tree=tree, target_tree=tree, plan=plan)


# ---------------------------------------------------------------------------
# Equivariance (exact, on the dense path)
# ---------------------------------------------------------------------------


def test_translation_invariance(device):
    """Outputs are unchanged under a rigid translation of all positions."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(60, 10, 2, 3, device, torch.float64)
    out_s0, out_v0 = layer.forward_reference(s, p, v, a)
    shift = torch.randn(3, dtype=torch.float64, device=device)
    out_s1, out_v1 = layer.forward_reference(s, p + shift, v, a)
    assert (out_s0 - out_s1).abs().max() < 1e-9
    assert (out_v0 - out_v1).abs().max() < 1e-9


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("vector_invariants", [False, True])
def test_orthogonal_equivariance(device, reflection, vector_invariants):
    """Scalars are invariant and vectors equivariant under O(D) transforms."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10,
            vector_dim=2,
            heads=4,
            dim_head=8,
            vector_invariants=vector_invariants,
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(60, 10, 2, 3, device, torch.float64)
    q = _orthogonal(3, reflection, torch.float64, device)

    out_s0, out_v0 = layer.forward_reference(s, p, v, a)
    # Transform geometry and input vectors consistently.
    out_s1, out_v1 = layer.forward_reference(s, p @ q.T, v @ q.T, a)

    # Scalar output invariant; vector output co-transforms.
    assert (out_s1 - out_s0).abs().max() < 1e-9
    assert (out_v1 - out_v0 @ q.T).abs().max() < 1e-9


def test_vectors_influence_scalar_output(device):
    """With vector_invariants on, changing input vectors changes scalar output."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(
            scalar_dim=10, vector_dim=2, heads=4, dim_head=8, vector_invariants=True
        )
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(60, 10, 2, 3, device, torch.float64)
    out_a, _ = layer.forward_reference(s, p, v, a)
    out_b, _ = layer.forward_reference(s, p, 2.0 * v, a)
    assert (out_a - out_b).abs().max() > 1e-6


# ---------------------------------------------------------------------------
# Discretization-invariance (quadrature property)
# ---------------------------------------------------------------------------


def test_discretization_invariance(device):
    """Splitting each source into equal-area copies leaves the output unchanged.

    The layer is an area-weighted quadrature of an integral operator, so the
    result depends on the sources only through their area-weighted measure.
    Tested via cross-attention to a fixed set of query points (so refining the
    sources does not change the query set).
    """
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(50, 10, 2, 3, device, torch.float64, seed=0)
    qs, qv, qp, _ = _inputs(20, 10, 2, 3, device, torch.float64, seed=99)

    out_coarse, outv_coarse = layer.forward_reference(
        s, p, v, a, query_scalars=qs, query_vectors=qv, query_positions=qp
    )

    # Refine: duplicate every source at the same position with half the area.
    s2 = torch.cat([s, s], dim=0)
    v2 = torch.cat([v, v], dim=0)
    p2 = torch.cat([p, p], dim=0)
    a2 = torch.cat([a, a], dim=0) * 0.5
    out_fine, outv_fine = layer.forward_reference(
        s2, p2, v2, a2, query_scalars=qs, query_vectors=qv, query_positions=qp
    )

    assert (out_coarse - out_fine).abs().max() < 1e-9
    assert (outv_coarse - outv_fine).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradients_flow_and_are_finite(device):
    """Backprop populates finite gradients on all parameters and inputs."""
    torch.manual_seed(0)
    layer = MeshAttention(scalar_dim=10, vector_dim=2, heads=4, dim_head=8).to(device)
    s, v, p, a = _inputs(60, 10, 2, 3, device, torch.float32)
    s.requires_grad_(True)
    v.requires_grad_(True)
    out_s, out_v = layer(s, p, v, a, theta=0.5)
    (out_s.sum() + out_v.sum()).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_gradients_match_dense_at_theta_zero(device):
    """Parameter gradients of the tree forward equal the dense ones at theta=0."""
    torch.manual_seed(0)
    layer = (
        MeshAttention(scalar_dim=8, vector_dim=2, heads=2, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(40, 8, 2, 3, device, torch.float64)

    def grads(use_hier):
        layer.zero_grad(set_to_none=True)
        if use_hier:
            out_s, out_v = layer(s, p, v, a, theta=0.0)
        else:
            out_s, out_v = layer.forward_reference(s, p, v, a)
        (out_s.square().sum() + out_v.square().sum()).backward()
        return {n: pr.grad.clone() for n, pr in layer.named_parameters()}

    g_dense = grads(use_hier=False)
    g_hier = grads(use_hier=True)
    for name in g_dense:
        assert (g_dense[name] - g_hier[name]).abs().max() < 1e-7, name


# ---------------------------------------------------------------------------
# MeshTransformerBlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vector_dim", [0, 2])
def test_block_forward_shapes(device, vector_dim):
    """Block preserves scalar/vector shapes and produces finite output."""
    torch.manual_seed(0)
    block = MeshTransformerBlock(
        scalar_dim=16, vector_dim=vector_dim, heads=4, dim_head=8
    ).to(device)
    s, v, p, a = _inputs(100, 16, vector_dim, 3, device, torch.float32)
    out_s, out_v = block(s, p, v, a, theta=1.0)
    assert out_s.shape == (100, 16) and torch.isfinite(out_s).all()
    if vector_dim > 0:
        assert out_v.shape == (100, vector_dim, 3)
    else:
        assert out_v is None


def test_block_equivariance(device):
    """The block is O(D)-equivariant (scalars invariant, vectors equivariant)."""
    torch.manual_seed(0)
    block = (
        MeshTransformerBlock(scalar_dim=16, vector_dim=2, heads=4, dim_head=8)
        .double()
        .to(device)
    )
    s, v, p, a = _inputs(60, 16, 2, 3, device, torch.float64)
    q = _orthogonal(3, reflection=False, dtype=torch.float64, device=device)
    out_s0, out_v0 = block(s, p, v, a, theta=0.0)
    out_s1, out_v1 = block(s, p @ q.T, v @ q.T, a, theta=0.0)
    assert (out_s1 - out_s0).abs().max() < 1e-9
    assert (out_v1 - out_v0 @ q.T).abs().max() < 1e-9


def test_block_shared_tree_stack(device):
    """A stack of blocks can share one prebuilt tree/plan and run end to end."""
    torch.manual_seed(0)
    s, v, p, a = _inputs(150, 16, 2, 3, device, torch.float32)
    tree = ClusterTree.from_points(p, areas=a)
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=1.0)
    blocks = torch.nn.ModuleList(
        [MeshTransformerBlock(16, 2, heads=4, dim_head=8).to(device) for _ in range(3)]
    )
    ss, vv = s, v
    for block in blocks:
        ss, vv = block(ss, p, vv, a, source_tree=tree, plan=plan, theta=1.0)
    assert ss.shape == (150, 16) and vv.shape == (150, 2, 3)
    assert torch.isfinite(ss).all() and torch.isfinite(vv).all()
