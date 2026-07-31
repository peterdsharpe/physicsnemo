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

"""Barnes--Hut decode backend (task #41): partition properties, near-path
bitwise exactness, theta-controlled far-field deviation, and the decoder
contracts (query-set independence, zero drive, trace mode) under the
hierarchical backend."""

import pytest
import torch
from test_kernel_decoder import _icosphere  # shared closed-surface fixture

from physicsnemo.experimental.nn.mesh_attention import barnes_hut as bh
from physicsnemo.experimental.nn.mesh_attention.attention import ScalarVectorState
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    NonlinearZeroKernelBasisCrossDecoder,
    exact_double_layer_member,
    exact_single_layer_member,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import MEASURE_WEIGHTS_KEY
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree


def _sphere_mesh(device, subdivisions: int = 2) -> Mesh:
    points, cells = _icosphere(subdivisions)
    return Mesh(points=points.to(device), cells=cells.to(device))


def _decoder_pair(device, mesh: Mesh, *, theta: float, seed: int = 7, **overrides):
    """A dense decoder and a BH decoder with IDENTICAL weights + one cache each."""
    kwargs = dict(
        n_spatial_dims=3,
        operator_scalar_dim=3,
        operator_vector_dim=2,
        drive_scalar_dim=4,
        drive_vector_dim=2,
        heads=1,
        include_polynomial_members=False,
        include_single_layer_member=True,
        log_radial_features=True,
        mlp_members=4,
        mlp_hidden_dim=16,
        query_chunk_size=4096,
    )
    kwargs.update(overrides)
    torch.manual_seed(seed)
    dense = NonlinearZeroKernelBasisCrossDecoder(**kwargs).to(
        device=device, dtype=torch.float64
    )
    hier = NonlinearZeroKernelBasisCrossDecoder(
        **kwargs, decode_backend="barnes_hut", bh_theta=theta, bh_leaf_size=8
    ).to(device=device, dtype=torch.float64)
    hier.load_state_dict(dense.state_dict())
    n = mesh.n_cells
    generator = torch.Generator().manual_seed(seed + 1)

    def _state():
        return ScalarVectorState(
            torch.randn(n, 3, generator=generator, dtype=torch.float64).to(device),
            torch.randn(n, 2, 3, generator=generator, dtype=torch.float64).to(device),
        )

    operator = _state()
    drive = ScalarVectorState(
        torch.randn(n, 4, generator=generator, dtype=torch.float64).to(device),
        torch.randn(n, 2, 3, generator=generator, dtype=torch.float64).to(device),
    )
    dense_cache = dense.build_source_cache(mesh, operator, drive)
    hier_cache = hier.build_source_cache(mesh, operator, drive)
    return dense, hier, dense_cache, hier_cache


def _exterior_queries(device, n: int = 64, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    radii = 1.05 + 3.0 * torch.rand(n, 1, generator=generator, dtype=torch.float64)
    return (directions * radii).to(device)


# ---------------------------------------------------------------------------
# Partition properties.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_size", [1, 4, 16])
@pytest.mark.parametrize("theta", [0.3, 0.7])
def test_partition_covers_every_pair_exactly_once(device, leaf_size, theta):
    """Near pairs plus far-node subtrees tile (Q x S) with no gap or overlap."""
    generator = torch.Generator().manual_seed(3)
    sources = torch.randn(97, 3, generator=generator, dtype=torch.float64).to(device)
    queries = torch.cat(
        (
            torch.randn(23, 3, generator=generator, dtype=torch.float64).to(device),
            sources[:5],  # coincident with sources: gap distance zero
        )
    )
    tree = ClusterTree.from_points(sources, leaf_size=leaf_size)
    part = bh.single_tree_partition(queries, tree, theta)

    counts = torch.zeros(
        queries.shape[0], sources.shape[0], dtype=torch.long, device=device
    )
    counts[part.near_query, part.near_source] += 1
    # Each far node contributes its whole morton-contiguous subtree range.
    for q, node in zip(part.far_query.tolist(), part.far_node.tolist()):
        start = int(tree.node_range_start[node])
        count = int(tree.node_range_count[node])
        members = tree.sorted_source_order[start : start + count]
        counts[q, members] += 1
    assert torch.all(counts == 1), "partition must cover every pair exactly once"


def test_partition_far_nodes_satisfy_opening_criterion(device):
    """Every far admission has node diagonal < theta times the gap distance."""
    generator = torch.Generator().manual_seed(5)
    sources = torch.randn(200, 3, generator=generator, dtype=torch.float64).to(device)
    queries = torch.randn(31, 3, generator=generator, dtype=torch.float64).to(device)
    theta = 0.6
    tree = ClusterTree.from_points(sources, leaf_size=4)
    part = bh.single_tree_partition(queries, tree, theta)
    points = queries[part.far_query]
    gap = torch.clamp(
        tree.node_aabb_min[part.far_node] - points, min=0.0
    ) + torch.clamp(points - tree.node_aabb_max[part.far_node], min=0.0)
    gap_sq = gap.square().sum(dim=-1).double()
    assert torch.all(
        tree.node_diameter_sq[part.far_node] < theta * theta * gap_sq + 1e-30
    )


# ---------------------------------------------------------------------------
# Near path: bitwise per-pair exactness against the dense closed forms.
# ---------------------------------------------------------------------------


def test_pairwise_members_bitwise_match_dense_entries(device):
    """The pair re-expressions equal the broadcast closed forms bitwise."""
    mesh = _sphere_mesh(device, 1)
    queries = _exterior_queries(device, 40)
    vertices = mesh.points[mesh.cells]
    normals = mesh.cell_normals
    dense_dl = exact_double_layer_member(queries, vertices, normals)
    dense_sl = exact_single_layer_member(queries, vertices)
    q_idx, s_idx = torch.meshgrid(
        torch.arange(queries.shape[0], device=device),
        torch.arange(mesh.n_cells, device=device),
        indexing="ij",
    )
    q_idx = q_idx.reshape(-1)
    s_idx = s_idx.reshape(-1)
    pair_dl = bh.pair_triangle_double_layer(
        queries[q_idx], vertices[s_idx], normals[s_idx]
    )
    pair_sl = bh.pair_triangle_single_layer(queries[q_idx], vertices[s_idx])
    assert torch.equal(pair_dl, dense_dl.reshape(-1))
    assert torch.equal(pair_sl, dense_sl.reshape(-1))


# ---------------------------------------------------------------------------
# Whole-operator equivalence and theta control.
# ---------------------------------------------------------------------------


def test_theta_to_zero_recovers_dense(device):
    """At a vanishing opening threshold every pair is near: BH == dense to
    reduction-order tolerance (the segment sums use a different, equally
    fixed, reduction order than the dense fixed-axis sums)."""
    mesh = _sphere_mesh(device, 2)
    dense, hier, dense_cache, hier_cache = _decoder_pair(device, mesh, theta=1e-6)
    queries = _exterior_queries(device, 48)
    out_dense = dense(queries, dense_cache)
    out_hier = hier(queries, hier_cache)
    assert torch.allclose(out_dense.scalars, out_hier.scalars, rtol=1e-10, atol=1e-12)
    assert torch.allclose(out_dense.vectors, out_hier.vectors, rtol=1e-10, atol=1e-12)


def test_theta_to_zero_recovers_dense_with_nonuniform_public_measure(device):
    """Both backends distinguish panel geometry from effective quadrature."""
    base = _sphere_mesh(device, 2)
    factors = torch.linspace(
        0.2,
        5.0,
        base.n_cells,
        dtype=base.points.dtype,
        device=base.points.device,
    )
    mesh = base.with_data(cell_data={MEASURE_WEIGHTS_KEY: factors})
    dense, hier, dense_cache, hier_cache = _decoder_pair(device, mesh, theta=1.0e-6)
    queries = _exterior_queries(device, 48)
    out_dense = dense(queries, dense_cache)
    out_hier = hier(queries, hier_cache)

    torch.testing.assert_close(
        dense_cache.panel_areas, base.cell_areas, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(dense_cache.measure_factors, factors, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        dense_cache.quadrature_measures,
        base.cell_areas * factors,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        hier_cache.quadrature_measures,
        dense_cache.quadrature_measures,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.allclose(out_dense.scalars, out_hier.scalars, rtol=1e-10, atol=1e-12)
    assert torch.allclose(out_dense.vectors, out_hier.vectors, rtol=1e-10, atol=1e-12)


def _theta_error_curve(device, mesh, queries, thetas, **overrides):
    errors = {}
    for theta in thetas:
        dense, hier, dense_cache, hier_cache = _decoder_pair(
            device, mesh, theta=theta, **overrides
        )
        out_dense = dense(queries, dense_cache)
        out_hier = hier(queries, hier_cache)
        num = (out_hier.scalars - out_dense.scalars).norm()
        den = out_dense.scalars.norm().clamp_min(1e-30)
        errors[theta] = float(num / den)
    return errors


def test_far_field_exact_members_error_is_theta_controlled(device):
    """Exact singular members (DL + SL): the far field keeps the analytic
    decay (point dipole/monopole against exact channel-resolved density
    aggregates), so the deviation is theta-controlled at roughly second
    order (measured ~theta^2 on this sphere) and small at production
    theta."""
    mesh = _sphere_mesh(device, 2)
    queries = _exterior_queries(device, 48)
    errors = _theta_error_curve(
        device,
        mesh,
        queries,
        (0.25, 0.5, 1.0),
        mlp_members=0,
        log_radial_features=False,
    )
    assert errors[0.25] <= errors[0.5] <= errors[1.0] * (1 + 1e-9)
    assert errors[1.0] > 3.0 * errors[0.25], (
        "far-field error should grow superlinearly in theta for the exact "
        f"members, got {errors}"
    )
    assert errors[0.25] < 2e-2, f"theta=0.25 deviation {errors[0.25]:.3e}"
    assert errors[0.5] < 5e-2, f"theta=0.5 deviation {errors[0.5]:.3e}"


def test_far_field_smooth_members_error_decreases_with_theta(device):
    """Smooth MLP members: the virtual-source far field is exact for the
    aggregated learned DENSITIES but approximates the member value by one
    evaluation per (query, node).  Nothing forces an untrained random MLP
    to decay with distance, so for THIS test's random weights the deviation
    magnitude is O(0.1) and only weakly theta-controlled -- the honest
    guarantee is monotone improvement as theta shrinks plus exact recovery
    at theta -> 0 (separate test).  Fidelity at production theta on a
    TRAINED checkpoint (whose members can and do learn decay through the
    log-radial features) is measured by the acceptance-A calibration, not
    pinned here."""
    mesh = _sphere_mesh(device, 2)
    queries = _exterior_queries(device, 48)
    errors = _theta_error_curve(device, mesh, queries, (0.25, 0.5, 1.0))
    assert errors[0.25] <= errors[0.5] <= errors[1.0] * (1 + 1e-9)
    assert errors[1.0] < 1.0, f"theta=1.0 deviation {errors[1.0]:.3e}"


def test_zero_drive_gives_exactly_zero(device):
    """All aggregates are linear in the values: zero drive -> zero output."""
    mesh = _sphere_mesh(device, 1)
    _, hier, _, _ = _decoder_pair(device, mesh, theta=0.5)
    n = mesh.n_cells
    generator = torch.Generator().manual_seed(23)
    operator = ScalarVectorState(
        torch.randn(n, 3, generator=generator, dtype=torch.float64).to(device),
        torch.randn(n, 2, 3, generator=generator, dtype=torch.float64).to(device),
    )
    zero_drive = ScalarVectorState(
        torch.zeros(n, 4, dtype=torch.float64, device=device),
        torch.zeros(n, 2, 3, dtype=torch.float64, device=device),
    )
    cache = hier.build_source_cache(mesh, operator, zero_drive)
    out = hier(_exterior_queries(device, 16), cache)
    assert torch.all(out.scalars == 0.0)
    assert torch.all(out.vectors == 0.0)


def test_query_set_independence_is_bitwise_under_bh(device):
    """Decoding a subset reproduces those rows bitwise (single-tree descent:
    each query's interaction list reads its own position and the source tree
    alone; per-query reductions are fixed-order cumsum segments)."""
    mesh = _sphere_mesh(device, 2)
    _, hier, _, hier_cache = _decoder_pair(device, mesh, theta=0.5)
    queries = _exterior_queries(device, 32)
    full = hier(queries, hier_cache)
    subset = torch.tensor([3, 11, 17, 29], device=device)
    part = hier(queries[subset], hier_cache)
    assert torch.equal(full.scalars[subset], part.scalars)
    assert torch.equal(full.vectors[subset], part.vectors)


def test_trace_self_pairs_are_near_and_corrected(device):
    """On-boundary queries with declared self panels: the own pair is always
    near (gap distance zero) and carries the exact exterior limit +1/2, so a
    tiny-theta BH trace decode matches the dense trace decode."""
    mesh = _sphere_mesh(device, 1)
    dense, hier, dense_cache, hier_cache = _decoder_pair(device, mesh, theta=1e-6)
    queries = mesh.cell_centroids
    self_indices = torch.arange(mesh.n_cells, device=device)
    out_dense = dense(queries, dense_cache, self_indices=self_indices)
    out_hier = hier(queries, hier_cache, self_indices=self_indices)
    assert torch.allclose(out_dense.scalars, out_hier.scalars, rtol=1e-10, atol=1e-12)


def test_bh_rejects_unsupported_configurations():
    """Each v1 scope limit is a loud error, not a silent approximation."""
    base = dict(
        n_spatial_dims=3,
        operator_scalar_dim=2,
        operator_vector_dim=1,
        drive_scalar_dim=2,
        drive_vector_dim=1,
        heads=1,
        decode_backend="barnes_hut",
    )
    with pytest.raises(NotImplementedError, match="polynomial"):
        NonlinearZeroKernelBasisCrossDecoder(
            **{**base, "include_polynomial_members": True}
        )
    with pytest.raises(NotImplementedError, match="3D"):
        NonlinearZeroKernelBasisCrossDecoder(
            **{
                **base,
                "n_spatial_dims": 2,
                "include_polynomial_members": False,
            }
        )
    with pytest.raises(NotImplementedError, match="checkpoint"):
        NonlinearZeroKernelBasisCrossDecoder(
            **{
                **base,
                "include_polynomial_members": False,
                "checkpoint_query_chunks": True,
            }
        )
    with pytest.raises(ValueError, match="bh_theta"):
        NonlinearZeroKernelBasisCrossDecoder(
            **{**base, "include_polynomial_members": False, "bh_theta": -1.0}
        )


def test_gradients_flow_through_both_fields(device):
    """Backward reaches the coefficient map through near pairs AND far
    aggregates (the aggregates are linear in the learned tensors)."""
    mesh = _sphere_mesh(device, 1)
    _, hier, _, _ = _decoder_pair(device, mesh, theta=0.7)
    n = mesh.n_cells
    generator = torch.Generator().manual_seed(31)
    operator = ScalarVectorState(
        torch.randn(n, 3, generator=generator, dtype=torch.float64).to(device),
        torch.randn(n, 2, 3, generator=generator, dtype=torch.float64).to(device),
    )
    drive = ScalarVectorState(
        torch.randn(
            n, 4, generator=generator, dtype=torch.float64, requires_grad=True
        ).to(device),
        torch.randn(n, 2, 3, generator=generator, dtype=torch.float64).to(device),
    )
    cache = hier.build_source_cache(mesh, operator, drive)
    out = hier(_exterior_queries(device, 8), cache)
    out.scalars.square().sum().backward()
    grads = [
        p.grad
        for p in hier.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert grads, "no nonzero parameter gradients under the BH backend"
