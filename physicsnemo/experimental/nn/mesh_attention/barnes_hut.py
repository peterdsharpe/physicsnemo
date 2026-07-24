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

r"""Barnes--Hut (single-tree) acceleration for the kernel decoder.

The dense decoder evaluates every (query, source) pair: :math:`O(QS)` work
per decode.  This module supplies the hierarchical alternative pre-registered
as task #41 (design: ``studies/notes/hierarchical_decode_design.md``):

- **Near field** (source panels whose subtended size exceeds the opening
  threshold :math:`\theta`): evaluated PAIRWISE through per-pair
  re-expressions of the SAME closed forms and the SAME smooth-member MLP the
  dense path uses.  Per-pair values are bitwise identical to the dense
  entries (asserted by tests; requires the dense closed forms' mul+sum
  reduction idiom -- keep the two in sync).  Trace-mode self pairs are
  always near (their gap distance is zero) and receive the same
  exterior-trace correction.
- **Far field** (well-separated cluster nodes): the learned content is
  aggregated EXACTLY -- the per-source density products
  :math:`\rho_{s,m,h,f} = C_{smh} V_{shf}` are summed per node, channel
  resolved, so no learned quantity is averaged heuristically -- and only the
  GEOMETRY is approximated: exact singular members collapse to their
  analytic far limits (single layer: monopole :math:`A/4\pi r`; double
  layer: the aggregated dipole vector :math:`\sum A_s n_s \rho_s` against
  :math:`\nabla G`), and smooth members are evaluated once per (query, node)
  at the node's area-weighted virtual source.  The single knob is
  :math:`\theta`; :math:`\theta \to 0` recovers the dense operator.

  **Honesty note on the smooth members**: the exact members decay
  analytically, so their far-field error is :math:`O(\theta^2)`
  (measured).  A LEARNED member is only as far-field-friendly as what it
  learned: nothing forces the MLP to decay with distance, and for a
  non-decaying member the virtual-source truncation error is not
  :math:`\theta`-controlled (for random init it is O(0.1) and nearly
  :math:`\theta`-flat -- see the test suite).  Fidelity at production
  :math:`\theta` is therefore a measured property of each trained
  checkpoint (the acceptance-A calibration), not a theorem.

Contract (design doc section 4): deterministic given (source mesh,
:math:`\theta`, leaf size); **bitwise query-set independent** (single-tree
descent -- each query's interaction list depends on its own position and the
source tree alone; per-query reductions are fixed-shape binary-tree folds
over the query's own pairs, never atomics or global scans); similarity-equivariant
(the tree lives in the model's normalized frame); zero drive gives exactly
zero output (all aggregates are linear in the values).  Deviation from the
dense oracle is APPROXIMATE, measured, and :math:`\theta`-controlled -- see
the acceptance suite in ``test_barnes_hut.py`` and the calibration artifacts.

Current scope (v1): 3D triangle boundaries only (the DrivAerML acceptance
target); 2D raises ``NotImplementedError``.  ``monopole_free_single_layer``
and the ``local_pair_features`` probe block are rejected with clear errors
(no far-field treatment is defined for them yet).
"""

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from jaxtyping import Float, Int

from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

_FOUR_PI = 4.0 * torch.pi


# ---------------------------------------------------------------------------
# Pairwise re-expressions of the exact closed forms.
#
# These mirror ``_triangle_double_layer_member`` / ``_triangle_single_layer
# _member`` in ``kernel_decoder.py`` with the (Q, S) broadcast replaced by a
# flat pair axis: every operation is the same elementwise op in the same
# order on the same scalars, so per-pair values are bitwise identical to the
# corresponding dense entries (pinned by ``test_barnes_hut.py``).  Keep the
# two implementations in sync -- the bitwise test is the coupling.
# ---------------------------------------------------------------------------


def pair_triangle_double_layer(
    query_points: Float[torch.Tensor, "p 3"],
    panel_vertices: Float[torch.Tensor, "p 3 3"],
    cell_normals: Float[torch.Tensor, "p 3"],
) -> Float[torch.Tensor, " p"]:
    """Per-pair exact double-layer values; see the module header."""
    a = panel_vertices[:, 0, :] - query_points
    b = panel_vertices[:, 1, :] - query_points
    c = panel_vertices[:, 2, :] - query_points
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = (
        la * lb * lc
        + (a * b).sum(dim=-1) * lc
        + (b * c).sum(dim=-1) * la
        + (c * a).sum(dim=-1) * lb
    )
    winding_normal = torch.cross(
        panel_vertices[:, 1, :] - panel_vertices[:, 0, :],
        panel_vertices[:, 2, :] - panel_vertices[:, 0, :],
        dim=-1,
    )
    sigma = torch.sign((winding_normal * cell_normals).sum(dim=-1))
    return -sigma * 2.0 * torch.atan2(numerator, denominator) / _FOUR_PI


def pair_triangle_single_layer(
    query_points: Float[torch.Tensor, "p 3"],
    panel_vertices: Float[torch.Tensor, "p 3 3"],
) -> Float[torch.Tensor, " p"]:
    """Per-pair exact single-layer values; see the module header."""
    a = panel_vertices[:, 0, :] - query_points
    b = panel_vertices[:, 1, :] - query_points
    c = panel_vertices[:, 2, :] - query_points
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    winding_normal = torch.cross(
        panel_vertices[:, 1, :] - panel_vertices[:, 0, :],
        panel_vertices[:, 2, :] - panel_vertices[:, 0, :],
        dim=-1,
    )
    unit_normal = winding_normal / winding_normal.norm(dim=-1, keepdim=True)
    height = (a * unit_normal).sum(dim=-1)
    tiny = torch.finfo(query_points.dtype).tiny
    relative = (a, b, c)
    edge_terms = query_points.new_zeros(a.shape[0])
    for start, end in ((0, 1), (1, 2), (2, 0)):
        p = relative[start]
        q = relative[end]
        edge = panel_vertices[:, end, :] - panel_vertices[:, start, :]
        edge_tangent = edge / edge.norm(dim=-1, keepdim=True)
        s_start = (p * edge_tangent).sum(dim=-1)
        s_end = (q * edge_tangent).sum(dim=-1)
        mu = (p - s_start.unsqueeze(-1) * edge_tangent).norm(dim=-1)
        mu = mu.clamp_min(tiny)
        in_plane_distance = (p * torch.cross(edge_tangent, unit_normal, dim=-1)).sum(
            dim=-1
        )
        edge_terms = edge_terms + in_plane_distance * (
            torch.asinh(s_end / mu) - torch.asinh(s_start / mu)
        )
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = (
        la * lb * lc
        + (a * b).sum(dim=-1) * lc
        + (b * c).sum(dim=-1) * la
        + (c * a).sum(dim=-1) * lb
    )
    solid_angle = 2.0 * torch.atan2(numerator, denominator).abs()
    return (edge_terms - height.abs() * solid_angle) / _FOUR_PI


# ---------------------------------------------------------------------------
# Single-tree partition.
# ---------------------------------------------------------------------------


@dataclass
class BHPartition:
    """Near pairs and far (query, node) pairs from the single-tree descent.

    ``near_query``/``near_source`` index (query row, ORIGINAL source index)
    pairs sorted by (query, source); ``far_query``/``far_node`` index
    (query row, tree node) pairs sorted by query.  Together with the far
    nodes' subtree contents, every (query, source) pair is covered exactly
    once (pinned by the completeness/exclusivity property test).
    """

    near_query: Int[torch.Tensor, " p"]
    near_source: Int[torch.Tensor, " p"]
    far_query: Int[torch.Tensor, " f"]
    far_node: Int[torch.Tensor, " f"]


def single_tree_partition(
    query_points: Float[torch.Tensor, "q spatial_dims"],
    tree: ClusterTree,
    theta: float,
) -> BHPartition:
    r"""Classify every (query, source-subtree) interaction as near or far.

    Level-batched descent from the root: a frontier (query, node) pair is
    FAR when :math:`D_S < \theta\, d(x, \mathrm{AABB})` (node AABB diagonal
    against the query's gap distance to the node box -- the point
    specialization of the upstream dual criterion, so a node containing the
    query is never far); otherwise it opens into its children, and leaves
    expand into exact near pairs.  Each query's classification reads its own
    position and the source tree alone -- the decoder's query-set
    independence survives the approximation bitwise (module header).
    """
    device = query_points.device
    n_queries = query_points.shape[0]
    if theta < 0.0:
        raise ValueError(f"theta must be >= 0, got {theta}")
    if tree.n_sources == 0 or n_queries == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return BHPartition(empty, empty.clone(), empty.clone(), empty.clone())
    theta_sq = theta * theta

    frontier_query = torch.arange(n_queries, dtype=torch.long, device=device)
    frontier_node = torch.zeros(n_queries, dtype=torch.long, device=device)

    near_query_parts: list[torch.Tensor] = []
    near_source_parts: list[torch.Tensor] = []
    far_query_parts: list[torch.Tensor] = []
    far_node_parts: list[torch.Tensor] = []

    max_iters = int(tree.max_depth.item()) + 2
    for _ in range(max_iters):
        if frontier_query.numel() == 0:
            break
        points = query_points[frontier_query]
        aabb_min = tree.node_aabb_min[frontier_node]
        aabb_max = tree.node_aabb_max[frontier_node]
        gap = torch.clamp(aabb_min - points, min=0.0) + torch.clamp(
            points - aabb_max, min=0.0
        )
        gap_sq = gap.square().sum(dim=-1)
        # Far iff D_S^2 < theta^2 * d^2; d = 0 (query inside the box) can
        # never satisfy this, so containing nodes always open.
        far_mask = tree.node_diameter_sq[frontier_node] < theta_sq * gap_sq
        if far_mask.any():
            far_query_parts.append(frontier_query[far_mask])
            far_node_parts.append(frontier_node[far_mask])
        open_query = frontier_query[~far_mask]
        open_node = frontier_node[~far_mask]
        left = tree.node_left_child[open_node]
        is_leaf = left < 0
        if is_leaf.any():
            leaf_query = open_query[is_leaf]
            leaf_node = open_node[is_leaf]
            counts = tree.leaf_count[leaf_node]
            starts = tree.leaf_start[leaf_node]
            # Expand each (query, leaf) hit into its sources via a ragged
            # arange over the morton-sorted contiguous leaf ranges.
            total = int(counts.sum().item())
            if total > 0:
                offsets = torch.repeat_interleave(starts, counts)
                cumulative = torch.cumsum(counts, dim=0) - counts
                ragged = torch.arange(
                    total, dtype=torch.long, device=device
                ) - torch.repeat_interleave(cumulative, counts)
                near_query_parts.append(torch.repeat_interleave(leaf_query, counts))
                near_source_parts.append(tree.sorted_source_order[offsets + ragged])
        internal_query = open_query[~is_leaf]
        internal_node = open_node[~is_leaf]
        frontier_query = torch.cat((internal_query, internal_query))
        frontier_node = torch.cat(
            (
                tree.node_left_child[internal_node],
                tree.node_right_child[internal_node],
            )
        )
    if frontier_query.numel() != 0:
        # Count-midpoint splits bound the depth, so this is unreachable --
        # but if the split rule ever changes, dropping leftover pairs would
        # silently truncate sums (the cluster_tree review's hardening note).
        raise RuntimeError(
            "single_tree_partition: traversal frontier not empty after "
            f"max_depth+2 iterations ({frontier_query.numel()} pairs left)"
        )

    if near_query_parts:
        near_query = torch.cat(near_query_parts)
        near_source = torch.cat(near_source_parts)
        order = torch.argsort(near_query * tree.n_sources + near_source)
        near_query = near_query[order]
        near_source = near_source[order]
    else:
        near_query = torch.empty(0, dtype=torch.long, device=device)
        near_source = near_query.clone()
    if far_query_parts:
        far_query = torch.cat(far_query_parts)
        far_node = torch.cat(far_node_parts)
        order = torch.argsort(far_query * tree.n_nodes + far_node)
        far_query = far_query[order]
        far_node = far_node[order]
    else:
        far_query = torch.empty(0, dtype=torch.long, device=device)
        far_node = far_query.clone()
    return BHPartition(near_query, near_source, far_query, far_node)


# ---------------------------------------------------------------------------
# Channel-resolved node aggregates.
# ---------------------------------------------------------------------------


def _node_range_sums(
    tree: ClusterTree,
    per_source: Float[torch.Tensor, "s ..."],
) -> Float[torch.Tensor, "n ..."]:
    """Exact per-node subtree sums of a per-source tensor.

    Sorts into morton order and takes cumulative-sum differences over each
    node's contiguous subtree range -- deterministic (no atomics), one pass,
    exact for every node including internal ones.
    """
    sorted_vals = per_source[tree.sorted_source_order]
    flat = sorted_vals.reshape(sorted_vals.shape[0], -1)
    zero = flat.new_zeros((1, flat.shape[1]))
    cumulative = torch.cat((zero, torch.cumsum(flat, dim=0)), dim=0)
    start = tree.node_range_start
    stop = start + tree.node_range_count
    sums = cumulative[stop] - cumulative[start]
    return sums.reshape((tree.n_nodes,) + per_source.shape[1:])


@dataclass
class BHNodeAggregates:
    r"""Per-node far-field content: exact learned-density sums + virtual geometry.

    The density products :math:`\rho = C \cdot V` are aggregated exactly
    (channel resolved); only geometry is approximated at evaluation time.
    ``sl_*``/``dl_*`` carry the exact singular members' aggregates
    (:math:`\sum A\rho` and :math:`\sum A n \rho`); ``smooth_*`` carry the
    measure-weighted smooth-member density sums evaluated against the node's
    virtual source (area-weighted centroid/normal/state vectors).
    """

    centroid: Float[torch.Tensor, "n 3"]
    unit_normal: Float[torch.Tensor, "n 3"]
    mean_pair_vectors: Float[torch.Tensor, "n channels 3"]
    # Exact-member aggregates, channel resolved over (heads, value channels).
    sl_scalars: Float[torch.Tensor, "n heads fs"] | None
    sl_vectors: Float[torch.Tensor, "n heads fv 3"] | None
    sl_pseudos: Float[torch.Tensor, "n heads fp"] | None
    dl_scalars: Float[torch.Tensor, "n 3 heads fs"]
    dl_vectors: Float[torch.Tensor, "n 3 heads fv 3"]
    dl_pseudos: Float[torch.Tensor, "n 3 heads fp"] | None
    # Smooth-member aggregates (measure-weighted), per smooth member.
    smooth_scalars: Float[torch.Tensor, "n m heads fs"] | None
    smooth_vectors: Float[torch.Tensor, "n m heads fv 3"] | None
    smooth_pseudos: Float[torch.Tensor, "n m heads fp"] | None


def build_node_aggregates(
    tree: ClusterTree,
    *,
    areas: Float[torch.Tensor, " s"],
    centroids: Float[torch.Tensor, "s 3"],
    normals: Float[torch.Tensor, "s 3"],
    pair_vectors: Float[torch.Tensor, "s channels 3"],
    coefficients: Float[torch.Tensor, "s members heads"],
    value_scalars: Float[torch.Tensor, "s heads fs"],
    value_vectors: Float[torch.Tensor, "s heads fv 3"],
    value_pseudos: Float[torch.Tensor, "s heads fp"] | None,
    include_single_layer: bool,
    n_smooth_members: int,
) -> BHNodeAggregates:
    """Assemble every per-node aggregate the far field needs (module header)."""
    area_sum = _node_range_sums(tree, areas).clamp_min(torch.finfo(areas.dtype).tiny)
    centroid = _node_range_sums(tree, areas[:, None] * centroids) / area_sum[:, None]
    normal_sum = _node_range_sums(tree, areas[:, None] * normals)
    unit_normal = normal_sum / normal_sum.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(areas.dtype).tiny
    )
    mean_pair_vectors = (
        _node_range_sums(tree, areas[:, None, None] * pair_vectors)
        / area_sum[:, None, None]
    )

    def _density(member: int, weight: Float[torch.Tensor, " s"]):
        rho = coefficients[:, member, :] * weight[:, None]  # (S, H)
        scalars = _node_range_sums(tree, rho[:, :, None] * value_scalars)
        vectors = _node_range_sums(tree, rho[:, :, None, None] * value_vectors)
        pseudos = (
            _node_range_sums(tree, rho[:, :, None] * value_pseudos)
            if value_pseudos is not None
            else None
        )
        return scalars, vectors, pseudos

    def _dipole(member: int, weight: Float[torch.Tensor, " s"]):
        rho = coefficients[:, member, :] * weight[:, None]  # (S, H)
        rho_n = rho[:, None, :] * normals[:, :, None]  # (S, 3, H)
        scalars = _node_range_sums(tree, rho_n[:, :, :, None] * value_scalars[:, None])
        vectors = _node_range_sums(
            tree, rho_n[:, :, :, None, None] * value_vectors[:, None]
        )
        pseudos = (
            _node_range_sums(tree, rho_n[:, :, :, None] * value_pseudos[:, None])
            if value_pseudos is not None
            else None
        )
        return scalars, vectors, pseudos

    dl_scalars, dl_vectors, dl_pseudos = _dipole(0, areas)
    if include_single_layer:
        sl_scalars, sl_vectors, sl_pseudos = _density(1, areas)
        first_smooth = 2
    else:
        sl_scalars = sl_vectors = sl_pseudos = None
        first_smooth = 1

    if n_smooth_members > 0:
        parts = [_density(first_smooth + m, areas) for m in range(n_smooth_members)]
        smooth_scalars = torch.stack([p[0] for p in parts], dim=1)
        smooth_vectors = torch.stack([p[1] for p in parts], dim=1)
        smooth_pseudos = (
            torch.stack([p[2] for p in parts], dim=1)
            if value_pseudos is not None
            else None
        )
    else:
        smooth_scalars = smooth_vectors = smooth_pseudos = None

    return BHNodeAggregates(
        centroid=centroid,
        unit_normal=unit_normal,
        mean_pair_vectors=mean_pair_vectors,
        sl_scalars=sl_scalars,
        sl_vectors=sl_vectors,
        sl_pseudos=sl_pseudos,
        dl_scalars=dl_scalars,
        dl_vectors=dl_vectors,
        dl_pseudos=dl_pseudos,
        smooth_scalars=smooth_scalars,
        smooth_vectors=smooth_vectors,
        smooth_pseudos=smooth_pseudos,
    )


# ---------------------------------------------------------------------------
# Deterministic per-query segment sums.
# ---------------------------------------------------------------------------


_SEGMENT_FOLD_BLOCK = 128


def segment_sum_by_query(
    values,
    query_index: Int[torch.Tensor, " p"],
    n_queries: int,
    *,
    block: int = _SEGMENT_FOLD_BLOCK,
    checkpoint_blocks: bool = False,
) -> Float[torch.Tensor, "q ..."]:
    """Sum pair rows into query rows deterministically and set-independently.

    ``values`` is either a ``(P, ...)`` tensor of pair rows, or a callable
    ``sel -> (len(sel), ...)`` that produces the rows for a batch of pair
    indices lazily.  The lazy form is how the decoder keeps memory bounded:
    per-pair products (members x coefficients x values) are computed for at
    most ``n_queries * block`` pairs at a time and reduced immediately,
    never materialized for the full pair list.

    With ``checkpoint_blocks=True`` (lazy form under grad only), each
    block's row computation + fold is gradient-checkpointed: activations
    are recomputed in backward instead of stored, so TRAINING memory is
    also bounded by one block rather than the total pair count (measured:
    the uncheckpointed backward at 50k sources exhausts even a 276 GB
    device).  The row producer must be deterministic -- ours are -- making
    the recomputation bitwise.

    ``query_index`` must be sorted ascending (the partition guarantees it).
    Each query's sum is a fixed binary-tree fold over its own pair rows in
    their sorted order, evaluated in blocks of ``block`` ranks.  The fold
    shape depends only on the query's own pair count and ``block`` -- never
    on any other query's pairs -- which is what the bitwise
    query-set-independence contract requires (``block`` must therefore be a
    fixed constant per call site, not data dependent).  A global cumulative
    sum with boundary differences would couple queries through accumulated
    rounding.  No atomics: every padded slot has exactly one writer.
    """
    if block < 1 or block & (block - 1):
        raise ValueError(f"block must be a power of two, got {block}")
    lazy = callable(values)
    if query_index.shape[0] == 0:
        rows = values(query_index[:0]) if lazy else values
        return rows.new_zeros((n_queries,) + rows.shape[1:])
    counts = torch.bincount(query_index, minlength=n_queries)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = (
        torch.arange(query_index.shape[0], device=query_index.device)
        - starts[query_index]
    )

    def _block_partial(sel: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        rows = values(sel) if lazy else values[sel]
        flat = rows.reshape(rows.shape[0], -1)
        padded = flat.new_zeros((n_queries, block, flat.shape[1]))
        padded[query_index[sel], slot] = flat
        width = block
        while width > 1:
            width //= 2
            padded = padded[:, :width] + padded[:, width : 2 * width]
        return padded[:, 0]

    use_checkpoint = checkpoint_blocks and lazy and torch.is_grad_enabled()
    total = None
    n_blocks = (int(counts.max().item()) + block - 1) // block
    for j in range(n_blocks):
        in_block = (rank >= j * block) & (rank < (j + 1) * block)
        sel = in_block.nonzero(as_tuple=False).squeeze(-1)
        slot = rank[sel] - j * block
        if use_checkpoint:
            # Non-reentrant so gradients reach closure-captured parameters;
            # the decoder is RNG-free, so no rng-state round-trip.
            partial = torch.utils.checkpoint.checkpoint(
                _block_partial,
                sel,
                slot,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            partial = _block_partial(sel, slot)
        # Trailing zero blocks add +0.0 -- a bitwise no-op -- so the block
        # count (a max over the batch) does not leak across queries.
        total = partial if total is None else total + partial
    tail_shape = (values(query_index[:0]) if lazy else values).shape[1:]
    return total.reshape((n_queries,) + tail_shape)
