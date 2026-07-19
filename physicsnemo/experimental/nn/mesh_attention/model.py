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

r"""A mesh-native global transformer for boundary-driven PDE surrogates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.mesh import (
    DomainMesh,
    FieldLayout,
    Mesh,
    RankSpecDict,
    ScalarVectorFields,
    flatten_rank_spec,
    validate_rank_spec,
)

from .attention import AttentionMoments, ScalarVectorState, TypedProjection
from .block import (
    GeometryConditionedLinear,
    LinearMeshFieldBlock,
    MeshOperatorBlock,
    NonlinearZeroMeshFieldBlock,
    PointwiseGeometryBlock,
    QuadraticFieldReadIn,
)
from .kernel_decoder import (
    KernelDecoderCache,
    LinearKernelBasisCrossDecoder,
    NonlinearZeroKernelBasisCrossDecoder,
)

FieldMode = Literal["linear", "zero_preserving_nonlinear", "quadratic"]
QueryDecoder = Literal["moment", "kernel"]
#: A :data:`~physicsnemo.mesh.RankSpecDict` whose leaves may additionally be
#: the string rank token ``"0o"``, declaring a 2D pseudoscalar field
#: (rotation invariant, sign-flipping under reflection).  Pseudoscalar leaves
#: have scalar shape -- ``()`` for global fields, ``(N,)`` for cell data --
#: and differ from rank-0 leaves only in their transformation law.
PseudoAwareRankSpecDict: TypeAlias = dict[
    str, Union[int, str, "PseudoAwareRankSpecDict"]
]
FieldRoleRanks: TypeAlias = dict[str, PseudoAwareRankSpecDict]
_FIELD_ROLES = ("operator", "drive")
#: The irrep rank token declaring a 2D pseudoscalar field.
_PSEUDO_RANK = "0o"


def _split_pseudoscalar_spec(
    rank_spec: PseudoAwareRankSpecDict,
    *,
    source_label: str,
) -> tuple[RankSpecDict, RankSpecDict]:
    r"""Split ``"0o"`` leaves out of a rank spec.

    Returns ``(tensor_spec, pseudo_spec)``: the first contains every
    integer-rank leaf unchanged, the second contains the pseudoscalar leaves
    re-expressed at rank 0 (a pseudoscalar packs exactly like a scalar; only
    its transformation law differs, which the model tracks separately).
    String leaves other than ``"0o"`` are rejected here with the full token
    menu; integer leaves are validated downstream by
    :func:`~physicsnemo.mesh.validate_rank_spec`.
    """
    tensor_spec: RankSpecDict = {}
    pseudo_spec: RankSpecDict = {}
    for key, value in rank_spec.items():
        if isinstance(value, dict):
            sub_tensor, sub_pseudo = _split_pseudoscalar_spec(
                value, source_label=f"{source_label}[{key!r}]"
            )
            if sub_tensor:
                tensor_spec[key] = sub_tensor
            if sub_pseudo:
                pseudo_spec[key] = sub_pseudo
        elif isinstance(value, str):
            if value != _PSEUDO_RANK:
                raise ValueError(
                    f"Rank for {key!r} in {source_label} must be 0, 1, or "
                    f"the pseudoscalar token {_PSEUDO_RANK!r}, got {value!r}"
                )
            pseudo_spec[key] = 0
        else:
            tensor_spec[key] = value
    return tensor_spec, pseudo_spec


@dataclass
class MetaData(ModelMetaData):
    """Runtime capabilities of :class:`MeshTransformer`."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = False


@dataclass(frozen=True)
class EncodedBoundary:
    r"""Reusable boundary state returned by :meth:`MeshTransformer.encode`.

    This is the sole public cache object.  It binds the dimensionless source
    quadrature, encoded operator and drive states, source frame, domain-level
    operator/drive data, query-block source moments, and the default query
    mesh.  It intentionally contains no tree, neighbourhood, or
    query-dependent state.
    """

    source_mesh: Mesh
    operator_state: ScalarVectorState
    drive_state: ScalarVectorState
    center: Float[torch.Tensor, " spatial_dims"]
    # The resolved scale gauge: the declared ``reference_length_key`` scalar
    # if the model has one, otherwise the intrinsic measure-weighted RMS
    # boundary radius.
    reference_length: Float[torch.Tensor, ""]
    global_operator_state: ScalarVectorState
    global_drive_state: ScalarVectorState
    query_moments: tuple[AttentionMoments, ...]
    query_mesh: Mesh
    global_data: TensorDict
    # Populated only by models constructed with ``query_decoder="kernel"``:
    # the query-independent source cache of the dense kernel-basis decoder
    # (normalized cell vertices, kernel coefficients, and projected values).
    kernel_cache: KernelDecoderCache | None = None
    # Populated only by models constructed with a declared boundary-trace
    # mode (``trace_of``): the declared boundary's contiguous cell range in
    # the merged source, whose cell centroids the query mesh is declared to
    # be (index-aligned, in cell order).  ``None`` keeps caches from
    # trace-free models valid, and lets trace models reject them loudly.
    trace_slice: slice | None = None
    # Populated only under the CONTRACT-BREAKING DIAGNOSTIC
    # ``diagnostic_local_query_features``: per-trace-cell (scalars, unit
    # normals) where scalars = (log relative area, curvature *
    # reference_length**n_manifold_dims), both similarity-invariant by
    # construction.
    diagnostic_query_features: (
        tuple[Float[torch.Tensor, "s 2"], Float[torch.Tensor, "s spatial_dims"]] | None
    ) = None


def _role_spec(
    spec: FieldRoleRanks, role: str, *, label: str
) -> PseudoAwareRankSpecDict:
    """Extract and validate one role's rank spec from a declared
    ``FieldRoleRanks`` dict (ranks 0/1 only; the operator role additionally
    may not carry a pseudoscalar sector)."""
    if not isinstance(spec, dict):
        raise TypeError(f"{label} must be a dict, got {type(spec).__name__}")
    unexpected = set(spec) - set(_FIELD_ROLES)
    if unexpected:
        raise ValueError(
            f"{label} contains unknown field roles {sorted(unexpected)}; "
            f"expected only {_FIELD_ROLES}"
        )
    value = spec.get(role, {})
    if not isinstance(value, dict):
        raise TypeError(f"{label}[{role!r}] must be a dict, got {type(value).__name__}")
    tensor_spec, pseudo_spec = _split_pseudoscalar_spec(
        value, source_label=f"{label}[{role!r}]"
    )
    validate_rank_spec(
        tensor_spec,
        allowed_ranks=(0, 1),
        source_label=f"{label}[{role!r}]",
    )
    if role == "operator" and (pseudo_fields := flatten_rank_spec(pseudo_spec)):
        raise ValueError(
            f"{label}[{role!r}] declares pseudoscalar ({_PSEUDO_RANK!r}) "
            f"fields {sorted(pseudo_fields)}, but the operator stream is "
            "parity-even and carries no pseudoscalar sector; pseudoscalars "
            "are supported for the 'drive' role and for output fields only"
        )
    return value


def _require_int(name: str, value: int, *, minimum: int) -> None:
    """Raise unless ``value`` is a true int (bools rejected) >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        relation = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}, got {value}")


def _rank_entries(
    rank_spec: RankSpecDict,
    path: tuple[str, ...] = (),
) -> list[tuple[str, tuple[str, ...], int]]:
    """Flatten a nested rank spec into ``(dotted_name, path, rank)`` leaves
    in deterministic declaration order."""
    entries: list[tuple[str, tuple[str, ...], int]] = []
    for key, value in rank_spec.items():
        leaf_path = (*path, key)
        if isinstance(value, dict):
            entries.extend(_rank_entries(value, leaf_path))
        else:
            entries.append((".".join(leaf_path), leaf_path, value))
    return entries


def _td_get(data: TensorDict, path: tuple[str, ...]) -> torch.Tensor:
    """Fetch a declared field from a TensorDict, raising a ``ValueError``
    that names the missing dotted path instead of a bare ``KeyError``."""
    key: str | tuple[str, ...] = path[0] if len(path) == 1 else path
    try:
        return data[key]
    except KeyError:
        raise ValueError(f"Missing declared field {'.'.join(path)!r}") from None


class MeshTransformer(Module):
    r"""Global similarity-covariant transformer on a :class:`DomainMesh`.

    The source tokens are codimension-one boundary cells with geometric
    measure.  The query tokens are ``domain.interior.points``.  All learned
    interactions are quadrature-weighted global integrals -- signed separable
    moments, plus (with ``query_decoder="kernel"``) one dense
    invariant pair kernel for the boundary-to-query read.  No graph edge,
    neighbour radius, Fourier coordinate, or absolute Cartesian component is
    part of the model.

    Parameters
    ----------
    n_spatial_dims : int
        Ambient dimension ``D``.
    output_field_ranks : PseudoAwareRankSpecDict
        Named point predictions, with rank 0 for invariant scalars, rank 1
        for polar vectors, and the string rank token ``"0o"`` for 2D
        pseudoscalars (rotation invariant, sign-flipping under reflection;
        requires ``n_spatial_dims == 2`` and ``drive_pseudo_dim > 0``).
    boundary_field_ranks : dict[str, FieldRoleRanks]
        Schema for each boundary-condition name.  Each value may contain an
        ``"operator"`` mapping (geometry/material conditioning) and a
        ``"drive"`` mapping (boundary data whose zero defines the homogeneous
        zero-input test).  Declared boundary fields are read from cell data.
        Drive fields may use the ``"0o"`` pseudoscalar token; operator fields
        may not (the operator stream is parity-even).
    global_field_ranks : FieldRoleRanks, optional
        Domain-level operator and drive fields read from
        ``DomainMesh.global_data``.  Drive fields may use the ``"0o"``
        pseudoscalar token (e.g. a circulation).
    reference_length_key : str or None, optional
        Scale gauge of the geometric nondimensionalization: coordinates and
        codimension-one measures are divided by ``L`` and ``L**(D-1)``.
        With the default ``None`` the gauge is intrinsic: ``L`` is the
        measure-weighted RMS radius of the boundary about its
        measure-weighted centroid (the radius of gyration),
        :math:`L=\sqrt{\sum_j\omega_j\lVert y_j-c\rVert^2/\sum_j\omega_j}`,
        accumulated in float64 and differentiable through the mesh.
        Because that statistic is positively homogeneous of degree one in
        the geometry, scale equivariance is unconditional -- there is no
        caller-supplied convention that can drift between training and
        inference -- and the estimate is refinement-convergent and smooth
        in the boundary.  (Rejected intrinsic gauges: whitening grants
        affine invariance, which is the wrong physics; total boundary
        measure is wrinkliness-sensitive; diameter is non-smooth; conformal
        radius is PDE-specific and requires a solve.)  A string instead
        names a positive scalar ``global_data`` leaf that supplies ``L``
        explicitly for canonically dimensioned applications; with a key the
        model consumes exactly the declared scalar, bitwise identical to
        models predating the intrinsic default.
    field_mode : {"linear", "zero_preserving_nonlinear", "quadratic"}
        ``linear`` guarantees fixed-geometry superposition.  The nonlinear
        mode guarantees zero drive produces zero output but does not claim
        superposition.  ``quadratic`` DECLARES the drive degree the way the
        linear mode declares linearity: the output is exactly a polynomial
        of degree :math:`\le 2` in the drive at fixed geometry, for any
        weights.  Structurally the quadratic mode is the linear machinery
        (drive lift, ``LinearMeshFieldBlock`` stack, linear kernel/moment
        query decode -- each exactly drive-linear and zero-preserving) plus
        ONE bilinear, operator-conditioned typed composition of the
        assembled query field state
        (:class:`~physicsnemo.experimental.nn.mesh_attention.block.QuadraticFieldReadIn`,
        the third field-mode class) applied immediately before the
        drive-linear output projection, so
        :math:`F = u + B(L_2 u, L_3 u)` with :math:`u` drive-linear.
        Zero preservation is inherited.  Why this mode exists (iteration
        34 diagnosis, ``studies/nonlinear_fragility.py``): the
        ``zero_preserving_nonlinear`` read-in has IMPLICIT drive degree
        ~21 (measured by the pre-registered amplitude probe on trained
        euler_bernoulli arms) against targets of exactly degree 1
        (velocity) and 2 (pressure); training suppresses the on-range
        amplitude of the spurious high degrees, not their existence, so
        off-range drive amplification -- direct (amplitude OOD,
        :math:`10^2`--:math:`10^8` blowups) or physical (near-eigenvalue
        operator amplification feeding the drive,
        :math:`10^6`--:math:`10^{13}`) -- detonates.  Declaring the degree
        removes the mechanism rather than damping it: no drive monomial
        above the target degree exists to be amplified.  Pre-registered
        test (iteration 35, logged before the q2 retraining): on
        euler_rotational the quadratic arm's near-eigenvalue error must
        fall from the nonlinear arms' :math:`10^6`--:math:`10^{13}` to
        below 1.0 (stretch: the ~0.24--0.30 renormalized-ordinary level of
        iteration 34); on euler_bernoulli circulation-OOD must fall below
        the nonlinear arm's 0.35 toward the ID level (< 0.15);
        in-distribution must stay within :math:`2\times` seed sd of the
        nonlinear arms; and the drive-scaling structural test must pass at
        machine precision.  Falsifier: if the quadratic arm matches
        on-range but still detonates off-range, the degree diagnosis was
        incomplete.
    query_decoder : {"moment", "kernel"}
        Boundary-to-query decoding operator.  ``moment`` is the separable
        signed-moment decoder with :math:`O(N_s+N_q)` cost; in linear mode
        with scalar outputs and no physical query-side vector it has the
        exact :math:`m\le2` angular ceiling documented in the README.
        ``kernel`` replaces the query cross blocks with a dense
        operator-conditioned pair-kernel decoder
        (:mod:`.kernel_decoder`) whose dictionary combines exact
        cell-integrated double-layer quadrature with learned smooth members,
        lifting that ceiling at a documented dense :math:`O(N_qN_s)` decode
        cost.  ``kernel_include_double_layer_member=False`` removes the
        exact double-layer base member (MLP-only ablation; see
        KernelBasisCrossDecoder's Parameters -- the member dictionary must
        stay non-empty).  ``kernel_include_single_layer_member=True`` adds the exact
        single-layer (monopole) member, which a double-layer-only dictionary
        lacks on multiply connected domains (default ``False`` preserves the
        pruned dictionary bitwise).  It requires 2D segment or 3D triangle
        boundary cells.  The boundary-to-boundary drive blocks are identical
        in both modes.
    bounded_output_gate_invariants : bool, default=False
        Feed the output projection's sigmoid gates compactified (bounded)
        geometry invariants instead of the raw ones: vector channels are
        rescaled by :math:`1/\sqrt{1+|v|^2}` before their Gram products
        (making the Gram diagonal the compactified radial invariant
        :math:`|v|^2/(1+|v|^2)`) and scalars pass through
        :math:`s/\sqrt{1+s^2}`.  The default ``False`` is bitwise identical
        to the historical model.  Rationale: the query operator state mixes
        the normalized query position into every channel, so the raw gate
        invariants grow like :math:`|x|^2` and the gate saturates
        doubly-exponentially beyond the training query radius -- the
        measured far-field collapse of the exterior potential-flow arms.
        With bounded gate inputs no saturation regime exists; radial
        structure reaches the output only through the non-saturating
        field-linear branches (and the kernel dictionary's exactly
        extrapolating members).  Measured (p17, 2026-07-04): the gate
        stops saturating and in-distribution error improves, but this
        knob ALONE does not fix the far field -- it unmasks polynomial
        growth in the geometry-vector output branches that the saturating
        gate had been suppressing.  See
        :class:`~physicsnemo.experimental.nn.mesh_attention.block.GeometryConditionedLinear`
        for the full design note, the pre-registered retraining test, and
        its measured outcome; ``bounded_query_geometry`` below is the
        source-side completion of that fix.
    bounded_query_geometry : bool, default=False
        Inject the *compactified* nondimensional query position
        :math:`\hat x = x/\sqrt{1+|x|^2}` (direction preserved, magnitude
        compactified into ``[0, 1)``) into the query-side operator state
        instead of the raw :math:`x`, so every learned quantity downstream
        of that injection -- the operator lift's Gram invariants, the
        pointwise input block, the drive lift's query-side geometry
        conditioning, the output projection's gates, and, critically, the
        geometry *vectors* that the direct-drive linear branches
        (``vector_from_scalar``/``vector_from_pseudo`` etc. in
        :class:`~physicsnemo.experimental.nn.mesh_attention.block.GeometryConditionedLinear`)
        multiply -- is a bounded function of the query radius by
        construction, converging to a direction-dependent limit as
        :math:`|x|\to\infty`.  All unbounded radial structure is then
        carried exclusively by the kernel decoder's pair-invariant
        features and exact singular members, which read the RAW normalized
        query points through a separate argument of :meth:`decode` (they
        are the physics and extrapolate analytically; this knob never
        touches them).  The map is O(D)-equivariant (a rescale of each
        position by a function of its own invariant norm) and acts on the
        already-nondimensionalized coordinate, so similarity equivariance
        is unchanged.  Source-side cell centroids stay raw: the boundary
        is compact (RMS radius one in the intrinsic gauge), so no source
        quantity has an unbounded radial direction to begin with.  Default
        ``False`` is bitwise identical to the historical model; the knob
        adds no parameters, so state dicts are interchangeable and the
        flag alone selects the parameterization.  History of the far-field
        ladder this knob closes: the far-field strong-inference study
        (iteration 27) diagnosed the exterior far-field failure as
        query-side coefficient extrapolation -- the output gate collapsed
        doubly-exponentially beyond the training annulus while the exact
        kernel members extrapolated exactly; bounding the GATE inputs only
        (iteration 28, ``bounded_output_gate_invariants``) fired its
        falsifier instructively: the collapse had been load-bearing,
        suppressing polynomially GROWING query-side direct-drive linear
        branches (lifted drive norms grow like :math:`r`--:math:`r^2`,
        geometry vector channels like :math:`r`), and unmasking them sent
        ``farfield_queries`` from 0.694 to 2.98.  Root cause: the PAIR of
        unbounded query-radius dependences.  This knob is the principled
        fix -- bound at the SOURCE, so both halves (gate inputs and
        linear-branch geometry vectors) are bounded at once.  For the
        query path it therefore subsumes ``bounded_output_gate_invariants``
        (whose gate inputs become bounded functions of radius either way);
        the two knobs stay independent and composable because they bound
        at different places -- this one at the model's query injection,
        the gate knob inside any ``GeometryConditionedLinear`` -- and the
        gate knob remains the measured instrument of iteration 28 with its
        own trained artifacts.  MEASURED OUTCOME (p18_ff_boundedquery_s17,
        2026-07-04, both knobs on, original annulus, 3000 steps, seed 17):
        the knob does exactly what it claims -- every traced query-side
        coefficient now continues smoothly beyond the training edge (M3
        coefficient extrapolation EXCLUDED for the first time on this
        benchmark: worst far/near trace ratio 1.03, zero oscillations;
        gate 0.62 -> 0.50 smoothly out to r=12) and in-distribution error
        is preserved (0.0259 in the 0.0215-0.0269 band) -- but the far
        field is only partially rescued: ``farfield_queries`` 0.452
        (baseline 0.694, gate-only regression 2.98), missing the
        pre-registered < 0.35 support bar, with mean signed far-band
        exponent delta +2.07 (vs the baseline's -24 collapse and the
        gate-only arm's +4.9 growth).  The residual pathology is purely
        representational (M2): (i) BOUNDED is not DECAYING -- the
        direct-drive part converges to a direction-dependent constant
        (~0.24, far exponent ~0) while the exact exterior field decays
        like :math:`r^{-2}`; (ii) the kernel message state carries the
        exact single-layer member's :math:`\log r` tail (measured far
        exponent ~:math:`1/\ln r`); the trained arm fits the annulus by
        near-cancellation of these two O(1) pieces, which degrades far
        out.  Honest conclusion of the three-iteration ladder: source
        bounding fixes the coefficient half by construction; exterior
        far-field ACCURACY additionally requires either training coverage
        of the far annulus (the M1 discriminator's partial 74% rescue) or
        explicit far-field decay structure in the parameterization -- a
        decaying radial multiplier on the query-side direct-drive
        branches, and monopole-controlled single-layer coefficients.
        Those two structural pieces are ``decaying_direct_drive`` and
        ``kernel_monopole_free_single_layer`` below (iteration 30, fourth
        rung of the ladder).
    decaying_direct_drive : bool, default=False
        Multiply the query-side direct-drive contribution -- the lifted
        global drive at the query, i.e. the
        :class:`~physicsnemo.experimental.nn.mesh_attention.block.GeometryConditionedLinear`
        output path fed by the query operator state rather than by the
        kernel message -- by the fixed analytic decay envelope

        .. math:: \frac{1}{1 + |x|^2}

        of the RAW (uncompactified) nondimensional query radius.  The
        envelope is dimensionless (:math:`x` is already divided by the
        reference length), O(D)-equivariant (a function of the invariant
        :math:`|x|^2` alone), exact rather than learned (no parameters; the
        default ``False`` is bitwise identical to the historical model and
        state dicts are interchangeable), and applied ONLY to the direct
        contribution: the output projection is exactly linear in its field
        state, so scaling the lifted query drive before the kernel message
        is added multiplies the direct part of the decoded output by
        exactly this factor while the member-mediated message part is
        untouched -- the exact members remain the sole carriers of the true
        radial physics.  Requires ``query_decoder="kernel"`` (the moment
        decoder has no direct/member split for the premise to act on).
        CHOICE OF POWER (exterior expansion, velocity target): in a 2D
        exterior domain the disturbance velocity of a harmonic flow is
        :math:`u = \Gamma x^\perp/(2\pi|x|^2) + Q\,x/(2\pi|x|^2) +
        O(r^{-2})`; for a closed body with no net flux :math:`Q=0`, and the
        circulation :math:`r^{-1}` term is topological -- carried exactly
        by the kernel members (the double layer's :math:`r^{-1}` tail) --
        so the leading order of the single-valued zero-flux disturbance
        velocity is the doublet, :math:`|u| \sim r^{-2}`.  The direct drive
        is a query-local learned function of constant global drive data
        with no boundary-integral structure, so nothing it carries may
        outlive the physical leading order: :math:`1/(1+|x|^2)\to 1` at the
        body (near field re-learnable through the compactified radius,
        which is invertible on the annulus) and :math:`\sim r^{-2}` far
        out.  In 3D the same envelope is the conservative
        monopole-velocity order (:math:`r^{-2}`); zero-flux 3D fields
        decay faster (:math:`r^{-3}`) and the members carry the exact
        rate.  History (far-field ladder; restated from iterations 27-29):
        27 diagnosed the exterior far-field failure as query-side
        coefficient extrapolation (gate collapse, M3); 28
        (``bounded_output_gate_invariants``) showed the collapse was
        load-bearing over polynomially GROWING direct-drive linear
        branches (farfield 0.694 -> 2.98); 29
        (``bounded_query_geometry``) bounded the query geometry at the
        source -- coefficient extrapolation is now excluded by
        construction (M3 traces flat) -- but farfield plateaued at 0.452
        because BOUNDED IS NOT DECAYING: the direct-drive contribution
        converges to a direction-dependent constant (~0.24) while the
        exact disturbance velocity decays like :math:`r^{-2}`.  This knob
        turns that bounded limit into the physical decay rate.  MEASURED
        OUTCOME (p19_ff_decay_s17, 2026-07-04; this knob and
        ``kernel_monopole_free_single_layer`` jointly, on top of both
        bounding knobs, original annulus, 3000 steps, seed 17):
        pre-registered support bar met on the fourth attempt --
        ``farfield_queries`` 0.694 -> 2.98 -> 0.452 -> **0.133** (< 0.35
        supported; the < 0.10 fully-fixed bar remains open),
        in-distribution preserved at 0.0229, circulation OOD improved to
        0.0227, and the far-band exponent departure shrank from +2.07
        (mean signed) to +0.34 (mean absolute; direct-drive far exponent
        -1.95, i.e. the envelope's :math:`r^{-2}`, with near bands exact
        to 0.03).  The remaining slow leak is the deflated single layer's
        licensed :math:`r^{-1}` dipole survivor (member exponent -1.27 at
        r in [8,12]), whose zero-circulation cancellation to :math:`r^{-2}`
        is fitted rather than structural.
    kernel_monopole_free_single_layer : bool, default=False
        With ``query_decoder="kernel"`` and
        ``kernel_include_single_layer_member=True``, deflate the exact
        single-layer member column by its measure-weighted boundary mean
        (a linear, differentiable, rank-one projection -- exactly
        equivalent to projecting the conditioned single-layer charge
        density onto the zero-net-charge subspace for every head and value
        channel at once; see
        :class:`~physicsnemo.experimental.nn.mesh_attention.kernel_decoder.KernelBasisCrossDecoder`).
        Every deflated member carries exactly zero net charge, so the
        single layer's monopole tail (:math:`\log r` in 2D, :math:`1/r` in
        3D) is structurally absent no matter what the learned coefficients
        do.  PHYSICS LICENSE: in 2D exterior problems the single-layer
        :math:`\log r` tail corresponds to net monopole charge; for the
        disturbance velocity of a closed body with no net flux the
        monopole vanishes identically, so the projection removes only
        unphysical directions.  It must stay opt-in because it is NOT
        licensed for problems with genuine net flux -- screened operators,
        volumetric sources or sinks, nonzero-mean Neumann data, and
        multiply connected domains where the single-layer member exists
        precisely to carry net flux through handles.  History (iteration
        29 measurement motivating this knob): the trained singpair arm's
        exact single-layer member carried a :math:`\log r` tail (message
        state norm 2.4 at r=12) that the annulus fit controlled by
        near-cancellation instead of structure, and the cancellation
        degraded far out.  Default ``False`` is bitwise identical (no
        parameters added; the flag alone selects the parameterization).
        MEASURED OUTCOME (p19_ff_decay_s17, 2026-07-04, jointly with
        ``decaying_direct_drive``; see that knob's entry for the full
        numbers): the single-layer message tail dropped from the log-r
        flattening (exponent +0.44 at r in [8,12]) to a decaying
        :math:`r^{-1.27}` dipole survivor, and the trained arm still
        conditions a large net monopole (vector-channel norm ~34 on every
        probed case, recorded as the study's ``sl_net_monopole``
        diagnostic) that the deflation removes structurally -- the
        projection is load-bearing, not vacuous.
    query_chunk_size : int, default=65536
        Maximum number of independent query points decoded together.
    kernel_checkpoint_query_chunks : bool, default=False
        With ``query_decoder="kernel"``, recompute each decode chunk's dense
        pair activations in the backward pass instead of retaining them
        (:func:`torch.utils.checkpoint.checkpoint`, non-reentrant).  Query
        chunking alone bounds only the *forward* peak: during training,
        autograd retains every chunk's :math:`O(\text{chunk}\times N_s)`
        decode intermediates, so training memory grows as the full dense
        :math:`O(N_qN_s)` -- the measured product-scope wall
        (:math:`\approx742` GiB extrapolated at :math:`10^5` boundary cells
        :math:`\times\ 10^6` queries; see the scale study).  With the knob
        on, retained decode activations drop to one chunk plus the
        :math:`O(N_q)` outputs, at the cost of one extra decode forward
        inside backward.  The decoder is RNG-free and its chunk evaluation
        is shape-deterministic, so the recomputation -- and therefore every
        gradient -- is bitwise identical to the retained-graph result.
        Default ``False`` preserves the historical autograd graph exactly
        (the knob adds no parameters and does not change the forward
        output in either state).  Ignored by the moment decoder, which has
        no dense pair term.
    kernel_auxiliary_scale_key : str or None, default=None
        With ``query_decoder="kernel"``, the name of a DECLARED rank-0
        global operator field (dotted paths address nested leaves, as with
        ``reference_length_key``) whose per-case value is the dimensionless
        ratio :math:`\lambda=\delta/L_{\mathrm{ref}}` of a declared
        auxiliary length scale to the model's scale gauge.  The kernel
        decoder then appends a second copy of its pair invariants rescaled
        to the :math:`\delta` gauge (:math:`a/\lambda^2`,
        :math:`b/\lambda`, :math:`v\cdot r/\lambda`), feeding ONLY the
        learned smooth members -- the exact singular members are untouched
        -- so the knob requires ``kernel_mlp_members > 0`` (no carrier
        otherwise; rejected at construction).  MOTIVATION (AirFRANS, steady
        RANS at :math:`\mathrm{Re}\sim4\times10^6`): the measured velocity
        error concentrates at the wall -- 49% of the MSE inside
        :math:`d/c<10^{-4}` -- because the boundary layer lives at
        :math:`\delta/c\sim\mathrm{Re}^{-1/2}\approx5\times10^{-4}`, a
        scale the kernels previously saw only through per-source scalar
        conditioning while all pair-radial structure was chord-scale.  The
        knob is the per-problem CONTRACT that fixes this: declare
        :math:`\lambda` (e.g. :math:`\mathrm{Re}^{-1/2}`) as a
        dimensionless global input, and the boundary-layer scale appears
        at order one in the smooth members' radial argument.  PHYSICS
        LICENSE: use exactly when the problem class carries a known second
        physical scale (a boundary layer, a screening length); the value
        is a physical declaration like ``reference_length_key``, read RAW
        from ``global_data`` at encode time (before the operator lift) and
        required to be finite and positive -- it is never a learned
        feature, although the same declared field also conditions the
        encoder like any other operator scalar.  Similarity covariance is
        preserved by construction (:math:`r` is already
        :math:`L_{\mathrm{ref}}`-normalized and :math:`\lambda` is
        dimensionless), and parity typing is unchanged (scalar division
        alters no transformation law).  Default ``None`` adds no
        parameters and is bitwise identical to the pre-extension model
        (state dict and outputs).  MEASURED OUTCOME (H4 verdict,
        book/10-notebook.qmd): REFUTED by its own pre-registered capacity
        control -- ``kernel_mlp_members=8`` WITHOUT the declared scale
        beat the declared arm ~6x on AirFRANS velocity, and dividing the
        invariants by :math:`\lambda\sim5\times10^{-4}` fed the member
        MLP features scaled by ~4e6 (an optimization handicap).  The knob
        stays runnable (refuted configurations remain reproducible); the
        successor is ``kernel_log_radial_features`` below.
    kernel_log_radial_features : bool, default=False
        With ``query_decoder="kernel"``, append the log-radial pair-feature
        block -- ``ln(a + eps)`` plus the scale-free normalized alignments
        ``b / sqrt(a + eps)`` and ``v . r / sqrt(a + eps)`` -- to the
        kernel decoder's smooth-member MLP input, after every other block
        (see
        :class:`~physicsnemo.experimental.nn.mesh_attention.kernel_decoder.KernelBasisCrossDecoder`).
        MOTIVATION (H4-L, pre-registered as V4 in the velocity-front
        fan-out): any power-law auxiliary length scale
        :math:`\delta=L_{\mathrm{ref}}\,\Pi^\alpha` built from a
        dimensionless group :math:`\Pi` is LINEAR in log space,
        :math:`\ln(r/\delta)=\ln r-\alpha\ln\Pi`, so handing the smooth
        members ``ln a`` alongside the dimensionless-group conditioning
        they already receive (e.g. :math:`\ln\mathrm{Re}` as a global
        operator scalar) makes ANY power-law scale learnable -- no
        declared exponent, no semantically named input, PDE-general.
        LINEAGE: the declared-scale contract
        (``kernel_auxiliary_scale_key`` above) was REFUTED by its own
        capacity control (plain members won ~6x on AirFRANS velocity);
        this knob keeps the winning members and turns the scale's
        exponent into learnable log-space structure instead of a
        declaration.  The block feeds ONLY the learned smooth members, so
        ``kernel_mlp_members > 0`` is required (no carrier otherwise;
        rejected at construction), and it composes freely with the
        auxiliary-scale contract (independent feature blocks; the MLP
        input widths add).  Similarity covariance and parity typing are
        preserved feature by feature (``ln(a + eps)`` is even like ``a``;
        the normalized alignments are odd in the normal and state vector
        exactly like their parents).  Default ``False`` adds no
        parameters and is bitwise identical to the pre-extension model
        (state dict and outputs).
    attention_chunk_size : int or None, default=65536
        Maximum entities passed through a typed attention projection at once.
        Chunking changes temporary memory, not the moment operator. ``None``
        disables projection chunking.
    per_boundary_moment_pool_balanced : bool, default=False
        With ``per_boundary_moment_pool=True``, offset each boundary's
        moment-pool log-gain by ln(mean boundary measure) - ln(boundary
        measure) per sample, so at initialization every boundary
        contributes equally instead of by raw measure (the external-review
        "balanced" arm; a reparameterized initialization, not a smaller
        hypothesis class -- the gains can learn back the plain sum).
        Requires the per-boundary pool.
    per_boundary_moment_pool : bool, default=False
        Compute every encoder attention moment per declared boundary and
        combine the per-boundary moments through learned, dimensionless
        per-boundary/per-head log-gains:
        :math:`M=\sum_b e^{g_b} M_b=\sum_b e^{g_b+\ln A_b}\,\bar M_b`
        with :math:`\bar M_b` the boundary's measure-averaged moment and
        :math:`A_b` its total measure.  PHYSICS LICENSE: Green's identity
        splits over boundary components, so a per-component decomposition
        of the boundary integrals recombined by pure numbers is exact
        structure; the gain is dimensionless per (boundary, head), so
        similarity covariance, parity typing, drive-linearity (the gain is
        drive-independent), and zero preservation are all unchanged.
        MOTIVATION (DrivAerML all-boundary probe, 2026-07): the moments are
        quadrature-weighted with raw cell measures, and the tunnel panels
        carry ~1e6x the vehicle's measure -- at init the tunnel/vehicle
        moment-contribution ratio is ~5e6 (operator stream) to ~1e9 (drive
        stream), and 500 trained epochs suppressed drive values by only
        ~20x of the needed ~1e7: an optimization pathology, since ignoring
        a boundary requires the type-conditioned features to fight the
        measure ratio multiplicatively.  With the pool, that correction is
        an additive shift of a per-boundary log-scale parameter.  At
        initialization the gains are zero, so the pooled moments reproduce
        the plain quadrature sum exactly (up to floating-point summation
        order).  The default ``False`` adds no parameters and does not
        change the forward pass (state dicts and outputs are bitwise the
        historical model).  Scope: the encoder's operator/drive blocks and
        the moment decoder's query cross moments; the kernel decoder's
        dense pair read is per-cell-weighted and is not pooled.

    trace_of : str or None, default=None
        Declare that the query mesh IS the named boundary's cell centroids,
        index-aligned -- query ``i`` is cell ``i`` of
        ``boundary_field_ranks[trace_of]``, in cell order -- turning the
        decode into a declared **boundary-trace** (boundary-to-boundary)
        read instead of a boundary-to-interior map.  Requires
        ``query_decoder="kernel"``.  MOTIVATION (the GeoTransolver-gap
        verdict, book/18-notebook.qmd @sec-nb-geot-gap-verdict): on
        surface-only tasks every query sits at the kernels' singular limit
        with no identity, and the exact double-layer member evaluated AT a
        query on its own panel was measured returning the INTERIOR jump
        branch (:math:`-1/2`) while the physical surface values are the
        exterior trace (:math:`+1/2` at :math:`+\epsilon` outside) -- a
        term that flips sign under infinitesimal displacement, repairable
        by no learned coefficient.  The declaration licenses exactly two
        structural changes, both denied to bare interior queries:

        * **Side-corrected self-terms.**  Every query's own-panel
          double-layer entry is replaced by the exact one-sided limit of
          the declared trace side: :math:`+1/2`, in both 2D and 3D, for
          any flat panel -- the side the cell normals point toward
          (external aerodynamics: normals out of the body, into the
          fluid).  The on-panel principal value of a flat panel's own
          integral is exactly zero (:math:`n\cdot(x-y)\equiv 0` in the
          panel plane), so the jump correction IS the whole self-term;
          off-panel entries are regular and unchanged, so the corrected
          trace equals the exterior limit exactly (pinned numerically:
          constant density on a closed polygon/triangulation sums to the
          analytic exterior value 0 to roundoff).  The exact single-layer
          member needs NO value correction and receives none: the
          single-layer potential is continuous across its own layer (only
          its normal derivative jumps, by minus the density) and its
          closed forms already evaluate the exact finite on-panel value.
          See :func:`~.kernel_decoder.exterior_trace_self_entries`.
        * **Own-cell typed read-outs.**  Each query receives bias-free
          :class:`~.attention.TypedProjection` read-outs of its own cell's
          post-attention encoded states: the cell's operator state joins
          the query operator state (the query now knows *which* panel it
          is -- encoded local geometry, normal direction), and the cell's
          drive state joins the query field state (the boundary solve
          already computes the trace; Dirichlet data IS the trace on
          Dirichlet patches).  The drive read-out is an exactly linear
          typed channel mix (no invariant lift, no bias), so
          drive-linearity (linear and quadratic modes) and zero
          preservation (nonlinear mode) survive; both read-outs are typed
          projections of typed state, so O(D)/similarity equivariance
          survives.  Query independence survives as independence *given
          the declared identity map*: the cell identity is declared input,
          not inferred from the query set, each query still reads only its
          own declared cell, and chunking remains a pure memory control.

        ``decode`` validates the declaration loudly: the query count must
        equal the declared boundary's cell count (the ORDER of the
        alignment is the caller's declaration -- the recipe pipeline pins
        it by test).  Default ``None`` adds no parameters and is bitwise
        identical to the pre-extension model (state dict and outputs).
        Pre-registered acceptance (the verdict memo): trace-mode singpair
        beats plain singpair :math:`\ge 5\times` on DrivAerML surface val
        at matched protocol; falsifier: no improvement means the trace
        defect was cosmetic and capacity (H-C) owns the gap.

    diagnostic_local_query_features : bool, default=False
        CONTRACT-BREAKING DIAGNOSTIC -- never a keeper (pre-registered as
        probe P2 of the H-C decomposition, book/18-notebook.qmd
        @sec-nb-aga-fleet).  Requires ``trace_of``.  Each trace query
        additionally reads its own cell's local geometry directly: the unit
        cell normal (injected through the query slot of the existing typed
        normal channel, inheriting the source-side transformation law), the
        log cell area relative to the sample's median cell area, and the
        cell curvature nondimensionalized by
        ``reference_length**n_manifold_dims`` (both scalars appended as two
        channels that are zero on source rows).  All three features are
        similarity-invariant or equivariant, so the O(D)/similarity
        contract SURVIVES; what breaks is the boundary-integral
        *information diet* -- the query no longer learns its local geometry
        through the kernel path but is handed it.  Purpose: if this arm
        recovers most of GeoTransolver's early-epoch efficiency, the
        efficiency lives in local-geometry shortcuts, and the principled
        response is a declared local corrector, not feature injection.
        Default ``False`` adds no parameters and is bitwise identical to
        the pre-extension model.

    trace_self_correction : bool, default=True
        Decomposition switch for the declared boundary-trace mode (the trace
        factorial, external-review P1): whether the exact exterior jump
        correction (+1/2 own-panel self-entries; see
        :func:`~.kernel_decoder.exterior_trace_self_entries`) is applied.
        ``False`` leaves own-panel entries on the closed form's accidental
        signed-zero branch, exactly as a non-trace model evaluates
        on-boundary queries.  Requires ``trace_of``; default preserves
        every existing trace configuration bitwise.
    trace_readouts : bool, default=True
        Decomposition switch for the declared boundary-trace mode: whether
        the two own-cell read-outs (operator and drive typed projections of
        the query's own post-attention states) are constructed and applied.
        ``False`` adds no read-out parameters (state dict matches a
        pre-read-out model).  With both this and
        ``trace_self_correction`` False, trace mode reduces to pure query
        placement (queries are the declared boundary's cell centroids).
        Requires ``trace_of``; default preserves existing configurations
        bitwise.
    kernel_local_pair_features : str or None, default=None
        Local-corrector probe modes (task #53; pre-registered in the
        program notebook).  ``"windowed"`` (probe A) appends five even
        similarity-invariant scalar channels to the smooth-member MLP's
        per-pair features -- the subtended-angle window
        :math:`\theta/(1+\theta)` and the windowed query-/source-side
        squashed local scalars -- so the SAME information as the P2
        query-stream diagnostic enters through the KERNEL instead.
        ``"near_only"`` (probe B) replaces the window by a C^1 smoothstep
        that is exactly zero for :math:`\theta\le\theta_c/2` (compact
        near-field support; :math:`\theta_c` = ``kernel_near_theta``).
        ``"global_control"`` (probe C) keeps the channel width (matched
        parameter count) but carries only per-sample measure-weighted
        pooled scalars -- the sharing control.  The subtended angle is
        dimension-generic (:func:`.kernel_decoder.subtended_angle`:
        :math:`h=\mu^{1/m}`, never an area over a distance).  Requires
        ``query_decoder='kernel'``, ``kernel_mlp_members > 0``, and
        ``trace_of``.  Default ``None`` adds no parameters and is bitwise
        identical to the pre-extension model.
    kernel_decode_backend : {"dense", "barnes_hut"}, default="dense"
        Decode evaluation backend (``query_decoder="kernel"``).  ``"dense"``
        is the exact :math:`O(QS)` pairwise operator (bitwise unchanged
        default).  ``"barnes_hut"`` is the hierarchical approximation of
        task #41: near pairs (subtended size above ``kernel_bh_theta``)
        evaluate through the same closed forms and smooth-member MLP
        pairwise, bitwise per pair; far cluster nodes contribute through
        exactly-aggregated channel-resolved densities against analytic far
        kernels (exact members) and node virtual sources (smooth members).
        Deterministic and query-set independent by construction; deviation
        from dense is measured and theta-controlled (theta -> 0 recovers
        dense).  v1 scope: 3D, no polynomial members, no monopole-free
        deflation, no local_pair_features, no checkpoint_query_chunks (each
        rejected loudly; see the decoder docstring).
    kernel_bh_theta : float, default=0.5
        Barnes--Hut opening threshold: a source node whose AABB diagonal
        subtends less than ``bh_theta`` from the query is evaluated in the
        far field.  Smaller is more accurate and slower.
    kernel_bh_leaf_size : int, default=32
        Cluster-tree leaf size; a throughput knob only (larger leaves mean
        fewer tree levels and more exact near pairs).
    kernel_near_theta : float, default=0.25
        Near-field threshold :math:`\theta_c` for the ``"near_only"``
        probe; calibrate the resulting near fraction on a real sample
        before drawing locality conclusions (see
        ``scratch/measure_near_fraction.py``).

    drive_pseudo_dim : int, default=0
        Width of the drive stream's 2D pseudoscalar (``0o``) channel sector.
        The default 0 disables the sector and is bitwise identical to the
        pre-extension model (state dict and outputs).  A positive width
        requires ``n_spatial_dims == 2`` and is required whenever any field
        is declared with the ``"0o"`` token; it is also legal without pseudo
        declarations, in which case pseudo channels arise internally from
        vector wedges.  See :mod:`.attention` for the measured failures that
        motivated the sector and the closed 2D product set that implements
        it.

    Notes
    -----
    All declared physical fields are expected to be nondimensional.  Rank-1
    leaves are polar vectors and must be transformed together with the mesh
    under rotations or reflections.  Pseudoscalar (``"0o"``) leaves have
    scalar shape but flip sign under reflections of the frame and data; they
    are supported in two dimensions only.  Axial vectors (the 3D analogue)
    and higher tensor types require a future representation extension and
    are rejected rather than silently treated as channels.

    Setting ``operator_vector_dim=drive_vector_dim=0`` (which requires
    ``vector_rank=0``) selects the scalar-only ablation mode: the encoder
    carries no oriented (rank-1) state between cells, and boundary normals
    reach the model only through scalar geometric invariants (the operator
    lift's Gram invariants of position and normal, and -- with
    ``query_decoder="kernel"`` -- the decoder's pair invariants and exact
    double-layer member).  The vector dimensions must be both zero or both
    positive; mixed settings are rejected because they leave one stream's
    vectors without an equivariant read-out path.  In scalar-only mode,
    rank-1 output and drive fields are rejected for the same reason, while
    rank-1 operator fields remain legal as Gram-invariant scalars.
    """

    def __init__(
        self,
        n_spatial_dims: int,
        output_field_ranks: PseudoAwareRankSpecDict,
        boundary_field_ranks: dict[str, FieldRoleRanks],
        global_field_ranks: FieldRoleRanks | None = None,
        reference_length_key: str | None = None,
        field_mode: FieldMode = "linear",
        query_decoder: QueryDecoder = "moment",
        kernel_mlp_members: int = 8,
        kernel_include_double_layer_member: bool = True,
        kernel_include_polynomial_members: bool = True,
        kernel_include_single_layer_member: bool = False,
        kernel_monopole_free_single_layer: bool = False,
        kernel_checkpoint_query_chunks: bool = False,
        kernel_auxiliary_scale_key: str | None = None,
        kernel_log_radial_features: bool = False,
        kernel_local_pair_features: str | None = None,
        kernel_near_theta: float = 0.25,
        kernel_decode_backend: str = "dense",
        kernel_bh_theta: float = 0.5,
        kernel_bh_leaf_size: int = 32,
        bounded_output_gate_invariants: bool = False,
        bounded_query_geometry: bool = False,
        decaying_direct_drive: bool = False,
        operator_scalar_dim: int = 32,
        operator_vector_dim: int = 8,
        drive_scalar_dim: int = 64,
        drive_vector_dim: int = 16,
        drive_pseudo_dim: int = 0,
        operator_layers: int = 3,
        drive_layers: int = 2,
        query_layers: int | None = None,
        heads: int = 4,
        scalar_rank: int = 8,
        vector_rank: int = 4,
        query_chunk_size: int = 65536,
        attention_chunk_size: int | None = 65536,
        per_boundary_moment_pool: bool = False,
        per_boundary_moment_pool_balanced: bool = False,
        trace_of: str | None = None,
        trace_self_correction: bool = True,
        trace_readouts: bool = True,
        diagnostic_local_query_features: bool = False,
    ) -> None:
        """Validate the declared schema and build the full stack (encoder
        blocks, drive lift, query decoder, typed outputs).  Every public
        parameter is documented in the class docstring; the coherence rules
        enforced below (scalar-only mode, pseudo planarity, trace and
        diagnostic knob requirements) are stated there as well."""
        _require_int("n_spatial_dims", n_spatial_dims, minimum=2)
        if not isinstance(per_boundary_moment_pool, bool):
            raise TypeError(
                "per_boundary_moment_pool must be a bool, got "
                f"{per_boundary_moment_pool!r}"
            )
        if not isinstance(per_boundary_moment_pool_balanced, bool):
            raise TypeError(
                "per_boundary_moment_pool_balanced must be a bool, got "
                f"{per_boundary_moment_pool_balanced!r}"
            )
        if per_boundary_moment_pool_balanced and not per_boundary_moment_pool:
            raise ValueError(
                "per_boundary_moment_pool_balanced=True requires "
                "per_boundary_moment_pool=True: the balance offsets the "
                "per-boundary moment-pool log-gains (external-review "
                "balanced arm)"
            )
        if not isinstance(boundary_field_ranks, dict):
            raise TypeError("boundary_field_ranks must be a dict")
        if not boundary_field_ranks:
            raise ValueError("boundary_field_ranks must declare at least one boundary")
        if global_field_ranks is not None and not isinstance(global_field_ranks, dict):
            raise TypeError("global_field_ranks must be a dict or None")
        if reference_length_key is not None and (
            not isinstance(reference_length_key, str) or not reference_length_key
        ):
            raise TypeError("reference_length_key must be a non-empty string or None")
        if kernel_auxiliary_scale_key is not None and (
            not isinstance(kernel_auxiliary_scale_key, str)
            or not kernel_auxiliary_scale_key
        ):
            raise TypeError(
                "kernel_auxiliary_scale_key must be a non-empty string or None"
            )
        if field_mode not in ("linear", "zero_preserving_nonlinear", "quadratic"):
            raise ValueError(
                "field_mode must be 'linear', 'zero_preserving_nonlinear', "
                "or 'quadratic'"
            )
        if query_decoder not in ("moment", "kernel"):
            raise ValueError("query_decoder must be 'moment' or 'kernel'")
        if query_decoder == "kernel" and n_spatial_dims not in (2, 3):
            raise ValueError(
                "query_decoder='kernel' requires n_spatial_dims 2 or 3: its "
                "exact double-layer member is dimension-dispatched to "
                "straight-segment and flat-triangle quadrature"
            )
        if decaying_direct_drive and query_decoder != "kernel":
            raise ValueError(
                "decaying_direct_drive requires query_decoder='kernel': the "
                "knob's premise is that the kernel dictionary's exact members "
                "are the sole carriers of the true radial physics while the "
                "query-side direct drive is a bounded local term; the moment "
                "decoder has no such direct/member split"
            )
        if kernel_auxiliary_scale_key is not None and query_decoder != "kernel":
            raise ValueError(
                "kernel_auxiliary_scale_key requires query_decoder='kernel': "
                "the declared auxiliary scale enters only the kernel "
                "decoder's pair invariants, and the moment decoder has no "
                "pair-radial argument for it to rescale"
            )
        if kernel_log_radial_features and query_decoder != "kernel":
            raise ValueError(
                "kernel_log_radial_features requires query_decoder='kernel': "
                "the log-radial features extend the kernel decoder's pair "
                "invariants, and the moment decoder has no dense pair "
                "features to extend"
            )
        if not isinstance(trace_self_correction, bool):
            raise ValueError("trace_self_correction must be a bool")
        if not isinstance(trace_readouts, bool):
            raise ValueError("trace_readouts must be a bool")
        if trace_of is None and not (trace_self_correction and trace_readouts):
            raise ValueError(
                "trace_self_correction/trace_readouts decompose the declared "
                "boundary-trace mode and are meaningful only with trace_of; "
                "leave them at their defaults for trace-free models"
            )
        if trace_of is not None:
            if not isinstance(trace_of, str) or not trace_of:
                raise TypeError("trace_of must be a non-empty string or None")
            if trace_of not in boundary_field_ranks:
                raise ValueError(
                    f"trace_of={trace_of!r} must name a declared boundary in "
                    f"boundary_field_ranks; declared boundaries: "
                    f"{sorted(boundary_field_ranks)}"
                )
            if query_decoder != "kernel":
                raise ValueError(
                    "trace_of requires query_decoder='kernel': the declared "
                    "boundary-trace mode side-corrects the exact singular "
                    "members' own-panel entries per the jump relation, and "
                    "the moment decoder has no exact members to correct"
                )
        if not isinstance(diagnostic_local_query_features, bool):
            raise TypeError(
                "diagnostic_local_query_features must be a bool, got "
                f"{diagnostic_local_query_features!r}"
            )
        if kernel_local_pair_features is not None:
            if query_decoder != "kernel":
                raise ValueError(
                    "kernel_local_pair_features requires query_decoder="
                    "'kernel': the probe block lives in the kernel decoder"
                )
            if trace_of is None:
                raise ValueError(
                    "kernel_local_pair_features requires trace_of: the "
                    "windowed and near-only modes read query-side local "
                    "scalars through the declared trace identity map, and "
                    "the probe protocol (task #53) is defined on trace arms"
                )
        if diagnostic_local_query_features and trace_of is None:
            raise ValueError(
                "diagnostic_local_query_features requires trace_of: the "
                "diagnostic features (own-cell normal, log relative area, "
                "nondimensional curvature) are defined by the query's "
                "declared own cell, which only the boundary-trace mode "
                "provides.  This knob is a CONTRACT-BREAKING DIAGNOSTIC "
                "(the query reads its own local geometry directly instead "
                "of through the boundary-integral information diet); it "
                "exists to locate representational-efficiency gaps and "
                "must never ship as a default"
            )
        for name, value, minimum in (
            ("operator_scalar_dim", operator_scalar_dim, 1),
            ("operator_vector_dim", operator_vector_dim, 0),
            ("drive_scalar_dim", drive_scalar_dim, 1),
            ("drive_vector_dim", drive_vector_dim, 0),
            ("drive_pseudo_dim", drive_pseudo_dim, 0),
            ("operator_layers", operator_layers, 0),
            ("drive_layers", drive_layers, 0),
            ("heads", heads, 1),
            ("scalar_rank", scalar_rank, 0),
            ("vector_rank", vector_rank, 0),
            ("query_chunk_size", query_chunk_size, 1),
        ):
            _require_int(name, value, minimum=minimum)
        # Coherence rule for the encoder's vector (rank-1) channel widths:
        # either both are zero (scalar-only ablation mode) or both are
        # positive (the default vector-carrying encoder).  A mixed setting is
        # rejected rather than silently supported because a vector-less
        # operator stream leaves declared drive vectors with no equivariant
        # read-out path, and a vector-less drive stream makes operator
        # vectors unreadable by the field decoder.  In scalar-only mode a
        # positive ``vector_rank`` would be silently ignored by every block
        # (there is no vector basis to project keys from), so it must be 0.
        if (operator_vector_dim == 0) != (drive_vector_dim == 0):
            raise ValueError(
                "operator_vector_dim and drive_vector_dim must be both zero "
                "(scalar-only mode) or both positive; got "
                f"operator_vector_dim={operator_vector_dim} and "
                f"drive_vector_dim={drive_vector_dim}"
            )
        scalar_only = operator_vector_dim == 0
        if scalar_only and vector_rank:
            raise ValueError(
                "scalar-only mode (operator_vector_dim=drive_vector_dim=0) "
                f"requires vector_rank=0, got vector_rank={vector_rank}"
            )
        if attention_chunk_size is not None:
            _require_int("attention_chunk_size", attention_chunk_size, minimum=1)
        if query_layers is None:
            # The quadratic mode's query machinery is the linear machinery
            # (its degree is added by the read-in composition, not by depth).
            query_layers = 1 if field_mode in ("linear", "quadratic") else 2
        _require_int("query_layers", query_layers, minimum=1)
        if scalar_rank + vector_rank == 0:
            raise ValueError("at least one attention rank must be positive")

        if not isinstance(output_field_ranks, dict):
            raise TypeError("output_field_ranks must be a dict")
        output_tensor_ranks, output_pseudo_ranks = _split_pseudoscalar_spec(
            output_field_ranks, source_label="output_field_ranks"
        )
        validate_rank_spec(
            output_tensor_ranks,
            allowed_ranks=(0, 1),
            source_label="output_field_ranks",
        )
        if not flatten_rank_spec(output_field_ranks):
            raise ValueError("output_field_ranks must contain at least one field")
        if scalar_only:
            # Without vector channels the model carries no oriented state, so
            # rank-1 predictions have no equivariant basis and rank-1 drive
            # inputs would be silently dead (their only read-out paths dot
            # them against operator vectors).  Rank-1 *operator* inputs stay
            # legal: the operator lift folds them into scalar Gram
            # invariants against the position and normal channels.
            if vector_outputs := sorted(
                name
                for name, rank in flatten_rank_spec(output_field_ranks).items()
                if rank == 1
            ):
                raise ValueError(
                    "scalar-only mode cannot predict rank-1 output fields "
                    f"{vector_outputs}; use positive vector dimensions or "
                    "declare only rank-0 outputs"
                )

        global_field_ranks = {} if global_field_ranks is None else global_field_ranks
        boundary_names = sorted(boundary_field_ranks)
        for name in boundary_names:
            if not isinstance(name, str) or not name:
                raise ValueError("boundary condition names must be non-empty strings")
            for role in _FIELD_ROLES:
                _role_spec(
                    boundary_field_ranks[name],
                    role,
                    label=f"boundary_field_ranks[{name!r}]",
                )
            operator_names = set(
                flatten_rank_spec(boundary_field_ranks[name].get("operator", {}))
            )
            drive_names = set(
                flatten_rank_spec(boundary_field_ranks[name].get("drive", {}))
            )
            if overlap := operator_names & drive_names:
                raise ValueError(
                    f"Boundary {name!r} fields cannot have both operator and "
                    f"drive roles: {sorted(overlap)}"
                )
            if scalar_only:
                drive_ranks = flatten_rank_spec(
                    boundary_field_ranks[name].get("drive", {})
                )
                if vector_drives := sorted(
                    field for field, rank in drive_ranks.items() if rank == 1
                ):
                    raise ValueError(
                        f"scalar-only mode cannot consume rank-1 drive fields "
                        f"{vector_drives} on boundary {name!r}: with no "
                        "operator vector channels they would have no "
                        "equivariant read-out path"
                    )
        for role in _FIELD_ROLES:
            _role_spec(global_field_ranks, role, label="global_field_ranks")
        global_operator_names = set(
            flatten_rank_spec(global_field_ranks.get("operator", {}))
        )
        global_drive_names = set(flatten_rank_spec(global_field_ranks.get("drive", {})))
        if overlap := global_operator_names & global_drive_names:
            raise ValueError(
                "Global fields cannot have both operator and drive roles: "
                f"{sorted(overlap)}"
            )
        if scalar_only:
            global_drive_ranks = flatten_rank_spec(global_field_ranks.get("drive", {}))
            if vector_drives := sorted(
                field for field, rank in global_drive_ranks.items() if rank == 1
            ):
                raise ValueError(
                    "scalar-only mode cannot consume rank-1 global drive "
                    f"fields {vector_drives}: with no operator vector "
                    "channels they would have no equivariant read-out path"
                )

        # Pseudoscalar ("0o") coherence rules.  Declarations are legal for
        # drive and output roles only (enforced in ``_role_spec``); here the
        # sector-wide constraints: 2D only, a positive channel width, and no
        # scalar-only mode (the wedge and rotation products that give pseudo
        # channels their equivariant read-out paths need vector channels).
        pseudo_declared = sorted(
            {
                field
                for name in boundary_names
                for field, rank in flatten_rank_spec(
                    boundary_field_ranks[name].get("drive", {})
                ).items()
                if rank == _PSEUDO_RANK
            }
            | {
                field
                for field, rank in flatten_rank_spec(
                    global_field_ranks.get("drive", {})
                ).items()
                if rank == _PSEUDO_RANK
            }
            | set(flatten_rank_spec(output_pseudo_ranks))
        )
        if (pseudo_declared or drive_pseudo_dim) and n_spatial_dims != 2:
            raise ValueError(
                "the pseudoscalar ('0o') sector requires n_spatial_dims == 2: "
                "it is two-dimensional by design, and in 3D the analogous "
                "parity-odd object is the axial vector, which is out of "
                f"scope; got n_spatial_dims={n_spatial_dims} with "
                f"pseudoscalar fields {pseudo_declared} and "
                f"drive_pseudo_dim={drive_pseudo_dim}"
            )
        if (pseudo_declared or drive_pseudo_dim) and scalar_only:
            raise ValueError(
                "scalar-only mode (operator_vector_dim=drive_vector_dim=0) "
                "cannot carry the pseudoscalar sector: the wedge and rotation "
                "products that read pseudoscalars in and out require vector "
                f"channels; got pseudoscalar fields {pseudo_declared} and "
                f"drive_pseudo_dim={drive_pseudo_dim}"
            )
        if pseudo_declared and not drive_pseudo_dim:
            raise ValueError(
                f"pseudoscalar ('0o') fields {pseudo_declared} require "
                "drive_pseudo_dim > 0: the pseudo channel width is a "
                "constructor knob that defaults to 0 (sector off)"
            )

        if reference_length_key is not None:
            declared_global_names = global_operator_names | global_drive_names
            if reference_length_key in declared_global_names:
                raise ValueError(
                    "reference_length_key is used only for geometric "
                    "nondimensionalization and must not also be a learned field"
                )

        if kernel_auxiliary_scale_key is not None:
            # Unlike reference_length_key, the auxiliary scale MUST be a
            # declared operator field: the same per-case value both
            # conditions the encoder (a legitimate operator scalar) and is
            # read raw as the decoder's pair-invariant rescale.
            operator_ranks = flatten_rank_spec(global_field_ranks.get("operator", {}))
            if kernel_auxiliary_scale_key not in operator_ranks:
                if kernel_auxiliary_scale_key in global_drive_names:
                    raise ValueError(
                        f"kernel_auxiliary_scale_key "
                        f"{kernel_auxiliary_scale_key!r} is declared as a "
                        "global DRIVE field; the auxiliary scale describes "
                        "the problem, not the drive, and must be declared "
                        "under the 'operator' role of global_field_ranks"
                    )
                raise ValueError(
                    f"kernel_auxiliary_scale_key {kernel_auxiliary_scale_key!r} "
                    "must name a declared rank-0 global operator field in "
                    "global_field_ranks['operator']"
                )
            if operator_ranks[kernel_auxiliary_scale_key] != 0:
                raise ValueError(
                    f"kernel_auxiliary_scale_key {kernel_auxiliary_scale_key!r} "
                    "must be a rank-0 (dimensionless scalar) global operator "
                    "field; got rank "
                    f"{operator_ranks[kernel_auxiliary_scale_key]!r}"
                )

        # Freeze caller-owned mutable schemas before constructing layouts or
        # checkpoint metadata. Public configuration must not drift away from
        # the modules if a caller later edits their original dictionaries.
        output_field_ranks = deepcopy(output_field_ranks)
        boundary_field_ranks = deepcopy(boundary_field_ranks)
        global_field_ranks = deepcopy(global_field_ranks)

        super().__init__(meta=MetaData())
        self._args["output_field_ranks"] = deepcopy(output_field_ranks)
        self._args["boundary_field_ranks"] = deepcopy(boundary_field_ranks)
        self._args["global_field_ranks"] = deepcopy(global_field_ranks)
        self.n_spatial_dims = n_spatial_dims
        self.output_field_ranks = output_field_ranks
        self.boundary_field_ranks = boundary_field_ranks
        self.global_field_ranks = global_field_ranks
        self.reference_length_key = reference_length_key
        self.kernel_auxiliary_scale_key = kernel_auxiliary_scale_key
        self.field_mode = field_mode
        self.query_decoder = query_decoder
        self.bounded_query_geometry = bool(bounded_query_geometry)
        self.decaying_direct_drive = bool(decaying_direct_drive)
        self.operator_scalar_dim = operator_scalar_dim
        self.operator_vector_dim = operator_vector_dim
        self.drive_scalar_dim = drive_scalar_dim
        self.drive_vector_dim = drive_vector_dim
        self.drive_pseudo_dim = drive_pseudo_dim
        self.operator_layers = operator_layers
        self.drive_layers = drive_layers
        self.query_layers = query_layers
        self.heads = heads
        self.scalar_rank = scalar_rank
        self.vector_rank = vector_rank
        self.query_chunk_size = query_chunk_size
        self.attention_chunk_size = attention_chunk_size
        self.boundary_names = tuple(boundary_names)
        self.per_boundary_moment_pool = per_boundary_moment_pool
        self.per_boundary_moment_pool_balanced = per_boundary_moment_pool_balanced
        self.trace_of = trace_of
        self.trace_self_correction = trace_self_correction
        self.trace_readouts = trace_readouts
        self.diagnostic_local_query_features = diagnostic_local_query_features
        self.kernel_local_pair_features = kernel_local_pair_features
        # One moment-pool segment per declared boundary when the knob is on;
        # 0 keeps every block bitwise pre-extension (no gain parameters).
        n_moment_segments = len(boundary_names) if per_boundary_moment_pool else 0

        self._boundary_layouts: dict[str, dict[str, FieldLayout | None]] = {
            role: {} for role in _FIELD_ROLES
        }
        # Pseudoscalar boundary fields pack through their own rank-0 layouts
        # (identical shapes, different transformation law).
        self._boundary_pseudo_layouts: dict[str, dict[str, FieldLayout | None]] = {
            role: {} for role in _FIELD_ROLES
        }
        self._boundary_names_by_rank: dict[str, dict[int | str, tuple[str, ...]]] = {}
        for role in _FIELD_ROLES:
            union: dict[str, int | str] = {}
            for name in boundary_names:
                rank_spec = _role_spec(
                    boundary_field_ranks[name],
                    role,
                    label=f"boundary_field_ranks[{name!r}]",
                )
                flat = flatten_rank_spec(rank_spec)
                for field_name, rank in flat.items():
                    previous = union.get(field_name)
                    if previous is not None and previous != rank:
                        raise ValueError(
                            f"Boundary field {field_name!r} has conflicting ranks "
                            f"{previous} and {rank}"
                        )
                    union[field_name] = rank
                tensor_spec, pseudo_spec = _split_pseudoscalar_spec(
                    rank_spec,
                    source_label=f"boundary_field_ranks[{name!r}][{role!r}]",
                )
                self._boundary_layouts[role][name] = (
                    FieldLayout(tensor_spec, n_spatial_dims)
                    if flatten_rank_spec(tensor_spec)
                    else None
                )
                self._boundary_pseudo_layouts[role][name] = (
                    FieldLayout(pseudo_spec, n_spatial_dims)
                    if flatten_rank_spec(pseudo_spec)
                    else None
                )
            self._boundary_names_by_rank[role] = {
                rank: tuple(
                    sorted(name for name, value in union.items() if value == rank)
                )
                for rank in (0, 1, _PSEUDO_RANK)
            }

        self._global_entries = {
            role: tuple(
                sorted(
                    _rank_entries(
                        _role_spec(
                            global_field_ranks,
                            role,
                            label="global_field_ranks",
                        )
                    ),
                    key=lambda item: item[0],
                )
            )
            for role in _FIELD_ROLES
        }

        boundary_operator_scalars = len(self._boundary_names_by_rank["operator"][0])
        boundary_operator_vectors = len(self._boundary_names_by_rank["operator"][1])
        global_operator_scalars = sum(
            rank == 0 for _, _, rank in self._global_entries["operator"]
        )
        global_operator_vectors = sum(
            rank == 1 for _, _, rank in self._global_entries["operator"]
        )
        # BC one-hot + (source, query) association indicators.
        raw_operator_scalars = (
            boundary_operator_scalars
            + global_operator_scalars
            + len(boundary_names)
            + 2
        )
        # DIAGNOSTIC ONLY: two extra scalar channels (log relative cell
        # area, nondimensional own-cell curvature) that are zero on source
        # rows and populated on trace-query rows.  Changes the lift width,
        # so knob-off stays bitwise identical to pre-knob models.
        if diagnostic_local_query_features:
            raw_operator_scalars += 2
        # Boundary/global vectors + normalized position + source normal.
        raw_operator_vectors = boundary_operator_vectors + global_operator_vectors + 2
        self.operator_lift = TypedProjection(
            raw_operator_scalars,
            raw_operator_vectors,
            operator_scalar_dim,
            operator_vector_dim,
            scalar_bias=True,
        )
        # A shared nonlinear typed feature map gives source and query
        # coordinates a rich finite-rank basis before global interaction.
        # It is pointwise (not a neighbourhood heuristic); boundary-wide
        # information still enters only through the signed moments below.
        self.operator_input_block = PointwiseGeometryBlock(
            operator_scalar_dim, operator_vector_dim
        )
        self.operator_blocks = nn.ModuleList(
            [
                MeshOperatorBlock(
                    operator_scalar_dim,
                    operator_vector_dim,
                    heads=heads,
                    scalar_rank=scalar_rank,
                    vector_rank=vector_rank,
                    entity_chunk_size=attention_chunk_size,
                    n_moment_segments=n_moment_segments,
                    moment_pool_balanced=per_boundary_moment_pool_balanced,
                )
                for _ in range(operator_layers)
            ]
        )

        boundary_drive_scalars = len(self._boundary_names_by_rank["drive"][0])
        boundary_drive_vectors = len(self._boundary_names_by_rank["drive"][1])
        boundary_drive_pseudos = len(
            self._boundary_names_by_rank["drive"][_PSEUDO_RANK]
        )
        self._boundary_drive_scalars = boundary_drive_scalars
        self._boundary_drive_vectors = boundary_drive_vectors
        self._boundary_drive_pseudos = boundary_drive_pseudos
        global_drive_scalars = sum(
            rank == 0 for _, _, rank in self._global_entries["drive"]
        )
        global_drive_vectors = sum(
            rank == 1 for _, _, rank in self._global_entries["drive"]
        )
        global_drive_pseudos = sum(
            rank == _PSEUDO_RANK for _, _, rank in self._global_entries["drive"]
        )
        self._global_drive_scalars = global_drive_scalars
        self._global_drive_vectors = global_drive_vectors
        self._global_drive_pseudos = global_drive_pseudos
        raw_drive_scalars = boundary_drive_scalars + global_drive_scalars
        raw_drive_vectors = boundary_drive_vectors + global_drive_vectors
        raw_drive_pseudos = boundary_drive_pseudos + global_drive_pseudos
        if raw_drive_scalars + raw_drive_vectors + raw_drive_pseudos == 0:
            raise ValueError(
                "At least one boundary or global field must have the 'drive' role"
            )
        self.drive_lift = GeometryConditionedLinear(
            operator_scalar_dim,
            operator_vector_dim,
            raw_drive_scalars,
            raw_drive_vectors,
            drive_scalar_dim,
            drive_vector_dim,
            field_pseudo_dim=raw_drive_pseudos,
            out_pseudo_dim=drive_pseudo_dim,
        )

        # The quadratic mode deliberately reuses the LINEAR field blocks and
        # decoders: they are the drive-linear machinery whose states the
        # single bilinear read-in composition below combines, and reusing
        # them is what makes the declared degree provable (every stage
        # before the composition is exactly linear in the drive).
        block_type = (
            NonlinearZeroMeshFieldBlock
            if field_mode == "zero_preserving_nonlinear"
            else LinearMeshFieldBlock
        )
        block_kwargs = dict(
            geometry_scalar_dim=operator_scalar_dim,
            geometry_vector_dim=operator_vector_dim,
            field_scalar_dim=drive_scalar_dim,
            field_vector_dim=drive_vector_dim,
            heads=heads,
            scalar_rank=scalar_rank,
            vector_rank=vector_rank,
            entity_chunk_size=attention_chunk_size,
            field_pseudo_dim=drive_pseudo_dim,
            n_moment_segments=n_moment_segments,
            moment_pool_balanced=per_boundary_moment_pool_balanced,
        )
        self.drive_blocks = nn.ModuleList(
            [block_type(**block_kwargs) for _ in range(drive_layers)]
        )
        if query_decoder == "kernel":
            # The dense kernel decoder replaces every separable query cross
            # block; ``query_layers`` therefore configures only the moment
            # decoder.  The two field modes are separate decoder classes so a
            # nonlinearity cannot silently invalidate the linear contract.
            decoder_type = (
                NonlinearZeroKernelBasisCrossDecoder
                if field_mode == "zero_preserving_nonlinear"
                else LinearKernelBasisCrossDecoder
            )
            self.kernel_decoder = decoder_type(
                n_spatial_dims=n_spatial_dims,
                operator_scalar_dim=operator_scalar_dim,
                operator_vector_dim=operator_vector_dim,
                drive_scalar_dim=drive_scalar_dim,
                drive_vector_dim=drive_vector_dim,
                heads=heads,
                include_double_layer_member=kernel_include_double_layer_member,
                include_polynomial_members=kernel_include_polynomial_members,
                include_single_layer_member=kernel_include_single_layer_member,
                monopole_free_single_layer=kernel_monopole_free_single_layer,
                auxiliary_scale=kernel_auxiliary_scale_key is not None,
                log_radial_features=kernel_log_radial_features,
                checkpoint_query_chunks=kernel_checkpoint_query_chunks,
                mlp_members=kernel_mlp_members,
                drive_pseudo_dim=drive_pseudo_dim,
                local_pair_features=kernel_local_pair_features,
                near_theta=kernel_near_theta,
                decode_backend=kernel_decode_backend,
                bh_theta=kernel_bh_theta,
                bh_leaf_size=kernel_bh_leaf_size,
            )
            self.query_blocks = nn.ModuleList()
        else:
            self.kernel_decoder = None
            self.query_blocks = nn.ModuleList(
                [
                    block_type(
                        **block_kwargs,
                        # The first boundary-to-query operation is a read-in,
                        # not a perturbative residual update. Its learnable
                        # scale starts at one; later cross messages retain the
                        # small residual initialization used throughout the
                        # stack.
                        message_layer_scale=1.0 if index == 0 else None,
                    )
                    for index in range(query_layers)
                ]
            )

        # Declared boundary-trace read-outs: typed projections of the query's
        # OWN cell's post-attention encoded states (the declared identity map
        # query index <-> trace-boundary cell index).  The operator read-out
        # joins the query operator state (panel identity: encoded local
        # geometry, normal direction); the drive read-out joins the query
        # field state and is an exactly linear typed channel mix (no
        # invariant lift, no bias) so drive-linearity and zero preservation
        # survive.  None when the mode is off: the knob adds no parameters,
        # keeping default state dicts bitwise pre-extension.
        if trace_of is not None and trace_readouts:
            self.trace_operator_read_out = TypedProjection(
                operator_scalar_dim,
                operator_vector_dim,
                operator_scalar_dim,
                operator_vector_dim,
                scalar_bias=False,
            )
            self.trace_drive_read_out = TypedProjection(
                drive_scalar_dim,
                drive_vector_dim,
                drive_scalar_dim,
                drive_vector_dim,
                scalar_bias=False,
                include_vector_invariants=False,
                pseudo_in=drive_pseudo_dim,
                pseudo_out=drive_pseudo_dim,
            )
        else:
            self.trace_operator_read_out = None
            self.trace_drive_read_out = None

        # The quadratic mode's single declared bilinear composition, applied
        # to the assembled query field state in decode() immediately before
        # the (drive-linear) output projection.  None in the other modes:
        # the knob adds no parameters unless selected, so linear/nonlinear
        # state dicts are bitwise identical to the pre-extension model.
        if field_mode == "quadratic":
            self.quadratic_read_in = QuadraticFieldReadIn(
                operator_scalar_dim,
                operator_vector_dim,
                drive_scalar_dim,
                drive_vector_dim,
                field_pseudo_dim=drive_pseudo_dim,
            )
        else:
            self.quadratic_read_in = None

        # Pseudoscalar outputs unpack through a separate rank-0 layout: they
        # share the scalar packing shape but not the transformation law, so
        # they must never share channels with the true-scalar outputs.
        self.output_layout = (
            FieldLayout(output_tensor_ranks, n_spatial_dims)
            if flatten_rank_spec(output_tensor_ranks)
            else None
        )
        self._output_pseudo_layout = (
            FieldLayout(output_pseudo_ranks, n_spatial_dims)
            if flatten_rank_spec(output_pseudo_ranks)
            else None
        )
        self.output_projection = GeometryConditionedLinear(
            operator_scalar_dim,
            operator_vector_dim,
            drive_scalar_dim,
            drive_vector_dim,
            self.output_layout.n_scalars if self.output_layout is not None else 0,
            self.output_layout.n_vectors if self.output_layout is not None else 0,
            field_pseudo_dim=drive_pseudo_dim,
            out_pseudo_dim=(
                self._output_pseudo_layout.n_scalars
                if self._output_pseudo_layout is not None
                else 0
            ),
            # Only the final gate multiplies the whole prediction, so it is
            # the one whose saturation collapses the far field; the interior
            # gates (drive lift, residual blocks) modulate bounded residual
            # or additive terms and are deliberately left untouched.
            bounded_gate_invariants=bounded_output_gate_invariants,
        )

    def _validate_domain(self, domain: DomainMesh) -> None:
        """Raise unless ``domain`` matches the declared schema exactly:
        boundary names, spatial dimension, geometry dtype/device coherence,
        codimension-one non-empty boundaries."""
        if not isinstance(domain, DomainMesh):
            raise TypeError(f"domain must be a DomainMesh, got {type(domain).__name__}")
        actual_names = set(domain.boundaries.keys())
        expected_names = set(self.boundary_names)
        if actual_names != expected_names:
            raise ValueError(
                "Domain boundary names must exactly match the model schema; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        if domain.interior.n_spatial_dims != self.n_spatial_dims:
            raise ValueError(
                f"Expected {self.n_spatial_dims} spatial dimensions, got "
                f"{domain.interior.n_spatial_dims}"
            )
        if domain.interior.points.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Mesh geometry must use float32 or float64; mixed-precision "
                "execution should use autocast rather than reduced-precision points"
            )
        for name in self.boundary_names:
            mesh = domain.boundaries[name]
            if mesh.codimension != 1:
                raise ValueError(
                    f"Boundary {name!r} must be codimension one, got "
                    f"codimension={mesh.codimension}"
                )
            if mesh.n_cells == 0:
                raise ValueError(f"Boundary {name!r} must contain at least one cell")
            if mesh.points.device != domain.interior.points.device:
                raise ValueError("All boundary and query meshes must share a device")
            if mesh.points.dtype != domain.interior.points.dtype:
                raise ValueError("All boundary and query meshes must share a dtype")

    def _pack_boundary_role(
        self,
        domain: DomainMesh,
        role: str,
    ) -> ScalarVectorState:
        """Pack one role's declared per-boundary fields (all boundaries,
        source cell order) into a typed state, one sector per rank."""
        scalar_names = self._boundary_names_by_rank[role][0]
        vector_names = self._boundary_names_by_rank[role][1]
        pseudo_names = self._boundary_names_by_rank[role][_PSEUDO_RANK]
        scalar_index = {name: index for index, name in enumerate(scalar_names)}
        vector_index = {name: index for index, name in enumerate(vector_names)}
        pseudo_index = {name: index for index, name in enumerate(pseudo_names)}
        scalar_parts: list[torch.Tensor] = []
        vector_parts: list[torch.Tensor] = []
        pseudo_parts: list[torch.Tensor] = []

        for boundary_name in self.boundary_names:
            mesh = domain.boundaries[boundary_name]
            scalars = mesh.points.new_zeros(mesh.n_cells, len(scalar_names))
            vectors = mesh.points.new_zeros(
                mesh.n_cells, len(vector_names), self.n_spatial_dims
            )
            pseudos = mesh.points.new_zeros(mesh.n_cells, len(pseudo_names))
            layout = self._boundary_layouts[role][boundary_name]
            if layout is not None:
                packed = layout.pack(mesh.cell_data)
                if packed.scalars.dtype != mesh.points.dtype:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must have "
                        f"dtype {mesh.points.dtype}"
                    )
                if packed.scalars.device != mesh.points.device:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must be on "
                        f"{mesh.points.device}"
                    )
                for local, field_name in enumerate(layout.scalar_names):
                    scalars[:, scalar_index[field_name]] = packed.scalars[:, local]
                for local, field_name in enumerate(layout.vector_names):
                    vectors[:, vector_index[field_name], :] = packed.vectors[
                        :, local, :
                    ]
            pseudo_layout = self._boundary_pseudo_layouts[role][boundary_name]
            if pseudo_layout is not None:
                packed_pseudos = pseudo_layout.pack(mesh.cell_data)
                if packed_pseudos.scalars.dtype != mesh.points.dtype:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must have "
                        f"dtype {mesh.points.dtype}"
                    )
                if packed_pseudos.scalars.device != mesh.points.device:
                    raise ValueError(
                        f"Boundary {boundary_name!r} {role} fields must be on "
                        f"{mesh.points.device}"
                    )
                for local, field_name in enumerate(pseudo_layout.scalar_names):
                    pseudos[:, pseudo_index[field_name]] = packed_pseudos.scalars[
                        :, local
                    ]
            scalar_parts.append(scalars)
            vector_parts.append(vectors)
            pseudo_parts.append(pseudos)
        return ScalarVectorState(
            torch.cat(scalar_parts, dim=0),
            torch.cat(vector_parts, dim=0),
            torch.cat(pseudo_parts, dim=0),
        )

    def _pack_global_role(
        self,
        global_data: TensorDict,
        role: str,
        n_entities: int,
        reference: Float[torch.Tensor, ""],
    ) -> ScalarVectorState:
        """Broadcast one role's declared per-sample global fields to every
        source entity as a typed state (vectors carried in the normalized
        frame; shape-validated against the declared ranks)."""
        scalars: list[torch.Tensor] = []
        vectors: list[torch.Tensor] = []
        pseudos: list[torch.Tensor] = []
        for name, path, rank in self._global_entries[role]:
            value = _td_get(global_data, path)
            expected = (self.n_spatial_dims,) if rank == 1 else ()
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Global {role} field {name!r} is rank {rank} and must "
                    f"have shape {expected}, got {tuple(value.shape)}"
                )
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(
                    f"Global {role} field {name!r} must share mesh device and dtype"
                )
            if rank == 1:
                vectors.append(value.expand(n_entities, -1))
            elif rank == _PSEUDO_RANK:
                pseudos.append(value.expand(n_entities))
            else:
                scalars.append(value.expand(n_entities))
        return ScalarVectorState(
            torch.stack(scalars, dim=-1)
            if scalars
            else reference.new_empty(n_entities, 0),
            torch.stack(vectors, dim=1)
            if vectors
            else reference.new_empty(n_entities, 0, self.n_spatial_dims),
            torch.stack(pseudos, dim=-1)
            if pseudos
            else reference.new_empty(n_entities, 0),
        )

    def _intrinsic_reference_length(
        self,
        weights: torch.Tensor,
        centroids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        r"""Measure-weighted RMS boundary radius (radius of gyration).

        The intrinsic scale gauge used when ``reference_length_key`` is
        ``None``: :math:`L=\sqrt{\sum_j\omega_j\lVert y_j-c\rVert^2/
        \sum_j\omega_j}` over boundary cell centroids :math:`y_j`, cell
        measures :math:`\omega_j`, and their measure-weighted centroid
        :math:`c`.  Degree-1 positive homogeneity in the geometry makes the
        model's scale equivariance unconditional, and the statistic is
        refinement-convergent and smooth in the boundary shape.  It is
        accumulated in float64 and cast back to the geometry dtype; it
        depends only on the source boundary, never on query points, so the
        bitwise query-set-independence contracts are unaffected.  The
        computation is differentiable through the mesh geometry.
        """
        weights64 = weights.double()
        centroids64 = centroids.double()
        total = weights64.sum()
        center = torch.einsum("n,nd->d", weights64, centroids64) / total
        radius_sq = (centroids64 - center).square().sum(dim=-1)
        length = torch.sqrt(torch.einsum("n,n->", weights64, radius_sq) / total)
        length = length.to(dtype)
        if not torch.compiler.is_compiling() and (
            not torch.isfinite(length).item() or length.item() <= 0.0
        ):
            raise ValueError(
                "Intrinsic reference length (measure-weighted RMS boundary "
                "radius) must be finite and positive; it vanishes when every "
                "boundary cell centroid coincides with the boundary "
                "centroid.  Supply reference_length_key to override the "
                "intrinsic scale gauge for such degenerate geometries"
            )
        return length

    def _reference_length(
        self,
        global_data: TensorDict,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        r"""Explicit reference length declared in ``global_data``.

        Only called with ``reference_length_key`` set; this override path is
        bitwise identical to models predating the intrinsic default gauge.
        """
        path = tuple(self.reference_length_key.split("."))
        length = _td_get(global_data, path)
        if length.numel() != 1:
            raise ValueError(
                f"Reference length {self.reference_length_key!r} must be scalar"
            )
        if length.device != reference.device or length.dtype != reference.dtype:
            raise ValueError("Reference length must share mesh device and dtype")
        length = length.reshape(())
        if not torch.compiler.is_compiling() and (
            not torch.isfinite(length).item() or length.item() <= 0.0
        ):
            raise ValueError("Reference length must be finite and positive")
        return length

    def _kernel_auxiliary_scale(
        self,
        global_data: TensorDict,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        r"""Raw declared auxiliary-scale ratio :math:`\lambda` for the decoder.

        Only called with ``kernel_auxiliary_scale_key`` set.  The value is
        read from the RAW global operator input at encode time, before the
        operator lift: like ``reference_length_key``, it is a physical
        declaration (the dimensionless ratio
        :math:`\lambda=\delta/L_{\mathrm{ref}}`, e.g.
        :math:`\mathrm{Re}^{-1/2}`), not a learned feature, and a
        non-positive or non-finite value is a caller error rather than data.
        """
        path = tuple(self.kernel_auxiliary_scale_key.split("."))
        value = _td_get(global_data, path)
        if value.numel() != 1:
            raise ValueError(
                f"Auxiliary scale {self.kernel_auxiliary_scale_key!r} must be scalar"
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("Auxiliary scale must share mesh device and dtype")
        value = value.reshape(())
        if not torch.compiler.is_compiling() and (
            not torch.isfinite(value).item() or value.item() <= 0.0
        ):
            raise ValueError(
                f"Auxiliary scale {self.kernel_auxiliary_scale_key!r} must "
                "be finite and positive: it declares the physical "
                "length-scale ratio delta / L_ref"
            )
        return value

    def _local_cell_scalars(
        self, source_mesh: Mesh, length: torch.Tensor
    ) -> torch.Tensor:
        r"""Per-cell squashed local-geometry scalars, ``(n_cells, 2)``.

        Channel 0 is log-relative-measure, channel 1 nondimensional
        Gaussian curvature; both SQUASHED to ``(-1, 1)`` by smooth fixed
        maps (curvature through asinh -- log-like across its ~12 measured
        decades, smooth at zero -- before tanh; log-relative-measure is
        already log-scaled and takes a plain tanh).  The constants are
        FIXED, never per-sample, from DrivAerML run_1 quantiles
        (2026-07-11): 3.0 ~ p95 of ``log(A / median A)``; 13.0 ~
        ``asinh(p95 of K * L^m)``.  Both channels are even
        similarity-invariant scalars; the unbounded predecessors NaN'd
        training by epoch 25 (measured).  Shared by the P2 diagnostic
        (query-stream entry) and the task-#53 local-corrector probes
        (kernel-side entry), so the two entry points carry IDENTICAL
        information.  Measured caveat: soup-style cell subsampling zeroes
        the curvature channel (boundary-vertex angle defects are undefined
        -> NaN -> 0), leaving it harmlessly inert on
        non-topology-preserving pipelines.
        """
        from physicsnemo.mesh.curvature.gaussian import (
            gaussian_curvature_cells,
        )

        areas = source_mesh.cell_areas
        log_rel_area = torch.tanh(
            torch.log(areas / areas.median().clamp_min(torch.finfo(areas.dtype).tiny))
            / 3.0
        )
        curvature = gaussian_curvature_cells(source_mesh)
        curvature = torch.nan_to_num(curvature, nan=0.0) * length.pow(
            source_mesh.n_manifold_dims
        )
        curvature = torch.tanh(torch.asinh(curvature) / 13.0)
        return torch.stack((log_rel_area, curvature), dim=-1)

    def _source_operator_input(
        self,
        domain: DomainMesh,
        source_mesh: Mesh,
        boundary_operator: ScalarVectorState,
        global_operator: ScalarVectorState,
    ) -> ScalarVectorState:
        """Assemble the full source-side operator input: declared boundary
        and global operator fields, the per-boundary one-hot BC type, and
        the source/query association flags (plus zero-filled diagnostic
        channels when that probe knob is on)."""
        n = source_mesh.n_cells
        bc_one_hot = source_mesh.points.new_zeros(n, len(self.boundary_names))
        offset = 0
        for index, name in enumerate(self.boundary_names):
            count = domain.boundaries[name].n_cells
            bc_one_hot[offset : offset + count, index] = 1.0
            offset += count
        association = source_mesh.points.new_zeros(n, 2)
        association[:, 0] = 1.0
        scalar_parts = [
            boundary_operator.scalars,
            global_operator.scalars,
            bc_one_hot,
            association,
        ]
        if self.diagnostic_local_query_features:
            # Source rows carry no diagnostic features; the channels exist
            # (zero-valued) so source and query rows share one lift layout,
            # mirroring the association-indicator pattern above.
            scalar_parts.append(source_mesh.points.new_zeros(n, 2))
        return ScalarVectorState(
            torch.cat(tuple(scalar_parts), dim=-1),
            torch.cat(
                (
                    boundary_operator.vectors,
                    global_operator.vectors,
                    source_mesh.cell_centroids[:, None, :],
                    source_mesh.cell_normals[:, None, :],
                ),
                dim=1,
            ),
        )

    def _query_operator_input(
        self,
        points: torch.Tensor,
        global_operator: ScalarVectorState,
        diagnostic_scalars: torch.Tensor | None = None,
        diagnostic_normals: torch.Tensor | None = None,
    ) -> ScalarVectorState:
        r"""Raw operator-state injection at nondimensional query ``points``.

        With ``bounded_query_geometry`` the injected position channel is the
        compactified :math:`\hat x = x/\sqrt{1+|x|^2}` rather than the raw
        :math:`x`, so the learned-coefficient path (operator lift, gates,
        direct-drive geometry vectors) sees a bounded function of query
        radius by construction; see the constructor docstring for the
        far-field ladder that motivated this.  Only this injection changes:
        :meth:`decode` hands the RAW normalized points to the kernel
        decoder's pair-invariant features and exact singular members
        through a separate argument, so the physics members keep their
        exact, unbounded radial dependence.
        """
        n = points.shape[0]
        injected_points = points
        if self.bounded_query_geometry:
            injected_points = points * torch.rsqrt(
                1.0 + points.square().sum(dim=-1, keepdim=True)
            )
        boundary_scalars = points.new_zeros(
            n, len(self._boundary_names_by_rank["operator"][0])
        )
        boundary_vectors = points.new_zeros(
            n,
            len(self._boundary_names_by_rank["operator"][1]),
            self.n_spatial_dims,
        )
        bc_one_hot = points.new_zeros(n, len(self.boundary_names))
        association = points.new_zeros(n, 2)
        association[:, 1] = 1.0
        # DIAGNOSTIC ONLY: a trace query may read its own cell's unit
        # normal through the (otherwise zero) query slot of the existing
        # normal channel -- the same typed channel the sources use, so the
        # transformation law is inherited rather than reinvented.
        query_normal = (
            diagnostic_normals[:, None, :]
            if diagnostic_normals is not None
            else points.new_zeros(n, 1, self.n_spatial_dims)
        )
        scalar_parts = [
            boundary_scalars,
            global_operator.scalars.expand(n, -1),
            bc_one_hot,
            association,
        ]
        if self.diagnostic_local_query_features:
            scalar_parts.append(
                diagnostic_scalars
                if diagnostic_scalars is not None
                else points.new_zeros(n, 2)
            )
        return ScalarVectorState(
            torch.cat(tuple(scalar_parts), dim=-1),
            torch.cat(
                (
                    boundary_vectors,
                    global_operator.vectors.expand(n, -1, -1),
                    injected_points[:, None, :],
                    query_normal,
                ),
                dim=1,
            ),
        )

    def encode(self, domain: DomainMesh) -> EncodedBoundary:
        r"""Encode a boundary once for reuse at one or more query meshes."""
        self._validate_domain(domain)
        geometry_meshes = [
            domain.boundaries[name].with_data(
                point_data={}, cell_data={}, global_data={}
            )
            for name in self.boundary_names
        ]
        merged = Mesh.merge(geometry_meshes)
        length = (
            None
            if self.reference_length_key is None
            else self._reference_length(domain.global_data, merged.points)
        )
        # Geometry and quadrature construction stay outside ambient AMP. The
        # learned projections may autocast, but centering, normals, and source
        # measure are numerical mesh operations and retain the input geometry
        # precision.
        with torch.autocast(device_type=merged.points.device.type, enabled=False):
            weights = merged.cell_areas
            total_measure = weights.sum()
        if not torch.compiler.is_compiling():
            cells_are_valid = torch.isfinite(weights).all() & (weights > 0.0).all()
            total_is_valid = torch.isfinite(total_measure) & (total_measure > 0.0)
            # Keep valid execution to one host synchronization. The individual
            # predicate is inspected only on the exceptional path so the error
            # still identifies degenerate cells versus an overflowing total.
            if not (cells_are_valid & total_is_valid).item():
                if not cells_are_valid.item():
                    raise ValueError(
                        "Every boundary cell must have finite positive measure"
                    )
                raise ValueError("Boundary measure must be finite and positive")
        with torch.autocast(device_type=merged.points.device.type, enabled=False):
            if length is None:
                # Intrinsic scale gauge: the measure-weighted RMS boundary
                # radius, computed after the measure validation above so a
                # degenerate boundary reports its own error first.
                length = self._intrinsic_reference_length(
                    weights, merged.cell_centroids, merged.points.dtype
                )
            center = torch.einsum("n,nd->d", weights, merged.cell_centroids)
            center = center / total_measure
            source_mesh = Mesh(
                points=(merged.points - center) / length,
                cells=merged.cells,
            )
            # Populate the immutable geometric cache at full geometry precision
            # before learned layers are entered under any outer autocast scope.
            _ = source_mesh.cell_centroids
            _ = source_mesh.cell_areas
            _ = source_mesh.cell_normals

        boundary_operator = self._pack_boundary_role(domain, "operator")
        global_operator = self._pack_global_role(
            domain.global_data,
            "operator",
            source_mesh.n_cells,
            source_mesh.points,
        )
        operator = self.operator_input_block(
            self.operator_lift(
                self._source_operator_input(
                    domain,
                    source_mesh,
                    boundary_operator,
                    global_operator,
                )
            )
        )
        # Per-boundary moment-pool segments: cell ranges of the merged source
        # in ``self.boundary_names`` order (the merge order above).  ``None``
        # (knob off) leaves every attention call on the historical path.
        moment_segments: tuple[slice, ...] | None = None
        if self.per_boundary_moment_pool:
            segment_list: list[slice] = []
            offset = 0
            for name in self.boundary_names:
                count = domain.boundaries[name].n_cells
                segment_list.append(slice(offset, offset + count))
                offset += count
            moment_segments = tuple(segment_list)
        for block in self.operator_blocks:
            operator = block(source_mesh, operator, moment_segments=moment_segments)

        boundary_drive = self._pack_boundary_role(domain, "drive")
        global_drive = self._pack_global_role(
            domain.global_data,
            "drive",
            source_mesh.n_cells,
            source_mesh.points,
        )
        raw_drive = ScalarVectorState(
            torch.cat((boundary_drive.scalars, global_drive.scalars), dim=-1),
            torch.cat((boundary_drive.vectors, global_drive.vectors), dim=1),
            torch.cat((boundary_drive.pseudos, global_drive.pseudos), dim=-1),
        )
        drive = self.drive_lift(operator, raw_drive)
        for block in self.drive_blocks:
            drive = block(source_mesh, operator, drive, moment_segments=moment_segments)

        global_operator_single = self._pack_global_role(
            domain.global_data,
            "operator",
            1,
            source_mesh.points,
        )
        global_drive_single = self._pack_global_role(
            domain.global_data,
            "drive",
            1,
            source_mesh.points,
        )
        kernel_cache: KernelDecoderCache | None = None
        if self.query_decoder == "kernel":
            query_moments: tuple[AttentionMoments, ...] = ()
            auxiliary_scale = (
                None
                if self.kernel_auxiliary_scale_key is None
                else self._kernel_auxiliary_scale(
                    domain.global_data, source_mesh.points
                )
            )
            cache_local_scalars = (
                self._local_cell_scalars(source_mesh, length)
                if self.kernel_local_pair_features is not None
                else None
            )
            kernel_cache = self.kernel_decoder.build_source_cache(
                source_mesh,
                operator,
                drive,
                auxiliary_scale=auxiliary_scale,
                local_scalars=cache_local_scalars,
            )
        else:
            query_moments = tuple(
                block.build_source_moments(
                    source_mesh, operator, drive, moment_segments=moment_segments
                )
                for block in self.query_blocks
            )
        query_mesh = domain.interior.with_data(
            point_data={},
            cell_data={},
            global_data=domain.global_data,
        )
        # The declared trace boundary's cell range in the merged source (the
        # merge above concatenates boundaries in ``self.boundary_names``
        # order, and each boundary's cells keep their mesh order), recorded
        # so decode can align query i with its declared own cell.
        trace_slice: slice | None = None
        if self.trace_of is not None:
            offset = 0
            for name in self.boundary_names:
                count = domain.boundaries[name].n_cells
                if name == self.trace_of:
                    trace_slice = slice(offset, offset + count)
                    break
                offset += count
        diagnostic_query_features: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.diagnostic_local_query_features and trace_slice is not None:
            # DIAGNOSTIC ONLY (see constructor): the trace boundary's
            # per-cell local geometry, nondimensionalized so the similarity
            # contract is preserved -- what breaks is the INFORMATION diet
            # (a query reads its own geometry directly instead of through
            # the boundary-integral path), which is this probe's point.
            #
            # Both scalar channels are SQUASHED to (-1, 1) by smooth fixed
            # maps: raw nondimensional curvature spans ~12 decades on a
            # real DrivAerML vehicle (run_1 full mesh, 2026-07-11: K*L^2
            # p50=13, p95=1.9e5, absmax=6.2e8), and unbounded injection
            # measurably NaN'd training by epoch 25.  Curvature goes
            # through asinh (log-like across decades, smooth at zero)
            # before tanh so the bulk keeps resolution instead of being
            # crushed by the p95 scale; log-relative-area is already
            # log-scaled and takes a plain tanh.  The constants are FIXED
            # (never per-sample) for determinism and comparability, from
            # run_1 quantiles: 3.0 ~ p95 of log(A/median A) (reproduced on
            # a 10k-cell subsample); 13.0 ~ asinh(p95 of K*L^m).  Caveat,
            # measured: soup-style cell subsampling zeroes curvature
            # entirely (every vertex becomes a mesh-boundary vertex with
            # undefined angle defect -> NaN -> 0), leaving that channel
            # harmlessly inert on non-topology-preserving pipelines.
            diagnostic_query_features = (
                self._local_cell_scalars(source_mesh, length)[trace_slice],
                source_mesh.cell_normals[trace_slice],
            )
        return EncodedBoundary(
            source_mesh=source_mesh,
            operator_state=operator,
            drive_state=drive,
            center=center,
            reference_length=length,
            global_operator_state=global_operator_single,
            global_drive_state=global_drive_single,
            query_moments=query_moments,
            query_mesh=query_mesh,
            global_data=domain.global_data.copy(),
            kernel_cache=kernel_cache,
            trace_slice=trace_slice,
            diagnostic_query_features=diagnostic_query_features,
        )

    def decode(
        self,
        encoded: EncodedBoundary,
        query_mesh: Mesh | None = None,
    ) -> Mesh:
        r"""Evaluate an encoded boundary at arbitrary query mesh points."""
        if not isinstance(encoded, EncodedBoundary):
            raise TypeError("encoded must be an EncodedBoundary returned by encode")
        query_mesh = encoded.query_mesh if query_mesh is None else query_mesh
        if not isinstance(query_mesh, Mesh):
            raise TypeError(
                f"query_mesh must be a Mesh, got {type(query_mesh).__name__}"
            )
        if query_mesh.n_spatial_dims != self.n_spatial_dims:
            raise ValueError("query_mesh has the wrong spatial dimension")
        if (
            query_mesh.points.device != encoded.center.device
            or query_mesh.points.dtype != encoded.center.dtype
        ):
            raise ValueError("query_mesh must share encoded boundary device and dtype")

        if self.query_decoder == "kernel":
            if encoded.kernel_cache is None:
                raise ValueError(
                    "EncodedBoundary carries no kernel-decoder cache; it was "
                    "encoded by a model with query_decoder='moment'"
                )
        elif len(encoded.query_moments) != len(self.query_blocks):
            raise ValueError(
                "EncodedBoundary query moments do not match this decoder depth"
            )
        # Declared boundary-trace alignment: query i IS cell i of the trace
        # boundary (offset into the merged source by ``trace_slice.start``).
        # The count is validated loudly; the ORDER is the caller's
        # declaration and cannot be checked here.
        trace_start: int | None = None
        if self.trace_of is not None:
            if encoded.trace_slice is None:
                raise ValueError(
                    "EncodedBoundary carries no declared trace alignment; it "
                    "was encoded by a model without trace_of.  Re-encode with "
                    "this model so the declared boundary's cell range is "
                    "recorded"
                )
            n_trace = encoded.trace_slice.stop - encoded.trace_slice.start
            if query_mesh.n_points != n_trace:
                raise ValueError(
                    f"trace_of={self.trace_of!r} declares the query mesh to "
                    "be that boundary's cell centroids, index-aligned, but "
                    f"the query mesh has {query_mesh.n_points} points while "
                    f"the encoded boundary has {n_trace} cells; the declared "
                    "identity map requires exactly one query per cell, in "
                    "cell order"
                )
            trace_start = encoded.trace_slice.start
        scalar_outputs: list[torch.Tensor] = []
        vector_outputs: list[torch.Tensor] = []
        pseudo_outputs: list[torch.Tensor] = []
        n_queries = query_mesh.n_points
        starts = range(0, n_queries, self.query_chunk_size)
        slices = [
            slice(start, min(start + self.query_chunk_size, n_queries))
            for start in starts
        ]
        if not slices:
            slices = [slice(0, 0)]

        if self.diagnostic_local_query_features:
            if encoded.diagnostic_query_features is None:
                raise ValueError(
                    "diagnostic_local_query_features is enabled but the "
                    "EncodedBoundary carries no diagnostic features; "
                    "re-encode with this model"
                )
            diag_scalars_full, diag_normals_full = encoded.diagnostic_query_features
        for chunk in slices:
            normalized_points = (
                query_mesh.points[chunk] - encoded.center
            ) / encoded.reference_length
            diag_scalars = diag_normals = None
            if self.diagnostic_local_query_features:
                # The declared trace identity map is chunk-local: query rows
                # [chunk.start, chunk.stop) are exactly trace cells of the
                # same indices, so the features slice with the chunk.
                diag_scalars = diag_scalars_full[chunk]
                diag_normals = diag_normals_full[chunk]
            query_operator = self.operator_input_block(
                self.operator_lift(
                    self._query_operator_input(
                        normalized_points,
                        encoded.global_operator_state,
                        diagnostic_scalars=diag_scalars,
                        diagnostic_normals=diag_normals,
                    )
                )
            )
            own: slice | None = None
            if trace_start is not None:
                # The declared identity map, chunk-local: queries
                # [chunk.start, chunk.stop) are cells [own.start, own.stop)
                # of the merged source.  The own-cell operator read-out is
                # applied first so every downstream conditioning path (the
                # query drive lift, the output gates, the quadratic
                # read-in) sees the panel identity.
                own = slice(trace_start + chunk.start, trace_start + chunk.stop)
                if self.trace_operator_read_out is not None:
                    own_operator = self.trace_operator_read_out(
                        encoded.operator_state.slice(own)
                    )
                    query_operator = ScalarVectorState(
                        query_operator.scalars + own_operator.scalars,
                        query_operator.vectors + own_operator.vectors,
                        query_operator.pseudos + own_operator.pseudos,
                    )
            n_chunk = normalized_points.shape[0]
            # Global drive quantities (for example a prescribed far field)
            # are legitimate pointwise query inputs as well as boundary
            # inputs. With boundary-only drives there is no receiver field to
            # update residually: the first cross-attention message initializes
            # it directly and therefore must not be LayerScale-suppressed.
            query_drive: ScalarVectorState | None = None
            if (
                self._global_drive_scalars
                + self._global_drive_vectors
                + self._global_drive_pseudos
            ):
                raw_query_drive = ScalarVectorState(
                    torch.cat(
                        (
                            normalized_points.new_zeros(
                                n_chunk, self._boundary_drive_scalars
                            ),
                            encoded.global_drive_state.scalars.expand(n_chunk, -1),
                        ),
                        dim=-1,
                    ),
                    torch.cat(
                        (
                            normalized_points.new_zeros(
                                n_chunk,
                                self._boundary_drive_vectors,
                                self.n_spatial_dims,
                            ),
                            encoded.global_drive_state.vectors.expand(n_chunk, -1, -1),
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            normalized_points.new_zeros(
                                n_chunk, self._boundary_drive_pseudos
                            ),
                            encoded.global_drive_state.pseudos.expand(n_chunk, -1),
                        ),
                        dim=-1,
                    ),
                )
                query_drive = self.drive_lift(query_operator, raw_query_drive)
                if self.decaying_direct_drive:
                    # Fixed analytic decay envelope of the RAW (uncompactified)
                    # nondimensional query radius -- deliberately not the
                    # compactified injection, so the envelope keeps its exact
                    # r^-2 asymptotics regardless of bounded_query_geometry.
                    # The output projection is exactly linear in its field
                    # state, so scaling the lifted query drive here multiplies
                    # the direct (non-member-mediated) part of the decoded
                    # output by exactly this factor while the kernel message
                    # below is untouched.  See the constructor docstring for
                    # the exterior-expansion justification of the power.
                    envelope = 1.0 / (1.0 + normalized_points.square().sum(dim=-1))
                    query_drive = ScalarVectorState(
                        query_drive.scalars * envelope[:, None],
                        query_drive.vectors * envelope[:, None, None],
                        query_drive.pseudos * envelope[:, None],
                    )
            if self.query_decoder == "kernel":
                # The dense pair-kernel message is the read-in of the query
                # field state; a pointwise global drive keeps its additional
                # lifted path exactly as in moment mode.  In trace mode the
                # declared own-panel indices side-correct the exact
                # double-layer member's self-entries (the jump relation).
                message = self.kernel_decoder(
                    normalized_points,
                    encoded.kernel_cache,
                    self_indices=(
                        None
                        if own is None or not self.trace_self_correction
                        else torch.arange(
                            own.start, own.stop, device=normalized_points.device
                        )
                    ),
                )
                if query_drive is None:
                    query_drive = message
                else:
                    query_drive = ScalarVectorState(
                        query_drive.scalars + message.scalars,
                        query_drive.vectors + message.vectors,
                        query_drive.pseudos + message.pseudos,
                    )
            else:
                for block, source_moments in zip(
                    self.query_blocks, encoded.query_moments, strict=True
                ):
                    query_drive = block.evaluate_cross(
                        query_operator, source_moments, query_drive
                    )
            if query_drive is None:  # guarded by query_layers >= 1 at construction
                raise RuntimeError("query decoder produced no field state")
            if own is not None and self.trace_drive_read_out is not None:
                # The own-cell drive read-out: an exactly linear typed
                # channel mix of the declared cell's post-attention drive
                # state, added to the query field state.  Drive-linear and
                # zero-preserving by construction (bias-free, no invariant
                # lift), so every field-mode contract survives; applied
                # before the quadratic read-in so the declared drive degree
                # counts it as part of the drive-linear state u.
                trace_state = self.trace_drive_read_out(encoded.drive_state.slice(own))
                query_drive = ScalarVectorState(
                    query_drive.scalars + trace_state.scalars,
                    query_drive.vectors + trace_state.vectors,
                    query_drive.pseudos + trace_state.pseudos,
                )
            if self.quadratic_read_in is not None:
                # Every stage above is exactly linear in the drive at fixed
                # geometry; this one bilinear composition raises the state to
                # declared degree <= 2, and the output projection below is
                # exactly linear in the field state, so the prediction is a
                # degree <= 2 polynomial in the drive by construction.
                query_drive = self.quadratic_read_in(query_operator, query_drive)
            output = self.output_projection(query_operator, query_drive)
            scalar_outputs.append(output.scalars)
            vector_outputs.append(output.vectors)
            pseudo_outputs.append(output.pseudos)

        packed_output = ScalarVectorState(
            torch.cat(scalar_outputs, dim=0),
            torch.cat(vector_outputs, dim=0),
            torch.cat(pseudo_outputs, dim=0),
        )
        point_data: TensorDict | None = None
        if self.output_layout is not None:
            point_data = self.output_layout.unpack(
                ScalarVectorFields(packed_output.scalars, packed_output.vectors)
            )
        if self._output_pseudo_layout is not None:
            pseudo_data = self._output_pseudo_layout.unpack(
                ScalarVectorFields(
                    packed_output.pseudos,
                    packed_output.pseudos.new_empty(
                        packed_output.pseudos.shape[0], 0, self.n_spatial_dims
                    ),
                )
            )
            if point_data is None:
                point_data = pseudo_data
            else:
                # Leaf-wise merge: the two layouts may share nested groups
                # but never leaves (they were split from one declaration).
                for key in pseudo_data.keys(include_nested=True, leaves_only=True):
                    point_data[key] = pseudo_data[key]
        if point_data is None:  # unreachable: at least one output is declared
            raise RuntimeError("no output layout produced predictions")
        return query_mesh.with_data(
            point_data=point_data,
            cell_data={},
            global_data=encoded.global_data,
        )

    def forward(self, domain: DomainMesh) -> Mesh:
        r"""Encode ``domain.boundaries`` and predict at ``domain.interior``."""
        return self.decode(self.encode(domain))


__all__ = [
    "EncodedBoundary",
    "FieldMode",
    "FieldRoleRanks",
    "MeshTransformer",
    "QueryDecoder",
]
