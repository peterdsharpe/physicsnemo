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

r"""Strong-inference resolution of the exterior far-field failure.

Every velocity arm of the exterior potential-flow suite -- including the best
(``mesh_transformer_kernel_singpair_pseudo``, in-distribution relative L2
~0.02-0.05) -- degrades to relative L2 0.58-0.88 on the ``farfield_queries``
split (training queries at canonical radius r/L in [1.05, 4]; evaluation at
[4, 8]).  Three rival mechanisms are put up for EXCLUSION, not confirmation.
Each discriminator's decisive outcome is pre-registered here, before the
measurements are made.

M1 -- ANNULUS COVERAGE (covariate shift): the model simply never saw far
queries.  DISCRIMINATOR: retrain the same arm with training queries widened
to r/L in [1.05, 8] (``--train-query-outer 8`` on
``problems/potential_flow.py``; evaluation splits untouched) and read the
standard split report.  PRE-REGISTERED RULE, with ``F_wide`` the retrained
arm's ``farfield_queries`` error, ``I_wide`` its ``in_distribution`` error,
and ``F_base`` the default-annulus arm's ``farfield_queries`` error:

- M1 SUFFICIENT (supported) if ``F_wide <= max(2 * I_wide, 0.10)`` --
  far-field error joins the ID level once the annulus is covered;
- M1 EXCLUDED as the main cause if ``F_wide >= 0.5 * F_base`` -- widening
  the annulus barely moves the failure;
- AMBIGUOUS otherwise (partial rescue).

M2 -- MEMBER DECAY STRUCTURE (representational): the singpair dictionary's
far-field span is {log r (single layer), 1/r (double layer / single-layer
dipole)} per member, while the exact zero-circulation disturbance velocity
decays like r^-2 (doublet); the per-source kernel coefficients are
query-independent, so any far-field shape the message can take is a fixed
linear combination of member tails.  DISCRIMINATOR: on the trained
checkpoint, evaluate ``rms_theta |u_pred|`` and ``rms_theta |u_exact|`` on
canonical-radius rays r/L in [1.05, 12] for zero-circulation flows and fit
local power-law exponents ``d log|u| / d log r`` in the fixed bands
[1.05,2], [2,3], [3,4] (near) and [4,6], [6,8], [8,12] (far).
PRE-REGISTERED RULE, with ``delta(band) = exponent_pred - exponent_exact``
averaged over cases:

- M2 SUPPORTED if mean |delta| over the far bands > 0.5 while mean |delta|
  over the near bands <= 0.5, AND the departure points toward the member
  tails: both member tails (log r and 1/r) sit ABOVE the exact r^-2, so a
  member-decay pathology surfaces as signed mean far delta > 0 (flattening
  or growth), with the per-member message decomposition (also recorded)
  showing the surviving tail;
- M2 EXCLUDED if mean |delta| over the far bands <= 0.25 (the prediction
  keeps the exact decay law beyond r = 4), or if the far departure is
  large but points AWAY from the member tails (signed mean far delta < 0,
  super-decay): the member basis cannot produce a faster-than-exact decay,
  so such a departure must live in the coefficient values, not the member
  basis values;
- AMBIGUOUS otherwise (including near-band misfit > 0.5, which would
  invalidate the probe).

M3 -- COEFFICIENT EXTRAPOLATION: the query-side conditioned coefficients --
every learned function of query-position invariants that multiplies the
decoded field: the output projection's sigmoid gates, the lifted global
drive at the query (constant drive data times learned functions of |x|),
and the query-operator geometry-vector modulation |v(x)| / |x| -- may
extrapolate wildly beyond the training range of |x|.  DISCRIMINATOR: trace
each of these on the same rays.  PRE-REGISTERED RULE, per trace ``c(r)``
per case, with ``R = max_{r in (4,12]} |c| / max_{r in [1.05,4]} |c|`` and
``n_osc`` the number of direction reversals of ``c`` against log r beyond
r = 4 whose swing exceeds 10% of the trace's full range:

- M3 SUPPORTED if any trace of any case has ``R > 3`` (divergence) or
  ``n_osc >= 2`` (oscillation);
- M3 EXCLUDED if every trace of every case has ``R <= 1.5`` and
  ``n_osc <= 1`` -- smooth continuation beyond the training edge;
- AMBIGUOUS otherwise.

M2 and M3 are distinguished by WHERE the pathology lives: M2 in the member
basis values (the kernel message, decomposed per exact member here), M3 in
the query-side coefficient values (gates, lifted drive, geometry
modulation).  The decoder's output is exactly linear in its field state, so
the prediction splits exactly into a boundary-message part (member basis
territory) and a direct-drive part (query-side-coefficient territory); the
per-branch and per-part magnitudes are recorded as mechanism attribution.

Everything below the verdicts (per-member message norms, per-branch
contributions, error-vs-radius profile with circulation on, and -- added in
iteration 30 -- the conditioned single-layer net-monopole diagnostic) is
supporting evidence, not part of the pre-registered rules.  The analysis
runs unchanged on every rung of the far-field ladder: the checkpoint flags
(``bounded_gates``, ``bounded_query``, ``decaying_drive``,
``monopole_free_sl``) select the parameterization, and the replicated decode
paths reproduce the corresponding structure (decay envelope on the lifted
query drive; measure-weighted single-layer deflation) before verification.

Usage (defaults point at the p16 runs)::

    python studies/farfield_inference.py \
        --checkpoint  scratch/brev/results/phase9/p16_ff_base_s17/out/*.pt \
        --baseline-report scratch/brev/results/phase9/p16_ff_base_s17/out/*.json \
        --retrain-report  scratch/brev/results/phase9/p16_ff_annulus8_s17/out/*.json

This is a benchmark-local research diagnostic; it reaches into private
model internals (`_query_operator_input`, decoder caches) deliberately and
verifies every replicated path against the public ``decode`` output.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import _paths  # noqa: F401
import torch
from conformal_laplace import complex_to_points
from potential_flow import (
    SPLITS,
    _build_model,
    body_to_physical,
    build_potential_flow_velocity_sample,
    disturbance_velocity,
    evaluate_splits,
)

from physicsnemo.experimental.nn.mesh_attention.attention import (
    ScalarVectorState,
    _vector_perp,
)
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    exact_double_layer_member,
    exact_single_layer_member,
)
from physicsnemo.mesh import Mesh

ARM = "mesh_transformer_kernel_singpair_pseudo"
FAMILY = "potential_flow_velocity"
MEMBER_NAMES = ("double_layer", "single_layer")

#: Fixed log-radius bands (canonical preimage radius r/L) for exponent fits.
BANDS: tuple[tuple[float, float], ...] = (
    (1.05, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 12.0),
)
NEAR_BAND_INDICES = (0, 1, 2)
FAR_BAND_INDICES = (3, 4, 5)

#: Pre-registered decision constants (see module docstring).
RULES = {
    "m1_id_factor": 2.0,
    "m1_absolute_floor": 0.10,
    "m1_exclusion_fraction": 0.5,
    "m2_far_support_delta": 0.5,
    "m2_near_valid_delta": 0.5,
    "m2_far_exclude_delta": 0.25,
    "m3_ratio_support": 3.0,
    "m3_ratio_exclude": 1.5,
    "m3_oscillation_support": 2,
    "m3_oscillation_exclude": 1,
    "m3_swing_fraction": 0.1,
}

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the singpair_pseudo velocity arm and load its trained state.

    Honors the checkpoint's recorded ``bounded_gates``, ``bounded_query``,
    ``decaying_drive``, and ``monopole_free_sl`` flags (absent on pre-fix
    checkpoints, defaulting to the historical architecture) so the same
    analysis runs unchanged on collapsed, gate-fixed, source-bounded, and
    decay-structured arms.  None of the knobs adds parameters, so the flags
    alone select the parameterization the state dict is read as.
    """

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model") != ARM or payload.get("family") != FAMILY:
        raise ValueError(
            f"checkpoint {path} holds {payload.get('model')!r}/"
            f"{payload.get('family')!r}, expected {ARM!r}/{FAMILY!r}"
        )
    model = _build_model(
        ARM,
        FAMILY,
        bounded_gates=bool(payload.get("bounded_gates", False)),
        bounded_query=bool(payload.get("bounded_query", False)),
        decaying_drive=bool(payload.get("decaying_drive", False)),
        monopole_free_sl=bool(payload.get("monopole_free_sl", False)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model.kernel_decoder.n_members != len(MEMBER_NAMES):
        raise RuntimeError(
            "the singpair arm must carry exactly the double- and single-layer "
            f"members, got n_members={model.kernel_decoder.n_members}"
        )
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, metadata


# ---------------------------------------------------------------------------
# Ray probes
# ---------------------------------------------------------------------------


def _ray_grid(n_radii: int, n_angles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-spaced canonical radii in [1.05, 12] and uniform angles."""

    radii = torch.logspace(
        math.log10(1.05), math.log10(12.0), n_radii, dtype=torch.float64
    )
    angles = torch.arange(n_angles, dtype=torch.float64) * (_TWO_PI / n_angles)
    return radii, angles


def _rms_over_angles(values: torch.Tensor, n_radii: int, n_angles: int) -> torch.Tensor:
    """Root-mean-square over the angle axis of a flat (radius-major) probe."""

    return values.reshape(n_radii, n_angles).square().mean(dim=1).sqrt()


@torch.no_grad()
def probe_case(
    model: torch.nn.Module,
    seed: int,
    *,
    n_radii: int,
    n_angles: int,
    circulation_on: bool,
    device: torch.device,
) -> dict:
    """Evaluate one flow on canonical rays and record every traced quantity.

    Rays are radius-major: point ``(i, k)`` sits at canonical preimage
    ``radii[i] * exp(1j * angles[k])``; all traces below are RMS over the
    angle axis at fixed radius.  The replicated decode paths (member
    decomposition, direct/message split, output branches) are verified
    against the public ``decode`` output before anything is recorded.
    """

    spec = dict(SPLITS[FAMILY]["in_distribution"])
    if not circulation_on:
        spec["circulation_magnitude_range"] = (0.0, 0.0)
    sample = build_potential_flow_velocity_sample(
        seed, device=device, dtype=torch.float32, **spec
    )
    radii, angles = _ray_grid(n_radii, n_angles)
    preimages = torch.polar(
        radii[:, None].expand(-1, n_angles),
        angles[None, :].expand(n_radii, -1),
    ).reshape(-1)
    exact = disturbance_velocity(
        sample.body, sample.canonical_freestream, sample.circulation, preimages
    )
    physical = complex_to_points(body_to_physical(sample.body, preimages))
    points32 = physical.to(device=device, dtype=torch.float32)

    encoded = model.encode(sample.domain)
    decoded = model.decode(encoded, Mesh(points=points32))
    prediction = decoded.point_data["velocity"]

    # --- replicated decode with intermediates -----------------------------
    normalized = (points32 - encoded.center) / encoded.reference_length
    query_operator = model.operator_input_block(
        model.operator_lift(
            model._query_operator_input(normalized, encoded.global_operator_state)
        )
    )
    n = normalized.shape[0]
    raw_query_drive = ScalarVectorState(
        torch.cat(
            (
                normalized.new_zeros(n, model._boundary_drive_scalars),
                encoded.global_drive_state.scalars.expand(n, -1),
            ),
            dim=-1,
        ),
        torch.cat(
            (
                normalized.new_zeros(n, model._boundary_drive_vectors, 2),
                encoded.global_drive_state.vectors.expand(n, -1, -1),
            ),
            dim=1,
        ),
        torch.cat(
            (
                normalized.new_zeros(n, model._boundary_drive_pseudos),
                encoded.global_drive_state.pseudos.expand(n, -1),
            ),
            dim=-1,
        ),
    )
    lifted = model.drive_lift(query_operator, raw_query_drive)
    if getattr(model, "decaying_direct_drive", False):
        # Iteration 30's decay envelope: decode() multiplies the lifted query
        # drive by the fixed analytic 1/(1+|x|^2) of the RAW normalized
        # radius before the kernel message is added; replicate it so the
        # direct/message split below still reproduces decode() exactly.
        envelope = 1.0 / (1.0 + normalized.square().sum(dim=-1))
        lifted = ScalarVectorState(
            lifted.scalars * envelope[:, None],
            lifted.vectors * envelope[:, None, None],
            lifted.pseudos * envelope[:, None],
        )
    message = model.kernel_decoder(normalized, encoded.kernel_cache)

    # Exact linearity of the output in its field state splits the prediction
    # into a direct-drive part and a boundary-message part.
    out_direct = model.output_projection(query_operator, lifted)
    out_message = model.output_projection(query_operator, message)
    u_direct = out_direct.vectors[:, 0, :]
    u_message = out_message.vectors[:, 0, :]
    _verify(
        u_direct + u_message,
        prediction,
        "direct + message decomposition vs decode()",
    )

    member_states = _member_messages(model.kernel_decoder, encoded, normalized)
    _verify(
        sum(state.vectors for state in member_states.values()),
        message.vectors,
        "per-member message sum vs kernel decoder (vectors)",
    )
    _verify(
        sum(state.scalars for state in member_states.values()),
        message.scalars,
        "per-member message sum vs kernel decoder (scalars)",
    )
    member_output = {
        name: model.output_projection(query_operator, state).vectors[:, 0, :]
        for name, state in member_states.items()
    }

    # Supporting evidence (not pre-registered): the conditioned single-layer
    # net monopole per head/value channel, ||sum_s w_s C_SL(s,h) V(s,h,f)||.
    # On monopole-free checkpoints this is the charge the deflation removes
    # (the effective decoded monopole is exactly zero by construction); on
    # undeflated checkpoints it sources the measured log-r tail.
    cache = encoded.kernel_cache
    sl_charge = (
        cache.weights.double()[:, None]
        * cache.coefficients.double()[:, MEMBER_NAMES.index("single_layer"), :]
    )  # (S, H)
    sl_net_monopole = {
        "scalars": float(
            torch.einsum("sh,shf->hf", sl_charge, cache.value_scalars.double()).norm()
        ),
        "vectors": float(
            torch.einsum("sh,shfd->hfd", sl_charge, cache.value_vectors.double()).norm()
        ),
        "pseudos": float(
            torch.einsum("sh,shf->hf", sl_charge, cache.value_pseudos.double()).norm()
        ),
    }

    branch_direct = _vector_branches(model.output_projection, query_operator, lifted)
    branch_message = _vector_branches(model.output_projection, query_operator, message)
    _verify(
        sum(branch_direct.values()) + sum(branch_message.values()),
        prediction,
        "branch sum vs decode()",
    )

    # --- query-side conditioned coefficient traces (the M3 set) -----------
    # The module's own gate-invariant map, so the trace is faithful for both
    # the raw and the bounded (compactified) gate parameterizations.
    invariants = model.output_projection._geometry_invariants(query_operator)
    gate = 2.0 * torch.sigmoid(model.output_projection.vector_gate(invariants))
    rho = normalized.norm(dim=-1)
    geometry_vector_norm = query_operator.vectors.norm(dim=-1).mean(dim=-1)

    def trace(values: torch.Tensor) -> list[float]:
        return _rms_over_angles(values.double(), n_radii, n_angles).tolist()

    traces_m3 = {
        "output_gate": trace(gate[:, 0]),
        "lift_scalar_norm": trace(lifted.scalars.norm(dim=-1)),
        "lift_vector_norm": trace(lifted.vectors.norm(dim=(-2, -1))),
        "lift_pseudo_norm": trace(lifted.pseudos.norm(dim=-1)),
        "geometry_vector_modulation": trace(geometry_vector_norm / rho),
    }
    magnitudes = {
        "pred": trace(prediction.norm(dim=-1)),
        "exact": _rms_over_angles(exact.norm(dim=-1), n_radii, n_angles).tolist(),
        "direct": trace(u_direct.norm(dim=-1)),
        "message": trace(u_message.norm(dim=-1)),
        **{
            f"member_{name}_message_vector": trace(state.vectors.norm(dim=(-2, -1)))
            for name, state in member_states.items()
        },
        **{
            f"member_{name}_output": trace(vector.norm(dim=-1))
            for name, vector in member_output.items()
        },
        **{
            f"branch_direct_{name}": trace(vector.norm(dim=-1))
            for name, vector in branch_direct.items()
        },
        **{
            f"branch_message_{name}": trace(vector.norm(dim=-1))
            for name, vector in branch_message.items()
        },
    }
    error = (prediction.double() - exact).norm(dim=-1)
    band_relative_l2 = {}
    radius_of_point = radii[:, None].expand(-1, n_angles).reshape(-1)
    for low, high in BANDS:
        mask = (radius_of_point >= low) & (radius_of_point < high)
        band_relative_l2[f"[{low}, {high})"] = float(
            error[mask].square().sum().sqrt()
            / exact.norm(dim=-1)[mask].square().sum().sqrt().clamp_min(1e-30)
        )
    return {
        "seed": seed,
        "circulation": sample.circulation,
        "radii": radii.tolist(),
        "normalized_radius": trace(rho),
        "magnitudes": magnitudes,
        "traces_m3": traces_m3,
        "band_relative_l2": band_relative_l2,
        "sl_net_monopole": sl_net_monopole,
    }


def _verify(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Guard the replicated internals against drift from the public path."""

    scale = expected.double().norm().clamp_min(1e-12)
    residual = (actual.double() - expected.double()).norm() / scale
    if not float(residual) < 1e-3:
        raise RuntimeError(
            f"replicated path diverged from decode(): {label} "
            f"(relative residual {float(residual):.3e})"
        )


def _member_messages(
    decoder: torch.nn.Module,
    encoded,
    normalized_points: torch.Tensor,
) -> dict[str, ScalarVectorState]:
    """Kernel message split per exact member (double layer, single layer).

    Replicates ``KernelBasisCrossDecoder._evaluate_chunk`` for the singpair
    dictionary with the member axis kept, applying the per-channel message
    scale to each member (it is linear, so the member sum reproduces the
    decoder's output exactly; verified by the caller).  With iteration 30's
    ``monopole_free_single_layer`` knob the single-layer column is deflated
    by its measure-weighted boundary mean exactly as in the decoder, so the
    replication stays faithful on decay-structured checkpoints.
    """

    cache = encoded.kernel_cache
    single_layer = exact_single_layer_member(normalized_points, cache.panel_vertices)
    if getattr(decoder, "monopole_free_single_layer", False):
        fraction = cache.weights.to(single_layer.dtype)
        fraction = fraction / fraction.sum()
        single_layer = (
            single_layer - single_layer.sum(dim=-1, keepdim=True) * fraction[None, :]
        )
    members = torch.stack(
        (
            exact_double_layer_member(
                normalized_points, cache.panel_vertices, cache.normals
            ),
            single_layer,
        ),
        dim=-1,
    )  # (Q, S, 2)
    states: dict[str, ScalarVectorState] = {}
    for index, name in enumerate(MEMBER_NAMES):
        kernel = (
            members[..., index, None] * cache.coefficients[None, :, index, :]
        )  # (Q, S, H)
        scalar_heads = torch.einsum("qsh,shf->qhf", kernel, cache.value_scalars)
        vector_heads = torch.einsum("qsh,shfd->qhfd", kernel, cache.value_vectors)
        pseudo_heads = torch.einsum("qsh,shf->qhf", kernel, cache.value_pseudos)
        scalars = scalar_heads.flatten(1) @ decoder.scalar_output_weight.T
        vectors = torch.einsum(
            "ohf,qhfd->qod", decoder.vector_output_weight, vector_heads
        )
        pseudos = pseudo_heads.flatten(1) @ decoder.pseudo_output_weight.T
        states[name] = decoder.message_scale(
            ScalarVectorState(scalars, vectors, pseudos)
        )
    return states


def _vector_branches(
    projection: torch.nn.Module,
    geometry: ScalarVectorState,
    field: ScalarVectorState,
) -> dict[str, torch.Tensor]:
    """The output projection's vector output split by Clebsch-Gordan branch.

    Mirrors ``GeometryConditionedLinear.forward`` for the vector sector; each
    entry already carries the invariant gate, so the values sum to the
    projected velocity (verified by the caller).
    """

    invariants = projection._geometry_invariants(geometry)
    gate = 2.0 * torch.sigmoid(projection.vector_gate(invariants))[:, :, None]
    n = field.n_entities
    branches: dict[str, torch.Tensor] = {}
    if projection.vector_from_vector is not None:
        branches["vector_from_vector"] = torch.einsum(
            "of,nfd->nod", projection.vector_from_vector, field.vectors
        )
    if projection.vector_from_vector_dots is not None:
        dots = torch.einsum("nfd,ngd->nfg", field.vectors, geometry.vectors).flatten(1)
        coefficients = projection.vector_from_vector_dots(dots).reshape(
            n, projection.out_vector_dim, projection.geometry_vector_dim
        )
        branches["vector_from_vector_dots"] = torch.einsum(
            "nog,ngd->nod", coefficients, geometry.vectors
        )
    if projection.vector_from_scalar is not None:
        coefficients = projection.vector_from_scalar(field.scalars).reshape(
            n, projection.out_vector_dim, projection.geometry_vector_dim
        )
        branches["vector_from_scalar"] = torch.einsum(
            "nog,ngd->nod", coefficients, geometry.vectors
        )
    if projection.vector_from_pseudo is not None:
        coefficients = projection.vector_from_pseudo(field.pseudos).reshape(
            n, projection.out_vector_dim, projection.geometry_vector_dim
        )
        branches["vector_from_pseudo"] = torch.einsum(
            "nog,ngd->nod", coefficients, _vector_perp(geometry.vectors)
        )
    return {name: (value * gate)[:, 0, :] for name, value in branches.items()}


# ---------------------------------------------------------------------------
# Fits and pre-registered metrics
# ---------------------------------------------------------------------------


def fit_band_exponent(
    radii: list[float], magnitudes: list[float], band: tuple[float, float]
) -> float | None:
    """Least-squares slope of log|u| against log r inside one radial band."""

    pairs = [
        (math.log(r), math.log(max(m, 1e-30)))
        for r, m in zip(radii, magnitudes, strict=True)
        if band[0] <= r < band[1]
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var <= 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var


def trace_metrics(radii: list[float], values: list[float]) -> dict:
    """Far/near max ratio and filtered oscillation count of one M3 trace."""

    near = [abs(v) for r, v in zip(radii, values, strict=True) if r <= 4.0]
    far = [abs(v) for r, v in zip(radii, values, strict=True) if r > 4.0]
    ratio = max(far) / max(max(near), 1e-30)
    swing_floor = RULES["m3_swing_fraction"] * (max(values) - min(values))
    oscillations = 0
    previous_direction = 0
    last_extremum = None
    for r, v in zip(radii, values, strict=True):
        if r <= 4.0:
            last_extremum = v
            continue
        if last_extremum is None:
            last_extremum = v
            continue
        step = v - last_extremum
        if abs(step) < max(swing_floor, 1e-30):
            continue
        direction = 1 if step > 0 else -1
        if previous_direction and direction != previous_direction:
            oscillations += 1
        previous_direction = direction
        last_extremum = v
    return {"far_near_ratio": ratio, "oscillations": oscillations}


# ---------------------------------------------------------------------------
# Verdicts (pre-registered rules; see module docstring)
# ---------------------------------------------------------------------------


def verdict_m1(baseline: dict | None, retrain: dict | None) -> dict:
    if baseline is None or retrain is None:
        return {"verdict": "pending", "reason": "run reports not available"}
    f_base = baseline["splits"]["farfield_queries"]
    f_wide = retrain["splits"]["farfield_queries"]
    i_wide = retrain["splits"]["in_distribution"]
    threshold = max(RULES["m1_id_factor"] * i_wide, RULES["m1_absolute_floor"])
    numbers = {
        "farfield_baseline": f_base,
        "farfield_widened": f_wide,
        "in_distribution_widened": i_wide,
        "sufficiency_threshold": threshold,
        "exclusion_threshold": RULES["m1_exclusion_fraction"] * f_base,
    }
    if f_wide <= threshold:
        return {"verdict": "supported (sufficient)", **numbers}
    if f_wide >= RULES["m1_exclusion_fraction"] * f_base:
        return {"verdict": "excluded (as main cause)", **numbers}
    return {"verdict": "ambiguous", **numbers}


def verdict_m2(band_deltas: dict[str, float | None]) -> dict:
    labels = [f"[{low}, {high})" for low, high in BANDS]
    near = [
        band_deltas[labels[i]]
        for i in NEAR_BAND_INDICES
        if band_deltas[labels[i]] is not None
    ]
    far = [
        band_deltas[labels[i]]
        for i in FAR_BAND_INDICES
        if band_deltas[labels[i]] is not None
    ]
    near_mean = sum(abs(d) for d in near) / len(near)
    far_mean = sum(abs(d) for d in far) / len(far)
    far_signed = sum(far) / len(far)
    numbers = {
        "near_mean_abs_delta": near_mean,
        "far_mean_abs_delta": far_mean,
        "far_mean_signed_delta": far_signed,
    }
    if (
        far_mean > RULES["m2_far_support_delta"]
        and near_mean <= RULES["m2_near_valid_delta"]
    ):
        if far_signed > 0.0:
            return {"verdict": "supported", **numbers}
        # Departure away from the member tails: the member basis (log r,
        # 1/r) cannot out-decay the exact r^-2, so a super-decaying
        # prediction locates the pathology in the coefficient values.
        return {
            "verdict": "excluded",
            "reason": "far departure is super-decay, opposite the member tails",
            **numbers,
        }
    if far_mean <= RULES["m2_far_exclude_delta"]:
        return {
            "verdict": "excluded",
            "reason": "prediction keeps the exact decay law beyond r=4",
            **numbers,
        }
    return {"verdict": "ambiguous", **numbers}


def verdict_m3(per_case_metrics: list[dict[str, dict]]) -> dict:
    worst_ratio, worst_ratio_trace = 0.0, None
    worst_osc, worst_osc_trace = 0, None
    for case_index, metrics in enumerate(per_case_metrics):
        for name, entry in metrics.items():
            if entry["far_near_ratio"] > worst_ratio:
                worst_ratio = entry["far_near_ratio"]
                worst_ratio_trace = f"case{case_index}:{name}"
            if entry["oscillations"] > worst_osc:
                worst_osc = entry["oscillations"]
                worst_osc_trace = f"case{case_index}:{name}"
    numbers = {
        "worst_far_near_ratio": worst_ratio,
        "worst_far_near_ratio_trace": worst_ratio_trace,
        "worst_oscillations": worst_osc,
        "worst_oscillations_trace": worst_osc_trace,
    }
    if (
        worst_ratio > RULES["m3_ratio_support"]
        or worst_osc >= RULES["m3_oscillation_support"]
    ):
        return {"verdict": "supported", **numbers}
    if (
        worst_ratio <= RULES["m3_ratio_exclude"]
        and worst_osc <= RULES["m3_oscillation_exclude"]
    ):
        return {"verdict": "excluded", **numbers}
    return {"verdict": "ambiguous", **numbers}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    example_dir = Path(__file__).resolve().parents[1]
    repo_root = example_dir.parents[2]
    p16 = repo_root / "scratch" / "brev" / "results" / "phase9"
    run_file = f"{FAMILY}_{ARM}_seed17"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=p16 / "p16_ff_base_s17" / "out" / f"{run_file}.pt",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=p16 / "p16_ff_base_s17" / "out" / f"{run_file}.json",
    )
    parser.add_argument(
        "--retrain-report",
        type=Path,
        default=p16 / "p16_ff_annulus8_s17" / "out" / f"{run_file}.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=example_dir / "results" / f"farfield_inference_{date.today()}.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--radii", type=int, default=64)
    parser.add_argument("--eval-cases", type=int, default=4)
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    model, metadata = load_checkpoint(arguments.checkpoint, device)

    # Authenticity check: the loaded checkpoint must reproduce the reported
    # failure signature (ID low, farfield high) on the frozen eval banks.
    reproduction = evaluate_splits(
        model,
        family=FAMILY,
        eval_seed=97_000_037,
        n_cases=arguments.eval_cases,
        device=device,
        dtype=torch.float32,
    )

    zero_circulation_cases = [
        probe_case(
            model,
            424_243 + 101 * case,
            n_radii=arguments.radii,
            n_angles=arguments.angles,
            circulation_on=False,
            device=device,
        )
        for case in range(arguments.cases)
    ]
    circulation_cases = [
        probe_case(
            model,
            777_121 + 211 * case,
            n_radii=arguments.radii,
            n_angles=arguments.angles,
            circulation_on=True,
            device=device,
        )
        for case in range(arguments.cases)
    ]

    band_labels = [f"[{low}, {high})" for low, high in BANDS]

    def mean_band_exponents(cases: list[dict], key: str) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for band, label in zip(BANDS, band_labels, strict=True):
            fits = [
                fit_band_exponent(case["radii"], case["magnitudes"][key], band)
                for case in cases
            ]
            fits = [fit for fit in fits if fit is not None]
            result[label] = sum(fits) / len(fits) if fits else None
        return result

    exponents = {
        key: mean_band_exponents(zero_circulation_cases, key)
        for key in (
            "pred",
            "exact",
            "direct",
            "message",
            "member_double_layer_message_vector",
            "member_single_layer_message_vector",
        )
    }
    band_deltas = {
        label: (
            None
            if exponents["pred"][label] is None or exponents["exact"][label] is None
            else exponents["pred"][label] - exponents["exact"][label]
        )
        for label in band_labels
    }
    m3_metrics = [
        {
            name: trace_metrics(case["radii"], values)
            for name, values in case["traces_m3"].items()
        }
        for case in zero_circulation_cases
    ]

    mean_band_error = {
        label: sum(case["band_relative_l2"][label] for case in circulation_cases)
        / len(circulation_cases)
        for label in band_labels
    }

    verdicts = {
        "M1_annulus_coverage": verdict_m1(
            _load_report(arguments.baseline_report),
            _load_report(arguments.retrain_report),
        ),
        "M2_member_decay_structure": verdict_m2(band_deltas),
        "M3_coefficient_extrapolation": verdict_m3(m3_metrics),
    }

    artifact = {
        "generated": str(date.today()),
        "arm": ARM,
        "family": FAMILY,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_metadata": metadata,
        "checkpoint_split_reproduction": reproduction,
        "decision_rules": RULES,
        "bands": band_labels,
        "near_bands": [band_labels[i] for i in NEAR_BAND_INDICES],
        "far_bands": [band_labels[i] for i in FAR_BAND_INDICES],
        "exponents_mean_over_cases": exponents,
        "band_exponent_deltas_pred_minus_exact": band_deltas,
        "band_relative_l2_circulation_on": mean_band_error,
        "m3_trace_metrics_per_case": m3_metrics,
        "verdicts": verdicts,
        "zero_circulation_cases": zero_circulation_cases,
        "circulation_cases": circulation_cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2))

    print(f"\ncheckpoint split reproduction ({arguments.eval_cases} cases/split):")
    for name, value in sorted(reproduction.items()):
        print(f"  {name:24s} {value:.4f}")
    print("\nband exponents  d log|u| / d log r  (mean over zero-circulation cases):")
    header = f"  {'band':>12s} {'exact':>8s} {'pred':>8s} {'direct':>8s} "
    header += f"{'message':>8s} {'DL msg':>8s} {'SL msg':>8s} {'relL2':>8s}"
    print(header)
    for label in band_labels:

        def cell(key: str, label: str = label) -> str:
            value = exponents[key][label]
            return f"{value:8.2f}" if value is not None else "     n/a"

        print(
            f"  {label:>12s} {cell('exact')} {cell('pred')} {cell('direct')} "
            f"{cell('message')} {cell('member_double_layer_message_vector')} "
            f"{cell('member_single_layer_message_vector')} "
            f"{mean_band_error[label]:8.3f}"
        )
    print("\nverdicts:")
    for name, entry in verdicts.items():
        detail = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in entry.items()
            if k != "verdict"
        )
        print(f"  {name:32s} {entry['verdict']:28s} {detail}")
    print(f"\nartifact: {arguments.output}")


if __name__ == "__main__":
    main()
