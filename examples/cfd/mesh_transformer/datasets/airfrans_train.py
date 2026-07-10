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

r"""Training and evaluation driver for the pre-registered AirFRANS campaign.

The industrial sibling of :mod:`dataset_train_ns`, executing the protocol
pre-registered in the book's AirFRANS chapter (``book/07-airfrans.qmd``) on
the catalog written by :mod:`airfrans_preprocess`.  The mapping is

.. math::

   \{\text{airfoil boundary (geometry only)},\ \hat U_\infty,\
   \ln\mathrm{Re}_c\} \longmapsto
   \{\Delta U/|U_\infty|,\ C_p,\ \ln(1+\nu_t/\nu)\}

with the freestream as a global rank-1 drive, the Reynolds number as one
dimensionless global operator scalar, and the model on its intrinsic scale
gauge (no declared reference length) -- the pre-registration's parsimony
test against GLOBE's two-reference-length construction.

ARMS (all on the program's reference configuration -- one encoder layer,
one full-width head at ranks 48/16, the two-member "singpair" kernel
dictionary -- in ``field_mode="zero_preserving_nonlinear"``; RANS is
nonlinear, so drive-linearity is not declared):

- ``mt_nl`` -- the plain architecture (H1: baseline competence).
- ``mt_nl_pseudo`` -- adds ``drive_pseudo_dim=8`` with NO declared ``"0o"``
  fields: the pseudo sector is fed by internally generated wedge products
  of vector drives, legal per the model contract and exactly the
  ``ns_cavity_star`` nl_pseudo construction (H2: lift/circulation is
  handed content; sector liveness is pinned by the test suite).
- ``mt_nl_decay`` -- the far-field ladder's closed recipe exactly as the
  winning ``farfield_inference`` checkpoint was built
  (``bounded_output_gate_invariants`` + ``bounded_query_geometry`` +
  ``decaying_direct_drive``) MINUS ``kernel_monopole_free_single_layer``,
  which stays OFF: a lifting airfoil carries net circulation, exactly the
  topological sector the deflation structurally kills, so its physics
  license does not hold here (H3: exterior far-field structure transfers).
- ``mt_nl_scale`` -- singpair plus 8 learned smooth members fed by the
  DECLARED auxiliary viscous scale
  (``kernel_auxiliary_scale_key="viscous_scale"``, lambda = Re^(-1/2),
  declared as a rank-0 global operator field by this arm alone).  H4: the
  pilot's velocity error is wall-concentrated (49% of MSE inside
  d/c < 1e-4) while the boundary layer lives at delta/c ~ Re^(-1/2) ~
  5e-4, a scale the singpair dictionary has no learned pair-radial
  pathway to resolve; this arm declares the scale and hands the smooth
  members r/delta-normalized invariants.
- ``mt_nl_members`` -- identical 8 smooth members WITHOUT the auxiliary
  scale: H4's pre-registered capacity control, isolating "declared scale"
  from "more members".
- ``mt_nl_members_log`` -- identical 8 smooth members plus
  ``kernel_log_radial_features=True`` (H4-L, pre-registered as V4 in the
  velocity-front fan-out): the members arm's feature map gains ln(a+eps)
  and the scale-free normalized alignments, making any power-law radial
  scale linearly learnable (H4's verdict refuted the DECLARED scale --
  the capacity control won ~6x -- so the scale exponent becomes learnable
  log-space structure instead).  Capacity-matched to ``mt_nl_members``
  except the feature map: only the member MLP's input width differs, so
  parameter counts are nearly equal (both recorded in the reports).
  Falsifier: no improvement over ``mt_nl_members``.

PROTOCOL (pre-registered; deviations would be protocol violations):

- Tasks ``scarce`` (200 train) then ``full`` (800 train); the official
  ``reynolds`` / ``aoa`` extrapolation tasks are evaluated zero-shot from
  full-trained checkpoints via ``--eval-checkpoint`` / ``--eval-task``.
- Full airfoil boundary every step (no boundary downsampling); 4,096
  volume query points per iteration (GLOBE-matched), freshly drawn from a
  seeded generator.
- Loss: Huber (``delta=1``) summed over the three target fields with
  per-field NaN-row exclusion (the published pathology masks arrive as NaN
  rows from the preprocessor).  No per-field weights: GLOBE's error scales
  are not pre-registered, so they are not applied.
- Optimizer/schedule: the fixed-dataset conventions of
  :mod:`dataset_train_ns` (AdamW at ``3e-4`` / weight decay ``1e-6``,
  gradient clipping at ``1.0``, reproducibly shuffled epochs, gradient
  accumulation over ``--batch-cases``, best-validation checkpoint restored
  before the final evaluation).  Validation follows GLOBE: the task's test
  list (``full_test`` for ``scarce``), capped at
  :data:`MAX_VALIDATION_CASES` cases.

METRICS, two z-score conventions side by side (see the
``metrics_conventions`` design note): (i) ``zscore_mse/...``, the
POOLED_TRAIN convention -- per-field MSE for ``u_x``, ``u_y``, ``p`` over
volume points and ``p`` on-surface, normalized by the mean/std of the RAW
physical fields pooled over every valid query point of the task's TRAIN
split (the mean cancels in the MSE; both constants recorded);
(ii) ``globe_zscore_mse/...``, the REFERENCE_PERSAMPLE convention -- an
exact port of GLOBE's ``benchmark.py`` (which matched the AirFRANS
reference *code*): nondimensional fields (``U/|U_inf|`` split into
``_x``/``_y``/``_mag``, ``C_p``, ``C_pt``, ``ln(1+nut/nu)``; pressure
fields also ``_surface_only``), per-sample unbiased-std normalization,
unweighted mean over samples.  The published-table columns u_x/u_y/p/p_s
are :data:`GLOBE_HEADLINE_METRICS` (tables render with a x10^-2 column
multiplier; stored values are raw).  Plus (iii) physically nondimensional
MAE (:math:`|\Delta U|` error per :math:`|U_\infty|`, :math:`C_p` error),
the more honest unit, reported alongside.

Every evaluation additionally decomposes the per-field z-score MSE and the
delta-velocity nondimensional MAE into WALL-DISTANCE BANDS of ``d/c``
(:data:`DISTANCE_BAND_EDGES`; exact minimum distance to the boundary
segments, vectorized and chunked) with per-band valid-point counts -- the
diagnostic for the near-wall-concentration hypothesis (at Re ~ 4e6 the
boundary layer lives at ``d/c ~ Re^{-1/2} ~ 5e-4``), available eval-only
against existing pilot checkpoints via ``--eval-checkpoint``.

LEARNING-RATE PROTOCOLS: ``--schedule flat`` (default, protocol-v0) is the
constant-AdamW pilot protocol, byte-identical.  ``--schedule globe``
(protocol-v1) adopts GLOBE's scaled base learning rate
(``1e-3 * sqrt(world_size * queries_per_step / 2048)``, floor at
``peak/64``, no warmup) with a deterministic cosine from peak to floor over
the run in place of GLOBE's validation-coupled plateau rule (same
endpoints; see :func:`resolve_schedule`).  The resolved spec is recorded in
every report under ``"schedule"``.

Long runs survive walltime kills via ``--checkpoint-every N`` (a single
overwritten resume file: model, optimizer, generators, best-so-far state,
history) plus ``--resume``; a resumed run continues bit-identically.
``--validate-every N`` (default 100) sets the validation cadence; each
validation epoch's JSON line carries both conventions' headline metrics,
so multi-hour runs are inspectable mid-flight.

Every report JSON carries the arm, task, seed, epochs, metrics, history,
and provenance (catalog path plus the SHA-256 of
``preprocess_manifest.json``), matching the program's archive style.

This is a benchmark-local research driver, not a proposed public API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import airfrans_dataset
import dataset_catalog
import torch
from airfrans_dataset import TARGET_FIELDS, AirFRANSCase, case_domain
from models import MeshTransformerConfig
from potential_flow import _KERNEL_ARM_KWARGS, _ONE_HEAD_CONFIG
from torch import nn
from torch.nn import functional as F  # noqa: N812 - torch convention

from physicsnemo.mesh import DomainMesh

FAMILY = "airfrans"
TRAIN_TASKS = ("scarce", "full")
MAX_VALIDATION_CASES = 8

OUTPUT_FIELD_RANKS: dict[str, int] = {
    "delta_velocity": 1,
    "pressure_coefficient": 0,
    "log_nut_ratio": 0,
}

#: Geometry-only no-slip boundary; the drive is the global freestream
#: direction, the operator scalar is ln(Re_c) (see the module docstring).
_SCHEMA: dict[str, dict] = {
    "boundary_field_ranks": {"airfoil": {"operator": {}, "drive": {}}},
    "global_field_ranks": {
        "operator": {"log_reynolds": 0},
        "drive": {"freestream_direction": 1},
    },
}

_SINGPAIR_KWARGS = _KERNEL_ARM_KWARGS["mesh_transformer_kernel_singpair"]

ARM_NAMES: tuple[str, ...] = (
    "mt_nl",
    "mt_nl_pseudo",
    "mt_nl_decay",
    "mt_nl_scale",
    "mt_nl_members",
    "mt_nl_members_pseudo",
    "mt_nl_members16",
    "mt_nl_members32",
    "mt_nl_members_log",
    "mt_nl_members_wide",
    "mt_nl_members_log_pseudo",
)

#: Arm-specific model kwargs on top of the shared reference construction.
#: The two H4 arms (book/07-airfrans.qmd) differ from ``mt_nl`` only by the
#: 8 learned smooth members, plus (scale arm only) the declared auxiliary
#: viscous scale entering the pair kernels as r/delta-normalized invariants;
#: ``mt_nl_members`` is the pre-registered capacity control.
_ARM_KWARGS: dict[str, dict] = {
    "mt_nl": {},
    "mt_nl_pseudo": {"drive_pseudo_dim": 8},
    "mt_nl_decay": {
        "bounded_output_gate_invariants": True,
        "bounded_query_geometry": True,
        "decaying_direct_drive": True,
    },
    "mt_nl_scale": {
        "kernel_mlp_members": 8,
        "kernel_auxiliary_scale_key": "viscous_scale",
    },
    "mt_nl_members": {"kernel_mlp_members": 8},
    # Composition of the two confirmed wins: the parity (0o) sector (H2,
    # 4/4 seeds) and the learned smooth members (H4's capacity-control
    # verdict: members >> declared scale). Pre-registered expectation
    # (@sec-nb-h4-verdict): the mechanisms are independent (typed drive
    # content vs pair-radial capacity), so the gains compose.
    "mt_nl_members_pseudo": {"kernel_mlp_members": 8, "drive_pseudo_dim": 8},
    # Capacity-response ladder for the velocity front (@sec-nb-velocity-sweep):
    # if velocity error falls monotonically with member count, the front is
    # capacity-limited; if flat from 8, it is structure-limited and wider
    # members are not bought (parsimony bar pre-registered in the notebook).
    "mt_nl_members16": {"kernel_mlp_members": 16},
    "mt_nl_members32": {"kernel_mlp_members": 32},
    # H4-L / V4 (@sec-nb-velocity-sweep): the members arm with log-radial
    # pair features -- a power-law scale r/(L*Pi^alpha) is LINEAR in log
    # space (ln r - alpha ln Pi), so ln(a+eps) plus the ln Re conditioning
    # the members already receive makes ANY power-law scale learnable, with
    # no declared exponent (the declared-scale contract was refuted by its
    # capacity control, @sec-nb-h4-verdict).  Capacity-matched to
    # mt_nl_members except the feature map: only the member MLP's input
    # width differs (params nearly equal; both recorded in the reports).
    "mt_nl_members_log": {
        "kernel_mlp_members": 8,
        "kernel_log_radial_features": True,
    },
    # V5 (width response): doubles the encoder/decoder channel widths at
    # fixed members=8 -- decides whether width binds after the members
    # rung.  Falsifier: no improvement at 2x width -- the reference width
    # is sufficient and wide is not bought (parsimony bar).  These keys
    # override the shared reference capacity in build_arm.
    "mt_nl_members_wide": {
        "kernel_mlp_members": 8,
        "scalar_rank": 96,
        "vector_rank": 32,
        "operator_scalar_dim": 64,
        "operator_vector_dim": 16,
        "drive_scalar_dim": 96,
        "drive_vector_dim": 24,
    },
    # Exploratory positioning of the likely-best composition (V4-dependent:
    # if V4 is null this reads as a members_pseudo replicate whose extra
    # log features are inert).
    "mt_nl_members_log_pseudo": {
        "kernel_mlp_members": 8,
        "kernel_log_radial_features": True,
        "drive_pseudo_dim": 8,
    },
}

#: The four pooled-convention headline metrics (validation model selection
#: uses their mean, unchanged from the pilot protocol); ``nut`` is reported
#: alongside for the published volume table but is not part of the
#: selection scalar.
HEADLINE_METRICS: tuple[str, ...] = (
    "zscore_mse/u_x",
    "zscore_mse/u_y",
    "zscore_mse/p",
    "zscore_mse/p_surface",
)

#: The published-table columns (u_x, u_y, p, p_s) under the REFERENCE
#: per-sample convention (see :func:`_globe_zscore_metrics`); these are the
#: numbers directly comparable to the AirFRANS/GLOBE/Transolver tables.
GLOBE_HEADLINE_METRICS: tuple[str, ...] = (
    "globe_zscore_mse/u_x",
    "globe_zscore_mse/u_y",
    "globe_zscore_mse/c_p",
    "globe_zscore_mse/c_p_surface_only",
)

#: Wall-distance band edges in units of chord (d/c) for the near-wall error
#: decomposition: band k spans ``[edges[k], edges[k+1])``, the last band is
#: unbounded.  At Re ~ 4e6 the boundary layer lives at d/c ~ Re^-1/2 ~ 5e-4
#: (bands 0-1); the diagnostic tests the near-wall-concentration hypothesis
#: for the velocity / surface-pressure gap.
DISTANCE_BAND_EDGES: tuple[float, ...] = (
    0.0,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    math.inf,
)
N_DISTANCE_BANDS = len(DISTANCE_BAND_EDGES) - 1

# --- Learning-rate protocols ----------------------------------------------
#
# ``flat`` is protocol-v0: the fixed-dataset convention of dataset_train_ns
# (constant AdamW at 3e-4), byte-identical to the pilot runs.  ``globe`` is
# protocol-v1: GLOBE's scaled base learning rate (train.py: base 1e-3 times
# sqrt(world_size * points_per_iter / 2048), floor at scaled/64, no warmup)
# with the plateau rule replaced by a deterministic cosine from the scaled
# peak to peak/64 over the run -- same endpoints, schedule shape fixed in
# advance instead of validation-coupled (GLOBE itself uses
# ReduceLROnPlateau(factor=0.5, patience=400, min_lr=peak/64, threshold=0)
# on RAdam; this driver keeps AdamW and the deterministic shape, disclosed).
SCHEDULES: tuple[str, ...] = ("flat", "globe")
_DEFAULT_BASE_LR: dict[str, float] = {"flat": 3.0e-4, "globe": 1.0e-3}
GLOBE_LR_REFERENCE_QUERIES = 2048
GLOBE_LR_FLOOR_FACTOR = 64.0
WORLD_SIZE = 1  # single-GPU driver; recorded in the report for the formula


def scaled_peak_lr(
    base_lr: float, queries_per_step: int, world_size: int = WORLD_SIZE
) -> float:
    """GLOBE's LR scaling: ``base * sqrt(world_size * queries / 2048)``."""

    return base_lr * (world_size * queries_per_step / GLOBE_LR_REFERENCE_QUERIES) ** 0.5


def resolve_schedule(
    schedule: str, learning_rate: float | None, queries_per_step: int
) -> dict:
    """Resolve the run's learning-rate protocol into a recorded spec dict.

    ``learning_rate=None`` selects the protocol's base (3e-4 flat, 1e-3
    globe); an explicit value overrides the base in either protocol.
    """

    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; available: {SCHEDULES}")
    base_lr = _DEFAULT_BASE_LR[schedule] if learning_rate is None else learning_rate
    if schedule == "flat":
        return {
            "name": "flat",
            "base_lr": base_lr,
            "peak_lr": base_lr,
            "floor_lr": base_lr,
            "world_size": WORLD_SIZE,
            "queries_per_step": queries_per_step,
            "warmup": None,
            "formula": "constant lr = base_lr (protocol-v0)",
        }
    peak = scaled_peak_lr(base_lr, queries_per_step)
    return {
        "name": "globe",
        "base_lr": base_lr,
        "peak_lr": peak,
        "floor_lr": peak / GLOBE_LR_FLOOR_FACTOR,
        "world_size": WORLD_SIZE,
        "queries_per_step": queries_per_step,
        "warmup": None,  # GLOBE has no warmup
        "formula": (
            "peak = base_lr * sqrt(world_size * queries_per_step / 2048); "
            "floor = peak / 64; lr(epoch) = floor + (peak - floor)/2 * "
            "(1 + cos(pi * (epoch - 1) / (epochs - 1))) for epoch = 1..epochs "
            "(protocol-v1; GLOBE endpoints, deterministic cosine shape)"
        ),
    }


def schedule_learning_rate(schedule_spec: dict, epoch: int, epochs: int) -> float:
    """Learning rate for one epoch under a resolved schedule spec."""

    peak, floor = schedule_spec["peak_lr"], schedule_spec["floor_lr"]
    if schedule_spec["name"] == "flat":
        return peak
    progress = 0.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


_DESIGN_NOTES = {
    "zscore_convention": (
        "POOLED_TRAIN convention ('zscore_mse/...'): constants are the "
        "per-field mean/std of the RAW physical fields (u_x, u_y in m/s; p "
        "in Pa; nut in m^2/s), reconstructed per case from the stored "
        "nondimensional targets via U = dU*|U_inf| + U_inf, p = C_p*q_inf, "
        "nut = expm1(ln(1+nut/nu))*nu, pooled over every valid (non-masked) "
        "query point of the task's TRAIN split (train-set normalization "
        "constants, test MSE reported in normalized units).  The mean "
        "cancels in the MSE; both constants are recorded"
    ),
    "metrics_conventions": (
        "two z-score conventions are reported side by side and labeled: "
        "'zscore_mse/...' is pooled_train (see zscore_convention); "
        "'globe_zscore_mse/...' is reference_persample -- an exact port of "
        "GLOBE's benchmark.py, which matched the AirFRANS reference CODE "
        "(the paper text says freestream normalization; the code z-scores). "
        "Reference_persample: metrics on the NONDIMENSIONAL fields "
        "(U/|U_inf| expanded to _x/_y/_mag, C_p, C_pt, ln(1+nut/nu)); "
        "surface variants mask off-surface points to NaN BEFORE the "
        "validity filter; per-variant error is normalized by the SAMPLE's "
        "own std of its true variant (torch-default UNBIASED std; "
        "std <= 0 variants skipped); per-sample point-mean of the squared "
        "normalized error; final = unweighted mean over samples, with "
        "skipped samples contributing zero but remaining in the "
        "denominator (the benchmark's scalar-accumulator reduction).  The "
        "published-table columns u_x/u_y/p/p_s are globe_zscore_mse/"
        "{u_x,u_y,c_p,c_p_surface_only}; published tables render with a "
        "x10^-2 column multiplier -- values stored here are RAW (no "
        "multiplier).  Under the unbiased std, the per-sample mean "
        "predictor floors every variant at exactly (N_valid-1)/N_valid "
        "(~1.0), pinned by the test suite.  Model selection stays on the "
        "pooled_train headline mean (pilot protocol, unchanged)"
    ),
    "pseudo_arm": (
        "mt_nl_pseudo declares NO '0o' fields: drive_pseudo_dim=8 alone is "
        "legal (the sector is fed by internally generated wedge products of "
        "vector drives -- the model contract, and bitwise the ns_cavity_star "
        "nl_pseudo construction); the test suite pins that a nonzero-AoA "
        "forward produces nonzero pseudo-sector activations"
    ),
    "decay_arm": (
        "mt_nl_decay mirrors the far-field ladder's closed recipe exactly "
        "as the winning farfield_inference checkpoint was constructed "
        "(bounded_output_gate_invariants + bounded_query_geometry + "
        "decaying_direct_drive) minus kernel_monopole_free_single_layer, "
        "which MUST stay off: a lifting airfoil carries net circulation -- "
        "the deflation's zero-net-flux license does not hold"
    ),
    "aux_scale_arms": (
        "mt_nl_scale (H4) declares the auxiliary viscous scale "
        "viscous_scale = Re_c^(-1/2) as a rank-0 global operator field and "
        "hands it to the kernel decoder "
        "(kernel_auxiliary_scale_key='viscous_scale'), so its 8 learned "
        "smooth members additionally see r/delta-normalized pair "
        "invariants at the boundary-layer scale (the banded diagnostic put "
        "49% of velocity MSE inside d/c < 1e-4, at delta/c ~ Re^(-1/2) ~ "
        "5e-4); mt_nl_members is the pre-registered capacity control -- "
        "identical 8 members, no declared scale -- separating 'declared "
        "contract' from 'more capacity'"
    ),
    "log_radial_arm": (
        "mt_nl_members_log (H4-L, pre-registered as V4) is mt_nl_members "
        "plus kernel_log_radial_features=True and nothing else: the member "
        "MLP additionally sees ln(a+eps) and the scale-free normalized "
        "alignments, so any power-law radial scale ln(r/(L*Pi^alpha)) = "
        "ln r - alpha*ln Pi is linearly learnable from features it already "
        "has (ln Re rides in as an operator scalar).  Lineage: H4's "
        "declared viscous scale was refuted by this very capacity control "
        "(~6x on velocity), so the exponent is learned, not declared.  "
        "Capacity-matched except the feature map -- only the MLP input "
        "width differs, parameter counts nearly equal, both recorded.  "
        "Falsifier: no improvement over mt_nl_members"
    ),
    "validation_protocol": (
        "GLOBE-matched: the scarce task defines no test list of its own and "
        "is validated/evaluated against full_test; validation for model "
        "selection uses the first MAX_VALIDATION_CASES cases of the task's "
        "test list at full query sets (GLOBE validates on the same list "
        "every epoch -- mirrored for budget parity, and disclosed)"
    ),
    "loss": (
        "Huber (delta=1) summed over the three target fields with per-field "
        "NaN-row exclusion; equal field weights (GLOBE's per-field error "
        "scales are not pre-registered and are not applied)"
    ),
    "gauge": (
        "no reference_length is declared: the arms run on the model's "
        "intrinsic gauge plus the single ln(Re_c) operator scalar -- the "
        "pre-registration's parsimony replacement for GLOBE's "
        "two-reference-length construction"
    ),
    "distance_bands": (
        "every evaluation decomposes the per-field z-score MSE (and the "
        "delta-velocity nondimensional MAE) into wall-distance bands of d/c "
        "(exact min distance to the boundary segments), keys "
        "'<metric>@band<k>' with band edges in 'band_edges' -- the "
        "diagnostic for the near-wall-concentration hypothesis (boundary "
        "layer at Re ~ 4e6 lives at d/c ~ 5e-4); aggregated over cases like "
        "the headline metrics, with per-band valid-point counts summed"
    ),
    "lr_schedule": (
        "--schedule flat (default, protocol-v0) is the byte-identical pilot "
        "protocol: constant AdamW at 3e-4.  --schedule globe (protocol-v1) "
        "uses GLOBE's scaled base LR (1e-3 * sqrt(world_size * "
        "queries_per_step / 2048), floor at peak/64, no warmup -- GLOBE "
        "train.py) with the validation-coupled ReduceLROnPlateau(0.5, "
        "patience=400, min_lr=peak/64) rule replaced by a deterministic "
        "cosine from peak to peak/64 over the run (same endpoints, shape "
        "fixed in advance); the resolved spec is recorded per run under "
        "'schedule'"
    ),
}


def build_arm(arm: str) -> nn.Module:
    """Instantiate one arm of the AirFRANS comparison (shared with tests).

    The reference construction: ``n_spatial_dims=2``, kernel query decoder
    with the two-member singpair dictionary, one encoder layer, the
    one-full-width-head capacity trade (ranks 48/16), zero-preserving
    nonlinear field mode, intrinsic scale gauge.  Arm-specific knobs come
    from :data:`_ARM_KWARGS`.
    """

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; available: {ARM_NAMES}")
    from physicsnemo.experimental.nn import MeshTransformer

    capacity = asdict(MeshTransformerConfig())
    capacity.update(_ONE_HEAD_CONFIG)
    capacity["operator_layers"] = 1  # the reference configuration's one layer
    kernel_kwargs = dict(_SINGPAIR_KWARGS)
    kernel_kwargs.update(_ARM_KWARGS[arm])
    # Capacity-axis arm knobs (the V5 width arms) override the shared
    # reference capacity instead of colliding with it at the constructor.
    for key in list(kernel_kwargs):
        if key in capacity:
            capacity[key] = kernel_kwargs.pop(key)
    global_field_ranks = {
        role: dict(fields) for role, fields in _SCHEMA["global_field_ranks"].items()
    }
    auxiliary_key = kernel_kwargs.get("kernel_auxiliary_scale_key")
    if auxiliary_key is not None:
        # H4 (book/07-airfrans.qmd): only the aux-scale arm declares the
        # viscous scale as a rank-0 global operator field; every other arm's
        # declared schema is untouched (the loader always carries the value
        # in global_data, where undeclared leaves are ignored).
        global_field_ranks["operator"][auxiliary_key] = 0
    return MeshTransformer(
        n_spatial_dims=2,
        output_field_ranks=dict(OUTPUT_FIELD_RANKS),
        boundary_field_ranks=_SCHEMA["boundary_field_ranks"],
        global_field_ranks=global_field_ranks,
        reference_length_key=None,  # intrinsic gauge (pre-registration)
        field_mode="zero_preserving_nonlinear",
        query_decoder="kernel",
        **kernel_kwargs,
        **capacity,
    )


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


class AirFRANSCaseBank:
    """Serve preprocessed cases as device-resident samples (cached)."""

    def __init__(
        self,
        directory: Path | str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache: bool = True,
    ) -> None:
        self._directory = Path(directory)
        self._device = device
        self._dtype = dtype
        self._cache: dict[str, AirFRANSCase] | None = {} if cache else None
        self._distance_cache: dict[str, torch.Tensor] | None = {} if cache else None

    def case(self, name: str) -> AirFRANSCase:
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        case = airfrans_dataset.load_case(
            self._directory, name, device=self._device, dtype=self._dtype
        )
        if self._cache is not None:
            self._cache[name] = case
        return case

    def distances(self, name: str) -> torch.Tensor:
        """Per-query wall distance (cached under the same policy as cases)."""

        if self._distance_cache is not None and name in self._distance_cache:
            return self._distance_cache[name]
        distances = airfrans_dataset.boundary_distances(self.case(name))
        if self._distance_cache is not None:
            self._distance_cache[name] = distances
        return distances


def _predictions(model: nn.Module, domain: DomainMesh) -> dict[str, torch.Tensor]:
    """Predict with the targets stripped at the benchmark boundary."""

    model_domain = DomainMesh(
        interior=domain.interior.with_data(point_data={}, cell_data={}, global_data={}),
        boundaries=domain.boundaries,
        global_data=domain.global_data,
    )
    point_data = model(model_domain).point_data
    return {name: point_data[name] for name in OUTPUT_FIELD_RANKS}


def masked_huber_loss(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Huber (``delta=1``) summed over fields, NaN target rows excluded.

    A row of a vector field is excluded when any component is NaN (the
    preprocessor NaNs whole rows, so the distinction is moot on real data).
    Fields with no valid rows in a draw contribute nothing; an all-masked
    draw returns a zero connected to the predictions so ``backward`` stays
    well-defined.
    """

    terms: list[torch.Tensor] = []
    for name, target in targets.items():
        prediction = predictions[name]
        valid = torch.isfinite(target)
        if target.ndim > 1:
            valid = valid.all(dim=-1)
        if bool(valid.any()):
            terms.append(F.huber_loss(prediction[valid], target[valid], delta=1.0))
    if not terms:
        return sum(value.sum() for value in predictions.values()) * 0.0
    return sum(terms)


# ---------------------------------------------------------------------------
# Metrics: train-split z-score constants, per-case evaluation
# ---------------------------------------------------------------------------

_RAW_FIELDS = ("u_x", "u_y", "p", "nut")


def _raw_fields(
    case: AirFRANSCase, delta: torch.Tensor, cp: torch.Tensor, log_nut: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Reconstruct raw physical fields (float64) from nondimensional ones."""

    u_inf = case.u_inf.to(delta.device)
    velocity = delta.double() * case.u_inf_magnitude + u_inf[None, :]
    return {
        "u_x": velocity[:, 0],
        "u_y": velocity[:, 1],
        "p": cp.double() * case.dynamic_pressure,
        "nut": torch.expm1(log_nut.double()) * case.nu,
    }


def compute_normalization(bank: AirFRANSCaseBank, names: list[str]) -> dict:
    """Train-split z-score constants (see ``zscore_convention`` note)."""

    sums = {field: 0.0 for field in _RAW_FIELDS}
    squares = {field: 0.0 for field in _RAW_FIELDS}
    counts = {field: 0 for field in _RAW_FIELDS}
    for name in names:
        case = bank.case(name)
        raw = _raw_fields(case, *(case.targets[key] for key in TARGET_FIELDS))
        for field, values in raw.items():
            valid = values[torch.isfinite(values)]
            sums[field] += float(valid.sum())
            squares[field] += float(valid.square().sum())
            counts[field] += int(valid.numel())
    fields = {}
    for field in _RAW_FIELDS:
        if counts[field] == 0:
            raise airfrans_dataset.AirFRANSCatalogError(
                f"no valid points to normalize field {field!r}"
            )
        mean = sums[field] / counts[field]
        variance = max(squares[field] / counts[field] - mean**2, 0.0)
        fields[field] = {
            "mean": mean,
            "std": max(variance**0.5, 1.0e-30),
            "n_points": counts[field],
        }
    return {
        "convention": _DESIGN_NOTES["zscore_convention"],
        "n_cases": len(names),
        "fields": fields,
    }


def _band_index(distances: torch.Tensor, chord: float) -> torch.Tensor:
    """Wall-distance band index (0..N-1) per query, in units of d/chord."""

    interior_edges = torch.tensor(
        DISTANCE_BAND_EDGES[1:-1], device=distances.device, dtype=distances.dtype
    )
    # right=True: band k spans [edges[k], edges[k+1]), final band unbounded.
    return torch.bucketize(distances / chord, interior_edges, right=True)


def json_band_edges() -> list[float | None]:
    """The band edges with the unbounded sentinel JSON-safe (inf -> None)."""

    return [edge if math.isfinite(edge) else None for edge in DISTANCE_BAND_EDGES]


def _globe_zscore_metrics(
    case: AirFRANSCase, predictions: dict[str, torch.Tensor]
) -> dict[str, float | None]:
    """One sample's REFERENCE_PERSAMPLE z-score MSE block (published table).

    An exact port of GLOBE's ``airfrans/benchmark/benchmark.py`` metric loop
    (which itself matched the AirFRANS reference *code*): metrics on the
    NONDIMENSIONAL fields as GLOBE outputs them -- ``U/|U_inf|`` (expanded
    to ``_x``/``_y``/``_mag`` variants), ``C_p``, ``C_pt``,
    ``ln(1+nut/nu)`` -- with ``_surface_only`` variants of the pressure
    fields produced by masking off-surface points to NaN BEFORE the
    validity filter.  Per variant: drop points where either side is NaN,
    normalize the error by the SAMPLE's own std of the filtered true values
    (torch-default UNBIASED std, exactly the benchmark's ``.std()``), skip
    the variant when that std is not > 0 (returned as ``None``; the
    aggregation keeps the skipped sample in the denominator, mirroring the
    benchmark's scalar accumulator), and take the point-mean of the squared
    normalized error.  Reconstruction from this driver's outputs:
    ``U/|U_inf| = delta_velocity + U_inf/|U_inf|``; ``C_p`` direct;
    ``ln(1+nut/nu)`` direct; predicted ``C_pt`` engineered as
    ``C_p + |U/|U_inf||^2`` (GLOBE's postprocess identity; the true
    ``C_pt`` is the stored diagnostic).  Float64 accumulation.

    Note the mean-predictor floor: predicting each sample's own true mean
    scores exactly ``(N_valid - 1) / N_valid`` per variant under the
    unbiased std (not exactly 1.0) -- the convention check with teeth.
    """

    delta_true = case.targets["delta_velocity"].double()
    delta_pred = predictions["delta_velocity"].double()
    direction = case.u_inf / torch.linalg.vector_norm(case.u_inf)
    direction = direction.to(device=delta_true.device, dtype=torch.float64)
    u_true = delta_true + direction[None, :]
    u_pred = delta_pred + direction[None, :]
    cp_true = case.targets["pressure_coefficient"].double()
    cp_pred = predictions["pressure_coefficient"].double()

    #: field -> (true, pred, has _surface_only variant); the benchmark's
    #: surface_only_fieldnames are the pressure fields.
    fields: dict[str, tuple[torch.Tensor, torch.Tensor, bool]] = {
        "u": (u_true, u_pred, False),
        "c_p": (cp_true, cp_pred, True),
        "c_pt": (case.cpt.double(), cp_pred + u_pred.square().sum(dim=-1), True),
        "log_nut_ratio": (
            case.targets["log_nut_ratio"].double(),
            predictions["log_nut_ratio"].double(),
            False,
        ),
    }
    nan = torch.tensor(math.nan, dtype=torch.float64, device=delta_true.device)
    record: dict[str, float | None] = {}
    for name, (true, pred, has_surface) in fields.items():
        variants: dict[str, tuple[torch.Tensor, torch.Tensor]] = {"": (true, pred)}
        if has_surface:
            surface = case.is_surface
            if true.ndim == 2:  # pragma: no cover - no surface vector fields
                surface = surface[:, None]
            variants["_surface_only"] = (
                torch.where(surface, true, nan),
                torch.where(surface, pred, nan),
            )
        if true.ndim == 2:
            expanded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for suffix, (true_v, pred_v) in variants.items():
                for axis in range(true.shape[1]):
                    expanded[f"{suffix}_{'xyz'[axis]}"] = (
                        true_v[:, axis],
                        pred_v[:, axis],
                    )
                expanded[f"{suffix}_mag"] = (
                    torch.linalg.vector_norm(true_v, dim=1),
                    torch.linalg.vector_norm(pred_v, dim=1),
                )
            variants = expanded
        for suffix, (true_v, pred_v) in variants.items():
            key = f"globe_zscore_mse/{name}{suffix}"
            valid = ~(torch.isnan(true_v) | torch.isnan(pred_v))
            true_vals, pred_vals = true_v[valid], pred_v[valid]
            if true_vals.numel() < 2:
                record[key] = None  # .std() of < 2 points is NaN: skipped
                continue
            true_std = true_vals.std()  # unbiased, the benchmark's default
            if not float(true_std) > 0:
                record[key] = None
                continue
            error = pred_vals - true_vals
            record[key] = float((error / true_std).square().mean())
    return record


def _case_metrics(
    case: AirFRANSCase,
    predictions: dict[str, torch.Tensor],
    normalization: dict,
    distances: torch.Tensor,
) -> dict:
    """One case's z-score MSE and nondimensional MAE record.

    Alongside the whole-domain metrics, every per-field z-score MSE and the
    delta-velocity nondimensional MAE are decomposed into the wall-distance
    bands of :data:`DISTANCE_BAND_EDGES` (``distances`` is the exact minimum
    distance of each query point to the boundary segments), with per-band
    valid-point counts -- the near-wall diagnostic.  Metric keys with no
    supporting points (e.g. no valid on-surface points, an empty band) are
    ``None`` and skipped by the aggregation.
    """

    std = {field: normalization["fields"][field]["std"] for field in _RAW_FIELDS}
    true_raw = _raw_fields(case, *(case.targets[key] for key in TARGET_FIELDS))
    pred_raw = _raw_fields(case, *(predictions[key] for key in TARGET_FIELDS))
    valid = {field: torch.isfinite(true_raw[field]) for field in _RAW_FIELDS}
    surface = case.is_surface & valid["p"]

    def zscore_mse(field: str, mask: torch.Tensor) -> float | None:
        if not bool(mask.any()):
            return None
        error = (pred_raw[field][mask] - true_raw[field][mask]) / std[field]
        return float(error.square().mean())

    delta_true = case.targets["delta_velocity"]
    delta_valid = torch.isfinite(delta_true).all(dim=-1)
    delta_error = (predictions["delta_velocity"] - delta_true).double()
    cp_true = case.targets["pressure_coefficient"]
    cp_valid = torch.isfinite(cp_true)
    cp_error = (predictions["pressure_coefficient"] - cp_true).double()
    log_nut_true = case.targets["log_nut_ratio"]
    log_nut_valid = torch.isfinite(log_nut_true)
    log_nut_error = (predictions["log_nut_ratio"] - log_nut_true).double()
    cp_surface = cp_valid & case.is_surface

    def mae(error: torch.Tensor, mask: torch.Tensor) -> float | None:
        if not bool(mask.any()):
            return None
        return float(error[mask].abs().mean())

    record = {
        "case": case.name,
        "u_inf": [float(case.u_inf[0]), float(case.u_inf[1])],
        "u_inf_magnitude": case.u_inf_magnitude,
        "n_valid": int(valid["p"].sum()),
        "n_surface_valid": int(surface.sum()),
        "zscore_mse/u_x": zscore_mse("u_x", valid["u_x"]),
        "zscore_mse/u_y": zscore_mse("u_y", valid["u_y"]),
        "zscore_mse/p": zscore_mse("p", valid["p"]),
        "zscore_mse/p_surface": zscore_mse("p", surface),
        "zscore_mse/nut": zscore_mse("nut", valid["nut"]),
        "mae/delta_velocity": mae(
            torch.linalg.vector_norm(delta_error, dim=-1), delta_valid
        ),
        "mae/pressure_coefficient": mae(cp_error, cp_valid),
        "mae/pressure_coefficient_surface": mae(cp_error, cp_surface),
        "mae/log_nut_ratio": mae(log_nut_error, log_nut_valid),
    }

    delta_error_norm = torch.linalg.vector_norm(delta_error, dim=-1)
    band = _band_index(distances, case.chord)
    for index in range(N_DISTANCE_BANDS):
        in_band = band == index
        for field in _RAW_FIELDS:
            record[f"zscore_mse/{field}@band{index}"] = zscore_mse(
                field, valid[field] & in_band
            )
        record[f"mae/delta_velocity@band{index}"] = mae(
            delta_error_norm, delta_valid & in_band
        )
        record[f"n_valid@band{index}"] = int((valid["p"] & in_band).sum())

    # The published-table convention block, side by side with the pooled one.
    record.update(_globe_zscore_metrics(case, predictions))
    return record


_METADATA_KEYS = ("case", "u_inf", "u_inf_magnitude", "n_valid", "n_surface_valid")


@torch.no_grad()
def _evaluate_cases(
    model: nn.Module,
    bank: AirFRANSCaseBank,
    names: list[str],
    normalization: dict,
) -> tuple[dict[str, float], list[dict]]:
    """Aggregate (mean over cases) and per-case metrics; full query sets."""

    model.eval()
    per_case: list[dict] = []
    sums: dict[str, list[float]] = {}
    for name in names:
        case = bank.case(name)
        domain, _ = case_domain(case)
        record = _case_metrics(
            case, _predictions(model, domain), normalization, bank.distances(name)
        )
        per_case.append(record)
        for key, value in record.items():
            if key in _METADATA_KEYS or value is None:
                continue
            sums.setdefault(key, []).append(value)
    aggregate: dict[str, float] = {}
    for key, values in sums.items():
        if key.startswith("n_valid@"):
            aggregate[key] = sum(values)
        elif key.startswith("globe_zscore_mse/"):
            # The benchmark's exact reduction: a scalar accumulator over
            # samples divided by the TOTAL sample count -- a std<=0 sample
            # adds nothing but still counts in the denominator.
            aggregate[key] = sum(values) / len(names)
        else:
            aggregate[key] = sum(values) / len(values)
    return aggregate, per_case


def _headline_score(aggregate: dict[str, float]) -> float:
    """Mean of the four published headline z-score MSEs (model selection)."""

    values = [aggregate[key] for key in HEADLINE_METRICS if key in aggregate]
    if not values:
        raise RuntimeError("no headline metrics available for validation")
    return sum(values) / len(values)


def _catalog_provenance(catalog_dir: Path, manifest: dict) -> dict:
    preprocess_manifest = catalog_dir / "preprocess_manifest.json"
    return {
        "catalog_dir": str(catalog_dir.resolve()),
        "preprocess_manifest_sha256": (
            dataset_catalog.sha256_of_file(preprocess_manifest)
            if preprocess_manifest.is_file()
            else None
        ),
        "manifest_split_sizes": {
            key: len(value)
            for key, value in manifest.items()
            if isinstance(value, list)
        },
    }


# ---------------------------------------------------------------------------
# Training and evaluation drivers
# ---------------------------------------------------------------------------


def resume_path(output_dir: Path | str, arm: str, task: str, seed: int) -> Path:
    """The run's single (overwritten) periodic-resume checkpoint file."""

    return Path(output_dir) / f"{arm}_{task}_seed{seed}_resume.pt"


def run_experiment(
    *,
    catalog_dir: Path | str,
    task: str,
    arm: str,
    epochs: int,
    seed: int,
    device: str,
    output_dir: Path | str,
    queries_per_step: int = 4096,
    batch_cases: int = 1,
    learning_rate: float | None = None,
    weight_decay: float = 1.0e-6,
    schedule: str = "flat",
    validate_every: int = 100,
    checkpoint_every: int | None = None,
    resume: bool = False,
    cache: bool = True,
) -> dict:
    """Train one arm on one AirFRANS task for a fixed epoch count.

    ``schedule`` selects the learning-rate protocol (see
    :func:`resolve_schedule`; ``learning_rate=None`` takes the protocol's
    base).  ``validate_every=N`` runs the validation evaluation every N
    epochs (plus always at the final epoch), printing one JSON line per
    epoch that -- on validation epochs -- carries both conventions'
    headline metrics (``validation_pooled_train`` and
    ``validation_reference_persample``), so multi-hour runs are
    inspectable mid-flight; model selection stays on the pooled headline
    mean.  ``checkpoint_every=N`` overwrites a single resume file
    (:func:`resume_path`: model, optimizer, generators, best-so-far state,
    history) every N epochs so walltime-killed runs lose at most N epochs;
    ``resume=True`` restores it when present (and starts fresh, with a
    printed note, when absent -- relaunching the same command is
    idempotent).  A resumed run continues bit-identically to the
    uninterrupted run: the shuffle and query generators are restored, and
    nothing in an epoch consumes global RNG.
    """

    if task not in TRAIN_TASKS:
        raise ValueError(f"training task must be one of {TRAIN_TASKS}, got {task!r}")
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if batch_cases < 1:
        raise ValueError("batch_cases must be positive")
    if queries_per_step < 1:
        raise ValueError("queries_per_step must be positive")
    if validate_every < 1:
        raise ValueError("validate_every must be a positive integer")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer or None")
    schedule_spec = resolve_schedule(schedule, learning_rate, queries_per_step)

    catalog_dir = Path(catalog_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = airfrans_dataset.load_manifest(catalog_dir)
    train_names = airfrans_dataset.split_case_names(manifest, task, "train")
    test_names = airfrans_dataset.split_case_names(manifest, task, "test")
    validation_names = test_names[:MAX_VALIDATION_CASES]

    torch.manual_seed(seed)
    device_t = torch.device(device)
    model = build_arm(arm).to(device_t)
    parameters = [p for p in model.parameters() if p.requires_grad]
    bank = AirFRANSCaseBank(catalog_dir, device=device_t, cache=cache)
    normalization = compute_normalization(bank, train_names)

    def validation_metrics() -> tuple[float, dict]:
        """Selection scalar (pooled headline mean) plus the full aggregate."""

        aggregate, _ = _evaluate_cases(model, bank, validation_names, normalization)
        return _headline_score(aggregate), aggregate

    def convention_blocks(aggregate: dict) -> dict[str, dict]:
        """Both conventions' headline metrics, for the per-epoch JSON line."""

        return {
            "validation_pooled_train": {
                key.split("/", 1)[1]: aggregate.get(key) for key in HEADLINE_METRICS
            },
            "validation_reference_persample": {
                key.split("/", 1)[1]: aggregate.get(key)
                for key in GLOBE_HEADLINE_METRICS
            },
        }

    history: list[dict[str, float | int]] = []
    best_state, best_val, best_epoch = None, float("inf"), 0
    start_time = time.perf_counter()
    resume_file = resume_path(output_dir, arm, task, seed)
    if parameters and epochs > 0:
        optimizer = torch.optim.AdamW(
            parameters, lr=schedule_spec["peak_lr"], weight_decay=weight_decay
        )
        shuffler = torch.Generator(device="cpu").manual_seed(seed)
        query_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        start_epoch = 1
        if resume and resume_file.is_file():
            payload = torch.load(resume_file, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            shuffler.set_state(payload["shuffler_state"])
            query_generator.set_state(payload["query_generator_state"])
            best_val = payload["best_val"]
            best_epoch = payload["best_epoch"]
            best_state = payload["best_state"]
            history = payload["history"]
            start_epoch = payload["epoch"] + 1
            print(
                json.dumps({"resumed_from": str(resume_file), "epoch": start_epoch}),
                flush=True,
            )
        elif resume:
            print(
                json.dumps({"resume_requested_but_no_file": str(resume_file)}),
                flush=True,
            )
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_lr = schedule_learning_rate(schedule_spec, epoch, epochs)
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
            order = torch.randperm(len(train_names), generator=shuffler).tolist()
            losses: list[float] = []
            for batch_start in range(0, len(order), batch_cases):
                batch = order[batch_start : batch_start + batch_cases]
                optimizer.zero_grad(set_to_none=True)
                for position in batch:
                    case = bank.case(train_names[position])
                    domain, _ = case_domain(
                        case, n_queries=queries_per_step, generator=query_generator
                    )
                    targets = {
                        key: domain.interior.point_data[key] for key in TARGET_FIELDS
                    }
                    loss = masked_huber_loss(_predictions(model, domain), targets)
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach().cpu()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "lr": epoch_lr,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if epoch % validate_every == 0 or epoch == epochs:
                validation, validation_aggregate = validation_metrics()
                record["validation_zscore_mse_mean"] = validation
                record.update(convention_blocks(validation_aggregate))
                if validation < best_val:
                    best_val = validation
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            history.append(record)
            print(json.dumps(record), flush=True)
            if checkpoint_every is not None and epoch % checkpoint_every == 0:
                torch.save(
                    {
                        "arm": arm,
                        "family": FAMILY,
                        "task": task,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs_target": epochs,
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                        "shuffler_state": shuffler.get_state(),
                        "query_generator_state": query_generator.get_state(),
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "best_state": best_state,
                        "history": history,
                        "schedule": schedule_spec,
                    },
                    resume_file,
                )
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_val, _ = validation_metrics()

    aggregate, per_case = _evaluate_cases(model, bank, test_names, normalization)

    checkpoint_name = f"{arm}_{task}_seed{seed}.pt"
    torch.save(
        {
            "arm": arm,
            "family": FAMILY,
            "task": task,
            "seed": seed,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "queries_per_step": queries_per_step,
            "normalization": normalization,
            "schedule": schedule_spec,
            "state_dict": model.state_dict(),
        },
        output_dir / checkpoint_name,
    )

    report = {
        "model": arm,
        "family": FAMILY,
        "task": task,
        "seed": seed,
        "epochs": epochs,
        "batch_cases": batch_cases,
        "queries_per_step": queries_per_step,
        "n_train_cases": len(train_names),
        "n_test_cases": len(test_names),
        "learning_rate": schedule_spec["base_lr"],
        "weight_decay": weight_decay,
        "schedule": schedule_spec,
        "parameters": sum(p.numel() for p in parameters),
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": history,
        "best_validation_zscore_mse_mean": best_val,
        "best_epoch": best_epoch,
        "validation_cases": len(validation_names),
        "normalization": normalization,
        "band_edges": json_band_edges(),
        "splits": {"test": aggregate},
        "per_case": {"test": per_case},
        "checkpoint": checkpoint_name,
        "provenance": _catalog_provenance(catalog_dir, manifest),
        "design_notes": _DESIGN_NOTES,
    }
    (output_dir / f"{arm}_{task}_seed{seed}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def evaluate_checkpoint(
    *,
    checkpoint: Path | str,
    catalog_dir: Path | str,
    eval_task: str,
    device: str,
    output_dir: Path | str,
) -> dict:
    """Evaluate a trained checkpoint on one task's test split (zero-shot).

    The intended use is the pre-registered ``reynolds`` / ``aoa``
    extrapolation readout from full-trained arms.  The z-score constants
    are recomputed from the EVAL task's train split (the constants under
    which that task's published table is stated), not the checkpoint's --
    recorded in the report either way.
    """

    if eval_task not in airfrans_dataset.TASKS:
        raise ValueError(
            f"eval task must be one of {airfrans_dataset.TASKS}, got {eval_task!r}"
        )
    catalog_dir = Path(catalog_dir)
    checkpoint = Path(checkpoint)
    device_t = torch.device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("family") != FAMILY:
        raise ValueError(
            f"checkpoint {checkpoint} holds family {payload.get('family')!r}, "
            f"expected {FAMILY!r}"
        )
    arm = payload["arm"]
    model = build_arm(arm)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device_t)
    model.eval()

    manifest = airfrans_dataset.load_manifest(catalog_dir)
    # Uncached bank: eval touches each case once; the eval-task train split
    # is streamed only for the normalization constants.
    bank = AirFRANSCaseBank(catalog_dir, device=device_t, cache=False)
    normalization = compute_normalization(
        bank, airfrans_dataset.split_case_names(manifest, eval_task, "train")
    )
    test_names = airfrans_dataset.split_case_names(manifest, eval_task, "test")
    start_time = time.perf_counter()
    aggregate, per_case = _evaluate_cases(model, bank, test_names, normalization)

    report = {
        "model": arm,
        "family": FAMILY,
        "task": payload.get("task"),
        "eval_task": eval_task,
        "eval_only": True,
        "seed": payload.get("seed"),
        "epochs": payload.get("epochs"),
        "checkpoint": str(checkpoint.resolve()),
        "n_test_cases": len(test_names),
        "elapsed_seconds": time.perf_counter() - start_time,
        "normalization": normalization,
        "band_edges": json_band_edges(),
        "splits": {eval_task: aggregate},
        "per_case": {eval_task: per_case},
        "provenance": _catalog_provenance(catalog_dir, manifest),
        "design_notes": _DESIGN_NOTES,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{arm}_{payload.get('task')}_seed{payload.get('seed')}_eval_{eval_task}.json"
    )
    (output_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TRAIN_TASKS, default="scarce")
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--queries-per-step", type=int, default=4096)
    parser.add_argument("--batch-cases", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="base learning rate; default is the schedule's (3e-4 flat, 1e-3 globe)",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="flat",
        help="LR protocol: flat (protocol-v0) or globe (protocol-v1 cosine)",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=100,
        help="validation cadence in epochs (always also at the final epoch)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="overwrite a single resume checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="restore the run's resume checkpoint when present",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="evaluation-only mode: path to a trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--eval-task",
        choices=airfrans_dataset.TASKS,
        default=None,
        help="task whose test split to evaluate (reynolds/aoa zero-shot)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.eval_checkpoint is not None:
        report = evaluate_checkpoint(
            checkpoint=args.eval_checkpoint,
            catalog_dir=args.catalog_dir,
            eval_task=args.eval_task
            or torch.load(
                args.eval_checkpoint, map_location="cpu", weights_only=True
            ).get("task"),
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        if args.arm is None or args.epochs is None:
            raise SystemExit("--arm and --epochs are required for training runs")
        report = run_experiment(
            catalog_dir=args.catalog_dir,
            task=args.task,
            arm=args.arm,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            queries_per_step=args.queries_per_step,
            batch_cases=args.batch_cases,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            schedule=args.schedule,
            validate_every=args.validate_every,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            cache=not args.no_cache,
        )
    summary_split = next(iter(report["splits"]))
    print(
        json.dumps(
            {
                "model": report["model"],
                "split": summary_split,
                "headline": {
                    key: report["splits"][summary_split].get(key)
                    for key in HEADLINE_METRICS
                },
            }
        )
    )


if __name__ == "__main__":
    main()
