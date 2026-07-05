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

r"""Strong-inference diagnosis of the nonlinear-mode extrapolation blowup.

EVIDENCE (restated).  The ``zero_preserving_nonlinear`` field mode explodes
under extrapolation on two independent axes:

(a) DRIVE AMPLITUDE (``euler_bernoulli``, iteration 25, archive key
    ``iteration_25_euler_bernoulli_multifield``, runs
    ``scratch/brev/results/phase9/p13_*``): the 200-step smoke showed a
    catastrophic circulation-OOD blowup of relative L2 37-147 in the
    nonlinear arms, settling to ~0.35 (best arm) by 3,000 steps -- an
    early-training transient in degree, but drive-amplitude extrapolation
    remains the worst split at full training (nl 0.74, nl_pseudo 0.42 at
    seed 17).

(b) OPERATOR PARAMETER (``euler_rotational``, iteration 33, archive key
    ``iteration_33_euler_rotational``, runs
    ``scratch/brev/results/p22_euler_rotational``): on the near-eigenvalue
    tier (:math:`\tilde c \in [2.25, 2.38]`, approaching :math:`j_{0,1} =
    2.4048`) the nonlinear arms at FULL training detonate to relative L2
    :math:`10^6`-:math:`10^{13}` on cases the linear arm handles at ~0.2
    (velocity); the pseudo sector damps the worst seed by ~:math:`10^3\times`
    without fixing anything.

Three RIVAL mechanisms are put up for EXCLUSION, not confirmation.  Each
discriminator's decisive outcome is pre-registered here, before the
measurements are made.  N1 and N2/N3 are NOT mutually exclusive across the
two axes -- the amplitude axis may be N1 while the parameter axis is N2/N3 --
so the verdicts are reported PER AXIS.

N1 -- IMPLICIT DRIVE-DEGREE (axis a): the nonlinear read-in's multiplicative
updates compose to a high effective polynomial degree in the drive; small
on-range high-degree components dominate off-range.  DISCRIMINATOR (no
training): on a trained ``euler_bernoulli`` ``mt_singpair_nl`` checkpoint,
evaluate the model at :math:`\alpha\cdot(\text{drive})` for :math:`\alpha
\in [0.1, 4]` at FIXED geometry (the drive is the global pair
:math:`(U, \Gamma)`, scaled jointly) and fit the log-log slope of the RMS
response per output field.  The exact targets are homogeneous of degree
EXACTLY 1 (velocity) and EXACTLY 2 (pressure) in the drive.
PRE-REGISTERED RULE, with ``n_on`` the fitted slope of
:math:`\log \mathrm{rms}` against :math:`\log\alpha` over the on-range band
:math:`\alpha \in [0.5, 1.5]` and ``n_ex`` over the extrapolated band
:math:`\alpha \in [2, 4]`, each the MEAN over the probe cases, per field
with target degree ``d`` (velocity 1, pressure 2):

- N1 SUPPORTED if any field has ``n_ex >= d + 1`` (the response runs at
  least one full polynomial degree above the target off-range);
- N1 EXCLUDED if every field has ``|n_ex - d| <= 0.5`` and
  ``|n_on - d| <= 0.5`` (degree matches the target on- and off-range);
- AMBIGUOUS otherwise.

N2 -- OPERATOR-CONDITIONING EXTRAPOLATION (axis b): the coefficients
conditioned on the operator parameter :math:`\tilde c` extrapolate wildly
beyond the training band -- the operator-stream analog of the far-field
gate/coefficient pathology (iterations 27-29) -- while the drive path is
innocent.  DISCRIMINATOR: on a trained ``euler_rotational``
``mt_singpair_nl`` checkpoint, sweep ``global_data["vorticity_coupling"]``
through the training band into the near-eigenvalue region AT FIXED drive
(one in-band flow's boundary velocity, pinned bitwise) and FIXED geometry
and queries, tracing intermediate state norms and conditioned-coefficient
magnitudes.  PRE-REGISTERED RULE, with, per trace :math:`t(\tilde c)`,
:math:`R(t) = \max_{\tilde c \in [2.25, 2.38]} |t| / \max_{\tilde c \in
[0.5, 1.8]} |t|`, ``R_out`` the larger of the two output-magnitude ratios
(velocity RMS, pressure RMS), everything at the WORST case over the pinned
drives:

- N2 SUPPORTED if ``R_out > 30`` -- the blowup reproduces with the drive
  pinned to an ordinary in-band value, so the divergence lives in the
  :math:`\tilde c`-conditioned path (the localization -- which conditioned
  tensor carries it -- is recorded from the trace ratios);
- N2 EXCLUDED if EVERY trace has ``R <= 3`` -- smooth continuation of the
  whole conditioned stack through the band edge;
- AMBIGUOUS otherwise.

N3 -- PHYSICAL-AMPLIFICATION TRACKING (axis b): near the eigenvalue the TRUE
solution amplifies (the :math:`m=0` interior/boundary ratio is
:math:`1/J_0(\tilde c)`, up to ~77 at :math:`\tilde c = 2.38`); the model's
mechanism for tracking that genuine growth is an unbounded learned
parameterization that overshoots catastrophically off-range -- some of the
blowup is the model chasing real divergence badly.  DISCRIMINATOR: evaluate
the trained model on near-eigenvalue cases RENORMALIZED -- drive scaled by
:math:`1/A`, velocity target by :math:`1/A`, pressure target by
:math:`1/A^2` with :math:`A(\tilde c) = 1/|J_0(\tilde c)|`; the rescaled
triple is again an exact Euler solution (all fields are homogeneous in the
mode coefficients), with ordinary interior amplitude but the SAME
:math:`\tilde c`.  Also fit the shape exponent ``k`` of
:math:`\log(\text{error})` against :math:`\log A` beyond the training band.
PRE-REGISTERED RULE, with ``E_raw`` the mean unrenormalized and ``E_ren``
the mean renormalized combined relative L2 over the near-eigenvalue cases:

- N3 SUPPORTED (amplitude-tracking failure) if ``E_ren <= 1.0`` (ordinary:
  at or below the no-response floor) while ``E_raw >= 1e3`` -- the
  conditioning-at-fixed-amplitude is fine and the failure rides the
  amplitude;
- N3 EXCLUDED if ``E_ren >= 100`` -- the blowup survives renormalization,
  so it is conditioning per se (N2's territory), not amplitude tracking;
- AMBIGUOUS otherwise.

N2 and N3 are distinguished by WHERE the divergence appears: N2 in the
conditioned tensors at fixed ordinary drive, N3 in the output scale chasing
the physical amplification curve.  If N2's traces stay smooth AND the
renormalized error is ordinary AND ``k > 1``, the parameter-axis failure is
the amplitude axis in disguise (the physical amplification feeds the
near-eigenvalue DRIVE, whose norm itself grows like :math:`A`): N3 is the
shape and N1 the suspected engine -- the recorded cross-axis amplitude probe
(the same :math:`\alpha`-sweep run on the rotational checkpoint at FIXED
in-band :math:`\tilde c`, :math:`\alpha` up to 128) is logged as supporting
evidence for exactly that reading.

Everything beyond the three rules (per-:math:`\alpha` relative errors, the
step-200 early-transient probe, the full trace tables, per-:math:`\tilde c`
error/amplification curves, the cross-axis probe) is supporting evidence,
not part of the pre-registered decisions.  NO FIX is attempted here; the fix
rung follows, informed by the verdicts.

Usage (defaults point at the p23 retraining, which added ``--save-checkpoint``
to both drivers -- p13/p22 archived only JSON reports)::

    python studies/nonlinear_fragility.py \
        --eb-checkpoint scratch/brev/results/p23_nonlinear_fragility/\
p23_eb_nl_s17/out/euler_bernoulli_mt_singpair_nl_seed17.pt \
        --er-checkpoint scratch/brev/results/p23_nonlinear_fragility/\
p23_er_nl_s17/out/euler_rotational_mt_singpair_nl_seed17.pt

This is a benchmark-local research diagnostic; it reaches into private model
internals (``_query_operator_input``, kernel caches) deliberately and
verifies every replicated decode path against the public output.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import _paths  # noqa: F401
import torch
from euler_bernoulli import SPLITS as EB_SPLITS
from euler_bernoulli import (
    build_euler_bernoulli_sample,
)
from euler_bernoulli import _build_model as _build_eb_model
from euler_bernoulli import evaluate_splits as evaluate_eb_splits
from euler_rotational import SPLITS as ER_SPLITS
from euler_rotational import (
    FIRST_DIRICHLET_EIGENVALUE,
    bessel_j,
    build_euler_rotational_sample,
)
from euler_rotational import _build_model as _build_er_model
from euler_rotational import evaluate_splits as evaluate_er_splits

from physicsnemo.mesh import DomainMesh, Mesh

EB_ARM = "mt_singpair_nl"
EB_FAMILY = "euler_bernoulli"
ER_ARM = "mt_singpair_nl"
ER_FAMILY = "euler_rotational"

#: The euler_bernoulli global drive is the joint pair (freestream, circulation);
#: both are scaled together by the amplitude factor alpha.
EB_DRIVE_KEYS = ("freestream_velocity", "circulation")

#: Exact homogeneity degree of each output field in the drive (both families).
TARGET_DEGREES = {"velocity": 1, "pressure": 2}

#: Amplitude grid and log-log fit bands for the N1 probe (axis a).
N1_ALPHAS = tuple(
    float(a) for a in torch.logspace(math.log10(0.1), math.log10(4.0), 25)
)
N1_ON_RANGE_BAND = (0.5, 1.5)
N1_EXTRAPOLATED_BAND = (2.0, 4.0)

#: Coupling grid for the fixed-drive N2 sweep: the training band and the gap
#: coarsely, the near-eigenvalue band densely, plus the approach to j_{0,1}
#: itself (supporting only; the pre-registered ratio uses [2.25, 2.38]).
N2_COUPLINGS = tuple(
    float(c) for c in torch.linspace(0.5, 2.2, 18)
) + tuple(float(c) for c in torch.linspace(2.25, 2.402, 12))
N2_TRAIN_BAND = (0.5, 1.8)
N2_NEAR_BAND = (2.25, 2.38)

#: Traces that are functions of (geometry, coupling) ONLY -- the pinned drive
#: never enters them -- versus traces that carry the (conditioned) drive.
N2_CONDITIONED_TRACES = (
    "operator_state_scalar_rms",
    "operator_state_vector_rms",
    "kernel_coefficients_max",
    "kernel_coefficients_rms",
    "query_operator_scalar_rms",
    "query_operator_vector_rms",
    "output_scalar_gate_rms",
    "output_vector_gate_rms",
)
N2_OUTPUT_TRACES = ("output_velocity_rms", "output_pressure_rms")

#: Coupling grid for the N3 error-vs-amplification sweep (full samples).
N3_COUPLINGS = (1.0, 1.4, 1.8, 1.9, 2.0, 2.1, 2.2, 2.25, 2.30, 2.34, 2.38)
N3_NEAR_COUPLINGS = (2.25, 2.30, 2.34, 2.38)
N3_SHAPE_FIT_MIN_COUPLING = 1.9

#: Cross-axis amplitude probe on the rotational checkpoint (supporting only).
CROSS_AXIS_ALPHAS = tuple(
    float(a) for a in torch.logspace(0.0, math.log10(128.0), 15)
)
CROSS_AXIS_BAND = (10.0, 128.0)
CROSS_AXIS_COUPLING = 1.15

#: Pre-registered decision constants (see module docstring).
RULES = {
    "n1_support_degree_excess": 1.0,
    "n1_exclude_degree_tolerance": 0.5,
    "n2_ratio_support": 30.0,
    "n2_ratio_exclude": 3.0,
    "n3_renormalized_ordinary": 1.0,
    "n3_unrenormalized_blowup": 1.0e3,
    "n3_renormalized_exclude": 100.0,
}

N_PROBE_CASES = 4
N_FIXED_DRIVE_CASES = 2


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_arm(
    path: Path,
    device: torch.device,
    *,
    arm: str,
    family: str,
    builder,
) -> tuple[torch.nn.Module, dict]:
    """Rebuild one nonlinear arm and load its trained (or snapshot) state."""

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model") != arm or payload.get("family") != family:
        raise ValueError(
            f"checkpoint {path} holds {payload.get('model')!r}/"
            f"{payload.get('family')!r}, expected {arm!r}/{family!r}"
        )
    model = builder(arm).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, metadata


def load_eb_checkpoint(path: Path, device: torch.device):
    """Load a trained ``euler_bernoulli`` ``mt_singpair_nl`` checkpoint."""

    return _load_arm(
        path, device, arm=EB_ARM, family=EB_FAMILY, builder=_build_eb_model
    )


def load_er_checkpoint(path: Path, device: torch.device):
    """Load a trained ``euler_rotational`` ``mt_singpair_nl`` checkpoint."""

    return _load_arm(
        path, device, arm=ER_ARM, family=ER_FAMILY, builder=_build_er_model
    )


# ---------------------------------------------------------------------------
# Shared numerics
# ---------------------------------------------------------------------------


def _rms(values: torch.Tensor) -> float:
    return float(values.double().square().mean().sqrt())


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction.double() - target.double())
        / torch.linalg.vector_norm(target.double()).clamp_min(1.0e-30)
    )


def _combined_relative_l2(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> float:
    numerator = sum(
        float((predictions[k].double() - targets[k].double()).square().sum())
        for k in targets
    )
    denominator = sum(float(targets[k].double().square().sum()) for k in targets)
    return math.sqrt(numerator / max(denominator, 1.0e-30))


def _verify(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Guard the replicated internals against drift from the public path."""

    scale = expected.double().norm().clamp_min(1e-12)
    residual = (actual.double() - expected.double()).norm() / scale
    if not float(residual) < 1e-3:
        raise RuntimeError(
            f"replicated path diverged from decode(): {label} "
            f"(relative residual {float(residual):.3e})"
        )


def fit_power_law(
    xs: list[float], ys: list[float], band: tuple[float, float]
) -> float | None:
    """Least-squares slope of log y against log x inside one x band.

    The effective-degree estimator: an exactly degree-``d`` homogeneous
    response has slope exactly ``d`` on every band.
    """

    pairs = [
        (math.log(x), math.log(max(abs(y), 1e-30)))
        for x, y in zip(xs, ys, strict=True)
        if band[0] <= x <= band[1] and math.isfinite(y)
    ]
    if len(pairs) < 3:
        return None
    logs_x, logs_y = zip(*pairs)
    n = len(logs_x)
    mean_x, mean_y = sum(logs_x) / n, sum(logs_y) / n
    variance = sum((x - mean_x) ** 2 for x in logs_x)
    if variance <= 0.0:
        return None
    return (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(logs_x, logs_y))
        / variance
    )


def amplification_factor(coupling: float) -> float:
    r"""The exact near-eigenvalue amplification :math:`A = 1/|J_0(\tilde c)|`.

    The :math:`m=0` mode's interior amplitude relative to its boundary trace;
    the :math:`m \ge 1` factors :math:`1/J_m(\tilde c)` stay :math:`O(1)`
    below :math:`j_{1,1} = 3.83`, so this is the family's amplification.
    """

    j0 = float(bessel_j(0, torch.tensor(coupling, dtype=torch.float64)))
    return 1.0 / max(abs(j0), 1e-30)


# ---------------------------------------------------------------------------
# Axis (a): the N1 amplitude probe on euler_bernoulli
# ---------------------------------------------------------------------------


def _scaled_drive_domain(domain: DomainMesh, alpha: float) -> DomainMesh:
    """The same euler_bernoulli problem with the global drive scaled by alpha."""

    global_data = {key: value for key, value in domain.global_data.items()}
    for key in EB_DRIVE_KEYS:
        global_data[key] = global_data[key] * alpha
    return DomainMesh(
        interior=domain.interior,
        boundaries=dict(domain.boundaries.items()),
        global_data=global_data,
    )


@torch.no_grad()
def amplitude_probe_case(
    model: torch.nn.Module,
    seed: int,
    *,
    alphas: tuple[float, ...],
    device: torch.device,
) -> dict:
    """One fixed-geometry euler_bernoulli flow swept over drive amplitude.

    The exact response under the joint scaling ``(U, Gamma) -> alpha (U,
    Gamma)`` is ``alpha * velocity`` and ``alpha**2 * pressure`` (the
    disturbance velocity is jointly linear in the drive and the Bernoulli
    pressure exactly quadratic), so the recorded relative errors compare
    against exact labels at every alpha.
    """

    sample = build_euler_bernoulli_sample(
        seed, device=device, dtype=torch.float32, **EB_SPLITS["in_distribution"]
    )
    rows = []
    for alpha in alphas:
        prediction = model(_scaled_drive_domain(sample.domain, alpha)).point_data
        exact = {
            "velocity": alpha * sample.targets["velocity"],
            "pressure": (alpha**2) * sample.targets["pressure"],
        }
        rows.append(
            {
                "alpha": alpha,
                **{f"{k}_rms": _rms(prediction[k]) for k in TARGET_DEGREES},
                **{f"{k}_exact_rms": _rms(exact[k]) for k in TARGET_DEGREES},
                **{
                    f"{k}_relative_l2": _relative_l2(prediction[k], exact[k])
                    for k in TARGET_DEGREES
                },
            }
        )
    alphas_list = [row["alpha"] for row in rows]
    slopes = {
        field: {
            "on_range": fit_power_law(
                alphas_list, [row[f"{field}_rms"] for row in rows], N1_ON_RANGE_BAND
            ),
            "extrapolated": fit_power_law(
                alphas_list,
                [row[f"{field}_rms"] for row in rows],
                N1_EXTRAPOLATED_BAND,
            ),
        }
        for field in TARGET_DEGREES
    }
    return {"seed": seed, "circulation": sample.circulation, "rows": rows,
            "slopes": slopes}


def mean_amplitude_slopes(cases: list[dict]) -> dict[str, dict[str, float]]:
    """Per-field mean fitted slopes over the probe cases (the N1 statistic)."""

    result: dict[str, dict[str, float]] = {}
    for field, degree in TARGET_DEGREES.items():
        entry: dict[str, float] = {"target_degree": float(degree)}
        for band in ("on_range", "extrapolated"):
            values = [
                case["slopes"][field][band]
                for case in cases
                if case["slopes"][field][band] is not None
            ]
            entry[band] = sum(values) / len(values)
        result[field] = entry
    return result


# ---------------------------------------------------------------------------
# Axis (b): the N2 fixed-drive coupling sweep on euler_rotational
# ---------------------------------------------------------------------------


def _with_coupling(domain: DomainMesh, coupling: float) -> DomainMesh:
    """The same rotational problem with only the operator scalar replaced."""

    global_data = {key: value for key, value in domain.global_data.items()}
    global_data["vorticity_coupling"] = torch.full_like(
        global_data["vorticity_coupling"], coupling
    )
    return DomainMesh(
        interior=domain.interior,
        boundaries=dict(domain.boundaries.items()),
        global_data=global_data,
    )


@torch.no_grad()
def fixed_drive_sweep_case(
    model: torch.nn.Module,
    seed: int,
    *,
    couplings: tuple[float, ...],
    device: torch.device,
) -> dict:
    """Sweep the operator parameter at bitwise-pinned drive/geometry/queries.

    The reference flow (and hence the boundary-velocity drive and the
    queries) is sampled once at the in-band coupling
    ``CROSS_AXIS_COUPLING``; only ``global_data["vorticity_coupling"]`` --
    the model's conditioning input -- changes between sweep points.  Every
    replicated internal is verified against the public forward output.
    """

    sample = build_euler_rotational_sample(
        seed,
        coupling_range=(CROSS_AXIS_COUPLING, CROSS_AXIS_COUPLING),
        modes=ER_SPLITS["in_distribution"]["modes"],
        device=device,
        dtype=torch.float32,
    )
    queries = sample.domain.interior.points
    traces: dict[str, list[float]] = {}

    def record(name: str, value: float) -> None:
        traces.setdefault(name, []).append(value)

    for coupling in couplings:
        domain = _with_coupling(sample.domain, coupling)
        encoded = model.encode(domain)
        decoded = model.decode(encoded, Mesh(points=queries))
        prediction = decoded.point_data

        # Replicated decode with intermediates (the family has no global
        # drive, so the query field state is exactly the kernel message).
        normalized = (queries - encoded.center) / encoded.reference_length
        query_operator = model.operator_input_block(
            model.operator_lift(
                model._query_operator_input(
                    normalized, encoded.global_operator_state
                )
            )
        )
        message = model.kernel_decoder(normalized, encoded.kernel_cache)
        output = model.output_projection(query_operator, message)
        _verify(
            output.vectors[:, 0, :],
            prediction["velocity"],
            "replicated velocity vs forward()",
        )
        _verify(
            output.scalars[:, 0],
            prediction["pressure"],
            "replicated pressure vs forward()",
        )

        projection = model.output_projection
        invariants = projection._geometry_invariants(query_operator)
        cache = encoded.kernel_cache
        record("operator_state_scalar_rms", _rms(encoded.operator_state.scalars))
        record("operator_state_vector_rms", _rms(encoded.operator_state.vectors))
        record("drive_state_scalar_rms", _rms(encoded.drive_state.scalars))
        record("drive_state_vector_rms", _rms(encoded.drive_state.vectors))
        record("kernel_coefficients_max", float(cache.coefficients.abs().max()))
        record("kernel_coefficients_rms", _rms(cache.coefficients))
        record("value_scalar_rms", _rms(cache.value_scalars))
        record("value_vector_rms", _rms(cache.value_vectors))
        record("query_operator_scalar_rms", _rms(query_operator.scalars))
        record("query_operator_vector_rms", _rms(query_operator.vectors))
        if projection.scalar_gate is not None:
            record(
                "output_scalar_gate_rms",
                _rms(2.0 * torch.sigmoid(projection.scalar_gate(invariants))),
            )
        if projection.vector_gate is not None:
            record(
                "output_vector_gate_rms",
                _rms(2.0 * torch.sigmoid(projection.vector_gate(invariants))),
            )
        record("message_scalar_rms", _rms(message.scalars))
        record("message_vector_rms", _rms(message.vectors))
        record("output_velocity_rms", _rms(prediction["velocity"]))
        record("output_pressure_rms", _rms(prediction["pressure"]))
    return {
        "seed": seed,
        "drive_coupling": sample.flow.coupling,
        "couplings": list(couplings),
        "traces": traces,
    }


def fixed_drive_ratios(case: dict) -> dict[str, float]:
    """Per-trace near-band / training-band max-magnitude ratio (N2 statistic)."""

    couplings = case["couplings"]
    ratios = {}
    for name, values in case["traces"].items():
        train = [
            abs(v)
            for c, v in zip(couplings, values, strict=True)
            if N2_TRAIN_BAND[0] <= c <= N2_TRAIN_BAND[1]
        ]
        near = [
            abs(v)
            for c, v in zip(couplings, values, strict=True)
            if N2_NEAR_BAND[0] <= c <= N2_NEAR_BAND[1]
        ]
        ratios[name] = max(near) / max(max(train), 1e-30)
    return ratios


# ---------------------------------------------------------------------------
# Axis (b): the N3 amplification comparison on euler_rotational
# ---------------------------------------------------------------------------


def _renormalized(sample, factor: float):
    """Drive / velocity scaled by ``factor``, pressure by ``factor**2``.

    Exact: every field is homogeneous in the mode coefficients (streamfunction
    and velocity degree 1, pressure degree 2), so the rescaled triple is
    itself an exact rotational Euler solution at the SAME coupling.
    """

    boundary = sample.domain.boundaries["dirichlet"]
    scaled_boundary = Mesh(
        points=boundary.points,
        cells=boundary.cells,
        cell_data={
            "boundary_velocity": boundary.cell_data["boundary_velocity"] * factor
        },
    )
    domain = DomainMesh(
        interior=sample.domain.interior,
        boundaries={"dirichlet": scaled_boundary},
        global_data={k: v for k, v in sample.domain.global_data.items()},
    )
    targets = {
        "velocity": sample.targets["velocity"] * factor,
        "pressure": sample.targets["pressure"] * (factor**2),
    }
    return domain, targets


@torch.no_grad()
def amplification_sweep(
    model: torch.nn.Module,
    *,
    couplings: tuple[float, ...],
    n_cases: int,
    device: torch.device,
) -> list[dict]:
    r"""Model error and exact amplification per pinned coupling (full samples).

    On the near-eigenvalue couplings each case is also evaluated
    RENORMALIZED (drive and targets scaled by the case's exact
    :math:`1/A(\tilde c)` homogeneity factors) -- the N3 discriminator.
    """

    results = []
    for index, coupling in enumerate(couplings):
        entries = []
        for case in range(n_cases):
            sample = build_euler_rotational_sample(
                555_001 + 7_919 * case + 1_000_003 * index,
                coupling_range=(coupling, coupling),
                modes=ER_SPLITS["in_distribution"]["modes"],
                device=device,
                dtype=torch.float32,
            )
            prediction = model(sample.domain).point_data
            predictions = {k: prediction[k] for k in TARGET_DEGREES}
            entry = {
                "combined": _combined_relative_l2(predictions, sample.targets),
                **{
                    k: _relative_l2(predictions[k], sample.targets[k])
                    for k in TARGET_DEGREES
                },
            }
            if coupling in N3_NEAR_COUPLINGS:
                factor = 1.0 / amplification_factor(sample.flow.coupling)
                domain, targets = _renormalized(sample, factor)
                renormalized_prediction = model(domain).point_data
                entry["renormalized_combined"] = _combined_relative_l2(
                    {k: renormalized_prediction[k] for k in TARGET_DEGREES},
                    targets,
                )
            entries.append(entry)
        result = {
            "coupling": coupling,
            "amplification": amplification_factor(coupling),
            "mean_combined": sum(e["combined"] for e in entries) / n_cases,
            **{
                f"mean_{k}": sum(e[k] for e in entries) / n_cases
                for k in TARGET_DEGREES
            },
            "cases": entries,
        }
        if any("renormalized_combined" in e for e in entries):
            values = [e["renormalized_combined"] for e in entries]
            result["mean_renormalized_combined"] = sum(values) / len(values)
        results.append(result)
    return results


def amplification_shape_exponent(sweep: list[dict]) -> float | None:
    """Fitted ``k`` of ``error ~ amplification**k`` beyond the training band."""

    beyond = [e for e in sweep if e["coupling"] >= N3_SHAPE_FIT_MIN_COUPLING]
    return fit_power_law(
        [e["amplification"] for e in beyond],
        [e["mean_combined"] for e in beyond],
        (0.0, float("inf")),
    )


@torch.no_grad()
def cross_axis_amplitude_probe(
    model: torch.nn.Module,
    seed: int,
    *,
    device: torch.device,
) -> dict:
    """SUPPORTING: the drive-amplitude sweep on the ROTATIONAL checkpoint.

    Fixed geometry and fixed IN-BAND coupling; the boundary-velocity drive is
    scaled by alpha up to 128 (covering the ~77x the physical amplification
    injects into near-eigenvalue drives).  Exact response: velocity degree 1,
    pressure degree 2, exactly as on axis (a).  Not verdict-bearing.
    """

    sample = build_euler_rotational_sample(
        seed,
        coupling_range=(CROSS_AXIS_COUPLING, CROSS_AXIS_COUPLING),
        modes=ER_SPLITS["in_distribution"]["modes"],
        device=device,
        dtype=torch.float32,
    )
    rows = []
    for alpha in CROSS_AXIS_ALPHAS:
        domain, targets = _renormalized(sample, alpha)
        prediction = model(domain).point_data
        rows.append(
            {
                "alpha": alpha,
                **{f"{k}_rms": _rms(prediction[k]) for k in TARGET_DEGREES},
                **{
                    f"{k}_relative_l2": _relative_l2(prediction[k], targets[k])
                    for k in TARGET_DEGREES
                },
            }
        )
    alphas = [row["alpha"] for row in rows]
    slopes = {
        field: fit_power_law(
            alphas, [row[f"{field}_rms"] for row in rows], CROSS_AXIS_BAND
        )
        for field in TARGET_DEGREES
    }
    return {"seed": seed, "rows": rows, "extrapolated_slopes": slopes}


# ---------------------------------------------------------------------------
# Verdicts (pre-registered rules; see module docstring)
# ---------------------------------------------------------------------------


def verdict_n1(mean_slopes: dict[str, dict[str, float]]) -> dict:
    numbers = {
        f"{field}_{key}": value
        for field, entry in mean_slopes.items()
        for key, value in entry.items()
    }
    supported = any(
        entry["extrapolated"]
        >= entry["target_degree"] + RULES["n1_support_degree_excess"]
        for entry in mean_slopes.values()
    )
    excluded = all(
        abs(entry["extrapolated"] - entry["target_degree"])
        <= RULES["n1_exclude_degree_tolerance"]
        and abs(entry["on_range"] - entry["target_degree"])
        <= RULES["n1_exclude_degree_tolerance"]
        for entry in mean_slopes.values()
    )
    if supported:
        return {"verdict": "supported", **numbers}
    if excluded:
        return {"verdict": "excluded", **numbers}
    return {"verdict": "ambiguous", **numbers}


def verdict_n2(per_case_ratios: list[dict[str, float]]) -> dict:
    """Worst case over the pinned drives decides (mirror of the M3 rule)."""

    worst_output, worst_output_trace = 0.0, None
    worst_conditioned, worst_conditioned_trace = 0.0, None
    worst_any = 0.0
    for case_index, ratios in enumerate(per_case_ratios):
        for name, ratio in ratios.items():
            if name in N2_OUTPUT_TRACES and ratio > worst_output:
                worst_output = ratio
                worst_output_trace = f"case{case_index}:{name}"
            if name in N2_CONDITIONED_TRACES and ratio > worst_conditioned:
                worst_conditioned = ratio
                worst_conditioned_trace = f"case{case_index}:{name}"
            worst_any = max(worst_any, ratio)
    numbers = {
        "worst_output_ratio": worst_output,
        "worst_output_trace": worst_output_trace,
        "worst_conditioned_ratio": worst_conditioned,
        "worst_conditioned_trace": worst_conditioned_trace,
        "worst_any_ratio": worst_any,
    }
    if worst_output > RULES["n2_ratio_support"]:
        return {"verdict": "supported", **numbers}
    if worst_any <= RULES["n2_ratio_exclude"]:
        return {"verdict": "excluded", **numbers}
    return {"verdict": "ambiguous", **numbers}


def verdict_n3(
    unrenormalized_mean: float,
    renormalized_mean: float,
    shape_exponent: float | None,
) -> dict:
    numbers = {
        "near_eigenvalue_mean_combined": unrenormalized_mean,
        "renormalized_mean_combined": renormalized_mean,
        "shape_exponent_k": shape_exponent,
    }
    if (
        renormalized_mean <= RULES["n3_renormalized_ordinary"]
        and unrenormalized_mean >= RULES["n3_unrenormalized_blowup"]
    ):
        return {"verdict": "supported", **numbers}
    if renormalized_mean >= RULES["n3_renormalized_exclude"]:
        return {"verdict": "excluded", **numbers}
    return {"verdict": "ambiguous", **numbers}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    example_dir = Path(__file__).resolve().parents[1]
    repo_root = example_dir.parents[2]
    p23 = repo_root / "scratch" / "brev" / "results" / "p23_nonlinear_fragility"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eb-checkpoint",
        type=Path,
        default=p23
        / "p23_eb_nl_s17"
        / "out"
        / f"{EB_FAMILY}_{EB_ARM}_seed17.pt",
    )
    parser.add_argument(
        "--eb-snapshot",
        type=Path,
        default=p23
        / "p23_eb_nl_s17"
        / "out"
        / f"{EB_FAMILY}_{EB_ARM}_seed17_step200.pt",
        help="optional raw step-200 state (the iteration-25 blowup regime); "
        "skipped if the file does not exist",
    )
    parser.add_argument(
        "--er-checkpoint",
        type=Path,
        default=p23
        / "p23_er_nl_s17"
        / "out"
        / f"{ER_FAMILY}_{ER_ARM}_seed17.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=example_dir / "results" / f"nonlinear_fragility_{date.today()}.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-cases", type=int, default=4)
    arguments = parser.parse_args()
    device = torch.device(arguments.device)

    # ----- axis (a): euler_bernoulli amplitude ---------------------------
    eb_model, eb_metadata = load_eb_checkpoint(arguments.eb_checkpoint, device)
    eb_reproduction = evaluate_eb_splits(
        eb_model,
        eval_seed=97_000_037,
        n_cases=arguments.eval_cases,
        device=device,
        dtype=torch.float32,
    )
    eb_cases = [
        amplitude_probe_case(
            eb_model, 424_243 + 101 * case, alphas=N1_ALPHAS, device=device
        )
        for case in range(N_PROBE_CASES)
    ]
    eb_mean_slopes = mean_amplitude_slopes(eb_cases)

    snapshot_probe = None
    if arguments.eb_snapshot.is_file():
        snapshot_model, snapshot_metadata = load_eb_checkpoint(
            arguments.eb_snapshot, device
        )
        snapshot_probe = {
            "metadata": snapshot_metadata,
            "cases": [
                amplitude_probe_case(
                    snapshot_model,
                    424_243 + 101 * case,
                    alphas=N1_ALPHAS,
                    device=device,
                )
                for case in range(N_PROBE_CASES)
            ],
        }
        snapshot_probe["mean_slopes"] = mean_amplitude_slopes(
            snapshot_probe["cases"]
        )

    # ----- axis (b): euler_rotational operator parameter ------------------
    er_model, er_metadata = load_er_checkpoint(arguments.er_checkpoint, device)
    er_reproduction = evaluate_er_splits(
        er_model,
        eval_seed=97_000_037,
        n_cases=arguments.eval_cases,
        device=device,
        dtype=torch.float32,
    )
    sweep_cases = [
        fixed_drive_sweep_case(
            er_model, 424_243 + 101 * case, couplings=N2_COUPLINGS, device=device
        )
        for case in range(N_FIXED_DRIVE_CASES)
    ]
    sweep_ratios = [fixed_drive_ratios(case) for case in sweep_cases]

    error_sweep = amplification_sweep(
        er_model, couplings=N3_COUPLINGS, n_cases=N_PROBE_CASES, device=device
    )
    near = [e for e in error_sweep if e["coupling"] in N3_NEAR_COUPLINGS]
    unrenormalized_mean = sum(e["mean_combined"] for e in near) / len(near)
    renormalized_mean = sum(
        e["mean_renormalized_combined"] for e in near
    ) / len(near)
    shape_exponent = amplification_shape_exponent(error_sweep)

    cross_axis = cross_axis_amplitude_probe(er_model, 424_243, device=device)

    verdicts = {
        "axis_a_amplitude": {
            "N1_implicit_drive_degree": verdict_n1(eb_mean_slopes),
        },
        "axis_b_operator_parameter": {
            "N2_operator_conditioning": verdict_n2(sweep_ratios),
            "N3_physical_amplification_tracking": verdict_n3(
                unrenormalized_mean, renormalized_mean, shape_exponent
            ),
        },
    }

    artifact = {
        "generated": str(date.today()),
        "eb_arm": EB_ARM,
        "eb_family": EB_FAMILY,
        "er_arm": ER_ARM,
        "er_family": ER_FAMILY,
        "eb_checkpoint": str(arguments.eb_checkpoint),
        "eb_checkpoint_metadata": eb_metadata,
        "er_checkpoint": str(arguments.er_checkpoint),
        "er_checkpoint_metadata": er_metadata,
        "decision_rules": RULES,
        "n1_bands": {
            "on_range": N1_ON_RANGE_BAND,
            "extrapolated": N1_EXTRAPOLATED_BAND,
        },
        "n2_bands": {"train": N2_TRAIN_BAND, "near_eigenvalue": N2_NEAR_BAND},
        "first_dirichlet_eigenvalue": FIRST_DIRICHLET_EIGENVALUE,
        "eb_checkpoint_split_reproduction": eb_reproduction,
        "er_checkpoint_split_reproduction": er_reproduction,
        "eb_amplitude_cases": eb_cases,
        "eb_mean_slopes": eb_mean_slopes,
        "eb_step200_snapshot_probe": snapshot_probe,
        "er_fixed_drive_sweeps": sweep_cases,
        "er_fixed_drive_ratios": sweep_ratios,
        "er_amplification_sweep": error_sweep,
        "er_cross_axis_amplitude_probe": cross_axis,
        "verdicts": verdicts,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2))

    def show(report: dict, keys: tuple[str, ...]) -> None:
        for name in keys:
            print(f"  {name:36s} {report[name]:.4g}")

    print(f"\neb checkpoint split reproduction ({arguments.eval_cases} cases):")
    show(eb_reproduction, tuple(sorted(eb_reproduction)))
    print(f"\ner checkpoint split reproduction ({arguments.eval_cases} cases):")
    show(er_reproduction, tuple(sorted(er_reproduction)))

    print("\naxis (a) effective drive degree (mean over cases):")
    print(f"  {'field':>10s} {'target':>8s} {'on-range':>10s} {'extrap':>10s}")
    for field, entry in eb_mean_slopes.items():
        print(
            f"  {field:>10s} {entry['target_degree']:8.1f} "
            f"{entry['on_range']:10.2f} {entry['extrapolated']:10.2f}"
        )
    if snapshot_probe is not None:
        print("  step-200 snapshot (supporting):")
        for field, entry in snapshot_probe["mean_slopes"].items():
            print(
                f"  {field:>10s} {entry['target_degree']:8.1f} "
                f"{entry['on_range']:10.2f} {entry['extrapolated']:10.2f}"
            )

    print("\naxis (b) fixed-drive near/train ratios (worst case):")
    for name in sorted(sweep_ratios[0]):
        worst = max(ratios[name] for ratios in sweep_ratios)
        print(f"  {name:36s} {worst:12.4g}")

    print("\naxis (b) error vs amplification:")
    print(
        f"  {'coupling':>9s} {'A':>9s} {'combined':>11s} "
        f"{'velocity':>11s} {'pressure':>11s} {'renorm':>11s}"
    )
    for entry in error_sweep:
        renormalized = entry.get("mean_renormalized_combined")
        renormalized_cell = (
            f"{renormalized:11.4g}" if renormalized is not None else "        n/a"
        )
        print(
            f"  {entry['coupling']:9.3f} {entry['amplification']:9.3f} "
            f"{entry['mean_combined']:11.4g} {entry['mean_velocity']:11.4g} "
            f"{entry['mean_pressure']:11.4g} {renormalized_cell}"
        )
    print(
        "  cross-axis amplitude probe (supporting), extrapolated slopes: "
        + ", ".join(
            f"{field}={value:.2f}" if value is not None else f"{field}=n/a"
            for field, value in cross_axis["extrapolated_slopes"].items()
        )
    )

    print("\nverdicts:")
    for axis, entries in verdicts.items():
        print(f"  {axis}:")
        for name, entry in entries.items():
            detail = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in entry.items()
                if k != "verdict"
            )
            print(f"    {name:38s} {entry['verdict']:12s} {detail}")
    print(f"\nartifact: {arguments.output}")


if __name__ == "__main__":
    main()
