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

r"""Iteration 35 acceptance: the DECLARED-degree fix rung, measured.

Iteration 34 (``studies/nonlinear_fragility.py``,
``results/nonlinear_fragility_2026-07-04.json``) proved both nonlinear-mode
extrapolation failures share one engine: the multiplicative zero-preserving
read-in has IMPLICIT drive degree ~21 (measured mean extrapolated log-log
slope 21.3 for velocity, 20.8 for pressure) against targets of exactly
degree 1 and 2; training suppresses the on-range amplitude of the spurious
high degrees, not their existence, so off-range drive amplification --
direct (euler_bernoulli amplitude axis) or physical (euler_rotational
near-eigenvalue :math:`1/J_0(\tilde c)` amplification feeding the drive) --
detonates.  The fix mirrors how the architecture declares linearity:
``MeshTransformer(field_mode="quadratic")`` DECLARES the drive degree --
drive-linear machinery end to end plus exactly one bilinear typed
composition at the query read-in, so the output is a degree-:math:`\le 2`
polynomial in the drive for any weights.

This study is the acceptance rung on the TRAINED ``mt_singpair_q2`` arms
(both suites, three seeds, 3,000 steps, checkpoints saved; jobs ``p24_*``).

PRE-REGISTERED (restated from the ``field_mode="quadratic"`` docstring and
the launch script, all logged before the q2 training ran):

1. STRUCTURAL DEGREE (the iteration-34 degree probe becomes the acceptance
   test): on trained checkpoints in float64, ``output(alpha * drive)`` at
   fixed geometry must be an exact degree-<=2 polynomial in ``alpha`` --
   per-entry fit residual ``<= 1e-9`` (machine precision for float64
   arithmetic on float32-trained weights) on both suites, INCLUDING the
   extrapolated band where iteration 34 measured slope ~21; the fitted
   log-log amplitude slopes must sit at the target degrees on- and
   off-range.
2. euler_rotational near-eigenvalue: the q2 arm's mean combined relative L2
   falls from the nonlinear arms' :math:`10^6`--:math:`10^{13}` to
   ``< 1.0`` (bar); stretch: the 0.24--0.30 renormalized-ordinary level
   iteration 34 measured (the physics-noise floor of the tier at ordinary
   amplitude).
3. euler_bernoulli circulation-OOD (the amplitude axis): mean combined
   ``< 0.15`` (bar; toward the ID level), below the best nonlinear arm's
   ~0.35.  MEASUREMENT NOTE (logged when the typed addendum launched,
   after the first untyped q2 report): the ~0.35 comparator is the TYPED
   nonlinear arm (``mt_singpair_nl_pseudo``) -- the only arm past the
   iteration-23 parity floor of ~0.647 on this split (the untyped
   circulation velocity :math:`\Gamma x^\perp/(2\pi|x|^2)` is
   parity-unrepresentable, independent of drive degree).  The untyped q2
   arm measures the degree mechanism at the floor; the matched-typing
   arm for rule 3 is ``mt_singpair_q2_pseudo`` (declared degree AND
   declared parity, jobs ``p24_eb_q2p_*``), evaluated when present.
4. In-distribution must not degrade: per family and field, the q2 3-seed
   mean exceeds the corresponding nonlinear arm's archived mean by at most
   ``2 x`` the larger of the two seed standard deviations (comparators:
   euler_rotational iteration 33 -- nl pressure ID 0.181, nl+0o 0.103;
   euler_bernoulli iteration 25).

FALSIFIER: if q2 matches on-range but still detonates off-range, the degree
diagnosis was incomplete -- report honestly.

Usage (defaults point at the pulled p24 artifacts)::

    python studies/declared_degree.py

This is a benchmark-local research diagnostic, not a proposed public API.
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
from euler_bernoulli import _build_model as _build_eb_model
from euler_bernoulli import build_euler_bernoulli_sample
from euler_bernoulli import evaluate_splits as evaluate_eb_splits
from euler_rotational import SPLITS as ER_SPLITS
from euler_rotational import _build_model as _build_er_model
from euler_rotational import build_euler_rotational_sample
from euler_rotational import evaluate_splits as evaluate_er_splits
from nonlinear_fragility import (
    N1_ALPHAS,
    N3_COUPLINGS,
    N3_NEAR_COUPLINGS,
    TARGET_DEGREES,
    _load_arm,
    _renormalized,
    _scaled_drive_domain,
    amplification_sweep,
    amplitude_probe_case,
    cross_axis_amplitude_probe,
    mean_amplitude_slopes,
)

ARM = "mt_singpair_q2"
#: The typed (declared degree + declared parity) euler_bernoulli addendum
#: arm; rule 3's matched-typing subject (see the docstring note).
TYPED_ARM = "mt_singpair_q2_pseudo"
SEEDS = (17, 29, 43)

#: Pre-registered decision constants (module docstring).
RULES = {
    "structural_fit_residual_max": 1.0e-9,
    "slope_degree_tolerance": 0.5,
    "er_near_eigenvalue_bar": 1.0,
    "er_near_eigenvalue_stretch": (0.24, 0.30),
    "eb_circulation_ood_bar": 0.15,
    "eb_circulation_ood_nl_level": 0.35,
    "id_degradation_sd_multiple": 2.0,
}

#: Alphas for the exact float64 polynomial fit (fit pair {1, 2}; probes
#: include the extrapolated band and sign flips).
STRUCTURAL_FIT_ALPHAS = (1.0, 2.0)
STRUCTURAL_PROBE_ALPHAS = (0.37, 3.1, 4.0, 11.0, -1.6)

#: Archived nonlinear comparators (checked-in archive keys).
ARCHIVE = "results/learned_bie_2026-07-02.json"
COMPARATOR_KEYS = {
    "euler_bernoulli": (
        "iteration_25_euler_bernoulli_multifield",
        "runs_3000steps_5seed",
    ),
    "euler_rotational": ("iteration_33_euler_rotational", "runs_3000steps_3seed"),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((v - center) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# The structural degree test (rule 1): exact polynomial fit in float64
# ---------------------------------------------------------------------------


@torch.no_grad()
def _flat_predictions(model, domain) -> torch.Tensor:
    point_data = model(domain).point_data
    return torch.cat(
        (point_data["velocity"].flatten(), point_data["pressure"].flatten())
    )


@torch.no_grad()
def structural_degree_test(model, domains_by_alpha) -> dict:
    """Exact degree-<=2 polynomial fit of the trained model's drive response.

    ``domains_by_alpha`` maps each alpha to the SAME problem with the whole
    drive scaled by alpha (geometry, queries, operator data pinned).  The
    fit pair {1, 2} determines (c1, c2) of ``c1 alpha + c2 alpha^2`` per
    output entry; every probe alpha (including the iteration-34
    extrapolated band and sign flips) must then reproduce the evaluation to
    the pre-registered residual, and alpha=0 must be exactly zero.
    """

    at_one = _flat_predictions(model, domains_by_alpha[1.0])
    at_two = _flat_predictions(model, domains_by_alpha[2.0])
    quadratic = (at_two - 2.0 * at_one) / 2.0
    linear = at_one - quadratic
    residuals = {}
    for alpha, domain in domains_by_alpha.items():
        if alpha in STRUCTURAL_FIT_ALPHAS:
            continue
        actual = _flat_predictions(model, domain)
        if alpha == 0.0:
            residuals["zero_drive_max_output"] = float(actual.abs().max())
            continue
        predicted = linear * alpha + quadratic * alpha**2
        residuals[f"alpha_{alpha:g}"] = float(
            (actual - predicted).abs().max() / actual.abs().max().clamp_min(1e-300)
        )
    worst = max(v for k, v in residuals.items() if k.startswith("alpha_"))
    return {
        "fit_alphas": list(STRUCTURAL_FIT_ALPHAS),
        "relative_residuals": residuals,
        "worst_relative_residual": worst,
        "passes": bool(
            worst <= RULES["structural_fit_residual_max"]
            and residuals["zero_drive_max_output"] == 0.0
        ),
    }


def eb_structural_domains(seed: int, device: torch.device) -> dict:
    """Fixed euler_bernoulli geometry, joint (U, Gamma) scaled per alpha."""

    sample = build_euler_bernoulli_sample(
        seed, device=device, dtype=torch.float64, **EB_SPLITS["in_distribution"]
    )
    alphas = (0.0, *STRUCTURAL_FIT_ALPHAS, *STRUCTURAL_PROBE_ALPHAS)
    return {a: _scaled_drive_domain(sample.domain, a) for a in alphas}


def er_structural_domains(seed: int, device: torch.device) -> dict:
    """Fixed euler_rotational geometry/coupling, boundary drive scaled."""

    sample = build_euler_rotational_sample(
        seed,
        coupling_range=(1.15, 1.15),
        modes=ER_SPLITS["in_distribution"]["modes"],
        device=device,
        dtype=torch.float64,
    )
    alphas = (0.0, *STRUCTURAL_FIT_ALPHAS, *STRUCTURAL_PROBE_ALPHAS)
    return {a: _renormalized(sample, a)[0] for a in alphas}


# ---------------------------------------------------------------------------
# Comparators (rule 4) from the checked-in archive
# ---------------------------------------------------------------------------


def load_comparators(example_dir: Path) -> dict:
    archive = json.loads((example_dir / ARCHIVE).read_text())
    comparators: dict[str, dict] = {}
    for family, (key, runs_key) in COMPARATOR_KEYS.items():
        runs = archive[key][runs_key]
        comparators[family] = {
            arm: {
                metric: {
                    "per_seed": [run[metric] for run in arm_runs],
                    "mean": _mean([run[metric] for run in arm_runs]),
                    "sd": _sd([run[metric] for run in arm_runs]),
                }
                for metric in arm_runs[0]
                if metric != "seed"
            }
            for arm, arm_runs in runs.items()
            if arm.startswith("mt_singpair_nl")
        }
    return comparators


def id_degradation_verdict(
    q2_values: dict[str, list[float]],
    comparators: dict,
    *,
    matched_arm: str | None = None,
) -> dict:
    """Rule 4: q2 ID means within 2x the larger seed sd of the nl arms.

    ``matched_arm`` names the comparator of identical typing (untyped q2
    against untyped nl; typed q2p against nl_pseudo): the parity sector is
    a separate, already-measured mechanism (iteration 23), so only the
    matched comparison isolates what the DEGREE declaration costs.  Every
    cross-arm check is still computed and reported.
    """

    checks = {}
    passes = True
    matched_passes = True
    for metric in ("in_distribution/velocity", "in_distribution/pressure"):
        q2 = q2_values[metric]
        for arm, arm_metrics in comparators.items():
            reference = arm_metrics[metric]
            tolerance = RULES["id_degradation_sd_multiple"] * max(
                _sd(q2), reference["sd"]
            )
            delta = _mean(q2) - reference["mean"]
            ok = delta <= tolerance
            passes = passes and ok
            if arm == matched_arm:
                matched_passes = matched_passes and ok
            checks[f"{metric}_vs_{arm}"] = {
                "q2_mean": _mean(q2),
                "q2_sd": _sd(q2),
                "reference_mean": reference["mean"],
                "reference_sd": reference["sd"],
                "delta": delta,
                "tolerance": tolerance,
                "passes": bool(ok),
                "matched_typing": bool(arm == matched_arm),
            }
    verdict = {"passes": bool(passes), "checks": checks}
    if matched_arm is not None:
        verdict["matched_arm"] = matched_arm
        verdict["matched_typing_passes"] = bool(matched_passes)
    return verdict


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    example_dir = Path(__file__).resolve().parents[1]
    repo_root = example_dir.parents[2]
    p24 = repo_root / "scratch" / "brev" / "results" / "p24_declared_degree"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=p24)
    parser.add_argument(
        "--output",
        type=Path,
        default=example_dir / "results" / f"declared_degree_{date.today()}.json",
    )
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)

    comparators = load_comparators(example_dir)
    families = {
        "euler_bernoulli": ("eb", _build_eb_model, evaluate_eb_splits),
        "euler_rotational": ("er", _build_er_model, evaluate_er_splits),
    }

    results: dict[str, dict] = {}
    for family, (tag, builder, evaluate) in families.items():
        per_seed = []
        for seed in SEEDS:
            run_dir = arguments.runs_root / f"p24_{tag}_q2_s{seed}" / "out"
            report = json.loads(
                (run_dir / f"{family}_{ARM}_seed{seed}.json").read_text()
            )
            model, metadata = _load_arm(
                run_dir / f"{family}_{ARM}_seed{seed}.pt",
                device,
                arm=ARM,
                family=family,
                builder=builder,
            )
            # Reproduce the report's evaluation (guards the checkpoint):
            reproduction = evaluate(
                model,
                eval_seed=97_000_037,
                n_cases=4,
                device=device,
                dtype=torch.float32,
            )
            # Structural degree test on the trained weights, float64.
            model64 = model.double().eval()
            domains = (
                eb_structural_domains(424_243, device)
                if family == "euler_bernoulli"
                else er_structural_domains(424_243, device)
            )
            structural = structural_degree_test(model64, domains)
            entry = {
                "seed": seed,
                "report_splits": report["splits"],
                "checkpoint_metadata": metadata,
                "reproduction_4case": reproduction,
                "structural_degree": structural,
            }
            model_fp32 = model64.float().eval()
            if family == "euler_bernoulli":
                probes = [
                    amplitude_probe_case(
                        model_fp32,
                        424_243 + 101 * case,
                        alphas=N1_ALPHAS,
                        device=device,
                    )
                    for case in range(4)
                ]
                entry["amplitude_probe_mean_slopes"] = mean_amplitude_slopes(probes)
                entry["amplitude_probe_cases"] = probes
            else:
                entry["amplification_sweep"] = amplification_sweep(
                    model_fp32, couplings=N3_COUPLINGS, n_cases=4, device=device
                )
                entry["cross_axis_amplitude_probe"] = cross_axis_amplitude_probe(
                    model_fp32, 424_243, device=device
                )
            per_seed.append(entry)
        results[family] = {"per_seed": per_seed}

    # ----- typed euler_bernoulli addendum (rule 3's matched-typing arm) ---
    typed_root = arguments.runs_root / "p24_eb_q2p_s17" / "out"
    if typed_root.is_dir() and any(typed_root.iterdir()):
        per_seed = []
        for seed in SEEDS:
            run_dir = arguments.runs_root / f"p24_eb_q2p_s{seed}" / "out"
            report = json.loads(
                (run_dir / f"euler_bernoulli_{TYPED_ARM}_seed{seed}.json").read_text()
            )
            model, metadata = _load_arm(
                run_dir / f"euler_bernoulli_{TYPED_ARM}_seed{seed}.pt",
                device,
                arm=TYPED_ARM,
                family="euler_bernoulli",
                builder=_build_eb_model,
            )
            model64 = model.double().eval()
            structural = structural_degree_test(
                model64, eb_structural_domains(424_243, device)
            )
            model_fp32 = model64.float().eval()
            probes = [
                amplitude_probe_case(
                    model_fp32, 424_243 + 101 * case, alphas=N1_ALPHAS, device=device
                )
                for case in range(4)
            ]
            per_seed.append(
                {
                    "seed": seed,
                    "report_splits": report["splits"],
                    "checkpoint_metadata": metadata,
                    "structural_degree": structural,
                    "amplitude_probe_mean_slopes": mean_amplitude_slopes(probes),
                    "amplitude_probe_cases": probes,
                }
            )
        results["euler_bernoulli_typed"] = {"arm": TYPED_ARM, "per_seed": per_seed}

    # ----- verdicts against the pre-registered rules ----------------------
    def collect(family: str, metric: str) -> list[float]:
        return [e["report_splits"][metric] for e in results[family]["per_seed"]]

    structural_pass = all(
        entry["structural_degree"]["passes"]
        for family in families
        for entry in results[family]["per_seed"]
    )
    worst_structural = max(
        entry["structural_degree"]["worst_relative_residual"]
        for family in families
        for entry in results[family]["per_seed"]
    )
    slope_entries = {
        f"seed{e['seed']}": e["amplitude_probe_mean_slopes"]
        for e in results["euler_bernoulli"]["per_seed"]
    }
    slopes_pass = all(
        abs(entry[field][band] - TARGET_DEGREES[field])
        <= RULES["slope_degree_tolerance"]
        for entry in slope_entries.values()
        for field in TARGET_DEGREES
        for band in ("on_range", "extrapolated")
    )

    er_near = collect("euler_rotational", "near_eigenvalue")
    er_near_sweep = [
        entry["mean_combined"]
        for e in results["euler_rotational"]["per_seed"]
        for entry in e["amplification_sweep"]
        if entry["coupling"] in N3_NEAR_COUPLINGS
    ]
    eb_circ = collect("euler_bernoulli", "circulation_ood")
    q2_id = {
        family: {
            metric: collect(family, metric)
            for metric in ("in_distribution/velocity", "in_distribution/pressure")
        }
        for family in families
    }

    stretch_low, stretch_high = RULES["er_near_eigenvalue_stretch"]
    verdicts = {
        "structural_degree": {
            "passes": bool(structural_pass and slopes_pass),
            "worst_fit_residual": worst_structural,
            "residual_rule": RULES["structural_fit_residual_max"],
            "amplitude_slopes": slope_entries,
            "slopes_at_target_degree": bool(slopes_pass),
        },
        "er_near_eigenvalue": {
            "per_seed_report": er_near,
            "mean_report": _mean(er_near),
            "mean_sweep_near_tier": _mean(er_near_sweep),
            "bar": RULES["er_near_eigenvalue_bar"],
            "passes_bar": bool(_mean(er_near) < RULES["er_near_eigenvalue_bar"]),
            "meets_stretch": bool(_mean(er_near) <= stretch_high + 1.0e-12),
            "stretch_band": [stretch_low, stretch_high],
        },
        "eb_circulation_ood": {
            "untyped": {
                "per_seed": eb_circ,
                "mean": _mean(eb_circ),
                "note": (
                    "the untyped arm is capped by the iteration-23 parity "
                    "floor (~0.647 circulation velocity), a mechanism "
                    "orthogonal to drive degree; rule 3's matched-typing "
                    "subject is the typed arm below"
                ),
            },
            "bar": RULES["eb_circulation_ood_bar"],
            "nl_level_typed_comparator": RULES["eb_circulation_ood_nl_level"],
        },
        "id_non_degradation": {
            family: id_degradation_verdict(
                q2_id[family],
                comparators[family],
                matched_arm="mt_singpair_nl",
            )
            for family in families
        },
    }
    if "euler_bernoulli_typed" in results:
        typed_circ = [
            e["report_splits"]["circulation_ood"]
            for e in results["euler_bernoulli_typed"]["per_seed"]
        ]
        verdicts["eb_circulation_ood"]["typed"] = {
            "arm": TYPED_ARM,
            "per_seed": typed_circ,
            "mean": _mean(typed_circ),
            "passes_bar": bool(_mean(typed_circ) < RULES["eb_circulation_ood_bar"]),
            "below_nl_level": bool(
                _mean(typed_circ) < RULES["eb_circulation_ood_nl_level"]
            ),
        }
        typed_structural = all(
            e["structural_degree"]["passes"]
            for e in results["euler_bernoulli_typed"]["per_seed"]
        )
        verdicts["structural_degree"]["typed_arm_passes"] = bool(typed_structural)
        verdicts["id_non_degradation"]["euler_bernoulli_typed"] = (
            id_degradation_verdict(
                {
                    metric: [
                        e["report_splits"][metric]
                        for e in results["euler_bernoulli_typed"]["per_seed"]
                    ]
                    for metric in (
                        "in_distribution/velocity",
                        "in_distribution/pressure",
                    )
                },
                {
                    "mt_singpair_nl_pseudo": comparators["euler_bernoulli"][
                        "mt_singpair_nl_pseudo"
                    ]
                },
                matched_arm="mt_singpair_nl_pseudo",
            )
        )

    artifact = {
        "generated": str(date.today()),
        "arm": ARM,
        "seeds": list(SEEDS),
        "decision_rules": RULES,
        "runs_root": str(arguments.runs_root),
        "comparator_archive_keys": {
            family: keys[0] for family, keys in COMPARATOR_KEYS.items()
        },
        "results": results,
        "verdicts": verdicts,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2))

    print("structural degree:")
    print(
        f"  worst fit residual {worst_structural:.3e} "
        f"(rule <= {RULES['structural_fit_residual_max']:.0e}); "
        f"passes={verdicts['structural_degree']['passes']}"
    )
    for label, slopes in slope_entries.items():
        for field, entry in slopes.items():
            print(
                f"  {label} {field:>9s}: target {entry['target_degree']:.0f}, "
                f"on-range {entry['on_range']:.3f}, "
                f"extrapolated {entry['extrapolated']:.3f}"
            )
    print("\ner near_eigenvalue combined (report, 16 cases):", er_near)
    print(
        f"  mean {_mean(er_near):.4g} "
        f"(bar < {RULES['er_near_eigenvalue_bar']}; stretch "
        f"{stretch_low}-{stretch_high}); "
        f"nl arms at full training: 1e6-1e13"
    )
    print("\neb circulation_ood combined (report, 16 cases):")
    print(
        f"  untyped q2 mean {_mean(eb_circ):.4g} {eb_circ} "
        "(parity-floor-capped; see verdict note)"
    )
    if "typed" in verdicts["eb_circulation_ood"]:
        typed = verdicts["eb_circulation_ood"]["typed"]
        print(
            f"  typed q2p mean {typed['mean']:.4g} {typed['per_seed']} "
            f"(bar < {RULES['eb_circulation_ood_bar']}; "
            f"typed nl level {RULES['eb_circulation_ood_nl_level']})"
        )
    print("\nid non-degradation:")
    for family, verdict in verdicts["id_non_degradation"].items():
        print(
            f"  {family}: all-arm passes={verdict['passes']}, "
            f"matched-typing passes={verdict.get('matched_typing_passes')}"
        )
        for name, check in verdict["checks"].items():
            print(
                f"    {name}: q2 {check['q2_mean']:.4f} vs "
                f"{check['reference_mean']:.4f} "
                f"(delta {check['delta']:+.4f}, tol {check['tolerance']:.4f})"
            )
    print(f"\nartifact: {arguments.output}")


if __name__ == "__main__":
    main()
