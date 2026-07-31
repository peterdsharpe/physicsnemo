"""Executable Phase-0 checks for the public source-measure contract.

This is a no-training diagnostic. It exercises the real public
``MeshTransformer`` path, not a monkeypatched geometric-area surrogate, and
records the exact linear Horvitz--Thompson control used by the reorientation
gate in the lab notebook.

Run from the repository root::

    uv run --no-sync python \
      examples/cfd/mesh_transformer/studies/phase0_measure_path.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from provenance import runtime_environment, source_provenance  # noqa: E402

from physicsnemo.datapipes._indexing import _cyclic_block_indices  # noqa: E402
from physicsnemo.experimental.nn.mesh_attention import MeshTransformer  # noqa: E402
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (  # noqa: E402
    exact_single_layer_member,
)
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import (  # noqa: E402
    MEASURE_WEIGHTS_KEY,
    cell_measure_weights,
)
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral  # noqa: E402


def _model(
    *,
    measure_normalization: bool | None,
    mlp_members: int,
) -> MeshTransformer:
    torch.manual_seed(2707)
    kwargs: dict[str, Any] = {}
    if measure_normalization is not None:
        kwargs["measure_normalization"] = measure_normalization
    return (
        MeshTransformer(
            n_spatial_dims=3,
            output_field_ranks={"u": 0, "v": 1},
            boundary_field_ranks={"surface": {"operator": {}, "drive": {"g": 0}}},
            global_field_ranks={"operator": {}, "drive": {"h": 0}},
            field_mode="linear",
            query_decoder="kernel",
            kernel_mlp_members=mlp_members,
            kernel_include_polynomial_members=False,
            kernel_include_single_layer_member=True,
            operator_scalar_dim=16,
            operator_vector_dim=4,
            drive_scalar_dim=16,
            drive_vector_dim=4,
            operator_layers=1,
            drive_layers=1,
            query_layers=1,
            heads=1,
            scalar_rank=16,
            vector_rank=4,
            **kwargs,
        )
        .eval()
        .to(torch.float64)
    )


def _domain(mesh: Mesh) -> DomainMesh:
    cell_data = {
        "g": torch.linspace(-1.0, 1.0, mesh.n_cells, dtype=torch.float64),
    }
    if MEASURE_WEIGHTS_KEY in mesh.cell_data:
        cell_data[MEASURE_WEIGHTS_KEY] = mesh.cell_data[MEASURE_WEIGHTS_KEY]
    return DomainMesh(
        interior=Mesh(
            points=torch.tensor(
                [[0.0, 0.0, 0.0], [0.3, 0.1, -0.2], [0.5, 0.5, 0.1]],
                dtype=torch.float64,
            )
        ),
        boundaries={"surface": mesh.with_data(cell_data=cell_data)},
        global_data={"h": torch.tensor(1.0, dtype=torch.float64)},
    )


def _with_weights(mesh: Mesh, weights: torch.Tensor) -> Mesh:
    return mesh.with_data(cell_data={MEASURE_WEIGHTS_KEY: weights})


def _relative_max_delta(actual: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (actual - reference).abs().max()
    denominator = reference.abs().max().clamp_min(torch.finfo(reference.dtype).tiny)
    return float(numerator / denominator)


def _public_path_checks(mesh: Mesh) -> dict[str, Any]:
    shape = torch.linspace(0.2, 5.0, mesh.n_cells, dtype=torch.float64)

    default = _model(measure_normalization=None, mlp_members=8)
    disabled = _model(measure_normalization=False, mlp_members=8)
    with torch.no_grad():
        default_output = default(_domain(mesh)).point_data["u"]
        disabled_output = disabled(_domain(mesh)).point_data["u"]
    default_bitwise = torch.equal(default_output, disabled_output)

    weighted_mesh = _with_weights(mesh, shape)
    with torch.no_grad():
        encoded = disabled.encode(_domain(weighted_mesh))
        weighted_output = disabled(_domain(weighted_mesh)).point_data["u"]
    retained = cell_measure_weights(encoded.source_mesh)
    retention_max_abs = float((retained - shape).abs().max())
    response_relative = _relative_max_delta(weighted_output, disabled_output)
    panel_geometry_max_abs = float(
        (encoded.kernel_cache.panel_areas - encoded.source_mesh.cell_areas).abs().max()
    )

    scale_results: dict[str, Any] = {}
    for members in (0, 8):
        arm: dict[str, Any] = {}
        for normalization in (False, True):
            model = _model(
                measure_normalization=normalization,
                mlp_members=members,
            )
            with torch.no_grad():
                values = [
                    model(_domain(_with_weights(mesh, shape * scale))).point_data["u"]
                    for scale in (1.0, 16.0, 880.0)
                ]
            arm["normalized" if normalization else "unnormalized"] = {
                "scale_1_output": values[0].tolist(),
                "scale_16_relative_drift": _relative_max_delta(values[1], values[0]),
                "scale_880_relative_drift": _relative_max_delta(values[2], values[0]),
            }
        scale_results[f"mlp_members_{members}"] = arm

    return {
        "no_weight_default_vs_explicit_false_bitwise": default_bitwise,
        "incoming_weight_retention_max_abs": retention_max_abs,
        "nonuniform_weight_response_relative_max": response_relative,
        "panel_geometry_identity_max_abs": panel_geometry_max_abs,
        "uniform_measure_scale": scale_results,
    }


def _linear_ht_control() -> dict[str, Any]:
    raw = sphere_icosahedral.load(radius=1.0, subdivisions=0)
    mesh = Mesh(points=raw.points.double(), cells=raw.cells)
    query = torch.tensor([[3.0, 0.2, -0.4]], dtype=torch.float64)
    per_panel = exact_single_layer_member(query, mesh.points[mesh.cells]).squeeze(0)
    full_total = per_panel.sum()
    n_cells = mesh.n_cells
    n_kept = 7
    bare = []
    ht = []
    for start in range(n_cells):
        indices = _cyclic_block_indices(n_cells, n_kept, start)
        subtotal = per_panel[indices].sum()
        bare.append(subtotal)
        ht.append(subtotal * (n_cells / n_kept))
    bare_mean = torch.stack(bare).mean()
    ht_mean = torch.stack(ht).mean()
    return {
        "n_cells": n_cells,
        "n_kept": n_kept,
        "n_starts": n_cells,
        "full_total": float(full_total),
        "bare_mean": float(bare_mean),
        "ht_mean": float(ht_mean),
        "bare_to_full_ratio": float(bare_mean / full_total),
        "predicted_bare_ratio": n_kept / n_cells,
        "ht_relative_error": float((ht_mean - full_total).abs() / full_total.abs()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_ROOT / "results" / "phase0_measure_path_2026-07-27.json",
    )
    args = parser.parse_args()

    raw = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    mesh = Mesh(points=raw.points.double(), cells=raw.cells)
    public = _public_path_checks(mesh)
    linear = _linear_ht_control()
    result = {
        "status": "phase0_no_training_diagnostic",
        "date": "2026-07-27",
        "question": (
            "Does the real public model retain incoming effective source "
            "measure while keeping geometry and nuisance scale distinct?"
        ),
        "source_mesh_cells": mesh.n_cells,
        "public_path": public,
        "linear_ht_control": linear,
        "gates": {
            "historical_no_weight_bitwise": public[
                "no_weight_default_vs_explicit_false_bitwise"
            ],
            "public_weights_retained": public["incoming_weight_retention_max_abs"]
            == 0.0,
            "nonuniform_shape_observable": public[
                "nonuniform_weight_response_relative_max"
            ]
            > 0.0,
            "panel_geometry_unchanged": public["panel_geometry_identity_max_abs"]
            == 0.0,
            "normalized_scale_880_drift_below_1e-9": all(
                member["normalized"]["scale_880_relative_drift"] < 1.0e-9
                for member in public["uniform_measure_scale"].values()
            ),
            "linear_ht_relative_error_below_1e-14": linear["ht_relative_error"]
            < 1.0e-14,
        },
        "runtime": runtime_environment(torch.device("cpu")),
        "source_provenance": source_provenance(),
    }
    result["all_gates_pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
