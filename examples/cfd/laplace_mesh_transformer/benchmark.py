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

"""Reproducible microbenchmarks for global mesh attention.

The benchmark constructs deterministic synthetic data and populates input-mesh
geometry caches before entering a timed region.  One invocation emits one JSON
record to stdout, making parameter sweeps straightforward to automate.

Examples
--------
Benchmark reusable attention moments on the default device::

    python benchmark.py --component attention --phase inference

Benchmark forward/backward training cost and check the factorized attention
oracle::

    python benchmark.py --component attention --phase training \
        --n-source 128 --n-query 128 --check
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

import psutil
import torch
from provenance import runtime_environment, source_provenance

from physicsnemo.experimental.nn.mesh_attention import (
    MeshAttention,
    MeshTransformer,
    ScalarVectorState,
)
from physicsnemo.mesh import DomainMesh, Mesh

Component = Literal["attention", "model"]
Phase = Literal["inference", "training"]
FieldMode = Literal["linear", "zero_preserving_nonlinear"]
T = TypeVar("T")


@dataclass(frozen=True)
class Architecture:
    """Small but representative scalar/vector architecture."""

    spatial_dims: int = 2
    operator_scalar_dim: int = 16
    operator_vector_dim: int = 4
    drive_scalar_dim: int = 24
    drive_vector_dim: int = 8
    operator_layers: int = 1
    drive_layers: int = 1
    query_layers: int = 1
    heads: int = 2
    scalar_rank: int = 4
    vector_rank: int = 2


ARCHITECTURE = Architecture()


@dataclass(frozen=True)
class AttentionInputs:
    """Resident tensors supplied to one attention benchmark case."""

    source_mesh: Mesh
    query: ScalarVectorState
    key: ScalarVectorState
    value: ScalarVectorState


@dataclass(frozen=True)
class Timing:
    """Summary of synchronized wall-clock samples."""

    median_ms: float
    minimum_ms: float
    maximum_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class Memory:
    """Incremental memory relative to resident model and input tensors."""

    method: str
    peak_increment_bytes: int
    retained_increment_bytes: int


def _optional_positive_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", choices=("attention", "model"), default="attention"
    )
    parser.add_argument(
        "--phase", choices=("inference", "training"), default="inference"
    )
    parser.add_argument(
        "--mode",
        choices=("linear", "zero_preserving_nonlinear"),
        default="linear",
    )
    parser.add_argument("--device", default="auto", help="Torch device or 'auto'")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--n-source", type=_positive_int, default=128)
    parser.add_argument("--n-query", type=_positive_int, default=128)
    parser.add_argument(
        "--attention-chunk-size", type=_optional_positive_int, default=65536
    )
    parser.add_argument("--query-chunk-size", type=_positive_int, default=65536)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--threads",
        type=_nonnegative_int,
        default=0,
        help="CPU intra-op threads; zero leaves the PyTorch setting unchanged",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check values and gradients against a small oracle",
    )
    args = parser.parse_args()
    if args.n_source < 3:
        parser.error("--n-source must be at least 3 for a closed polygon")
    if args.check and max(args.n_source, args.n_query) > 256:
        parser.error("--check is intentionally limited to at most 256 entities")
    return args


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _circle_geometry(
    n_source: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create CCW vertices, outward-oriented edge cells, and a smooth drive."""
    index = torch.arange(n_source, dtype=dtype)
    theta = 2.0 * torch.pi * index / n_source
    points = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    cells = torch.stack(
        (
            torch.arange(1, n_source + 1, dtype=torch.long) % n_source,
            torch.arange(n_source, dtype=torch.long),
        ),
        dim=-1,
    )
    midpoint_theta = theta + torch.pi / n_source
    drive = torch.cos(midpoint_theta) + 0.25 * torch.sin(3.0 * midpoint_theta)
    return points.to(device), cells.to(device), drive.to(device)


def _query_points(
    n_query: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample deterministic points strictly inside the unit disk."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    uniform = torch.rand(n_query, 2, generator=generator, dtype=dtype)
    radius = 0.95 * torch.sqrt(uniform[:, 0])
    theta = 2.0 * torch.pi * uniform[:, 1]
    points = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), -1)
    return points.to(device)


def _random_tensor(
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=dtype).to(device)


def _build_attention_inputs(
    n_source: int,
    n_query: int,
    mode: FieldMode,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionInputs:
    points, cells, _ = _circle_geometry(n_source, device=device, dtype=dtype)
    source_mesh = Mesh(points=points, cells=cells)
    # Geometry construction is input preparation, not attention execution.
    _ = source_mesh.cell_areas
    _ = source_mesh.cell_centroids
    _ = source_mesh.cell_normals

    architecture = ARCHITECTURE
    query_scalar_dim = architecture.operator_scalar_dim
    query_vector_dim = architecture.operator_vector_dim
    if mode == "zero_preserving_nonlinear":
        query_scalar_dim += architecture.drive_scalar_dim
        query_vector_dim += architecture.drive_vector_dim

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def state(n_entities: int, scalar_dim: int, vector_dim: int) -> ScalarVectorState:
        return ScalarVectorState(
            _random_tensor(
                (n_entities, scalar_dim),
                generator=generator,
                device=device,
                dtype=dtype,
            ),
            _random_tensor(
                (n_entities, vector_dim, architecture.spatial_dims),
                generator=generator,
                device=device,
                dtype=dtype,
            ),
        )

    return AttentionInputs(
        source_mesh=source_mesh,
        query=state(n_query, query_scalar_dim, query_vector_dim),
        key=state(n_source, query_scalar_dim, query_vector_dim),
        value=state(
            n_source,
            architecture.drive_scalar_dim,
            architecture.drive_vector_dim,
        ),
    )


def _build_domain(
    n_source: int,
    n_query: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DomainMesh:
    points, cells, drive = _circle_geometry(n_source, device=device, dtype=dtype)
    boundary = Mesh(
        points=points,
        cells=cells,
        cell_data={"boundary_value": drive},
    )
    # Prewarm every geometry property consumed from the input boundary.
    _ = boundary.cell_areas
    _ = boundary.cell_centroids
    _ = boundary.cell_normals
    interior = Mesh(
        points=_query_points(
            n_query,
            seed=seed + 1,
            device=device,
            dtype=dtype,
        )
    )
    return DomainMesh(
        interior=interior,
        boundaries={"dirichlet": boundary},
        global_data={"reference_length": points.new_tensor(1.0)},
    )


def _build_attention(
    mode: FieldMode,
    *,
    chunk_size: int | None,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MeshAttention:
    architecture = ARCHITECTURE
    query_scalar_dim = architecture.operator_scalar_dim
    query_vector_dim = architecture.operator_vector_dim
    if mode == "zero_preserving_nonlinear":
        query_scalar_dim += architecture.drive_scalar_dim
        query_vector_dim += architecture.drive_vector_dim
    # Modules are initialized on CPU and moved afterward. Seed only the CPU
    # generator so a CPU benchmark never initializes an otherwise idle GPU.
    torch.random.default_generator.manual_seed(seed)
    attention = MeshAttention(
        query_scalar_dim=query_scalar_dim,
        query_vector_dim=query_vector_dim,
        key_scalar_dim=query_scalar_dim,
        key_vector_dim=query_vector_dim,
        value_scalar_dim=architecture.drive_scalar_dim,
        value_vector_dim=architecture.drive_vector_dim,
        out_scalar_dim=architecture.drive_scalar_dim,
        out_vector_dim=architecture.drive_vector_dim,
        heads=architecture.heads,
        scalar_rank=architecture.scalar_rank,
        vector_rank=architecture.vector_rank,
        scalar_value_dim=architecture.drive_scalar_dim // architecture.heads,
        vector_value_dim=architecture.drive_vector_dim // architecture.heads,
        value_scalar_bias=False,
        value_include_vector_invariants=mode == "zero_preserving_nonlinear",
        output_scalar_bias=False,
        accumulation_dtype=dtype,
        entity_chunk_size=chunk_size,
    )
    return attention.to(device=device, dtype=dtype)


def _build_model(
    mode: FieldMode,
    *,
    attention_chunk_size: int | None,
    query_chunk_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MeshTransformer:
    architecture = ARCHITECTURE
    torch.random.default_generator.manual_seed(seed)
    model = MeshTransformer(
        n_spatial_dims=architecture.spatial_dims,
        output_field_ranks={"potential": 0},
        boundary_field_ranks={
            "dirichlet": {
                "operator": {},
                "drive": {"boundary_value": 0},
            }
        },
        global_field_ranks={"operator": {}, "drive": {}},
        reference_length_key="reference_length",
        field_mode=mode,
        operator_scalar_dim=architecture.operator_scalar_dim,
        operator_vector_dim=architecture.operator_vector_dim,
        drive_scalar_dim=architecture.drive_scalar_dim,
        drive_vector_dim=architecture.drive_vector_dim,
        operator_layers=architecture.operator_layers,
        drive_layers=architecture.drive_layers,
        query_layers=architecture.query_layers,
        heads=architecture.heads,
        scalar_rank=architecture.scalar_rank,
        vector_rank=architecture.vector_rank,
        query_chunk_size=query_chunk_size,
        attention_chunk_size=attention_chunk_size,
    )
    return model.to(device=device, dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(samples_ms: list[float]) -> Timing:
    return Timing(
        median_ms=statistics.median(samples_ms),
        minimum_ms=min(samples_ms),
        maximum_ms=max(samples_ms),
        samples_ms=tuple(samples_ms),
    )


def _time_call(
    function: Callable[[], T],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[Timing, T]:
    result: T | None = None
    for _ in range(warmup):
        result = function()
    _synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = function()
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
        else:
            start_ns = time.perf_counter_ns()
            result = function()
            samples_ms.append((time.perf_counter_ns() - start_ns) / 1.0e6)
    if result is None:
        raise RuntimeError("timing did not execute the benchmark function")
    return _summarize(samples_ms), result


def _time_training_step(
    module: torch.nn.Module,
    forward: Callable[[], T],
    loss_function: Callable[[T], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[Timing, Timing]:
    def execute() -> tuple[float, float]:
        module.zero_grad(set_to_none=True)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            middle = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward()
            middle.record()
            loss_function(output).backward()
            end.record()
            end.synchronize()
            return start.elapsed_time(middle), middle.elapsed_time(end)
        start_ns = time.perf_counter_ns()
        output = forward()
        middle_ns = time.perf_counter_ns()
        loss_function(output).backward()
        end_ns = time.perf_counter_ns()
        return (middle_ns - start_ns) / 1.0e6, (end_ns - middle_ns) / 1.0e6

    for _ in range(warmup):
        execute()
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    for _ in range(repeats):
        forward_ms, backward_ms = execute()
        forward_samples.append(forward_ms)
        backward_samples.append(backward_ms)
    return _summarize(forward_samples), _summarize(backward_samples)


def _measure_memory(
    function: Callable[[], T], *, device: torch.device
) -> tuple[Memory, T]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = function()
        _synchronize(device)
        return (
            Memory(
                method="torch_cuda_allocator",
                peak_increment_bytes=max(
                    0, torch.cuda.max_memory_allocated(device) - baseline
                ),
                retained_increment_bytes=max(
                    0, torch.cuda.memory_allocated(device) - baseline
                ),
            ),
            result,
        )

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.0005):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        result = function()
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        sampler.join()
    retained = process.memory_info().rss
    return (
        Memory(
            method="process_rss_sampler",
            peak_increment_bytes=max(0, peak[0] - baseline),
            retained_increment_bytes=max(0, retained - baseline),
        ),
        result,
    )


def _state_loss(state: ScalarVectorState) -> torch.Tensor:
    return state.scalars.square().mean() + state.vectors.square().mean()


def _model_loss(mesh: Mesh) -> torch.Tensor:
    return mesh.point_data["potential"].square().mean()


def _clone_state_for_grad(state: ScalarVectorState) -> ScalarVectorState:
    return ScalarVectorState(
        state.scalars.detach().clone().requires_grad_(),
        state.vectors.detach().clone().requires_grad_(),
    )


def _comparison_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (5.0e-5, 5.0e-6) if dtype == torch.float32 else (2.0e-10, 2.0e-11)


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().detach().cpu())


def _compare_gradients(
    actual: Sequence[torch.Tensor | None],
    expected: Sequence[torch.Tensor | None],
    *,
    rtol: float,
    atol: float,
) -> float:
    maximum = 0.0
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        if actual_gradient is None or expected_gradient is None:
            if actual_gradient is not expected_gradient:
                raise AssertionError("oracle paths use different parameter sets")
            continue
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=rtol, atol=atol
        )
        maximum = max(maximum, _max_abs_difference(actual_gradient, expected_gradient))
    return maximum


def _check_attention(
    attention: MeshAttention,
    inputs: AttentionInputs,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    rtol, atol = _comparison_tolerances(dtype)
    query = _clone_state_for_grad(inputs.query)
    key = _clone_state_for_grad(inputs.key)
    value = _clone_state_for_grad(inputs.value)
    differentiable: list[torch.Tensor] = [
        query.scalars,
        query.vectors,
        key.scalars,
        key.vectors,
        value.scalars,
        value.vectors,
        *attention.parameters(),
    ]

    actual = attention(inputs.source_mesh, query, key, value)
    actual_loss = _state_loss(actual)
    actual_gradients = torch.autograd.grad(
        actual_loss, differentiable, allow_unused=True
    )
    expected = attention.forward_reference(inputs.source_mesh, query, key, value)
    expected_loss = _state_loss(expected)
    expected_gradients = torch.autograd.grad(
        expected_loss, differentiable, allow_unused=True
    )

    torch.testing.assert_close(actual.scalars, expected.scalars, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual.vectors, expected.vectors, rtol=rtol, atol=atol)
    maximum_output_difference = max(
        _max_abs_difference(actual.scalars, expected.scalars),
        _max_abs_difference(actual.vectors, expected.vectors),
    )
    return {
        "oracle": "dense_all_pairs_attention",
        "passed": True,
        "rtol": rtol,
        "atol": atol,
        "maximum_output_absolute_difference": maximum_output_difference,
        "maximum_gradient_absolute_difference": _compare_gradients(
            actual_gradients, expected_gradients, rtol=rtol, atol=atol
        ),
    }


def _check_model(
    model: MeshTransformer,
    domain: DomainMesh,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    rtol, atol = _comparison_tolerances(dtype)
    reference = copy.deepcopy(model)
    reference.query_chunk_size = max(domain.interior.n_points, 1)
    reference.attention_chunk_size = None
    for submodule in reference.modules():
        if isinstance(submodule, MeshAttention):
            submodule.entity_chunk_size = None

    actual = model(domain).point_data["potential"]
    actual_gradients = torch.autograd.grad(
        actual.square().mean(), tuple(model.parameters()), allow_unused=True
    )
    expected = reference(domain).point_data["potential"]
    expected_gradients = torch.autograd.grad(
        expected.square().mean(), tuple(reference.parameters()), allow_unused=True
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {
        "oracle": "unchunked_mesh_transformer",
        "passed": True,
        "rtol": rtol,
        "atol": atol,
        "maximum_output_absolute_difference": _max_abs_difference(actual, expected),
        "maximum_gradient_absolute_difference": _compare_gradients(
            actual_gradients, expected_gradients, rtol=rtol, atol=atol
        ),
    }


def _inference_attention(
    attention: MeshAttention,
    inputs: AttentionInputs,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attention.eval()
    with torch.inference_mode():
        build_timing, moments = _time_call(
            lambda: attention.build_moments(
                inputs.source_mesh, inputs.key, inputs.value
            ),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        evaluate_timing, output = _time_call(
            lambda: attention.evaluate_moments(inputs.query, moments),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        del moments, output
        build_memory, moments = _measure_memory(
            lambda: attention.build_moments(
                inputs.source_mesh, inputs.key, inputs.value
            ),
            device=device,
        )
        evaluate_memory, output = _measure_memory(
            lambda: attention.evaluate_moments(inputs.query, moments),
            device=device,
        )
        del output
    return (
        {
            "build_moments": asdict(build_timing),
            "evaluate_moments": asdict(evaluate_timing),
            "source_entities_per_second": 1000.0
            * inputs.key.n_entities
            / build_timing.median_ms,
            "query_entities_per_second": 1000.0
            * inputs.query.n_entities
            / evaluate_timing.median_ms,
        },
        {
            "build_moments": asdict(build_memory),
            "evaluate_moments": asdict(evaluate_memory),
        },
    )


def _inference_model(
    model: MeshTransformer,
    domain: DomainMesh,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        encode_timing, encoded = _time_call(
            lambda: model.encode(domain),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        decode_timing, output = _time_call(
            lambda: model.decode(encoded),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        del encoded, output
        encode_memory, encoded = _measure_memory(
            lambda: model.encode(domain), device=device
        )
        decode_memory, output = _measure_memory(
            lambda: model.decode(encoded), device=device
        )
        del output
    return (
        {
            "encode": asdict(encode_timing),
            "decode": asdict(decode_timing),
            "source_entities_per_second": 1000.0
            * domain.boundaries["dirichlet"].n_cells
            / encode_timing.median_ms,
            "query_entities_per_second": 1000.0
            * domain.interior.n_points
            / decode_timing.median_ms,
        },
        {"encode": asdict(encode_memory), "decode": asdict(decode_memory)},
    )


def _training_benchmark(
    module: torch.nn.Module,
    forward: Callable[[], T],
    loss_function: Callable[[T], torch.Tensor],
    *,
    n_source: int,
    n_query: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module.train()
    forward_timing, backward_timing = _time_training_step(
        module,
        forward,
        loss_function,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    def step() -> T:
        module.zero_grad(set_to_none=True)
        output = forward()
        loss_function(output).backward()
        return output

    module.zero_grad(set_to_none=True)
    step_memory, output = _measure_memory(step, device=device)
    del output
    total_ms = forward_timing.median_ms + backward_timing.median_ms
    return (
        {
            "scope": "zero_grad_forward_backward_without_optimizer_step",
            "forward": asdict(forward_timing),
            "backward": asdict(backward_timing),
            "total_entities_per_second": 1000.0 * (n_source + n_query) / total_ms,
        },
        {"forward_backward": asdict(step_memory)},
    )


def main() -> None:
    """Run the requested timing or correctness benchmark and emit its report."""

    args = _parse_args()
    device = _resolve_device(args.device)
    dtype = _dtype(args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)

    mode: FieldMode = args.mode
    correctness: dict[str, Any] | None = None
    if args.component == "attention":
        inputs = _build_attention_inputs(
            args.n_source,
            args.n_query,
            mode,
            seed=args.seed + 1,
            device=device,
            dtype=dtype,
        )
        module: MeshAttention | MeshTransformer = _build_attention(
            mode,
            chunk_size=args.attention_chunk_size,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        if args.check:
            correctness = _check_attention(module, inputs, dtype=dtype)
            module.zero_grad(set_to_none=True)
        if args.phase == "inference":
            timings, memory = _inference_attention(
                module,
                inputs,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:
            timings, memory = _training_benchmark(
                module,
                lambda: module(
                    inputs.source_mesh, inputs.query, inputs.key, inputs.value
                ),
                _state_loss,
                n_source=args.n_source,
                n_query=args.n_query,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
    else:
        domain = _build_domain(
            args.n_source,
            args.n_query,
            seed=args.seed + 1,
            device=device,
            dtype=dtype,
        )
        module = _build_model(
            mode,
            attention_chunk_size=args.attention_chunk_size,
            query_chunk_size=args.query_chunk_size,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        if args.check:
            correctness = _check_model(module, domain, dtype=dtype)
            module.zero_grad(set_to_none=True)
        if args.phase == "inference":
            timings, memory = _inference_model(
                module,
                domain,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:
            timings, memory = _training_benchmark(
                module,
                lambda: module(domain),
                _model_loss,
                n_source=args.n_source,
                n_query=args.n_query,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )

    record = {
        "schema_version": 2,
        "component": args.component,
        "phase": args.phase,
        "mode": mode,
        "n_source": args.n_source,
        "n_query": args.n_query,
        "dtype": args.dtype,
        "attention_chunk_size": args.attention_chunk_size,
        "query_chunk_size": args.query_chunk_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "architecture": asdict(ARCHITECTURE),
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "environment": runtime_environment(device),
        "source": source_provenance(),
        "correctness": correctness,
        "timings": timings,
        "memory": memory,
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
