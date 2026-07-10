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

"""Decode cost split: exact singular members vs the smooth-member MLP.

Input to the FLARE-direction design note
(``studies/notes/flare_separable_members.md``): how much of the dense
kernel decode is spent in the pairwise smooth-member MLP (the
FLARE-shaped, factorizable part) versus the exact singular panel
integrals (the hierarchical thread's part)?  Measured, not estimated:
identical decoder caches, two timed configurations —

- ``singpair``  : two exact members only (``mlp_members=0``);
- ``members8``  : singpair + 8 MLP members (the H4-verdict best-arm class)

— with the MLP path's cost isolated by differencing.

Timings are median-of-repeats wall clock with device synchronization;
peak memory is reported on CUDA.  Scales default to the AirFRANS step
(S=1,000 segments, Q=4,096) and the DrivAerML step (S=10,000 triangles
subsampled, Q=10,000; run in 2D here — the pair-feature and MLP costs
scale identically and only the exact-member closed forms differ by a
constant factor between 2D/3D, which is noted in the output).

Usage::

    python studies/decode_cost_split.py [--device cuda] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import _paths  # noqa: F401
import torch

from physicsnemo.experimental.nn.mesh_attention.attention import ScalarVectorState
from physicsnemo.experimental.nn.mesh_attention.kernel_decoder import (
    NonlinearZeroKernelBasisCrossDecoder,
)
from physicsnemo.mesh import Mesh


def _circle_boundary(n_cells: int, device: torch.device) -> Mesh:
    angles = 2.0 * torch.pi * torch.arange(n_cells, device=device) / n_cells
    points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    indices = torch.arange(n_cells, device=device)
    cells = torch.stack((torch.roll(indices, -1), indices), dim=-1)
    return Mesh(points=points.float(), cells=cells)


def _decoder(
    mlp_members: int, device: torch.device
) -> NonlinearZeroKernelBasisCrossDecoder:
    torch.manual_seed(11)
    decoder = NonlinearZeroKernelBasisCrossDecoder(
        n_spatial_dims=2,
        operator_scalar_dim=32,
        operator_vector_dim=8,
        drive_scalar_dim=48,
        drive_vector_dim=12,
        heads=1,
        include_polynomial_members=False,
        include_single_layer_member=True,
        mlp_members=mlp_members,
        # Memory hygiene: chunked decode bounds the transient pair tensors
        # to chunk x S.  A 65536 chunk at 10k x 10k materializes ~1e8 pairs
        # x ~14 features at once (>40 GB RAM) and nearly OOMed the dev
        # machine on 2026-07-07 -- never benchmark unchunked on a shared
        # host, and prefer `systemd-run --user -p MemoryMax=8G` wrapping
        # for any local run.
        query_chunk_size=2048,
    )
    return decoder.to(device)


def _state(
    n: int, scalars: int, vectors: int, device: torch.device
) -> ScalarVectorState:
    generator = torch.Generator(device="cpu").manual_seed(7)
    return ScalarVectorState(
        torch.randn(n, scalars, generator=generator).to(device),
        torch.randn(n, vectors, 2, generator=generator).to(device),
    )


def _time(fn, device: torch.device, repeats: int) -> dict:
    fn()  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "peak_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["1000x4096"],
        help=(
            "SxQ pairs. The DrivAerML-scale point (10000x10000) is opt-in: "
            "run it on a cluster node, not a shared dev machine (both paths "
            "are O(Q*S), so the small scale's cost RATIO transfers)."
        ),
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    report: dict = {"device": str(device), "scales": {}}
    for scale in args.scales:
        n_source, n_query = (int(v) for v in scale.split("x"))
        boundary = _circle_boundary(n_source, device)
        generator = torch.Generator(device="cpu").manual_seed(3)
        queries = 0.8 * (torch.rand(n_query, 2, generator=generator).to(device) - 0.5)

        row: dict = {}
        for label, members in (("singpair", 0), ("members8", 8)):
            decoder = _decoder(members, device)
            operator = _state(n_source, 32, 8, device)
            drive = _state(n_source, 48, 12, device)
            with torch.no_grad():
                cache = decoder.build_source_cache(boundary, operator, drive)

                def run(d=decoder, c=cache):
                    d(queries, c)

                row[label] = _time(run, device, args.repeats)
        mlp_ms = row["members8"]["median_ms"] - row["singpair"]["median_ms"]
        row["mlp_path_ms_by_difference"] = mlp_ms
        row["mlp_share_of_members8"] = mlp_ms / row["members8"]["median_ms"]
        report["scales"][scale] = row

    report["note"] = (
        "2D benchmark; the pair-feature+MLP path scales identically in 3D "
        "while the exact members' closed forms differ by a constant factor "
        "(triangle solid angles cost more than segment angles), so the MLP "
        "share reported here is an UPPER bound-ish estimate for 3D."
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
