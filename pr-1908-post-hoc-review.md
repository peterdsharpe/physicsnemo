# Post-hoc review of PR #1908: ShardTensor cross products

- Review target: [NVIDIA/physicsnemo#1908](https://github.com/NVIDIA/physicsnemo/pull/1908)
- Reviewed head: `bc6c81e114671a833c5b151aff5bfabe39170a3a`
- Review date: 2026-08-20
- Independent validator: Cursor Agent with `claude-opus-5-thinking-high`

## Verdict

**Request changes.** The new `torch.cross` / `torch.linalg.cross` ShardTensor
handlers are not safe as written. Five findings were independently confirmed
as defects in the new cross handlers. The sixth behavior is also real, but its
root cause is a pre-existing ShardTensor-wide layout-invariant gap rather than
PR #1908.

None of these findings depends on the `physicsnemo.mesh` package or a
Mesh–ShardTensor integration layer:

- Findings 1 and 3–6 affect any user who reaches `torch.cross`,
  `Tensor.cross`, or `torch.linalg.cross` with the relevant ShardTensor operand
  combination.
- Finding 2 is broader: ShardTensor does not generally validate that two equal
  placement tuples describe the same per-rank shard boundaries. The same
  misaligned tensors silently corrupt `torch.add` and `torch.mul` as well as
  cross products.
- PhysicsNeMo Mesh geometry calls cross products in many places, but the
  in-repository call sites inspected derive both operands from the same
  partitioned tensor or from `zeros_like`. Those aligned patterns were
  forward- and backward-correct in representative probes.

So these are all **general ShardTensor issues in the sense that no Mesh object
is required**. They are not all ShardTensor-wide: five are cross-specific and
one is a general binary-operand layout problem. This review does not establish
a general Mesh regression.

## Findings at a glance

| Priority | Finding | Independent verdict | Scope | Regression on PyTorch 2.12? | Failure mode |
|---|---|---|---|---|---|
| High | Replicated broadcast operand gradients are not reduced over sharded ranks | Confirm | Cross handler | Yes | Silent wrong gradient |
| High | Identical placements can hide different uneven shard boundaries | Partial: behavior confirmed, attribution corrected | ShardTensor-wide | No | Silent wrong values, rank-divergent errors, or collective failure |
| Medium | A fully replicated, same-shape operand is accepted but not aligned to the local shard | Confirm | Cross handler | No | Runtime shape error |
| Medium | A mixed DTensor chosen as the output reference is treated as a ShardTensor | Confirm | Cross handler | Yes | `AttributeError` |
| Medium | `torch.cross(dim=None)` searches the broadcast output for the default dimension | Confirm | Cross handler | Yes | Deterministic runtime error |
| Medium | `torch.compile` loses plain-tensor promotion for mixed operands | Confirm | Cross handler | No | Runtime type error during tracing |

Line references below are against the reviewed head.

## Version-dependent baseline

The independent audit unregistered PR #1908's new handlers in-process to
exercise the pre-PR fallback, then repeated key controls on PyTorch 2.10 and
2.12:

- On PyTorch 2.12, findings 1, 4, and 5 work correctly without the new
  handlers. They are regressions introduced by PR #1908, not merely missing
  features.
- Findings 2, 3, and 6 also fail without the handlers. Finding 2 is a generic
  ShardTensor defect; findings 3 and 6 are pre-existing unsupported paths that
  the new handler admits or should cover.
- On PyTorch 2.10, the repository's minimum supported version, native DTensor
  raises `NotImplementedError` because `aten.linalg_cross.default` has no
  registered sharding strategy. A ShardTensor handler is therefore required.
- On PyTorch 2.12, the native DTensor fallback can make the broadcast case
  correct by returning `Replicate()` output, which introduces an all-gather
  and discards the desired sharding. A correct local handler remains useful
  even where the fallback exists.

The appropriate response is to fix the handlers, not remove the feature.

## Reproduction setup

The distributed reproductions use CPU/Gloo, so they do not require a GPU. Save
the following prefix and one finding's body in `repro.py`, then launch it with
the command below, adjusting the process count as stated for that finding.

```python
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.domain_parallel import ShardTensor


dist.init_process_group("gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
mesh = DeviceMesh("cpu", list(range(world_size)))
```

Use the PR checkout itself when running the probes:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 repro.py
```

### 1. Replicated broadcast operand gets a rank-local gradient

Severity: **High** because training continues with a plausible but incorrect
gradient.

The function handler unwraps both operands with bare `to_local()` calls in
`physicsnemo/domain_parallel/custom_ops/_tensor_ops.py:382-385`. For a plain
operand, promotion first creates a replicated DTensor. Each domain rank then
computes only the gradient contribution from its local shard, and those
contributions are never summed before reaching the original tensor. An
explicitly constructed `Replicate()` ShardTensor operand follows the same
incorrect path; this is not limited to automatic promotion.

Append this body to the common prefix and run with two processes:

```python
full_a = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
)
local_a = full_a.chunk(world_size, dim=0)[rank].clone().requires_grad_(True)
sharded_a = ShardTensor.from_local(
    local_a,
    mesh,
    (Shard(0),),
    sharding_shapes="chunk",
    global_shape=tuple(full_a.shape),
)

axis = torch.tensor([[0.25, -0.5, 2.0]], requires_grad=True)
torch.linalg.cross(sharded_a, axis).full_tensor().sum().backward()

ref_a = full_a.clone().requires_grad_(True)
ref_axis = axis.detach().clone().requires_grad_(True)
torch.linalg.cross(ref_a, ref_axis).sum().backward()

print("observed:", axis.grad, "expected:", ref_axis.grad)
torch.testing.assert_close(axis.grad, ref_axis.grad)
```

Each rank observes `[[2, -4, 2]]`; the correct global gradient is
`[[4, -8, 4]]`.

The failure requires the replicated operand to require gradients. Its error
factor is the number of mesh ranks over which it is replicated: a 2-by-2 mesh
was wrong by a factor of four. The sharded operand's gradient remained correct
in every control. Existing PhysicsNeMo Mesh call sites inspected use constants
or co-derived operands rather than a learnable replicated direction, so this
audit did not find an in-repository training path currently realizing the
silent error. It remains a High-severity library regression for user models.

Proposed fix:

1. Normalize DTensor operands to ShardTensor before local execution so there is
   one metadata and autograd path.
2. When an operand is replicated on a mesh axis and the output reference is
   sharded on that axis, call `to_local()` with `Partial()` as that axis's
   gradient placement.
3. Let the existing ShardTensor autograd bridge resolve the resulting partial
   gradient to replicated form with an all-reduce in backward.

The essential rule is that a replicated value used against distinct domain
shards has a **partial local gradient**, even though its forward placement is
`Replicate()`. The prototype implements this rule and matches the existing
`param_grad_placements` pattern in `shard_utils/linear_patches.py`.

### 2. ShardTensor does not validate aligned uneven shard boundaries

Status: **Partially attributed to PR #1908.** The behavior is High severity,
but its root cause is pre-existing and ShardTensor-wide rather than specific to
the new cross handlers.

`_cross_output_ref()` checks that placement tuples match at
`_tensor_ops.py:321-329`, but it does not compare the per-rank shard shapes.
Two tensors can both report `(Shard(0),)` while assigning different global rows
to a rank.

Append this body and run with exactly two processes:

```python
a_full = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
b_full = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)

a = ShardTensor.from_local(
    (a_full[:0], a_full)[rank],
    mesh,
    (Shard(0),),
    sharding_shapes={0: [(0, 3), (2, 3)]},
)
b = ShardTensor.from_local(
    (b_full[:1], b_full[1:])[rank],
    mesh,
    (Shard(0),),
    sharding_shapes={0: [(1, 3), (1, 3)]},
)

actual = torch.linalg.cross(a, b).full_tensor()
expected = torch.linalg.cross(a_full, b_full)
print("actual:", actual, "expected:", expected)
torch.testing.assert_close(actual, expected)
```

The result is:

```text
actual   = [[0, -1, 0], [1, 0, 0]]
expected = [[0,  0, 1], [1, 0, 0]]
```

On rank 1, the singleton local shard of `b` broadcasts over two local rows of
`a`, even though those rows do not share the same global indices.

The same operands were silently wrong through `torch.add` and `torch.mul` with
the PR handlers removed. Other mismatched partitions produced a rank-divergent
exception, and a valid four-rank case escalated to a Gloo size-mismatch abort
when `full_tensor()` encountered output local sizes inconsistent with the
reference spec. Equal placement objects are therefore not a sufficient binary
operand compatibility check anywhere ShardTensor executes local elementwise
math.

Proposed fix:

- In PR #1908, retain a defensive check for every mesh axis where both operands
  use the same `Shard(dim)` placement. Compare the ordered per-rank extents from
  `spec.sharding_shapes()` and raise a deterministic “gather or reshard first”
  error when they differ.
- Do not silently redistribute one already-sharded operand inside the cross
  patch; that would add an unexpected collective and complicate compile
  behavior.
- File a separate ShardTensor-level issue for enforcing or validating aligned
  shard boundaries across binary operands. A cross-local check does not repair
  `add`, `mul`, or other fallback operations.

Aligned uneven shards remain valid and should have an explicit positive test.
This check is appropriate defense in depth, but the pre-existing generic defect
should not be presented as introduced by PR #1908.

### 3. A full replicated operand is accepted but not locally aligned

Severity: **Medium** because the failure is deterministic on every rank. It is
still actionable because `_cross_output_ref()` explicitly advertises a fully
replicated operand as supported.

The validation explicitly accepts a fully replicated operand at
`_tensor_ops.py:321-325`. It then calls the local kernel directly. With a
sharded `(4, 3)` tensor and replicated `(4, 3)` tensor on two ranks, the kernel
receives `(2, 3)` and `(4, 3)`, which cannot broadcast.

Append this body and run with two processes:

```python
a_full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
b_full = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)

a = ShardTensor.from_local(
    a_full.chunk(world_size)[rank],
    mesh,
    (Shard(0),),
    sharding_shapes="chunk",
    global_shape=tuple(a_full.shape),
)
b = ShardTensor.from_local(b_full, mesh, (Replicate(),))

actual = torch.linalg.cross(a, b)
torch.testing.assert_close(actual.full_tensor(), torch.linalg.cross(a_full, b_full))
```

The local `torch.linalg.cross` call fails on the `2` versus `4` leading
dimension. A singleton replicated operand such as `(1, 3)` happens to work,
which is why this case is easy to miss. The pre-PR fallback also fails this
configuration, so it is a broken newly admitted path rather than a regression.

Proposed fix:

- If a replicated operand has full extent along a dimension sharded by the
  reference, slice it to the reference rank's exact local boundary before the
  local cross product.
- Reuse `redistribute_local_shard_tensor` for the existing Replicate-to-Shard
  local slicing path and pass sizes derived from the reference's
  `sharding_shapes()`. This needs no forward collective because every rank
  already holds the full replicated value.
- Preserve a size-one dimension instead of slicing it so ordinary broadcasting
  still works.
- Combine this with the `Partial()` backward placement from finding 1 so the
  original replicated tensor receives the sum of all rank-local gradients.
- Explicitly reject repeated sharding of the same tensor dimension across
  multiple mesh axes until the redistribution size-hint representation can
  describe that layout unambiguously.

The prototype's slicing and backward composition were independently verified:
the slice backward zero-pads each rank's contribution before the
Partial-to-Replicate all-reduce reconstructs the correct full gradient. The
repeated-same-dimension diagnostic remains unverified because the attempted
layout failed an earlier shape contract on both trees; it needs a reachable
regression test if retained.

### 4. A DTensor output reference lacks ShardTensor metadata

Severity: **Medium** because the failure is deterministic rather than silent,
but mixed ShardTensor/DTensor input is explicitly accepted by the handler.

`_cross_output_ref()` may return either a ShardTensor or a DTensor. The wrapper
then unconditionally calls `ref._spec.sharding_shapes()` at
`_tensor_ops.py:392-396`; `DTensorSpec` has no such method. The dispatch path
makes the same assumption at `_tensor_ops.py:494-503`.

Append this body and run with one process:

```python
axis = ShardTensor.from_local(torch.randn(1, 3), mesh, (Replicate(),))
values = DTensor.from_local(
    torch.randn(4, 3),
    mesh,
    (Shard(0),),
    shape=torch.Size((4, 3)),
    stride=(3, 1),
)

torch.linalg.cross(axis, values)
```

Observed failure:

```text
AttributeError: 'DTensorSpec' object has no attribute 'sharding_shapes'
```

This path works through the pre-PR fallback on PyTorch 2.12, so the failure is
a regression introduced by the new handler.

Proposed fix:

- At function-handler level, convert DTensor operands with
  `ShardTensor.from_dtensor()` so the autograd bridge is retained.
- At dispatch level, use the existing pure DTensor-to-ShardTensor spec/data
  conversion helpers, because native autograd wraps above `__torch_dispatch__`.
- After normalization, let validation and output construction operate only on
  `ShardTensorSpec` rather than branching on two incompatible spec APIs.

The proposed normalization was independently verified in both operand orders,
at function and dispatch levels, and through backward.

### 5. `torch.cross(dim=None)` uses the wrong shape to choose its dimension

Severity: **Medium** because explicit `dim` and `torch.linalg.cross` are not
affected, but the default `torch.cross` behavior is incorrect.

PyTorch selects the first dimension of size three in the **first input**.
`_normalize_cross_dim()` instead scans the broadcast output at
`_tensor_ops.py:256-260`. Broadcasting can introduce an earlier size-three
dimension.

Append this body and run with one process:

```python
a_full = torch.randn(1, 4, 3)
b_full = torch.randn(3, 4, 3)
a = ShardTensor.from_local(a_full, mesh, (Replicate(),))
b = ShardTensor.from_local(b_full, mesh, (Replicate(),))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    expected = torch.cross(a_full, b_full)
    actual = torch.cross(a, b)

torch.testing.assert_close(actual.full_tensor(), expected)
```

Native PyTorch chooses dimension 2. The handler sees the broadcast output
shape `(3, 4, 3)`, chooses dimension 0, and fails because the first input has
length one there.

This disagreement can only fail loudly, not return a silently wrong cross
product. If the first size-three broadcast dimension precedes the native one,
the corresponding first-input extent must be one, so `linalg.cross` rejects
it. The pre-PR fallback is correct on PyTorch 2.12, making this another
regression.

Proposed fix:

- Pass the first input's global shape to `_normalize_cross_dim()`.
- For `dim=None`, scan only that shape.
- Continue returning a negative trailing offset so the selected dimension is
  stable through any permitted leading-dimension broadcasting.

The fix was independently verified against native semantics for multiple
first-input shapes and operand orders.

### 6. Compiled mixed plain/distributed input bypasses promotion

Severity: **Medium** because eager execution succeeds and the compiled failure
is explicit.

Before a registered function handler runs, `_promote_plain_handler_args()`
promotes only exact `torch.Tensor` and `nn.Parameter` types
(`physicsnemo/domain_parallel/shard_tensor.py:323-328`). During Dynamo/AOT
tracing, the plain input is represented by a fake/functional tensor subclass,
so the exact-type check intentionally leaves it alone. The cross handler then
rejects it at `_tensor_ops.py:365-373` because it is not distributed.

Append this body and run with one process:

```python
x = ShardTensor.from_local(
    torch.randn(2, 3),
    mesh,
    (Shard(0),),
    sharding_shapes="chunk",
    global_shape=(2, 3),
)
axis = torch.randn(1, 3)


@torch.compile(fullgraph=True, backend="aot_eager")
def compiled_cross(a, b):
    return torch.linalg.cross(a, b)


actual = compiled_cross(x, axis).full_tensor()
expected = torch.linalg.cross(x.full_tensor(), axis)
torch.testing.assert_close(actual, expected)
```

The same ShardTensor/ShardTensor control works under compile; the mixed plain
operand is the differentiating condition. The pre-PR fallback also fails for a
compiled mixed plain operand, so this is a pre-existing compile gap rather than
a regression.

Proposed fix:

- Keep the generic exact-type promotion rule unchanged; broadening it globally
  risks nesting distributed wrappers around unrelated tensor subclasses.
- In the cross handler only, and only while tracing with promotion enabled,
  recognize a non-distributed `torch.Tensor` subclass operand and wrap it as a
  replicated ShardTensor on the reference mesh.
- Route it through the same local alignment and partial-gradient handling as an
  eagerly promoted operand.

The prototype makes compiled forward and both gradients correct and still
raises when promotion is disabled. Before merging, tighten two details:

- Compare the promotion enum directly with
  `TensorPromotionMode.DISABLED`; do not compare its raw string value.
- Preserve `WARN` behavior and avoid an unexplained eager/compiled divergence
  where eager promotion produces a DTensor while tracing produces a
  ShardTensor. Handling the plain operand natively, following the
  `linear_wrapper`/gradient-reducer pattern, may be cleaner than a second
  promotion path.

## Consolidated patch shape

Five fixes belong directly in the cross patch. Finding 2 also warrants a
cross-local guard, but its root fix belongs at the ShardTensor binary-operation
boundary rather than in PR #1908:

1. Change `_normalize_cross_dim` to take the first input shape.
2. Add one helper that normalizes DTensor operands to ShardTensor, with separate
   autograd-aware function-level and pure dispatch-level conversions.
3. Add one defensive validation helper that compares actual per-rank shard
   boundaries, not only placement objects; track the generic invariant
   separately.
4. Add one local-alignment helper that slices a replicated full tensor to the
   output reference's exact shard while preserving singleton broadcasting.
5. Add one autograd-aware unwrap helper that marks replicated-to-sharded
   operands as `Partial()` in backward.
6. Handle a mixed plain tensor while tracing without weakening the generic
   tensor-subclass promotion guard; preserve promotion-mode semantics.
7. Use the same validation/alignment rules in both the function and dispatch
   implementations.

The forward path remains local for supported layouts. The only new collective
is the mathematically required backward reduction for a replicated operand.
Independent dispatch-level counting confirmed zero forward collectives and one
`_c10d_functional.all_reduce` in backward for the replicated full-size case.

No change is needed in `physicsnemo.mesh`. The fixes should remain at the
ShardTensor operator and invariant layers where the defects occur.

## Required regression coverage

The current tests are insufficient because a one-rank mesh cannot expose a
missing all-reduce or mismatched per-rank boundaries. The patch should add
multi-rank checks covering:

- Sharded plus plain singleton operand, forward and both input gradients.
- Sharded plus explicit Replicate operand, with full-size and singleton shapes,
  in both operand orders.
- Even and uneven partitions; include an empty shard.
- Two aligned uneven ShardTensors as a positive case.
- Two misaligned uneven ShardTensors as a deterministic rejection case.
- The same misaligned tensors through `torch.add` and `torch.mul` in a separate
  ShardTensor-level regression/follow-up, so the generic issue is not mistaken
  for fixed once cross rejects it.
- Mixed ShardTensor/DTensor operands in both orders, including backward.
- `torch.cross(dim=None)` where broadcasting creates an earlier size-three
  output dimension.
- Eager and `torch.compile` forward/backward, with two iterations and an eager
  operation consuming the compiled output.
- A 2-D mesh with distinct tensor dimensions sharded on its mesh axes.
- Explicit rejection of repeated sharding of the same tensor dimension when a
  replicated operand would need local alignment, using a construction that
  actually reaches the proposed guard.

At minimum, run the distributed tests with two ranks. Keep one four-rank 2-D
mesh test because a 1-D mesh cannot reveal mesh-axis mixups.

## Independent and prototype validation

The original audit and an independent Cursor Agent / Claude Opus 5 audit
exercised the reviewed head and the proposed design in separate worktrees. The
prototype passed the following focused checks:

- Existing cross-focused tests in a one-rank static run on both trees: 9
  passed, 9 skipped on each.
- Two-rank eager singleton-operand gradient equivalence.
- Full replicated operands with even and uneven sharding, including forward and
  backward, in both operand orders and at dispatch level.
- Aligned uneven sharded operands and deterministic rejection of misaligned
  uneven operands.
- Mixed DTensor eager forward/backward and compiled forward.
- Compiled full replicated forward/backward on two ranks.
- A 2-by-2 mesh with distinct sharded tensor dimensions, including singleton
  and full replicated operands and their gradients.
- Zero forward collectives and exactly one required backward all-reduce for a
  full replicated operand.
- Ruff lint, Ruff format check, and `git diff --check`.

This validates the direction, not a merge-ready patch. Before merging, the
probes should be converted into repository tests and run through the normal
multi-GPU suite, including the repeated-same-dimension rejection and a real
Inductor backend run. The tracing-promotion semantics described in finding 6
also need tightening.

## Recommendation

Request changes rather than reverting the feature:

1. Treat findings 1, 3, 4, 5, and 6 as PR #1908 work and fix them substantially
   as proposed above. Finding 1 is the primary blocker because it is a silent
   gradient regression on PyTorch 2.12.
2. Keep the cross-specific aligned-layout check as defense in depth, but file
   finding 2 separately as a ShardTensor-wide binary-operand invariant issue.
3. Add true multi-rank forward and gradient coverage. A one-rank mesh cannot
   expose the missing all-reduce or per-rank boundary mismatch.
4. Document why the handler exists: PyTorch 2.10 lacks a DTensor cross
   strategy, while PyTorch 2.12's fallback can preserve correctness by
   all-gathering to a replicated output.

## Non-findings and intentional limitations

- Cross product along a sharded vector dimension is explicitly unsupported;
  rejecting it is correct.
- `out=` is explicitly unsupported; rejecting it is correct.
- The independent audit could not reproduce the earlier exploratory
  compile-plus-`Partial` failure at one or two ranks on either tree. It remains
  excluded from the findings.
- The findings do not establish a general Mesh regression. Finding 2 does
  implicate unrelated ShardTensor binary operations; findings 1 and 3–6 do
  not.
- Roughly fifteen in-repository Mesh geometry call sites use
  `torch.linalg.cross`, but the inspected operands share aligned layouts.
  Representative co-derived and `zeros_like` patterns were forward- and
  backward-correct on the reviewed head.
