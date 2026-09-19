# Research archive layout

The live book reads the result JSONs and study artifacts at their original
paths. Cleanup does not change those measurements, preregistrations or hash
manifests.

## Shared payloads

The six historical `execution_source/` trees were identical. The canonical
copy is now
`results/historical_k10000_stage_a_archive_canary_v2_job306754_2026-07-28/execution_source/`.
The other five paths are relative symbolic links to it. Identical `.npz`
payloads likewise share one copy through relative links. Opening an original
artifact path still yields the original bytes, including for SHA-256 checks.

Keep these links when working from a complete checkout. When copying an
individual campaign directory elsewhere, dereference them with `cp -aL` or
`rsync -aL` so that the exported directory contains its own payloads.

## Artifacts retained in Git history

Two source checkpoints from the July SHIFT-SUV pilot and the unused
Navier–Stokes hero dataset are no longer in the working tree. Their exact
paths, sizes, Git blob IDs and SHA-256 digests are recorded in
[archived_artifacts.json](archived_artifacts.json). The manifest pins commit
`10b0990100f58ff9004cdcbd3b2d40db4c759b0e`, which is an ancestor of the cleanup
commit; restoring these artifacts does not require an external storage service.

To restore all three, run this from the repository root in a checkout that
contains that commit's history:

```python
import hashlib
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path(
    "examples/cfd/mesh_transformer/archived_artifacts.json"
).read_text())
for artifact in manifest["artifacts"]:
    path = Path(artifact["path"])
    data = subprocess.check_output([
        "git", "show", f"{manifest['git_revision']}:{path}"
    ])
    assert len(data) == artifact["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
```

The checkpoint configurations remain beside their original paths. Reproducing
the retired exact-kernel MeshTransformer also requires its code, retained at
the `mt1-final` tag.

## Historical book material

Superseded chapters, their figure script, irreplaceable figure PNGs and the
gallery dataset are grouped under [book/archive/](book/archive/). They are
excluded from the live book's chapter list. The archive's README describes
their execution context; the live ISLA book does not depend on them.

Generated transfer-program HTML and execution caches are ignored and can be
rebuilt from `research/transfer_program/` with `quarto render` in the project
Python environment.
