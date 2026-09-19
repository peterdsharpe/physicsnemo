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
# ruff: noqa: F401

import os
import shutil
from pathlib import Path
from typing import Callable

import fsspec
import pytest
import torch
import torch.nn as nn

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected
from test.conftest import requires_module


@pytest.fixture(params=["./checkpoints", "msc://checkpoint-test/checkpoints"])
def checkpoint_folder(request) -> str:
    return request.param


@pytest.fixture(params=["physicsnemo", "pytorch"])
def model_generator(request) -> Callable:
    # Create fully-connected NN generator function
    if request.param == "physicsnemo":

        def model(x):
            return FullyConnected(
                in_features=x,
                out_features=x,
                num_layers=2,
                layer_size=8,
            )

    else:

        def model(x):
            return nn.Sequential(
                nn.Linear(x, 8),
                nn.ReLU(),
                nn.Linear(8, x),
            )

    return model


@requires_module(["wandb", "mlflow", "boto3"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_model_checkpointing(
    device,
    model_generator,
    checkpoint_folder,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Test checkpointing util for model"""

    import boto3

    pytest.importorskip("moto")
    from moto import mock_aws

    with mock_aws():
        from physicsnemo.utils import load_checkpoint, save_checkpoint

        # Set up the mock with IAM credentials for access. These should match those in
        # the MSC Config file (./msc_config_checkpoint.yaml).
        os.environ["AWS_ACCESS_KEY_ID"] = "access-key-id"
        # Credentials for testing only
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret-access-key"  # noqa: S105

        # Ensure default region is set to match the MSC Config file.
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        os.environ["MSC_CONFIG"] = f"{current_dir}/msc_config_checkpoint.yaml"

        # Create a bucket using the mock directly to ensure that MSC accesses the correct location.
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="checkpoint-test-bucket")

        # Initialize DistributedManager first since save_checkpoint instantiates it
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        mlp_model_1 = model_generator(8).to(device)
        mlp_model_2 = model_generator(4).to(device)

        input_1 = torch.randn(4, 8).to(device)
        input_2 = torch.randn(4, 4).to(device)

        output_1 = mlp_model_1(input_1)
        output_2 = mlp_model_2(input_2)
        # Save model weights to checkpoint
        save_checkpoint(
            checkpoint_folder,
            models=[mlp_model_1, mlp_model_2],
            metadata={"model_type": "MLP"},
        )

        # Load twin set of models for importing weights
        mlp_model_1 = model_generator(8).to(device)
        mlp_model_2 = model_generator(4).to(device)

        new_output_1 = mlp_model_1(input_1)
        new_output_2 = mlp_model_2(input_2)
        # Assert models are now different
        assert not torch.allclose(output_1, new_output_1, rtol, atol)
        assert not torch.allclose(output_2, new_output_2, rtol, atol)

        # Load model weights from checkpoint
        load_checkpoint(
            checkpoint_folder, models=[mlp_model_1, mlp_model_2], device=device
        )

        loaded_output_1 = mlp_model_1(input_1)
        loaded_output_2 = mlp_model_2(input_2)

        assert torch.allclose(output_1, loaded_output_1, rtol, atol)
        assert torch.allclose(output_2, loaded_output_2, rtol, atol)

        # Also load the model with metadata
        metadata_dict = {}
        epoch = load_checkpoint(
            checkpoint_folder,
            models=[mlp_model_1, mlp_model_2],
            metadata_dict=metadata_dict,
            device=device,
        )

        assert epoch == 0
        assert metadata_dict["model_type"] == "MLP"

        # Clean up if writing to local file system (no need with object storage - files will disappear along with the mock).
        if fsspec.utils.get_protocol(checkpoint_folder) == "file":
            shutil.rmtree(checkpoint_folder)
        else:
            # if writing to object, the local cache must be cleared to allow multiple test runs
            local_cache = os.environ["HOME"] + "/.cache/physicsnemo"
            shutil.rmtree(local_cache)


def test_get_checkpoint_dir():
    from physicsnemo.utils import get_checkpoint_dir

    assert get_checkpoint_dir(".", "model") == "./checkpoints_model"
    assert get_checkpoint_dir("./", "model") == "./checkpoints_model"
    assert (
        get_checkpoint_dir("/Users/auser", "model") == "/Users/auser/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("/Users/auser/", "model") == "/Users/auser/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("msc://test_profile/bucket", "model")
        == "msc://test_profile/bucket/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("msc://test_profile/bucket/", "model")
        == "msc://test_profile/bucket/checkpoints_model"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("model_type", ["physicsnemo", "pytorch"])
def test_load_model_weights(
    tmp_path, device, model_type, rtol: float = 1e-3, atol: float = 1e-3
):
    """load_model_weights restores weights from a single file for non-distributed models."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available in the test environment")

    from physicsnemo.utils import load_model_weights

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    in_feats = 8

    if model_type == "physicsnemo":
        model = FullyConnected(
            in_features=in_feats, out_features=in_feats, num_layers=2, layer_size=8
        ).to(device)
        weights_file = str(tmp_path / "model.mdlus")
        model.save(weights_file)
    else:
        model = nn.Sequential(
            nn.Linear(in_feats, 8), nn.ReLU(), nn.Linear(8, in_feats)
        ).to(device)
        weights_file = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), weights_file)

    x = torch.randn(4, in_feats, device=device)
    with torch.no_grad():
        ref_output = model(x).clone()

    # Build a fresh model with different weights
    if model_type == "physicsnemo":
        model2 = FullyConnected(
            in_features=in_feats, out_features=in_feats, num_layers=2, layer_size=8
        ).to(device)
    else:
        model2 = nn.Sequential(
            nn.Linear(in_feats, 8), nn.ReLU(), nn.Linear(8, in_feats)
        ).to(device)

    with torch.no_grad():
        assert not torch.allclose(ref_output, model2(x), rtol=rtol, atol=atol)

    load_model_weights(model2, weights_file, device=device)

    with torch.no_grad():
        loaded_output = model2(x)
    assert torch.allclose(ref_output, loaded_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("epoch", [None, 3])
@pytest.mark.parametrize("missing_weights", ["deleted", "renamed"])
def test_load_checkpoint_refuses_missing_model_weights(
    tmp_path, model_generator, epoch, missing_weights
):
    """Missing weights cannot restore training state around a fresh model."""
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    model = model_generator(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    save_checkpoint(tmp_path, models=model, optimizer=optimizer, epoch=3)
    weights = next(
        p for p in tmp_path.iterdir() if not p.name.startswith("checkpoint.")
    )
    if missing_weights == "deleted":
        weights.unlink()
    else:
        weights.rename(weights.with_name("FormerName.0.3" + weights.suffix))

    fresh = model_generator(8)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.5)
    metadata = {}
    with pytest.raises(FileNotFoundError, match="uninitialized") as exc:
        load_checkpoint(
            tmp_path,
            models=fresh,
            optimizer=fresh_optimizer,
            epoch=epoch,
            metadata_dict=metadata,
        )
    assert type(fresh).__name__ in str(exc.value)
    assert "checkpoint.0.3.pt" in str(exc.value)
    assert fresh_optimizer.param_groups[0]["lr"] == 0.5
    assert metadata == {}


def test_load_checkpoint_without_training_state(tmp_path, model_generator):
    """Fresh runs, absent epochs, and weights-only exports remain loadable."""
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    fresh = model_generator(8)
    assert load_checkpoint(tmp_path / "nonexistent", models=fresh) == 0
    assert load_checkpoint(tmp_path, models=fresh) == 0

    source = model_generator(8)
    save_checkpoint(tmp_path, models=source, epoch=3)
    assert load_checkpoint(tmp_path, models=fresh, epoch=4) == 0
    (tmp_path / "checkpoint.0.3.pt").unlink()
    assert load_checkpoint(tmp_path, models=fresh) == 0
    for name, value in source.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], value)


@pytest.mark.parametrize("automatic_index", [False, True])
@pytest.mark.parametrize("latest_weights", ["complete", "deleted", "newer"])
def test_load_checkpoint_uses_training_checkpoint_index(
    tmp_path, model_generator, automatic_index, latest_weights
):
    """All weights come from the selected training checkpoint's filename index."""
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    source = model_generator(8)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    for epoch in (1, 2):
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(epoch)
        optimizer.param_groups[0]["lr"] = epoch * 0.01
        save_checkpoint(
            tmp_path,
            models=source,
            optimizer=optimizer,
            epoch=None if automatic_index else epoch,
            metadata={"generation": epoch},
        )
    index = 1 if automatic_index else 2
    if latest_weights == "deleted":
        next(
            p
            for p in tmp_path.glob(f"*.0.{index}.*")
            if not p.name.startswith("checkpoint.")
        ).unlink()
    elif latest_weights == "newer":
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(3)
        # Simulate a newer weights file without a matching training-state file.
        save_checkpoint(tmp_path, models=source)

    fresh = model_generator(8)
    before = {name: value.clone() for name, value in fresh.state_dict().items()}
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.5)
    metadata = {}
    if latest_weights == "deleted":
        with pytest.raises(FileNotFoundError, match=rf"checkpoint\.0\.{index}\.pt"):
            load_checkpoint(
                tmp_path,
                models=fresh,
                optimizer=fresh_optimizer,
                metadata_dict=metadata,
            )
        for name, value in fresh.state_dict().items():
            torch.testing.assert_close(value, before[name])
        assert fresh_optimizer.param_groups[0]["lr"] == 0.5
        assert metadata == {}
    else:
        restored_epoch = load_checkpoint(
            tmp_path, models=fresh, optimizer=fresh_optimizer, metadata_dict=metadata
        )
        assert restored_epoch == (0 if automatic_index else 2)
        for parameter in fresh.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 2))
        assert fresh_optimizer.param_groups[0]["lr"] == 0.02
        assert metadata == {"generation": 2}


@pytest.mark.parametrize("epoch", [None, 3])
def test_load_checkpoint_checks_all_models_before_restoring(
    tmp_path, model_generator, epoch
):
    """A missing second model cannot leave the first model partially restored."""
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    sources = [model_generator(8), model_generator(8)]
    save_checkpoint(tmp_path, models=sources, epoch=3)
    next(tmp_path.glob(f"{type(sources[1]).__name__}1.0.3.*")).unlink()
    fresh = [model_generator(8), model_generator(8)]
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in fresh]
    optimizer = torch.optim.Adam(fresh[0].parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scheduler_before = scheduler.state_dict().copy()
    metadata = {}
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            tmp_path,
            models=fresh,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metadata_dict=metadata,
        )
    for model, original in zip(fresh, before):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert scheduler.state_dict() == scheduler_before
    assert metadata == {}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compiled_model_checkpointing(
    tmp_path, device, rtol: float = 1e-3, atol: float = 1e-3
):
    """Ensure save/load utilities strip torch.compile wrappers correctly."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available in the test environment")

    from physicsnemo.utils import load_checkpoint, save_checkpoint

    # Create and compile a simple model
    in_feats = 4
    base_model = FullyConnected(
        in_features=in_feats,
        out_features=in_feats,
        num_layers=2,
        layer_size=8,
    ).to(device)

    compiled_model = torch.compile(base_model, backend="eager")

    # Prime the compiled model (compilation happens on first run)
    sample_input = torch.randn(2, in_feats, device=device)
    original_output = compiled_model(sample_input).detach().cpu()

    # Save the compiled model; the utility should unwrap the wrapper
    ckpt_dir = tmp_path / "compiled_ckpt"
    save_checkpoint(ckpt_dir.as_posix(), models=[compiled_model])

    # Build a fresh, *uncompiled* model and load the checkpoint
    uncompiled_model = FullyConnected(
        in_features=in_feats,
        out_features=in_feats,
        num_layers=2,
        layer_size=8,
    ).to(device)

    load_checkpoint(ckpt_dir.as_posix(), models=[uncompiled_model], device=device)

    new_output = uncompiled_model(sample_input).detach().cpu()

    assert torch.allclose(original_output, new_output, rtol=rtol, atol=atol)


def test_load_checkpoint_finds_legacy_class_name(tmp_path):
    """A checkpoint written under a class's former name loads into the renamed class
    when the class declares that name, instead of being refused; the loaded weights
    are the saved ones and the epoch is restored."""
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    class RenamedFC(FullyConnected):
        _legacy_class_names = ("FullyConnected",)

    model = FullyConnected(in_features=8, out_features=8, num_layers=2, layer_size=8)
    ckpt_dir = tmp_path / "run"
    save_checkpoint(
        str(ckpt_dir),
        models=model,
        optimizer=torch.optim.Adam(model.parameters()),
        epoch=3,
    )
    assert len(list(ckpt_dir.glob("FullyConnected.0.3.mdlus"))) == 1

    renamed = RenamedFC(in_features=8, out_features=8, num_layers=2, layer_size=8)
    assert load_checkpoint(str(ckpt_dir), models=renamed) == 3
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.equal(renamed(x), model(x))

    # Without the declaration the same directory is still refused (not silently skipped).
    fresh = type("UnrelatedFC", (FullyConnected,), {})(
        in_features=8, out_features=8, num_layers=2, layer_size=8
    )
    with pytest.raises(FileNotFoundError, match="uninitialized"):
        load_checkpoint(str(ckpt_dir), models=fresh)


def test_isla_declares_its_legacy_name(tmp_path):
    """ISLA checkpoints written before the rename (MeshTransformer2.*.mdlus) load into ISLA."""
    from physicsnemo.experimental.nn.isla import ISLA
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    assert ISLA._legacy_class_names == ("MeshTransformer2",)

    torch.manual_seed(0)
    model = ISLA(hidden=16, n_layers=1, n_slices=4)
    ckpt_dir = tmp_path / "isla"
    save_checkpoint(str(ckpt_dir), models=model, epoch=1)
    (ckpt_dir / "ISLA.0.1.mdlus").rename(ckpt_dir / "MeshTransformer2.0.1.mdlus")

    torch.manual_seed(1)
    fresh = ISLA(hidden=16, n_layers=1, n_slices=4)
    assert load_checkpoint(str(ckpt_dir), models=fresh) == 1
    pts = torch.randn(1, 20, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 20, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 20) + 0.5
    with torch.no_grad():
        assert torch.allclose(
            fresh(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
            model(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
            atol=1e-6,
        )


@pytest.mark.parametrize("older_name", ["ISLA", "MeshTransformer2"])
@pytest.mark.parametrize("missing_latest", [False, True])
def test_isla_legacy_checkpoint_uses_training_index(
    tmp_path, older_name, missing_latest
):
    """Legacy names obey the selected epoch even with stale weights present."""
    from physicsnemo.experimental.nn.isla import ISLA
    from physicsnemo.utils import load_checkpoint, save_checkpoint

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    model = ISLA(hidden=16, n_layers=1, n_slices=4)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in (1, 2):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epoch)
        optimizer.param_groups[0]["lr"] = 0.01 * epoch
        save_checkpoint(tmp_path, models=model, optimizer=optimizer, epoch=epoch)
        saved_name = older_name if epoch == 1 else "MeshTransformer2"
        if saved_name != "ISLA":
            (tmp_path / f"ISLA.0.{epoch}.mdlus").rename(
                tmp_path / f"{saved_name}.0.{epoch}.mdlus"
            )
    if missing_latest:
        (tmp_path / "MeshTransformer2.0.2.mdlus").unlink()

    fresh = ISLA(hidden=16, n_layers=1, n_slices=4)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.5)
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    if missing_latest:
        with pytest.raises(FileNotFoundError, match=r"checkpoint\.0\.2\.pt"):
            load_checkpoint(tmp_path, models=fresh, optimizer=fresh_optimizer)
        for name, value in fresh.state_dict().items():
            torch.testing.assert_close(value, before[name])
        assert fresh_optimizer.param_groups[0]["lr"] == 0.5
    else:
        assert load_checkpoint(tmp_path, models=fresh, optimizer=fresh_optimizer) == 2
        for name, value in fresh.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name])
        assert fresh_optimizer.param_groups[0]["lr"] == 0.02


def test_isla_reference_checkpoint_round_trip(tmp_path):
    """A reference-configuration ISLA saved with Module.save is rebuilt bitwise by
    Module.from_checkpoint from its recorded constructor arguments."""
    from physicsnemo.core.module import Module
    from physicsnemo.experimental.nn.isla import ISLA

    torch.manual_seed(0)
    model = ISLA(hidden=16, n_layers=1, n_slices=4, out_scalars=2, out_vectors=1).eval()
    path = str(tmp_path / "isla_reference.mdlus")
    model.save(path)
    loaded = Module.from_checkpoint(path).eval()
    assert type(loaded) is ISLA
    assert loaded._args["__args__"] == model._args["__args__"]
    pts = torch.randn(1, 20, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 20, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 20) + 0.5
    with torch.no_grad():
        assert torch.equal(
            loaded(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
            model(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
        )


def test_isla_research_checkpoint_args_load_when_at_former_defaults(tmp_path):
    """Checkpoints written by the research class recorded options the lean class no
    longer has. Recorded at their former defaults they load into the same network
    (and the re-saved arguments no longer carry them); a non-default value raises
    and names the tag where the research class is preserved."""
    from physicsnemo.core.module import Module
    from physicsnemo.experimental.nn.isla import ISLA
    from physicsnemo.experimental.nn.isla.model import RESEARCH_TAG

    torch.manual_seed(0)
    model = ISLA(hidden=16, n_layers=1, n_slices=4).eval()
    ### Simulate the research class's recorded arguments (JSON turns tuples into lists).
    model._args["__args__"].update(
        center_mode="plain", odd_head=False, anchor_topk=0, local_radii=[0.01, 0.03]
    )
    path = str(tmp_path / "isla_research_defaults.mdlus")
    model.save(path)
    loaded = Module.from_checkpoint(path).eval()
    assert not {"center_mode", "odd_head", "anchor_topk", "local_radii"} & set(
        loaded._args["__args__"]
    )
    pts = torch.randn(1, 20, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 20, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 20) + 0.5
    with torch.no_grad():
        assert torch.equal(
            loaded(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
            model(points=pts, normals=nrm, global_vectors=drv, measure_weights=w),
        )

    model._args["__args__"]["odd_head"] = True
    bad = str(tmp_path / "isla_research_odd_head.mdlus")
    model.save(bad)
    with pytest.raises(ValueError, match=RESEARCH_TAG):
        Module.from_checkpoint(bad)
