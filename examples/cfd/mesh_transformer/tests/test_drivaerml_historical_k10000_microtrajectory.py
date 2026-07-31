# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import errno
import functools
import hashlib
import importlib.util
import json
import marshal
import os
import random
import struct
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.modules.module as torch_module_runtime
import torch.optim.optimizer as torch_optimizer

from physicsnemo.optim import CombinedOptimizer, Muon


@pytest.fixture(scope="module")
def study():
    path = (
        Path(__file__).parents[1]
        / "studies"
        / "drivaerml_historical_k10000_microtrajectory.py"
    )
    spec = importlib.util.spec_from_file_location("microtrajectory_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ToyState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.activation = torch.nn.Tanh()
        self.register_buffer("persistent_counter", torch.tensor([3.0]))
        self.register_buffer(
            "scratch",
            torch.tensor([-0.0, 2.0]),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(value))


def _package() -> tuple[_ToyState, CombinedOptimizer, dict[str, torch.Generator]]:
    torch.manual_seed(7)
    model = _ToyState()
    model.train()
    model.activation.eval()
    optimizer = CombinedOptimizer(
        [
            Muon(
                [model.linear.weight],
                lr=1e-3,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW([model.linear.bias], lr=1e-3),
        ]
    )
    optimizer.optimizers[1].param_groups[0]["signed_zero_control"] = -0.0
    loss = model(torch.arange(6.0).reshape(2, 3)).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    return model, optimizer, {"probe": generator}


class _TrajectoryToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.25)
        self.register_buffer("persistent_counter", torch.tensor([0.25]))
        self.register_buffer(
            "scratch",
            torch.tensor([-0.5, 0.75]),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.persistent_counter.add_(1.0)
            self.scratch.mul_(-1.0).add_(0.125)
        output = self.dropout(self.activation(self.linear(value)))
        return output + 0.01 * self.persistent_counter + 0.001 * self.scratch.sum()


def _trajectory_package(study, *, seed: int, warmed: bool):
    torch.manual_seed(seed)
    model = _TrajectoryToy()
    model.train()
    model.activation.eval()
    model.dropout.eval()
    optimizer = CombinedOptimizer(
        [
            Muon(
                [model.linear.weight],
                lr=2e-3,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(
                [model.linear.bias],
                lr=3e-3,
                weight_decay=0.01,
            ),
        ]
    )
    if warmed:
        warm_output = model(torch.arange(6.0).reshape(2, 3) / 7.0)
        warm_output.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)
    return study.TrajectoryPackage(model, optimizer, {"probe": generator})


def _trajectory_rng_schedule(study, package) -> tuple[dict, ...]:
    random.seed(7001)
    np.random.seed(7002)
    torch.manual_seed(7003)
    package.explicit_generators["probe"].manual_seed(7004)
    states = []
    for _step in range(study.TRAJECTORY_STEP_COUNT):
        states.append(study.capture_rng_state(package.explicit_generators))
        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), generator=package.explicit_generators["probe"])
    study.restore_rng_state(states[0], package.explicit_generators)
    return tuple(states)


def _trajectory_callback(case_index: int):
    value = (
        torch.arange(6.0, dtype=torch.float32).reshape(2, 3) / 11.0 + case_index / 13.0
    )
    target = torch.full((2, 2), case_index / 17.0, dtype=torch.float32)

    def forward_loss(package):
        assert all(parameter.grad is None for parameter in package.model.parameters())
        rng_offset = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + float(torch.rand((), generator=package.explicit_generators["probe"]))
        )
        prediction = package.model(value) + 1e-3 * rng_offset
        loss = (prediction - target).square().mean()
        return {
            "pressure": prediction[:, 0],
            "wss": prediction,
        }, loss

    return forward_loss


def _strict_trajectory_package(study, *, seed: int, warmed: bool = True):
    package = _trajectory_package(study, seed=seed, warmed=warmed)
    package.explicit_generators.clear()
    return package


def _strict_trajectory_callback(case_index: int):
    value = (
        torch.arange(6.0, dtype=torch.float32).reshape(2, 3) / 11.0 + case_index / 13.0
    )
    target = torch.full((2, 2), case_index / 17.0, dtype=torch.float32)

    def forward_loss(package):
        assert all(parameter.grad is None for parameter in package.model.parameters())
        prediction = package.model(value)
        loss = (prediction - target).square().mean()
        return {
            "pressure": prediction[:, 0],
            "wss": prediction,
        }, loss

    return forward_loss


def _flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        tuple(
            parameter.detach().reshape(-1).float().cpu()
            for _name, parameter in model.named_parameters(remove_duplicate=False)
        )
    ).clone()


def _flat_gradients(model: torch.nn.Module) -> torch.Tensor:
    gradients = []
    for _name, parameter in model.named_parameters(remove_duplicate=False):
        assert parameter.grad is not None
        gradients.append(parameter.grad.detach().reshape(-1).float().cpu())
    return torch.cat(tuple(gradients)).clone()


def _test_learning_rates(optimizer) -> tuple[dict, ...]:
    records = []
    for member_index, member in enumerate(optimizer.optimizers):
        member_class = f"{type(member).__module__}.{type(member).__qualname__}"
        for group_index, group in enumerate(member.param_groups):
            records.append(
                {
                    "member_index": member_index,
                    "member_class": member_class,
                    "group_index": group_index,
                    "value_float64": float(group["lr"]),
                }
            )
    return tuple(records)


def _direct_transition_record(
    study,
    package,
    *,
    step_index: int,
    rng_state: dict,
    callback,
) -> tuple[dict, dict]:
    continuation = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    study.restore_rng_state(rng_state, package.explicit_generators)
    package.optimizer.zero_grad(set_to_none=True)
    parameter_names = tuple(
        name
        for name, _parameter in package.model.named_parameters(remove_duplicate=False)
    )
    assert study.assert_gradients_cleared(package.model) == parameter_names
    pre = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    before = _flat_parameters(package.model)
    learning_rates_pre = _test_learning_rates(package.optimizer)

    outputs, loss = callback(package)
    output_records = tuple(
        {
            "name": name,
            "value_float32": (
                outputs[name]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .clone()
            ),
        }
        for name in sorted(outputs)
    )
    loss_float64 = np.asarray([float(loss.detach().item())], dtype="<f8")
    loss.backward()
    gradient = _flat_gradients(package.model)
    package.optimizer.step()
    after = _flat_parameters(package.model)
    update = (after - before).contiguous().clone()
    learning_rates_post = _test_learning_rates(package.optimizer)
    package.optimizer.zero_grad(set_to_none=True)
    post = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )

    def hashes(state):
        return {
            "complete": study.stable_sha256(state),
            "model": study.stable_sha256(state["model"]),
            "optimizer": study.stable_sha256(state["optimizer"]),
            "rng": study.stable_sha256(state["rng"]),
        }

    return {
        "schema_version": study.TRAJECTORY_RECORD_SCHEMA_VERSION,
        "step_index": step_index,
        "parameter_names": parameter_names,
        "parameter_count": int(before.numel()),
        "parameter_order_sha256": study.stable_sha256(parameter_names),
        "outputs": output_records,
        "loss_float64": loss_float64,
        "gradient_float32": gradient,
        "parameter_update_float32": update,
        "learning_rates_pre": learning_rates_pre,
        "learning_rates_post": learning_rates_post,
        "state_sha256": {
            "continuation": hashes(continuation),
            "pre": hashes(pre),
            "post": hashes(post),
        },
        "rng_state": {
            "pre": pre["rng"],
            "post": post["rng"],
        },
    }, post


def _checkpoint(trace: dict, step: int) -> dict:
    return next(
        record["state"] for record in trace["checkpoints"] if record["step"] == step
    )


def test_frozen_panel_schedule_and_record_cardinalities(study) -> None:
    assert study.PANEL_RESOLUTION == 10_000
    assert study.EXPECTED_PARAMETER_COUNT == 1_278_268
    assert study.FIXED_CASE_SPECS == (
        (0, 0, "run_118"),
        (1, 12, "run_271"),
        (2, 24, "run_429"),
        (3, 35, "run_86"),
    )
    assert study.STEP_CASE_INDICES == (0, 1, 2, 3) * 4
    assert (
        study.TRAJECTORY_STEP_SCHEDULE
        == (
            "run_118",
            "run_271",
            "run_429",
            "run_86",
        )
        * 4
    )
    assert study.STATE_REGIMES == ("fresh_seed42", "checkpoint_epoch491")
    assert study.GEOMETRY_PATHS == ("legacy", "canonical")
    assert study.TRAJECTORY_CHECKPOINT_STEPS == (0, 1, 2, 4, 8, 16)
    assert study.TRAJECTORY_REPLAY_STEPS == (0, 1, 2, 4, 8)
    assert study.CROSSOVER_STEPS == (0, 4, 8, 16)
    assert study.CROSSOVER_HISTORIES == (
        "legacy_updated_state",
        "canonical_updated_state",
    )
    assert study.CROSSOVER_EVALUATION_GEOMETRIES == (
        "legacy_geometry",
        "canonical_geometry",
    )
    assert study.HISTORY_TO_PATH_INDEX == (0, 1)
    assert study.EVALUATION_GEOMETRY_TO_PATH_INDEX == (0, 1)
    assert study.MAIN_RECORD_COUNT == 64
    assert study.CHECKPOINT_RECORD_COUNT == 24
    assert study.REPLAY_RECORD_COUNT == 20
    assert study.CROSSOVER_RECORD_COUNT == 128
    assert study.EXECUTED_TRANSITION_COUNT == 212
    assert study.PACKAGE_RECORD_COUNT == 4
    assert study.CASE_RECORD_COUNT == 4
    assert study.MATCHED_MAIN_PAIR_COUNT == 32
    assert study.T0_IDENTITY_COMPARISON_COUNT == 16
    assert study.STATE_TREE_IDENTITY_COUNT == 448
    assert study.SCIENTIFIC_FIXED_ARRAY_COUNT == 1_231
    assert study.FIXED_NON_STATE_ARRAY_COUNT == 1_233
    assert study.NPZ_IDENTITY_ARRAY_FIELDS == (
        "attempt_id_utf8",
        "launch_manifest_sha256_ascii",
    )
    assert len(study.PARAMETER_LAYOUT_ARRAY_FIELDS) == 3
    assert len(study.CASE_CONTROL_ARRAY_FIELDS) == 4
    assert len(study.MAIN_ARRAY_FIELDS) == 7
    assert len(study.CHECKPOINT_ARRAY_FIELDS) == 1
    assert len(study.REPLAY_ARRAY_FIELDS) == 5
    assert len(study.CROSSOVER_ARRAY_FIELDS) == 5


def test_record_identities_are_unique_contiguous_and_self_describing(study) -> None:
    main = [
        study._main_record_identity(regime, path, step)
        for regime in range(2)
        for path in range(2)
        for step in range(16)
    ]
    checkpoints = [
        study._checkpoint_record_identity(regime, path, checkpoint)
        for regime in range(2)
        for path in range(2)
        for checkpoint in range(6)
    ]
    replays = [
        study._replay_record_identity(regime, path, replay)
        for regime in range(2)
        for path in range(2)
        for replay in range(5)
    ]
    crossovers = [
        study._crossover_record_identity(
            regime,
            checkpoint,
            case,
            history,
            geometry,
        )
        for regime in range(2)
        for checkpoint in range(4)
        for case in range(4)
        for history in range(2)
        for geometry in range(2)
    ]

    for records, count in (
        (main, 64),
        (checkpoints, 24),
        (replays, 20),
        (crossovers, 128),
    ):
        assert [record["record_ordinal"] for record in records] == list(range(count))
        prefixes = [record["prefix"] for record in records]
        assert len(set(prefixes)) == count
        assert all(study._RAW_NPZ_KEY.fullmatch(prefix) for prefix in prefixes)

    assert main[0]["prefix"] == (
        "main_m000_r00_fresh_seed42_p00_legacy_t00_to_t01_c00_o00_run_118"
    )
    assert main[-1]["prefix"] == (
        "main_m063_r01_checkpoint_epoch491_p01_canonical_t15_to_t16_c03_o35_run_86"
    )
    assert checkpoints[-1]["prefix"] == (
        "checkpoint_k023_r01_checkpoint_epoch491_p01_canonical_t16"
    )
    assert replays[-1]["prefix"] == (
        "replay_y019_r01_checkpoint_epoch491_p01_canonical_t08_to_t09_c00_o00_run_118"
    )
    assert crossovers[-1]["prefix"] == (
        "crossover_x127_r01_checkpoint_epoch491_t16_c03_o35_run_86"
        "_h01_canonical_updated_state_g01_canonical_geometry"
    )
    assert [record["case_id"] for record in main[:16]] == list(
        study.TRAJECTORY_STEP_SCHEDULE
    )


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    (
        ("_case_identity", (True,)),
        ("_main_record_identity", (0, 0, 16)),
        ("_checkpoint_record_identity", (0, -1, 0)),
        ("_replay_record_identity", (2, 0, 0)),
        ("_crossover_record_identity", (0, 0, 4, 0, 0)),
    ),
)
def test_record_identities_reject_out_of_range_or_boolean_indices(
    study,
    function_name,
    arguments,
) -> None:
    with pytest.raises(ValueError, match="index is out of range"):
        getattr(study, function_name)(*arguments)


def test_complete_state_round_trip_restores_every_component(study) -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    model, optimizer, generators = _package()
    checkpoint = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    expected_hash = study.stable_sha256(checkpoint)

    with torch.no_grad():
        model.linear.weight.add_(9.0)
        model.persistent_counter.mul_(4.0)
        model.scratch.fill_(8.0)
    model.eval()
    model.activation.train()
    optimizer.optimizers[0].param_groups[0]["lr"] = 0.25
    next(iter(optimizer.optimizers[0].state.values()))["momentum_buffer"].zero_()
    random.random()
    np.random.random()
    torch.rand(3)
    torch.rand(3, generator=generators["probe"])

    study.restore_complete_state(
        model,
        optimizer,
        checkpoint,
        explicit_generators=generators,
    )
    observed = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    assert study.stable_sha256(observed) == expected_hash
    assert model.training is True
    assert model.activation.training is False
    assert model.scratch.signbit().tolist() == [True, False]
    assert optimizer.optimizers[1].param_groups[0]["signed_zero_control"] == -0.0


def test_rng_restore_reproduces_global_and_explicit_draws(study) -> None:
    model, optimizer, generators = _package()
    checkpoint = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )

    def draws() -> tuple[float, float, torch.Tensor, torch.Tensor]:
        return (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
            torch.rand(4, generator=generators["probe"]),
        )

    study.restore_complete_state(
        model,
        optimizer,
        checkpoint,
        explicit_generators=generators,
    )
    first = draws()
    study.restore_complete_state(
        model,
        optimizer,
        checkpoint,
        explicit_generators=generators,
    )
    second = draws()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_optimizer_state_is_parameter_name_addressed(study) -> None:
    model, optimizer, _generators = _package()
    state = study.capture_optimizer_state(optimizer, model)
    members = state["members"]
    assert tuple(
        record["parameter"]
        for member in members
        for record in member["parameter_states"]
    ) == ("linear.weight", "linear.bias")
    assert set(members[0]["parameter_states"][0]["values"]) == {"momentum_buffer"}
    assert set(members[1]["parameter_states"][0]["values"]) == {
        "step",
        "exp_avg",
        "exp_avg_sq",
    }


def test_non_none_gradient_is_rejected(study) -> None:
    model, optimizer, generators = _package()
    model.linear.bias.grad = torch.zeros_like(model.linear.bias)
    with pytest.raises(ValueError, match="gradients are not None"):
        study.capture_complete_state(
            model,
            optimizer,
            explicit_generators=generators,
        )


def test_explicit_generator_inventory_is_exact(study) -> None:
    model, optimizer, generators = _package()
    checkpoint = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    with pytest.raises(ValueError, match="generator names differ"):
        study.restore_complete_state(
            model,
            optimizer,
            checkpoint,
            explicit_generators={},
        )


def test_hash_distinguishes_signed_zero_and_numeric_types(study) -> None:
    assert study.stable_sha256(-0.0) != study.stable_sha256(0.0)
    assert study.stable_sha256(False) != study.stable_sha256(0)
    assert study.stable_sha256(1) != study.stable_sha256(1.0)


def test_empty_optimizer_state_membership_is_preserved(study) -> None:
    model, optimizer, _generators = _package()
    adamw = optimizer.optimizers[1]
    adamw.state.clear()
    absent = study.capture_optimizer_state(optimizer, model)
    adamw.state[model.linear.bias] = {}
    present = study.capture_optimizer_state(optimizer, model)
    assert study.stable_sha256(absent) != study.stable_sha256(present)

    study.restore_optimizer_state(optimizer, model, absent)
    assert model.linear.bias not in adamw.state
    study.restore_optimizer_state(optimizer, model, present)
    assert model.linear.bias in adamw.state
    assert adamw.state[model.linear.bias] == {}


def test_state_attached_to_wrong_optimizer_member_is_rejected(study) -> None:
    model, optimizer, _generators = _package()
    optimizer.optimizers[0].state[model.linear.bias] = {
        "foreign": torch.zeros_like(model.linear.bias)
    }
    with pytest.raises(ValueError, match="wrong member"):
        study.capture_optimizer_state(optimizer, model)


def test_explicit_generator_alias_is_rejected(study) -> None:
    generator = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="identities must be unique"):
        study.capture_rng_state({"first": generator, "second": generator})
    with pytest.raises(ValueError, match="default generator"):
        study.capture_rng_state({"default": torch.default_generator})
    with pytest.raises(ValueError, match="nonempty exact strings"):
        study.capture_rng_state({1: generator})


def test_hidden_model_or_optimizer_generator_is_rejected(study) -> None:
    model, optimizer, _generators = _package()
    package = study.TrajectoryPackage(model, optimizer, {})
    study._assert_no_hidden_package_generators(package)

    model.hidden_generator = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="hidden torch.Generator"):
        study._assert_no_hidden_package_generators(package)
    del model.hidden_generator

    optimizer.optimizers[1].param_groups[0]["hidden_generator"] = torch.Generator(
        device="cpu"
    )
    with pytest.raises(ValueError, match="hidden torch.Generator"):
        study._assert_no_hidden_package_generators(package)
    del optimizer.optimizers[1].param_groups[0]["hidden_generator"]

    model.hidden_holder = SimpleNamespace(
        generator=torch.Generator(device="cpu"),
    )
    with pytest.raises(ValueError, match="hidden torch.Generator"):
        study._assert_no_hidden_package_generators(package)


def test_registered_buffer_storage_alias_is_rejected(study) -> None:
    model = _ToyState()
    model.register_buffer("scratch_alias", model.scratch)
    with pytest.raises(ValueError, match="storage alias"):
        study.capture_model_state(model)


def test_noncontiguous_parameter_is_rejected(study) -> None:
    class NoncontiguousParameter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3).T)

    model = NoncontiguousParameter()
    assert not model.weight.is_contiguous()
    with pytest.raises(ValueError, match="contiguous strided tensor"):
        study.capture_model_state(model)


def test_optimizer_tensor_leaf_alias_is_rejected(study) -> None:
    model, optimizer, _generators = _package()
    state = optimizer.optimizers[1].state[model.linear.bias]
    state["exp_avg_sq"] = state["exp_avg"]
    with pytest.raises(ValueError, match="storage alias"):
        study.capture_optimizer_state(optimizer, model)


def test_parameterless_module_alias_is_rejected(study) -> None:
    class AliasedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.first = torch.nn.ReLU()
            self.second = self.first

    with pytest.raises(ValueError, match="Aliased module objects"):
        study.capture_model_state(AliasedModule())


def test_distinct_storage_wrappers_overlapping_memory_are_rejected(study) -> None:
    class ExternalMemoryAlias(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backing = np.arange(4, dtype=np.float32)
            self.first = torch.nn.Parameter(torch.from_numpy(backing))
            self.second = torch.nn.Parameter(torch.from_numpy(backing))

    model = ExternalMemoryAlias()
    assert model.first.untyped_storage()._cdata != model.second.untyped_storage()._cdata
    assert model.first.data_ptr() == model.second.data_ptr()
    with pytest.raises(ValueError, match="storage alias"):
        study.capture_model_state(model)


def test_buffer_and_optimizer_state_gradients_are_rejected(study) -> None:
    model, optimizer, _generators = _package()
    model.scratch.requires_grad_(True)
    with pytest.raises(ValueError, match="must not require or carry gradients"):
        study.capture_model_state(model)

    model.scratch.requires_grad_(False)
    state = optimizer.optimizers[1].state[model.linear.bias]
    state["exp_avg"].requires_grad_(True)
    with pytest.raises(ValueError, match="must not require or carry gradients"):
        study.capture_optimizer_state(optimizer, model)


def test_module_training_flag_must_be_a_boolean(study) -> None:
    model = _ToyState()
    model.activation.training = 1
    with pytest.raises(ValueError, match="training flag must be Boolean"):
        study.capture_model_state(model)


def test_module_class_inventory_is_bound(study) -> None:
    class ConfigurableActivation(torch.nn.Module):
        def __init__(self, activation: torch.nn.Module) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.activation = activation

    source = ConfigurableActivation(torch.nn.ReLU())
    target = ConfigurableActivation(torch.nn.Sigmoid())
    state = study.capture_model_state(source)
    with pytest.raises(ValueError, match="Module class differs"):
        study.restore_model_state(target, state)


def test_combined_optimizer_runtime_methods_are_bound(study) -> None:
    model, optimizer, _generators = _package()
    optimizer.step_fns = [lambda: None, lambda: None]
    with pytest.raises(ValueError, match="ordinary member steps"):
        study.capture_optimizer_state(optimizer, model)


def test_hash_preserves_nan_payload_and_sign(study) -> None:
    values = [
        struct.unpack(">d", bytes.fromhex(bits))[0]
        for bits in (
            "7ff8000000000001",
            "7ff8000000000002",
            "fff8000000000001",
        )
    ]
    assert len({study.stable_sha256(value) for value in values}) == len(values)


def test_model_and_parameter_hooks_are_rejected(study) -> None:
    model = _ToyState()
    handle = model.register_forward_hook(lambda _module, _inputs, output: output + 1)
    with pytest.raises(ValueError, match="Module hooks are forbidden"):
        study.capture_model_state(model)
    handle.remove()

    parameter_handle = model.linear.weight.register_hook(lambda gradient: gradient)
    with pytest.raises(ValueError, match="Parameter hooks are forbidden"):
        study.capture_model_state(model)
    parameter_handle.remove()


def test_global_optimizer_hooks_are_rejected(study) -> None:
    model, optimizer, _generators = _package()
    handle = torch_optimizer.register_optimizer_step_pre_hook(
        lambda _optimizer, _args, _kwargs: None
    )
    try:
        with pytest.raises(ValueError, match="Global optimizer hooks are forbidden"):
            study.capture_optimizer_state(optimizer, model)
    finally:
        handle.remove()


def test_global_module_hooks_are_rejected(study) -> None:
    model = _ToyState()
    handle = torch_module_runtime.register_module_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(ValueError, match="Global module hooks are forbidden"):
            study.capture_model_state(model)
    finally:
        handle.remove()


def test_tensor_subclasses_are_rejected(study) -> None:
    class FancyParameter(torch.nn.Parameter):
        pass

    class ParameterSubclassModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = FancyParameter(torch.ones(1))

    with pytest.raises(ValueError, match="exact Parameter"):
        study.capture_model_state(ParameterSubclassModel())

    model, optimizer, _generators = _package()

    class FancyTensor(torch.Tensor):
        pass

    adamw_state = optimizer.optimizers[1].state[model.linear.bias]
    adamw_state["exp_avg"] = torch.Tensor._make_subclass(
        FancyTensor,
        adamw_state["exp_avg"],
        False,
    )
    with pytest.raises(ValueError, match="Tensor subclass"):
        study.capture_optimizer_state(optimizer, model)

    model, optimizer, _generators = _package()
    checkpoint = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators={},
    )
    checkpoint["model"]["parameters"][0]["value"] = checkpoint["model"]["parameters"][
        0
    ]["value"].as_subclass(FancyTensor)
    with pytest.raises(ValueError, match="Parameter schema differs"):
        study.restore_complete_state(
            model,
            optimizer,
            checkpoint,
            explicit_generators={},
        )


def test_registered_none_parameter_and_module_slots_are_bound(study) -> None:
    class OptionalRegistrations(torch.nn.Module):
        def __init__(self, *, register_optional: bool) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            if register_optional:
                self.register_parameter("optional_parameter", None)
                self.add_module("optional_module", None)

    source = OptionalRegistrations(register_optional=True)
    target = OptionalRegistrations(register_optional=False)
    state = study.capture_model_state(source)
    with pytest.raises(ValueError, match="Registered parameter schema differs"):
        study.restore_model_state(target, state)


@pytest.mark.parametrize("warmed", [False, True])
def test_persistent_trace_matches_independent_direct_loop(study, warmed) -> None:
    package = _trajectory_package(study, seed=41, warmed=warmed)
    rng_states = _trajectory_rng_schedule(study, package)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    initial_state = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    initial_parameters = _flat_parameters(package.model)

    trace = study.run_persistent_trajectory(
        package,
        frozen_rng_states=rng_states,
        forward_losses=callbacks,
    )
    assert trace["step_count"] == 16
    assert tuple(record["step"] for record in trace["checkpoints"]) == (
        0,
        1,
        2,
        4,
        8,
        16,
    )
    assert tuple(record["step_index"] for record in trace["transitions"]) == tuple(
        range(16)
    )
    assert not torch.equal(_flat_parameters(package.model), initial_parameters)
    assert study.assert_gradients_cleared(package.model)

    oracle = _trajectory_package(study, seed=999, warmed=not warmed)
    study.restore_complete_state(
        oracle.model,
        oracle.optimizer,
        initial_state,
        explicit_generators=oracle.explicit_generators,
    )
    oracle_records = []
    for step_index, (rng_state, callback) in enumerate(
        zip(rng_states, callbacks, strict=True)
    ):
        record, _post = _direct_transition_record(
            study,
            oracle,
            step_index=step_index,
            rng_state=rng_state,
            callback=callback,
        )
        oracle_records.append(record)

    assert tuple(
        study.stable_sha256(record) for record in trace["transitions"]
    ) == tuple(study.stable_sha256(record) for record in oracle_records)
    for previous, current in zip(
        trace["transitions"][:-1],
        trace["transitions"][1:],
        strict=True,
    ):
        assert (
            previous["state_sha256"]["post"]["complete"]
            == current["state_sha256"]["continuation"]["complete"]
        )
        assert study.stable_sha256(previous["rng_state"]["post"]) == (
            study.stable_sha256(current["rng_state"]["pre"])
        )


def test_checkpoint_replays_are_exact_and_mutations_fail(study) -> None:
    package = _trajectory_package(study, seed=51, warmed=True)
    rng_states = _trajectory_rng_schedule(study, package)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    trace = study.run_persistent_trajectory(
        package,
        frozen_rng_states=rng_states,
        forward_losses=callbacks,
    )
    persistent_hash = study.stable_sha256(
        study.capture_complete_state(
            package.model,
            package.optimizer,
            explicit_generators=package.explicit_generators,
        )
    )

    def hostile_factory():
        return _trajectory_package(study, seed=12345, warmed=False)

    for step in (0, 1, 2, 4, 8):
        observed, _post = study.verify_transition_replay(
            package,
            checkpoint=_checkpoint(trace, step),
            expected_transition=trace["transitions"][step],
            package_factory=hostile_factory,
            step_index=step,
            frozen_rng_state=rng_states[step],
            forward_loss=callbacks[step],
        )
        assert study.stable_sha256(observed) == study.stable_sha256(
            trace["transitions"][step]
        )
    assert (
        study.stable_sha256(
            study.capture_complete_state(
                package.model,
                package.optimizer,
                explicit_generators=package.explicit_generators,
            )
        )
        == persistent_hash
    )

    mutated_transition = copy.deepcopy(trace["transitions"][4])
    mutated_transition["gradient_float32"].view(torch.uint8)[0] ^= 1
    with pytest.raises(ValueError, match="Checkpoint replay differs"):
        study.verify_transition_replay(
            package,
            checkpoint=_checkpoint(trace, 4),
            expected_transition=mutated_transition,
            package_factory=hostile_factory,
            step_index=4,
            frozen_rng_state=rng_states[4],
            forward_loss=callbacks[4],
        )

    mutated_checkpoint = copy.deepcopy(_checkpoint(trace, 4))
    scratch = next(
        record
        for record in mutated_checkpoint["model"]["buffers"]
        if record["name"] == "scratch"
    )
    scratch["value"].add_(0.25)
    with pytest.raises(ValueError, match="Checkpoint replay differs"):
        study.verify_transition_replay(
            package,
            checkpoint=mutated_checkpoint,
            expected_transition=trace["transitions"][4],
            package_factory=hostile_factory,
            step_index=4,
            frozen_rng_state=rng_states[4],
            forward_loss=callbacks[4],
        )


def test_isolated_probe_is_transactional_on_success_and_failure(study) -> None:
    package = _trajectory_package(study, seed=61, warmed=True)
    rng_states = _trajectory_rng_schedule(study, package)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    trace = study.run_persistent_trajectory(
        package,
        frozen_rng_states=rng_states,
        forward_losses=callbacks,
    )

    def complete_hash():
        return study.stable_sha256(
            study.capture_complete_state(
                package.model,
                package.optimizer,
                explicit_generators=package.explicit_generators,
            )
        )

    before = complete_hash()
    record, _post = study.execute_isolated_transition(
        package,
        checkpoint=_checkpoint(trace, 16),
        package_factory=lambda: _trajectory_package(
            study,
            seed=777,
            warmed=False,
        ),
        step_index=16,
        frozen_rng_state=rng_states[0],
        forward_loss=_trajectory_callback(3),
    )
    assert record["step_index"] == 16
    assert complete_hash() == before

    def failing_callback(clone):
        _trajectory_callback(2)(clone)
        raise RuntimeError("injected probe failure")

    with pytest.raises(RuntimeError, match="injected probe failure"):
        study.execute_isolated_transition(
            package,
            checkpoint=_checkpoint(trace, 8),
            package_factory=lambda: _trajectory_package(
                study,
                seed=888,
                warmed=False,
            ),
            step_index=8,
            frozen_rng_state=rng_states[0],
            forward_loss=failing_callback,
        )
    assert complete_hash() == before

    def mutating_callback(_clone):
        with torch.no_grad():
            package.model.linear.weight.add_(1.0)
        raise RuntimeError("persistent package was touched")

    with pytest.raises(ValueError, match="mutated the persistent package"):
        study.execute_isolated_transition(
            package,
            checkpoint=_checkpoint(trace, 4),
            package_factory=lambda: _trajectory_package(
                study,
                seed=999,
                warmed=False,
            ),
            step_index=4,
            frozen_rng_state=rng_states[0],
            forward_loss=mutating_callback,
        )
    assert complete_hash() == before


def test_isolated_probe_rejects_shared_package_objects(study) -> None:
    package = _trajectory_package(study, seed=71, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    checkpoint = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    before = study.stable_sha256(checkpoint)

    with pytest.raises(ValueError, match="share module objects"):
        study.execute_isolated_transition(
            package,
            checkpoint=checkpoint,
            package_factory=lambda: package,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=_trajectory_callback(0),
        )
    assert (
        study.stable_sha256(
            study.capture_complete_state(
                package.model,
                package.optimizer,
                explicit_generators=package.explicit_generators,
            )
        )
        == before
    )


def test_isolated_probe_restores_rebound_persistent_objects(study) -> None:
    package = _trajectory_package(study, seed=72, warmed=True)
    rng_states = _trajectory_rng_schedule(study, package)
    checkpoint = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    original_bindings = (
        package.model,
        package.optimizer,
        package.explicit_generators,
    )
    replacement = _trajectory_package(study, seed=73, warmed=False)
    study.restore_complete_state(
        replacement.model,
        replacement.optimizer,
        checkpoint,
        explicit_generators=replacement.explicit_generators,
    )

    def rebind_persistent(clone):
        package.model = replacement.model
        package.optimizer = replacement.optimizer
        package.explicit_generators = replacement.explicit_generators
        return _trajectory_callback(0)(clone)

    with pytest.raises(ValueError, match="object bindings"):
        study.execute_isolated_transition(
            package,
            checkpoint=checkpoint,
            package_factory=lambda: _trajectory_package(
                study,
                seed=74,
                warmed=False,
            ),
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=rebind_persistent,
            require_rng_continuation=True,
        )
    assert package.model is original_bindings[0]
    assert package.optimizer is original_bindings[1]
    assert package.explicit_generators is original_bindings[2]
    assert study.stable_sha256(
        study.capture_complete_state(
            package.model,
            package.optimizer,
            explicit_generators=package.explicit_generators,
        )
    ) == study.stable_sha256(checkpoint)


def test_clone_cannot_alias_input_checkpoint_storage(study) -> None:
    package = _trajectory_package(study, seed=74, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    checkpoint = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    checkpoint_hash = study.stable_sha256(checkpoint)
    checkpoint_weight = next(
        record["value"]
        for record in checkpoint["model"]["parameters"]
        if record["name"] == "linear.weight"
    )

    def aliasing_factory():
        clone = _trajectory_package(study, seed=75, warmed=False)
        clone.model.linear.weight = torch.nn.Parameter(checkpoint_weight)
        return clone

    with pytest.raises(ValueError, match="storage alias"):
        study.execute_isolated_transition(
            package,
            checkpoint=checkpoint,
            package_factory=aliasing_factory,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=_trajectory_callback(0),
            require_rng_continuation=True,
        )
    assert study.stable_sha256(checkpoint) == checkpoint_hash


def test_interleaved_probes_leave_the_main_trace_byte_exact(study) -> None:
    source = _trajectory_package(study, seed=75, warmed=True)
    rng_states = _trajectory_rng_schedule(study, source)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    initial = study.capture_complete_state(
        source.model,
        source.optimizer,
        explicit_generators=source.explicit_generators,
    )
    baseline = study.run_persistent_trajectory(
        source,
        frozen_rng_states=rng_states,
        forward_losses=callbacks,
    )

    probed = _trajectory_package(study, seed=76, warmed=False)
    study.restore_complete_state(
        probed.model,
        probed.optimizer,
        initial,
        explicit_generators=probed.explicit_generators,
    )
    checkpoint = initial
    probed_transitions = []

    def run_probe(step):
        before = study.stable_sha256(
            study.capture_complete_state(
                probed.model,
                probed.optimizer,
                explicit_generators=probed.explicit_generators,
            )
        )
        study.execute_isolated_transition(
            probed,
            checkpoint=checkpoint,
            package_factory=lambda: _trajectory_package(
                study,
                seed=900 + step,
                warmed=False,
            ),
            step_index=step,
            frozen_rng_state=rng_states[(step + 5) % 16],
            forward_loss=_trajectory_callback((step + 1) % 4),
        )
        after = study.stable_sha256(
            study.capture_complete_state(
                probed.model,
                probed.optimizer,
                explicit_generators=probed.explicit_generators,
            )
        )
        assert after == before

    for step in range(16):
        if step in (0, 4, 8):
            run_probe(step)
        record, checkpoint = study.execute_stateful_transition(
            probed,
            step_index=step,
            frozen_rng_state=rng_states[step],
            forward_loss=callbacks[step],
        )
        probed_transitions.append(record)
    run_probe(16)

    assert tuple(study.stable_sha256(record) for record in probed_transitions) == tuple(
        study.stable_sha256(record) for record in baseline["transitions"]
    )
    assert study.stable_sha256(checkpoint) == study.stable_sha256(
        _checkpoint(baseline, 16)
    )


@pytest.mark.parametrize("component", ["python", "numpy", "torch", "explicit"])
def test_each_rng_component_skew_breaks_main_continuation(study, component) -> None:
    package = _trajectory_package(study, seed=81, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    checkpoint = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )
    study.restore_rng_state(rng_states[0], package.explicit_generators)
    if component == "python":
        random.random()
    elif component == "numpy":
        np.random.random()
    elif component == "torch":
        torch.rand(())
    else:
        torch.rand((), generator=package.explicit_generators["probe"])
    skewed = study.capture_rng_state(package.explicit_generators)
    study.restore_complete_state(
        package.model,
        package.optimizer,
        checkpoint,
        explicit_generators=package.explicit_generators,
    )

    with pytest.raises(ValueError, match="not the preceding continuation RNG"):
        study.execute_stateful_transition(
            package,
            step_index=0,
            frozen_rng_state=skewed,
            forward_loss=_trajectory_callback(0),
        )


def test_trajectory_requires_exact_schedule_and_inactive_stochastic_modules(
    study,
) -> None:
    package = _trajectory_package(study, seed=91, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    with pytest.raises(ValueError, match="exactly 16 RNG"):
        study.run_persistent_trajectory(
            package,
            frozen_rng_states=rng_states[:-1],
            forward_losses=callbacks,
        )
    with pytest.raises(ValueError, match="exactly 16 callbacks"):
        study.run_persistent_trajectory(
            package,
            frozen_rng_states=rng_states,
            forward_losses=callbacks[:-1],
        )

    package.model.dropout.train()
    with pytest.raises(ValueError, match="Active stochastic modules"):
        study.execute_stateful_transition(
            package,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=callbacks[0],
        )


def test_callback_cannot_change_modes_modules_or_optimizer_lr(study) -> None:
    package = _trajectory_package(study, seed=95, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    checkpoint = study.capture_complete_state(
        package.model,
        package.optimizer,
        explicit_generators=package.explicit_generators,
    )

    def activate_dropout(current):
        current.model.dropout.train()
        return _trajectory_callback(0)(current)

    with pytest.raises(ValueError, match="activated stochastic"):
        study.execute_stateful_transition(
            package,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=activate_dropout,
        )

    study.restore_complete_state(
        package.model,
        package.optimizer,
        checkpoint,
        explicit_generators=package.explicit_generators,
    )

    def replace_module(current):
        current.model.activation = torch.nn.Identity()
        return _trajectory_callback(0)(current)

    with pytest.raises(ValueError, match="model execution contract"):
        study.execute_stateful_transition(
            package,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=replace_module,
        )

    package = _trajectory_package(study, seed=96, warmed=False)
    rng_states = _trajectory_rng_schedule(study, package)
    package.optimizer.optimizers[0].param_groups[0]["lr"] = 0.0
    with pytest.raises(ValueError, match="LR is not positive"):
        study.execute_stateful_transition(
            package,
            step_index=0,
            frozen_rng_state=rng_states[0],
            forward_loss=_trajectory_callback(0),
        )


def test_transition_records_and_checkpoints_are_deeply_independent(study) -> None:
    package = _trajectory_package(study, seed=97, warmed=True)
    rng_states = _trajectory_rng_schedule(study, package)
    callbacks = tuple(_trajectory_callback(step % 4) for step in range(16))
    trace = study.run_persistent_trajectory(
        package,
        frozen_rng_states=rng_states,
        forward_losses=callbacks,
    )
    transition = trace["transitions"][0]
    checkpoint = _checkpoint(trace, 1)
    checkpoint_hash = study.stable_sha256(checkpoint)
    assert transition["rng_state"]["post"] is not checkpoint["rng"]
    assert (
        transition["rng_state"]["post"]["torch_cpu"]
        is not checkpoint["rng"]["torch_cpu"]
    )

    transition["rng_state"]["post"]["torch_cpu"][0] ^= 1
    transition["outputs"][0]["value_float32"].add_(1.0)
    assert study.stable_sha256(checkpoint) == checkpoint_hash


def test_matched_rng_transition_rejects_an_extra_unused_draw(study) -> None:
    source = _trajectory_package(study, seed=101, warmed=True)
    rng_states = _trajectory_rng_schedule(study, source)
    initial = study.capture_complete_state(
        source.model,
        source.optimizer,
        explicit_generators=source.explicit_generators,
    )

    def restored_package(seed):
        package = _trajectory_package(study, seed=seed, warmed=False)
        study.restore_complete_state(
            package.model,
            package.optimizer,
            initial,
            explicit_generators=package.explicit_generators,
        )
        return package

    callback = _trajectory_callback(0)
    left, _left_post = study.execute_stateful_transition(
        restored_package(102),
        step_index=0,
        frozen_rng_state=rng_states[0],
        forward_loss=callback,
    )
    right, _right_post = study.execute_stateful_transition(
        restored_package(103),
        step_index=0,
        frozen_rng_state=rng_states[0],
        forward_loss=callback,
    )
    study.assert_matched_rng_transition(left, right)

    def extra_draw(package):
        outputs, loss = callback(package)
        random.random()
        return outputs, loss

    skewed, _skewed_post = study.execute_stateful_transition(
        restored_package(104),
        step_index=0,
        frozen_rng_state=rng_states[0],
        forward_loss=extra_draw,
    )
    with pytest.raises(ValueError, match="post-step RNG states differ"):
        study.assert_matched_rng_transition(left, skewed)


def test_paired_microtrajectory_runs_main_replay_and_crossover_schedule(study) -> None:
    random.seed(1401)
    np.random.seed(1402)
    packages = {
        path: _trajectory_package(study, seed=1403, warmed=True)
        for path in study.GEOMETRY_PATHS
    }

    def package_factory():
        return _trajectory_package(study, seed=1403, warmed=True)

    factories = {path: package_factory for path in study.GEOMETRY_PATHS}
    case_callbacks = tuple(
        _trajectory_callback(case_index)
        for case_index in range(len(study.FIXED_CASE_SPECS))
    )
    result = study._run_paired_microtrajectory(
        packages,
        package_factories=factories,
        case_forward_losses={path: case_callbacks for path in study.GEOMETRY_PATHS},
    )

    assert result["step_count"] == 16
    assert tuple(result["main"]) == study.GEOMETRY_PATHS
    assert all(
        len(result["main"][path]) == study.TRAJECTORY_STEP_COUNT
        for path in study.GEOMETRY_PATHS
    )
    assert all(
        tuple(record["step"] for record in result["checkpoints"][path])
        == study.TRAJECTORY_CHECKPOINT_STEPS
        for path in study.GEOMETRY_PATHS
    )
    assert all(
        tuple(record["step"] for record in result["replays"][path])
        == study.TRAJECTORY_REPLAY_STEPS
        for path in study.GEOMETRY_PATHS
    )
    assert len(result["crossovers"]) == 64
    assert sum(record["checkpoint_step"] == 16 for record in result["crossovers"]) == 16
    for path in study.GEOMETRY_PATHS:
        terminal = study._capture_valid_complete_state(packages[path])
        assert study.stable_sha256(terminal) == study.stable_sha256(
            result["main"][path][-1]["post_state"]
        )


def test_paired_microtrajectory_rejects_final_sibling_mutation(study) -> None:
    random.seed(1411)
    np.random.seed(1412)
    packages = {
        path: _trajectory_package(study, seed=1413, warmed=True)
        for path in study.GEOMETRY_PATHS
    }
    callbacks = tuple(
        _trajectory_callback(case_index)
        for case_index in range(len(study.FIXED_CASE_SPECS))
    )
    canonical_final_case_calls = 0

    def mutate_legacy_on_final_occurrence(package):
        nonlocal canonical_final_case_calls
        outputs, loss = callbacks[3](package)
        canonical_final_case_calls += 1
        if canonical_final_case_calls == 4:
            with torch.no_grad():
                packages["legacy"].model.linear.bias.add_(1.0)
        return outputs, loss

    canonical_callbacks = (
        callbacks[0],
        callbacks[1],
        callbacks[2],
        mutate_legacy_on_final_occurrence,
    )

    def package_factory():
        return _trajectory_package(study, seed=1413, warmed=True)

    with pytest.raises(ValueError, match="Post-step 15.*mutated the legacy"):
        study._run_paired_microtrajectory(
            packages,
            package_factories={path: package_factory for path in study.GEOMETRY_PATHS},
            case_forward_losses={
                "legacy": callbacks,
                "canonical": canonical_callbacks,
            },
        )


def test_frozen_v3_transition_is_deterministic_and_revalidates_in_finally(
    study,
) -> None:
    package = _strict_trajectory_package(study, seed=1421)
    frozen_rng = study.capture_rng_state({})
    validations = []

    record, post_state = study.execute_frozen_v3_transition(
        package,
        step_index=0,
        frozen_rng_state=frozen_rng,
        forward_loss=_strict_trajectory_callback(0),
        execution_revalidator=lambda: validations.append("validated"),
        synchronize_cuda=False,
    )

    assert validations == ["validated", "validated"]
    assert study.stable_sha256(record["rng_state"]["pre"]) == study.stable_sha256(
        record["rng_state"]["post"]
    )
    assert record["state_sha256"]["post"]["complete"] == study.stable_sha256(post_state)


@pytest.mark.parametrize("rng_source", ("python", "numpy"))
def test_frozen_v3_transition_rejects_unrecorded_rng_consumption(
    study,
    rng_source,
) -> None:
    package = _strict_trajectory_package(study, seed=1422)
    frozen_rng = study.capture_rng_state({})
    deterministic = _strict_trajectory_callback(0)

    def random_callback(current):
        if rng_source == "python":
            random.random()
        else:
            np.random.random()
        return deterministic(current)

    with pytest.raises(ValueError, match="consumed RNG"):
        study.execute_frozen_v3_transition(
            package,
            step_index=0,
            frozen_rng_state=frozen_rng,
            forward_loss=random_callback,
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
        )


@pytest.mark.parametrize("operation", ("rand", "dropout"))
def test_frozen_v3_transition_rejects_seeded_aten_operations(
    study,
    operation,
) -> None:
    package = _strict_trajectory_package(study, seed=1423)
    frozen_rng = study.capture_rng_state({})
    deterministic = _strict_trajectory_callback(0)

    def random_callback(current):
        if operation == "rand":
            torch.rand(())
        else:
            torch.nn.functional.dropout(torch.ones(4), p=0.5, training=True)
        return deterministic(current)

    with pytest.raises(RuntimeError, match="Seeded nondeterministic ATen"):
        study.execute_frozen_v3_transition(
            package,
            step_index=0,
            frozen_rng_state=frozen_rng,
            forward_loss=random_callback,
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
        )


def test_generic_transition_remains_stochastic_capable(study) -> None:
    package = _strict_trajectory_package(study, seed=1424)
    frozen_rng = study.capture_rng_state({})
    deterministic = _strict_trajectory_callback(0)

    def random_callback(current):
        torch.rand(())
        return deterministic(current)

    record, _post_state = study.execute_stateful_transition(
        package,
        step_index=0,
        frozen_rng_state=frozen_rng,
        forward_loss=random_callback,
    )
    assert study.stable_sha256(record["rng_state"]["pre"]) != study.stable_sha256(
        record["rng_state"]["post"]
    )


def test_frozen_v3_transition_rejects_package_generators(study) -> None:
    explicit = _trajectory_package(study, seed=1425, warmed=True)
    explicit_rng = study.capture_rng_state(explicit.explicit_generators)
    with pytest.raises(ValueError, match="inventory must be empty"):
        study.execute_frozen_v3_transition(
            explicit,
            step_index=0,
            frozen_rng_state=explicit_rng,
            forward_loss=_trajectory_callback(0),
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
        )

    hidden = _strict_trajectory_package(study, seed=1426)
    hidden.model.hidden_generator = torch.Generator()
    hidden_rng = study.capture_rng_state({})
    with pytest.raises(ValueError, match="hidden torch.Generator"):
        study.execute_frozen_v3_transition(
            hidden,
            step_index=0,
            frozen_rng_state=hidden_rng,
            forward_loss=_strict_trajectory_callback(0),
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
        )


@pytest.mark.parametrize("capture_kind", ("closure", "default", "bound", "partial"))
def test_frozen_v3_transition_rejects_callback_generators(
    study,
    capture_kind,
) -> None:
    package = _strict_trajectory_package(study, seed=1427)
    frozen_rng = study.capture_rng_state({})
    generator = torch.Generator()
    deterministic = _strict_trajectory_callback(0)

    if capture_kind == "closure":

        def new_callback(hidden_generator, underlying):
            def callback(current):
                hidden_generator.get_state()
                return underlying(current)

            return callback

        callback = new_callback(generator, deterministic)
    elif capture_kind == "default":

        def callback(current, hidden_generator=generator):
            hidden_generator.get_state()
            return deterministic(current)

    elif capture_kind == "bound":

        class Callback:
            def __init__(self):
                self.hidden_generator = generator

            def run(self, current):
                self.hidden_generator.get_state()
                return deterministic(current)

        callback = Callback().run
    else:

        def with_generator(current, *, hidden_generator):
            hidden_generator.get_state()
            return deterministic(current)

        callback = functools.partial(with_generator, hidden_generator=generator)

    with pytest.raises(ValueError, match="captures torch.Generator"):
        study.execute_frozen_v3_transition(
            package,
            step_index=0,
            frozen_rng_state=frozen_rng,
            forward_loss=callback,
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
        )


def test_frozen_v3_transition_chains_run_and_finally_failures(study) -> None:
    package = _strict_trajectory_package(study, seed=1428)
    frozen_rng = study.capture_rng_state({})
    validation_calls = 0

    def revalidate():
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise ValueError("post-transition boundary failed")

    def fail_transition(_package):
        raise RuntimeError("transition failed")

    with pytest.raises(ValueError, match="post-transition boundary failed") as caught:
        study.execute_frozen_v3_transition(
            package,
            step_index=0,
            frozen_rng_state=frozen_rng,
            forward_loss=fail_transition,
            execution_revalidator=revalidate,
            synchronize_cuda=False,
        )
    assert validation_calls == 2
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "transition failed"


def test_paired_microtrajectory_routes_exactly_212_strict_transitions(study) -> None:
    transition_steps = []

    def transition_executor(package, **kwargs):
        transition_steps.append(kwargs["step_index"])
        return study.execute_frozen_v3_transition(
            package,
            execution_revalidator=lambda: None,
            synchronize_cuda=False,
            **kwargs,
        )

    for regime_index in range(len(study.STATE_REGIMES)):
        random.seed(1430 + regime_index)
        np.random.seed(1440 + regime_index)
        packages = {
            path: _strict_trajectory_package(study, seed=1450 + regime_index)
            for path in study.GEOMETRY_PATHS
        }

        def package_factory(seed=1450 + regime_index):
            return _strict_trajectory_package(study, seed=seed)

        callbacks = tuple(
            _strict_trajectory_callback(case_index)
            for case_index in range(len(study.FIXED_CASE_SPECS))
        )
        before = len(transition_steps)
        trace = study._run_paired_microtrajectory(
            packages,
            package_factories={path: package_factory for path in study.GEOMETRY_PATHS},
            case_forward_losses={path: callbacks for path in study.GEOMETRY_PATHS},
            transition_executor=transition_executor,
        )
        assert len(transition_steps) - before == 106
        assert (
            sum(len(trace["main"][path]) for path in study.GEOMETRY_PATHS)
            + sum(len(trace["replays"][path]) for path in study.GEOMETRY_PATHS)
            + len(trace["crossovers"])
            == 106
        )

    assert len(transition_steps) == study.EXECUTED_TRANSITION_COUNT == 212


def test_raw_npz_writer_streams_one_canonical_zip64_archive(study, tmp_path) -> None:
    path = (tmp_path / "raw.npz").resolve()
    arrays = {
        "gradient_float32": np.arange(12, dtype="<f4").reshape(3, 4),
        "selected_ids_int64": np.asarray([7, 11, 13], dtype="<i8"),
        "rng_bytes_uint8": np.asarray([0, 255, 4], dtype="|u1"),
        "gradients_none_bool": np.asarray([True, True], dtype="|b1"),
        "loss_float64": np.asarray([1.25], dtype="<f8"),
    }
    with study._RawNpzWriter(path) as writer:
        for key, value in arrays.items():
            writer.add(key, value)
        manifest = writer.manifest()
        member_order = writer.member_order()
    arrays["gradient_float32"].fill(-9.0)

    assert tuple(manifest) == tuple(sorted(arrays))
    for key, record in manifest.items():
        expected = (
            np.arange(12, dtype="<f4").reshape(3, 4)
            if key == "gradient_float32"
            else arrays[key]
        )
        assert record == {
            "dtype": expected.dtype.str,
            "shape": list(expected.shape),
            "nbytes": expected.nbytes,
            "sha256": study._raw_array_sha256(expected),
        }

    with np.load(path, allow_pickle=False) as archive:
        assert tuple(sorted(archive.files)) == tuple(sorted(arrays))
        assert np.array_equal(
            archive["gradient_float32"],
            np.arange(12, dtype="<f4").reshape(3, 4),
        )
        for key in set(arrays).difference({"gradient_float32"}):
            assert np.array_equal(archive[key], arrays[key])
    with zipfile.ZipFile(path, "r") as archive:
        assert tuple(info.filename for info in archive.infolist()) == tuple(
            f"{key}.npy" for key in arrays
        )
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert all(info.extract_version >= 45 for info in archive.infolist())
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
    assert member_order == tuple(arrays)


def test_raw_npz_writer_rejects_noncanonical_or_duplicate_arrays(
    study,
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        study._RawNpzWriter(Path("relative.npz"))

    path = (tmp_path / "raw.npz").resolve()
    writer = study._RawNpzWriter(path)
    writer.add("valid_float32", np.ones(3, dtype="<f4"))
    with pytest.raises(ValueError, match="duplicated"):
        writer.add("valid_float32", np.ones(3, dtype="<f4"))
    with pytest.raises(ValueError, match="key is not canonical"):
        writer.add("../escape", np.ones(1, dtype="<f4"))
    with pytest.raises(TypeError, match="exact ndarray"):
        writer.add("not_array", [1, 2, 3])
    with pytest.raises(ValueError, match="C-contiguous"):
        writer.add("noncontiguous", np.ones((3, 4), dtype="<f4")[:, ::2])
    with pytest.raises(ValueError, match="forbidden dtype"):
        writer.add("big_endian", np.ones(2, dtype=">f4"))
    with pytest.raises(ValueError, match="forbidden dtype"):
        writer.add("object_array", np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="non-finite"):
        writer.add("nonfinite", np.asarray([np.nan], dtype="<f4"))
    scalar = np.asarray(7, dtype="<i8")
    writer.add("scalar_int64", scalar)
    assert writer.manifest()["scalar_int64"]["sha256"] == study._raw_array_sha256(
        scalar
    )
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.add("after_close", np.ones(1, dtype="<f4"))
    with np.load(path, allow_pickle=False) as archive:
        assert archive["scalar_int64"].shape == ()
        assert int(archive["scalar_int64"]) == 7
    with pytest.raises(FileExistsError):
        study._RawNpzWriter(path)


def test_atomic_raw_npz_publication_is_validated_deterministic_and_cleans_staging(
    study,
    tmp_path,
) -> None:
    arrays = (
        ("first_float32", np.asarray([-0.0, 2.5], dtype="<f4")),
        ("second_int64", np.asarray([3, 5, 8], dtype="<i8")),
    )

    def populate(writer):
        for key, value in arrays:
            writer.add(key, value)
        return {"sentinel": 17}

    outputs = tuple((tmp_path / f"atomic_{index}.npz").resolve() for index in range(2))
    results = tuple(
        study._write_validated_raw_npz_no_clobber(output, populate)
        for output in outputs
    )

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert outputs[0].stat().st_nlink == outputs[1].stat().st_nlink == 1
    assert results[0]["populate_result"] == {"sentinel": 17}
    assert results[0]["manifest"] == results[1]["manifest"]
    assert results[0]["member_order"] == tuple(key for key, _value in arrays)
    assert results[0]["validation"] == results[0]["staged_validation"]
    assert (
        results[0]["validation"]["sha256"]
        == hashlib.sha256(outputs[0].read_bytes()).hexdigest()
    )
    assert not tuple(tmp_path.glob(".atomic_*.npz.*.staging"))


@pytest.mark.parametrize("failure", ("value_error", "keyboard_interrupt", "close"))
def test_atomic_raw_npz_publication_failure_never_exposes_partial_output(
    study,
    tmp_path,
    monkeypatch,
    failure,
) -> None:
    output = (tmp_path / f"atomic_failure_{failure}.npz").resolve()

    def populate(writer):
        writer.add("first_float32", np.ones(2, dtype="<f4"))
        if failure == "value_error":
            raise ValueError("injected population failure")
        if failure == "keyboard_interrupt":
            raise KeyboardInterrupt
        return None

    if failure == "close":
        original_close = study._RawNpzWriter.close

        def failing_close(writer):
            original_close(writer)
            raise OSError("injected close failure")

        monkeypatch.setattr(study._RawNpzWriter, "close", failing_close)
        expected_error = OSError
    elif failure == "keyboard_interrupt":
        expected_error = KeyboardInterrupt
    else:
        expected_error = ValueError

    with pytest.raises(expected_error):
        study._write_validated_raw_npz_no_clobber(output, populate)
    assert not os.path.lexists(output)
    assert not tuple(tmp_path.glob(f".{output.name}.*.staging"))


def test_atomic_raw_npz_publication_rolls_back_post_link_validation_failure(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    output = (tmp_path / "atomic_post_link_failure.npz").resolve()
    original_validate = study._validate_raw_npz
    calls = 0

    def fail_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected final validation failure")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(study, "_validate_raw_npz", fail_second_validation)
    with pytest.raises(OSError, match="final validation"):
        study._write_validated_raw_npz_no_clobber(
            output,
            lambda writer: writer.add(
                "first_float32",
                np.ones(2, dtype="<f4"),
            ),
        )
    assert calls == 2
    assert not os.path.lexists(output)
    assert not tuple(tmp_path.glob(f".{output.name}.*.staging"))


def test_atomic_raw_npz_publication_preserves_no_clobber_collisions(
    study,
    tmp_path,
) -> None:
    for collision in ("existing", "broken_symlink", "race"):
        output = (tmp_path / f"atomic_collision_{collision}.npz").resolve()
        if collision == "existing":
            output.write_bytes(b"existing")
        elif collision == "broken_symlink":
            os.symlink("absent-target", output)

        def populate(writer):
            writer.add("first_float32", np.ones(2, dtype="<f4"))
            if collision == "race":
                output.write_bytes(b"racing-writer")

        expected = (
            b"existing"
            if collision == "existing"
            else b"racing-writer"
            if collision == "race"
            else None
        )
        with pytest.raises(FileExistsError):
            study._write_validated_raw_npz_no_clobber(output, populate)
        if collision == "broken_symlink":
            assert output.is_symlink()
            assert os.readlink(output) == "absent-target"
        else:
            assert output.read_bytes() == expected
        assert not tuple(tmp_path.glob(f".{output.name}.*.staging"))


def test_fixed_array_schemas_pin_every_dtype_shape_and_field(study) -> None:
    layout = study._fixed_array_schema(
        "parameter_layout",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    case = study._fixed_array_schema(
        "case_control",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    main = study._fixed_array_schema(
        "main",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    checkpoint = study._fixed_array_schema(
        "checkpoint",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    replay = study._fixed_array_schema(
        "replay",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    crossover = study._fixed_array_schema(
        "crossover",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )

    assert tuple(field for field, _dtype, _shape in layout) == (
        study.PARAMETER_LAYOUT_ARRAY_FIELDS
    )
    assert tuple(field for field, _dtype, _shape in case) == (
        study.CASE_CONTROL_ARRAY_FIELDS
    )
    assert tuple(field for field, _dtype, _shape in main) == study.MAIN_ARRAY_FIELDS
    assert tuple(field for field, _dtype, _shape in checkpoint) == (
        study.CHECKPOINT_ARRAY_FIELDS
    )
    assert tuple(field for field, _dtype, _shape in replay) == (
        study.REPLAY_ARRAY_FIELDS
    )
    assert tuple(field for field, _dtype, _shape in crossover) == (
        study.CROSSOVER_ARRAY_FIELDS
    )
    assert main[-2:] == (
        ("learning_rates_pre_float64", "<f8", (2,)),
        ("learning_rates_post_float64", "<f8", (2,)),
    )
    assert checkpoint == (("parameter_vector_float32", "<f4", (1_278_268,)),)
    assert crossover[-1] == (
        "proposed_parameter_update_float32",
        "<f4",
        (1_278_268,),
    )

    for kwargs in (
        {"parameter_tensor_count": True, "learning_rate_count": 2},
        {"parameter_tensor_count": 140, "learning_rate_count": 0},
    ):
        with pytest.raises(ValueError, match="positive integer"):
            study._fixed_array_schema("main", **kwargs)
    with pytest.raises(ValueError, match="Unknown"):
        study._fixed_array_schema(
            "unknown",
            parameter_tensor_count=140,
            learning_rate_count=2,
        )


def test_npz_identity_embeds_exact_attempt_and_launch_bindings(
    study,
    tmp_path,
) -> None:
    attempt_id = "microtrajectory-attempt_001"
    launch_sha256 = "ab" * 32
    path = (tmp_path / "identity.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        keys = study._write_npz_identity(
            writer,
            attempt_id=attempt_id,
            launch_manifest_sha256=launch_sha256,
        )
    assert keys == study.NPZ_IDENTITY_ARRAY_FIELDS
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(archive.files) == keys
        assert archive["attempt_id_utf8"].tobytes().decode("ascii") == attempt_id
        assert (
            archive["launch_manifest_sha256_ascii"].tobytes().decode("ascii")
            == launch_sha256
        )

    for bad_attempt, bad_sha, message in (
        ("", launch_sha256, "Attempt ID"),
        ("bad/path", launch_sha256, "Attempt ID"),
        (attempt_id, "A" * 64, "Launch-manifest"),
        (attempt_id, "0" * 63, "Launch-manifest"),
    ):
        rejected_path = (
            tmp_path / f"rejected_identity_{len(list(tmp_path.iterdir()))}.npz"
        ).resolve()
        writer = study._RawNpzWriter(rejected_path)
        with pytest.raises(ValueError, match=message):
            study._write_npz_identity(
                writer,
                attempt_id=bad_attempt,
                launch_manifest_sha256=bad_sha,
            )
        assert writer.manifest() == {}
        writer.close()


def test_raw_npz_reread_validator_authenticates_every_member_and_file_byte(
    study,
    tmp_path,
) -> None:
    path = (tmp_path / "validated.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        writer.add("z_scalar_int64", np.asarray(7, dtype="<i8"))
        writer.add(
            "a_float32",
            np.asarray([-0.0, 1.25, 9.5], dtype="<f4"),
        )
        writer.add(
            "m_boolean",
            np.asarray([False, True, True], dtype="|b1"),
        )
        manifest = writer.manifest()
        member_order = writer.member_order()

    assert member_order != tuple(sorted(member_order))
    result = study._validate_raw_npz(
        path,
        manifest=manifest,
        expected_order=member_order,
    )
    assert result == {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "member_count": 3,
    }

    with pytest.raises(ValueError, match="member order or names differ"):
        study._validate_raw_npz(
            path,
            manifest=manifest,
            expected_order=tuple(sorted(member_order)),
        )

    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["a_float32"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload SHA-256 differs"):
        study._validate_raw_npz(
            path,
            manifest=bad_manifest,
            expected_order=member_order,
        )

    appended = (tmp_path / "appended.npz").resolve()
    appended.write_bytes(path.read_bytes() + b"hidden trailing bytes")
    with pytest.raises(ValueError, match="ZIP end record"):
        study._validate_raw_npz(
            appended,
            manifest=manifest,
            expected_order=member_order,
        )

    symlink = (tmp_path / "symlink.npz").resolve()
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="opened safely"):
        study._validate_raw_npz(
            symlink,
            manifest=manifest,
            expected_order=member_order,
        )


def test_raw_npz_reread_validator_rejects_noncanonical_zip_writers(
    study,
    tmp_path,
) -> None:
    values = {
        "only_float32": np.arange(4, dtype="<f4"),
    }
    manifest = {
        "only_float32": {
            "dtype": "<f4",
            "shape": [4],
            "nbytes": 16,
            "sha256": study._raw_array_sha256(values["only_float32"]),
        }
    }
    order = ("only_float32",)

    ordinary = (tmp_path / "ordinary.npz").resolve()
    np.savez(ordinary, **values)
    with pytest.raises(ValueError, match="NPY header differs"):
        study._validate_raw_npz(
            ordinary,
            manifest=manifest,
            expected_order=order,
        )

    compressed = (tmp_path / "compressed.npz").resolve()
    np.savez_compressed(compressed, **values)
    with pytest.raises(ValueError, match="archive layout differs"):
        study._validate_raw_npz(
            compressed,
            manifest=manifest,
            expected_order=order,
        )


def test_raw_npz_reread_validator_handles_python_zip64_offset_threshold(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 200)
    path = (tmp_path / "forced_zip64_end.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        for index in range(4):
            writer.add(
                f"array_{index:02d}_float32",
                np.arange(4, dtype="<f4") + index,
            )
        manifest = writer.manifest()
        member_order = writer.member_order()

    payload = path.read_bytes()
    assert payload[-42:-38] == b"PK\x06\x07"
    result = study._validate_raw_npz(
        path,
        manifest=manifest,
        expected_order=member_order,
    )
    assert result["member_count"] == 4
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_raw_npz_reread_validator_requires_python_zip64_threshold_encoding(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    path = (tmp_path / "missing_zip64_end.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        writer.add("only_float32", np.arange(4, dtype="<f4"))
        manifest = writer.manifest()
        member_order = writer.member_order()

    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    with pytest.raises(ValueError, match="required ZIP64 end records are absent"):
        study._validate_raw_npz(
            path,
            manifest=manifest,
            expected_order=member_order,
        )


def test_raw_npz_reread_validator_rejects_fifo_without_blocking(
    study,
    tmp_path,
) -> None:
    path = (tmp_path / "adversarial_fifo.npz").resolve()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="not a regular file"):
        study._validate_raw_npz(
            path,
            manifest={
                "only_float32": {
                    "dtype": "<f4",
                    "shape": [1],
                    "nbytes": 4,
                    "sha256": "0" * 64,
                }
            },
            expected_order=("only_float32",),
        )


def test_raw_npz_reread_validator_checks_layout_before_zipfile_allocation(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    path = (tmp_path / "wrong_layout.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        writer.add("only_float32", np.arange(4, dtype="<f4"))
        manifest = writer.manifest()
        member_order = writer.member_order()

    called = False

    class ForbiddenZipFile:
        def __init__(self, *_args, **_kwargs):
            nonlocal called
            called = True
            raise AssertionError("ZipFile must not inspect a wrong-sized archive")

    with path.open("ab") as stream:
        stream.write(b"x")
    monkeypatch.setattr(zipfile, "ZipFile", ForbiddenZipFile)
    with pytest.raises(ValueError, match="ZIP end record"):
        study._validate_raw_npz(
            path,
            manifest=manifest,
            expected_order=member_order,
        )
    assert not called


def test_raw_npz_reread_validator_rejects_hard_link_alias(
    study,
    tmp_path,
) -> None:
    path = (tmp_path / "bfloat16.npz").resolve()
    alias = (tmp_path / "float32_alias.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        writer.add("only_float32", np.arange(4, dtype="<f4"))
        manifest = writer.manifest()
        member_order = writer.member_order()
    os.link(path, alias)

    with pytest.raises(ValueError, match="exactly one hard link"):
        study._validate_raw_npz(
            path,
            manifest=manifest,
            expected_order=member_order,
        )
    with pytest.raises(ValueError, match="exactly one hard link"):
        study._validate_raw_npz(
            alias,
            manifest=manifest,
            expected_order=member_order,
        )


def test_frozen_v3_loader_executes_exact_authenticated_bytes(study) -> None:
    script_path = Path(study.__file__).resolve()
    producer, legacy, runtime, canonical = study._load_frozen_v3_support(script_path)
    assert producer.__name__ == "frozen_microtrajectory_v3_producer"
    assert Path(producer.__file__) == script_path.with_name(
        study.FROZEN_V3_PRODUCER_FILENAME
    )
    assert (
        producer.RESOLUTION,
        producer.FRESH_SEED,
        producer.CHECKPOINT_EPOCH,
        producer.EXPECTED_PARAMETER_COUNT,
    ) == (
        study.PANEL_RESOLUTION,
        study.FRESH_SEED,
        study.CHECKPOINT_EPOCH,
        study.EXPECTED_PARAMETER_COUNT,
    )
    assert tuple(module.__name__ for module in (legacy, runtime, canonical)) == (
        "frozen_one_step_legacy_support",
        "frozen_one_step_runtime",
        "frozen_one_step_canonical_support",
    )
    assert tuple(
        Path(module.__file__).name for module in (legacy, runtime, canonical)
    ) == (
        producer.LEGACY_SUPPORT_FILENAME,
        producer.RUNTIME_HELPER_FILENAME,
        producer.CANONICAL_HELPER_FILENAME,
    )


def test_frozen_v3_loader_rejects_tampered_producer_before_execution(
    study,
    tmp_path,
) -> None:
    script_path = (tmp_path / "microtrajectory.py").resolve()
    source_path = (
        Path(study.__file__).resolve().with_name(study.FROZEN_V3_PRODUCER_FILENAME)
    )
    producer_path = script_path.with_name(study.FROZEN_V3_PRODUCER_FILENAME)
    producer_path.write_bytes(source_path.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="Frozen v3 producer SHA-256 differs"):
        study._load_frozen_v3_support(script_path)


def test_frozen_v3_loader_rejects_tampered_support_before_execution(
    study,
    tmp_path,
) -> None:
    script_path = (tmp_path / "microtrajectory.py").resolve()
    source_directory = Path(study.__file__).resolve().parent
    producer_source = source_directory / study.FROZEN_V3_PRODUCER_FILENAME
    producer_path = script_path.with_name(study.FROZEN_V3_PRODUCER_FILENAME)
    producer_path.write_bytes(producer_source.read_bytes())

    producer, *_supports = study._load_frozen_v3_support(Path(study.__file__).resolve())
    support_names = (
        producer.LEGACY_SUPPORT_FILENAME,
        producer.RUNTIME_HELPER_FILENAME,
        producer.CANONICAL_HELPER_FILENAME,
    )
    for filename in support_names:
        payload = (source_directory / filename).read_bytes()
        if filename == producer.RUNTIME_HELPER_FILENAME:
            payload += b"\n# tampered\n"
        (tmp_path / filename).write_bytes(payload)

    with pytest.raises(ValueError, match="Frozen support-module SHA-256 differs"):
        study._load_frozen_v3_support(script_path)


def test_stable_small_file_rejects_fifo_without_blocking(study, tmp_path) -> None:
    path = (tmp_path / "support_fifo.py").resolve()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="Small-file contract differs"):
        study._stable_small_file(path)


def _sealed_input_specification(path: Path, payload: bytes) -> dict:
    return {
        "config": (
            path,
            hashlib.sha256(payload).hexdigest(),
            max(1, len(payload) + 16),
        )
    }


def test_sealed_static_input_rejects_wrong_type_hash_and_hardlink(
    study,
    tmp_path,
) -> None:
    payload = b"precision: bfloat16\n"
    regular = (tmp_path / "config.yaml").resolve()
    regular.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 differs"):
        study._SealedStaticInputBundle.from_paths(
            {
                "config": (
                    regular,
                    hashlib.sha256(b"different").hexdigest(),
                    len(payload) + 16,
                )
            }
        )
    with pytest.raises(ValueError, match="file contract differs"):
        study._SealedStaticInputBundle.from_paths(
            {
                "config": (
                    regular,
                    hashlib.sha256(payload).hexdigest(),
                    len(payload) - 1,
                )
            }
        )

    symlink = (tmp_path / "config_link.yaml").resolve()
    symlink.symlink_to(regular)
    with pytest.raises(ValueError, match="open static input safely"):
        study._SealedStaticInputBundle.from_paths(
            _sealed_input_specification(symlink, payload)
        )

    fifo = (tmp_path / "config.fifo").resolve()
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="file contract differs"):
        study._SealedStaticInputBundle.from_paths(
            _sealed_input_specification(fifo, payload)
        )

    alias = (tmp_path / "config_alias.yaml").resolve()
    os.link(regular, alias)
    with pytest.raises(ValueError, match="file contract differs"):
        study._SealedStaticInputBundle.from_paths(
            _sealed_input_specification(regular, payload)
        )


def test_sealed_static_input_rejects_write_grow_and_shrink(
    study,
    tmp_path,
) -> None:
    payload = b"model: sealed\n"
    path = (tmp_path / "config.yaml").resolve()
    path.write_bytes(payload)

    with study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(path, payload)
    ) as bundle:
        bundle.assert_sealed()
        assert bundle.attestation == {
            "config": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "seal_mask": study._SEALED_STATIC_INPUT_SEALS,
            }
        }
        descriptor = os.open(bundle.proc_paths["config"], os.O_RDWR | os.O_CLOEXEC)
        try:
            with pytest.raises(OSError) as write_error:
                os.pwrite(descriptor, b"X", 0)
            assert write_error.value.errno == errno.EPERM
            with pytest.raises(OSError) as shrink_error:
                os.ftruncate(descriptor, len(payload) - 1)
            assert shrink_error.value.errno == errno.EPERM
            with pytest.raises(OSError) as grow_error:
                os.ftruncate(descriptor, len(payload) + 1)
            assert grow_error.value.errno == errno.EPERM
        finally:
            os.close(descriptor)
        bundle.assert_sealed()


def test_sealed_static_input_supports_libc_memfd_fallback(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"fallback: libc\n"
    path = (tmp_path / "config.yaml").resolve()
    path.write_bytes(payload)
    monkeypatch.delattr(study.os, "memfd_create", raising=False)

    with study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(path, payload)
    ) as bundle:
        assert Path(bundle.proc_paths["config"]).read_bytes() == payload
        bundle.assert_sealed()


def test_sealed_static_input_is_independent_of_original_mutation(
    study,
    tmp_path,
) -> None:
    original = b"resolved: frozen\n"
    replacement = b"resolved: forged\n"
    assert len(original) == len(replacement)
    path = (tmp_path / "resolved.yaml").resolve()
    path.write_bytes(original)

    with study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(path, original)
    ) as bundle:
        snapshot_path = Path(bundle.proc_paths["config"])
        path.write_bytes(replacement)
        assert snapshot_path.read_bytes() == original
        bundle.assert_sealed()


def test_sealed_static_input_detects_missing_closed_and_substituted_descriptors(
    study,
    tmp_path,
) -> None:
    first_payload = b"first-static-input"
    first_path = (tmp_path / "first.bin").resolve()
    first_path.write_bytes(first_payload)
    missing = study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(first_path, first_payload)
    )
    missing_descriptor = int(missing.proc_paths["config"].rpartition("/")[2])
    os.close(missing_descriptor)
    with pytest.raises(ValueError, match="descriptor is unavailable"):
        missing.assert_sealed()
    missing.close()

    closed = study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(first_path, first_payload)
    )
    closed_path = closed.proc_paths["config"]
    closed.close()
    with pytest.raises(ValueError, match="bundle is closed"):
        closed.assert_sealed()
    with pytest.raises(FileNotFoundError):
        Path(closed_path).read_bytes()

    second_payload = b"other-static-input"
    assert len(first_payload) == len(second_payload)
    second_path = (tmp_path / "second.bin").resolve()
    second_path.write_bytes(second_payload)
    first = study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(first_path, first_payload)
    )
    second = study._SealedStaticInputBundle.from_paths(
        _sealed_input_specification(second_path, second_payload)
    )
    try:
        first_descriptor = int(first.proc_paths["config"].rpartition("/")[2])
        second_descriptor = int(second.proc_paths["config"].rpartition("/")[2])
        assert first_descriptor != second_descriptor
        os.dup2(second_descriptor, first_descriptor, inheritable=False)
        with pytest.raises(ValueError, match="descriptor identity differs"):
            first.assert_sealed()
        second.assert_sealed()
    finally:
        first.close()
        second.close()


def _toy_frozen_producer(
    study,
    *,
    drift_second_package: bool = False,
    corrupt_layout: bool = False,
    share_mutable_state: bool = False,
    share_custom_state: bool = False,
    share_bytearray_state: bool = False,
) -> ModuleType:
    producer = ModuleType("toy_frozen_v3")
    call_count = 0
    shared_state = {}
    shared_custom = SimpleNamespace(counter=0)
    shared_bytes = bytearray(b"shared")

    def parameter_order_sha256(names):
        return study.stable_sha256(tuple(names))

    def parameter_layout(model):
        named_parameters = list(model.named_parameters())
        names = [name for name, _parameter in named_parameters]
        module_names = list(
            dict.fromkeys(name.rpartition(".")[0] or "<root>" for name in names)
        )
        module_lookup = {name: index for index, name in enumerate(module_names)}
        starts = []
        stops = []
        module_indices = []
        offset = 0
        for name, parameter in named_parameters:
            starts.append(offset)
            offset += parameter.numel()
            stops.append(offset)
            module_indices.append(module_lookup[name.rpartition(".")[0] or "<root>"])
        if corrupt_layout:
            stops[-1] += 1
        return (
            {
                "parameter_count": offset,
                "parameter_names": names,
                "module_names": module_names,
                "ordered_parameter_names_sha256": parameter_order_sha256(names),
            },
            {
                "parameter_slice_starts_int64": np.asarray(starts, dtype="<i8"),
                "parameter_slice_stops_int64": np.asarray(stops, dtype="<i8"),
                "parameter_slice_module_indices_int64": np.asarray(
                    module_indices,
                    dtype="<i8",
                ),
            },
        )

    def new_model_optimizer(_runtime, *, regime, checkpoint_dir):
        nonlocal call_count
        assert checkpoint_dir.is_absolute()
        model, optimizer, _generators = _package()
        if share_mutable_state:
            model.shared_state = shared_state
        if share_custom_state:
            model.shared_custom = shared_custom
        if share_bytearray_state:
            model.shared_bytes = shared_bytes
        call_count += 1
        if drift_second_package and call_count == 2:
            with torch.no_grad():
                model.linear.bias.add_(1.0)
        loaded_epoch = None if regime == "fresh_seed42" else 491
        return model, optimizer, loaded_epoch

    producer._parameter_order_sha256 = parameter_order_sha256
    producer._parameter_layout = parameter_layout
    producer._new_model_optimizer = new_model_optimizer
    return producer


@pytest.mark.parametrize(
    "regime",
    ("fresh_seed42", "checkpoint_epoch491"),
)
def test_verified_package_pair_is_disjoint_and_byte_identical(
    study,
    tmp_path,
    monkeypatch,
    regime,
) -> None:
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    producer = _toy_frozen_producer(study)
    result = study._new_verified_package_pair(
        producer,
        object(),
        regime=regime,
        checkpoint_dir=tmp_path.resolve(),
        static_input_revalidator=lambda: {"static": "stable"},
    )
    assert result["legacy"] is not result["canonical"]
    assert result["legacy"].model is not result["canonical"].model
    assert result["legacy"].optimizer is not result["canonical"].optimizer
    assert result["legacy"].explicit_generators == {}
    assert result["canonical"].explicit_generators == {}
    assert (
        study.stable_sha256(study._capture_valid_complete_state(result["legacy"]))
        == study.stable_sha256(result["initial_state"])
        == study.stable_sha256(study._capture_valid_complete_state(result["canonical"]))
    )
    assert result["attestation"]["layout"]["parameter_count"] == 8
    assert len(result["attestation"]["learning_rates_initial"]) == 2


def test_verified_package_pair_rejects_constructor_or_layout_drift(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    with pytest.raises(ValueError, match="not byte-exact"):
        study._new_verified_package_pair(
            _toy_frozen_producer(study, drift_second_package=True),
            object(),
            regime="fresh_seed42",
            checkpoint_dir=tmp_path.resolve(),
            static_input_revalidator=lambda: {"static": "stable"},
        )
    with pytest.raises(ValueError, match="array differs"):
        study._new_verified_trajectory_package(
            _toy_frozen_producer(study, corrupt_layout=True),
            object(),
            regime="fresh_seed42",
            checkpoint_dir=tmp_path.resolve(),
            static_input_revalidator=lambda: {"static": "stable"},
        )
    with pytest.raises(ValueError, match="share mutable objects"):
        study._new_verified_package_pair(
            _toy_frozen_producer(study, share_mutable_state=True),
            object(),
            regime="fresh_seed42",
            checkpoint_dir=tmp_path.resolve(),
            static_input_revalidator=lambda: {"static": "stable"},
        )
    with pytest.raises(ValueError, match="share mutable objects"):
        study._new_verified_package_pair(
            _toy_frozen_producer(study, share_custom_state=True),
            object(),
            regime="fresh_seed42",
            checkpoint_dir=tmp_path.resolve(),
            static_input_revalidator=lambda: {"static": "stable"},
        )
    with pytest.raises(ValueError, match="share mutable objects"):
        study._new_verified_package_pair(
            _toy_frozen_producer(study, share_bytearray_state=True),
            object(),
            regime="fresh_seed42",
            checkpoint_dir=tmp_path.resolve(),
            static_input_revalidator=lambda: {"static": "stable"},
        )


def _toy_prepared_case(resolution: int, *, seed: int = 0) -> dict:
    selected_ids = np.arange(seed, seed + resolution, dtype="<i8")
    pressure = np.linspace(-1.0, 1.0, resolution, dtype="<f4")
    wss = np.arange(resolution * 3, dtype="<f4").reshape(resolution, 3)
    measure = np.linspace(1.0, 2.0, resolution, dtype="<f4")
    return {
        "domain": object(),
        "bundle": object(),
        "targets": {
            "pressure": torch.from_numpy(pressure.copy()),
            "wss": torch.from_numpy(wss.copy()),
        },
        "target_measure": torch.from_numpy(measure.copy()),
        "selected_ids": selected_ids,
        "target_pressure": pressure,
        "target_wss": wss,
        "target_measure_array": measure,
        "raw_source_geometry_sha256": "a" * 64,
        "global_inputs_sha256": "b" * 64,
        "batch_order_sha256": "c" * 64,
    }


def test_prepared_case_validation_binds_exact_tensor_and_raw_controls(
    study,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 5)
    case = _toy_prepared_case(5)
    control = study._validate_prepared_case(
        case,
        case_index=0,
        cohort_ordinal=0,
        case_id="run_118",
        historical_start=0,
    )
    assert control["case_id"] == "run_118"
    assert control["selected_ids_sha256"] == study._raw_array_sha256(
        case["selected_ids"]
    )
    assert control["target_measure_sha256"] == study._raw_array_sha256(
        case["target_measure_array"]
    )

    mismatched = copy.deepcopy(case)
    mismatched["target_pressure"][0] += np.float32(1.0)
    with pytest.raises(ValueError, match="tensor/array controls differ"):
        study._validate_prepared_case(
            mismatched,
            case_index=0,
            cohort_ordinal=0,
            case_id="run_118",
            historical_start=0,
        )

    nonpositive = copy.deepcopy(case)
    nonpositive["target_measure_array"][0] = np.float32(0.0)
    nonpositive["target_measure"][0] = 0.0
    with pytest.raises(ValueError, match="measure is not positive"):
        study._validate_prepared_case(
            nonpositive,
            case_index=0,
            cohort_ordinal=0,
            case_id="run_118",
            historical_start=0,
        )

    misbound = _toy_prepared_case(5, seed=1_000)
    with pytest.raises(ValueError, match="selected IDs or historical start differ"):
        study._validate_prepared_case(
            misbound,
            case_index=0,
            cohort_ordinal=0,
            case_id="run_118",
            historical_start=0,
        )


def test_flat_recipe_import_provenance_rejects_poisoned_module(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    repo_root = tmp_path.resolve()
    recipe_source = (
        repo_root
        / "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
    )
    recipe_source.mkdir(parents=True)
    required = {
        "collate",
        "datasets",
        "forward_kwargs",
        "loss",
        "output_normalize",
        "utils",
    }
    for name in required:
        path = recipe_source / f"{name}.py"
        path.write_text("# provenance sentinel\n", encoding="utf-8")
        module = ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", [str(recipe_source), *sys.path])

    with pytest.raises(ImportError, match="loaded before authentication"):
        study._preflight_recipe_import_namespace(repo_root)

    for name in required:
        monkeypatch.delitem(sys.modules, name)
    preflight = study._preflight_recipe_import_namespace(repo_root)
    for name in required:
        module = ModuleType(name)
        module.__file__ = str(recipe_source / f"{name}.py")
        monkeypatch.setitem(sys.modules, name, module)
    runtime = SimpleNamespace(
        normalize_output=lambda *_args: None,
        autocast_context=lambda *_args: None,
    )
    with pytest.raises(ImportError, match="provenance differs"):
        study._validate_recipe_import_provenance(
            repo_root,
            preflight=preflight,
            runtime=runtime,
        )


def test_authenticated_recipe_loader_ignores_timestamp_valid_pyc(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    repo_root = tmp_path.resolve()
    recipe_source = (
        repo_root
        / "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src"
    )
    recipe_source.mkdir(parents=True)
    sources = {
        "collate": "def build_collate_fn(): return 'source'\n",
        "datasets": (
            "def build_dataset(): return 'source'\n"
            "def find_normalizer(): return 'source'\n"
        ),
        "forward_kwargs": "def resolve_forward_kwargs(): return 'source'\n",
        "loss": (
            "class LossCalculator:\n    def __init__(self): self.origin = 'source'\n"
        ),
        "output_normalize": ("def normalize_output_to_tensordict(): return 'source'\n"),
        "utils": (
            "def build_muon_optimizer(): return 'source'\n"
            "def get_autocast_context(): return 'source'\n"
            "def set_seed(): return 'source'\n"
        ),
    }
    malicious = {
        name: "PYC_SENTINEL = True\n" + source.replace("'source'", "'pyc'")
        for name, source in sources.items()
    }
    for name, source in sources.items():
        monkeypatch.delitem(sys.modules, name, raising=False)
        path = recipe_source / f"{name}.py"
        path.write_text("SOURCE_SENTINEL = True\n" + source, encoding="utf-8")
        stat_result = path.stat()
        code = compile(malicious[name], str(path), "exec")
        pyc = Path(importlib.util.cache_from_source(str(path)))
        pyc.parent.mkdir(exist_ok=True)
        pyc.write_bytes(
            importlib.util.MAGIC_NUMBER
            + struct.pack(
                "<III",
                0,
                int(stat_result.st_mtime),
                stat_result.st_size,
            )
            + marshal.dumps(code)
        )

    preflight = study._preflight_recipe_import_namespace(repo_root)
    modules = {name: __import__(name) for name in sources}
    assert all(module.SOURCE_SENTINEL is True for module in modules.values())
    assert all(not hasattr(module, "PYC_SENTINEL") for module in modules.values())
    runtime = SimpleNamespace(
        normalize_output=modules["output_normalize"].normalize_output_to_tensordict,
        autocast_context=modules["utils"].get_autocast_context,
    )
    observed = study._validate_recipe_import_provenance(
        repo_root,
        preflight=preflight,
        runtime=runtime,
    )
    assert set(observed) == set(sources)


def test_verified_v3_case_loader_validates_all_36_before_preparing_four(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    case_ids = tuple(f"run_{index}" for index in range(36))
    fixed_specs = (
        (0, 0, case_ids[0]),
        (1, 12, case_ids[12]),
        (2, 24, case_ids[24]),
        (3, 35, case_ids[35]),
    )
    monkeypatch.setattr(study, "FIXED_CASE_SPECS", fixed_specs)
    specs = tuple(
        SimpleNamespace(
            cohort_ordinal=index,
            case_id=case_id,
            historical_start=index * 10,
        )
        for index, case_id in enumerate(case_ids)
    )
    producer = ModuleType("producer")
    legacy = ModuleType("legacy")
    runtime_support = ModuleType("runtime_support")
    canonical = ModuleType("canonical")
    runtime_support.CASE_SPECS = specs

    hashes = {
        "SOURCE_TREE": "1" * 64,
        "DATASET_MANIFEST": "2" * 64,
        "DATASET_CONFIG": "3" * 64,
        "RESOLVED_CONFIG": "4" * 64,
        "MODEL": "5" * 64,
        "TRAINING": "6" * 64,
        "NORMALIZATION": "7" * 64,
    }
    producer.EXPECTED_SOURCE_TREE_SHA256 = hashes["SOURCE_TREE"]
    producer.EXPECTED_DATASET_MANIFEST_SHA256 = hashes["DATASET_MANIFEST"]
    producer.EXPECTED_DATASET_CONFIG_SHA256 = hashes["DATASET_CONFIG"]
    producer.EXPECTED_RESOLVED_CONFIG_SHA256 = hashes["RESOLVED_CONFIG"]
    producer.EXPECTED_MODEL_CHECKPOINT_SHA256 = hashes["MODEL"]
    producer.EXPECTED_TRAINING_STATE_SHA256 = hashes["TRAINING"]
    producer.EXPECTED_NORMALIZATION_STATE_SHA256 = hashes["NORMALIZATION"]
    legacy.EXPECTED_EXECUTION_SOURCE_TREE_SHA256 = hashes["SOURCE_TREE"]
    legacy.EXPECTED_DATASET_MANIFEST_SHA256 = hashes["DATASET_MANIFEST"]
    legacy.EXPECTED_DATASET_CONFIG_SHA256 = hashes["DATASET_CONFIG"]
    legacy.EXPECTED_RESOLVED_CONFIG_SHA256 = hashes["RESOLVED_CONFIG"]
    legacy.EXPECTED_MODEL_SHA256 = hashes["MODEL"]
    legacy.EXPECTED_TRAINING_STATE_SHA256 = hashes["TRAINING"]
    legacy.EXPECTED_NORMALIZATION_SHA256 = hashes["NORMALIZATION"]
    legacy.EXPECTED_CURRENT_INFER_SHA256 = "8" * 64
    legacy.EXPECTED_CURRENT_MODEL_SOURCE_SHA256 = "9" * 64
    legacy.EXPECTED_GEOMETRY_MANIFEST_SHA256 = "a" * 64
    legacy.EXPECTED_TARGET_INPUT_MANIFEST_SHA256 = "b" * 64

    producer._validate_single_rank_environment = lambda: None
    legacy._validate_case_specs = lambda _runtime_support: None
    expected_static = {
        "Dataset manifest": hashes["DATASET_MANIFEST"],
        "Dataset config": hashes["DATASET_CONFIG"],
        "Resolved config": hashes["RESOLVED_CONFIG"],
        "Model checkpoint": hashes["MODEL"],
        "Training state": hashes["TRAINING"],
        "Normalization state": hashes["NORMALIZATION"],
        "Current inference source": legacy.EXPECTED_CURRENT_INFER_SHA256,
        "Current MeshTransformer source": (legacy.EXPECTED_CURRENT_MODEL_SOURCE_SHA256),
        "Current execution source tree": hashes["SOURCE_TREE"],
    }
    legacy._validate_static_inputs = lambda *_args, **_kwargs: expected_static
    geometry_records = [{"case_id": case_id} for case_id in case_ids]
    target_records = [{"case_id": case_id} for case_id in case_ids]
    legacy._verify_geometry_manifest = lambda *_args, **_kwargs: {
        "manifest_sha256": legacy.EXPECTED_GEOMETRY_MANIFEST_SHA256,
        "cases_verified": 36,
        "files_verified": 72,
        "case_records": geometry_records,
    }
    legacy._verify_target_input_manifest = lambda *_args, **_kwargs: {
        "manifest_sha256": legacy.EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
        "cases_verified": 36,
        "selected_ranges_verified": 72,
        "case_records": target_records,
    }
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(compile=True, precision="bfloat16"),
        loaded_epoch=491,
    )
    runtime_support._load_runtime = lambda **_kwargs: runtime
    legacy._validate_import_provenance = lambda _repo_root: {"physicsnemo": "ok"}
    legacy._validate_reader = lambda _runtime: None
    prepared_ids = []

    def prepare_case(**kwargs):
        prepared_ids.append(kwargs["spec"].case_id)
        return _toy_prepared_case(3, seed=kwargs["spec"].cohort_ordinal * 10)

    producer._prepare_case = prepare_case
    producer._batch_order_sha256 = lambda case_id, selected_ids: study.stable_sha256(
        {"case_id": case_id, "selected_ids": selected_ids}
    )
    producer._global_inputs_sha256 = lambda _legacy, domain: study.stable_sha256(
        str(id(domain))
    )

    original_prepare_case = producer._prepare_case

    def prepare_case_with_controls(**kwargs):
        case = original_prepare_case(**kwargs)
        case["batch_order_sha256"] = producer._batch_order_sha256(
            kwargs["spec"].case_id,
            case["selected_ids"],
        )
        case["global_inputs_sha256"] = producer._global_inputs_sha256(
            legacy,
            case["domain"],
        )
        return case

    producer._prepare_case = prepare_case_with_controls
    monkeypatch.setattr(
        study,
        "_preflight_recipe_import_namespace",
        lambda _repo_root: {
            "source_directory": str(_repo_root),
            "source_sha256": {},
        },
    )
    monkeypatch.setattr(
        study,
        "_validate_recipe_import_provenance",
        lambda _repo_root, **_kwargs: {"loss": "ok"},
    )
    monkeypatch.setattr(
        study,
        "_execution_backend_attestation",
        lambda _runtime: {"device": "cuda:0"},
    )
    paths = {
        name: (tmp_path / name).resolve()
        for name in (
            "repo_root",
            "dataset_root",
            "dataset_config",
            "resolved_config",
            "checkpoint_dir",
            "geometry_manifest",
            "target_input_manifest",
        )
    }
    result = study._load_verified_v3_cases(
        producer,
        legacy,
        runtime_support,
        canonical,
        **paths,
    )
    assert tuple(result["cases"]) == tuple(spec[2] for spec in fixed_specs)
    assert tuple(prepared_ids) == tuple(spec[2] for spec in fixed_specs)
    assert len(result["attestation"]["case_controls"]) == 4
    assert result["attestation"]["geometry_manifest"] == {
        "manifest_sha256": legacy.EXPECTED_GEOMETRY_MANIFEST_SHA256,
        "case_records_contract_checked": 36,
        "geometry_file_records_path_type_size_checked": 72,
        "prepared_case_count": 4,
        "prepared_case_geometry_memmap_files_sha256_verified": 8,
    }
    assert result["attestation"]["target_manifest"] == {
        "manifest_sha256": legacy.EXPECTED_TARGET_INPUT_MANIFEST_SHA256,
        "case_records_contract_checked": 36,
        "selected_target_range_records_contract_checked": 72,
        "prepared_case_count": 4,
        "prepared_case_selected_target_ranges_sha256_verified": 8,
    }

    legacy._verify_geometry_manifest = lambda *_args, **_kwargs: {
        "manifest_sha256": legacy.EXPECTED_GEOMETRY_MANIFEST_SHA256,
        "cases_verified": 36,
        "files_verified": 72,
        "case_records": list(reversed(geometry_records)),
    }
    with pytest.raises(ValueError, match="36-case input order differs"):
        study._load_verified_v3_cases(
            producer,
            legacy,
            runtime_support,
            canonical,
            **paths,
        )


class _FloatMapping(dict):
    def float(self):
        return self


class _CallbackMesh:
    def __init__(
        self,
        points,
        cells=None,
        *,
        point_data=None,
        cell_data=None,
        global_data=None,
    ):
        self.points = points
        self.cells = (
            torch.empty((0, 1), dtype=torch.long, device=points.device)
            if cells is None
            else cells
        )
        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.global_data = {} if global_data is None else global_data


def _callback_case(resolution: int = 3):
    points = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    cells = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    centroids = points[cells].mean(dim=1)
    areas = torch.ones(resolution)
    normals = torch.nn.functional.normalize(centroids + 1.0, dim=1)
    boundary = _CallbackMesh(points, cells)
    interior = _CallbackMesh(centroids.clone())
    domain = SimpleNamespace(
        interior=interior,
        boundaries={"vehicle": boundary},
        global_data={"reference_length": torch.tensor(8.0)},
    )
    bundle = SimpleNamespace(
        points=points.clone(),
        cells=cells.clone(),
        centroids=centroids.clone(),
        areas=areas,
        normals=normals,
        physical_center=torch.zeros(3, dtype=torch.float64),
        physical_length=5.0,
        model_reference_length=8.0,
    )
    pressure = torch.linspace(-1.0, 1.0, resolution)
    wss = torch.arange(resolution * 3, dtype=torch.float32).reshape(resolution, 3)
    measure = torch.linspace(1.0, 2.0, resolution)
    case = {
        "domain": domain,
        "bundle": bundle,
        "targets": _FloatMapping(
            pressure=pressure.clone(),
            wss=wss.clone(),
        ),
        "target_measure": measure.clone(),
        "selected_ids": np.arange(resolution, dtype="<i8"),
        "target_pressure": pressure.numpy().astype("<f4", copy=True),
        "target_wss": wss.numpy().astype("<f4", copy=True),
        "target_measure_array": measure.numpy().astype("<f4", copy=True),
        "raw_source_geometry_sha256": "a" * 64,
        "global_inputs_sha256": "b" * 64,
        "batch_order_sha256": "c" * 64,
    }
    return case


def _callback_runtime(events, *, normalize=None, mesh_type=_CallbackMesh):
    class Autocast:
        def __init__(self, precision):
            self.precision = precision

        def __enter__(self):
            events.append(("autocast_enter", self.precision))

        def __exit__(self, *_args):
            events.append(("autocast_exit", self.precision))

    def normalize_output(raw, _target_config, _output_type):
        events.append("normalize")
        return raw if normalize is None else normalize(raw)

    return SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(output_type="mesh"),
        autocast_context=Autocast,
        normalize_output=normalize_output,
        mesh_type=mesh_type,
    )


def _install_fake_loss(monkeypatch, events):
    module = ModuleType("loss")

    class LossCalculator:
        def __init__(
            self,
            target_config,
            *,
            loss_type,
            n_spatial_dims,
            normalize_by_channels,
        ):
            self.target_config = dict(target_config)
            self.loss_type = loss_type
            self.n_spatial_dims = n_spatial_dims
            self.normalize_by_channels = normalize_by_channels
            self.delta = 1.0
            self.total_channels = 4
            self.field_weights = {"pressure": 1.0, "wss": 1.0}

        def __call__(self, prediction, targets, measure):
            events.append("loss")
            pressure = (
                (prediction["pressure"] - targets["pressure"]).square() * measure
            ).sum() / measure.sum()
            wss_components = tuple(
                (
                    (prediction["wss"][:, index] - targets["wss"][:, index]).square()
                    * measure
                ).sum()
                / measure.sum()
                for index in range(3)
            )
            wss = sum(wss_components)
            total = (pressure + wss) / 4.0
            return total, _FloatMapping(
                {
                    "loss/pressure": pressure,
                    "loss/wss": wss,
                    "loss/total": total,
                }
            )

    module.LossCalculator = LossCalculator
    monkeypatch.setitem(sys.modules, "loss", module)


class _LegacyCallbackModel(torch.nn.Module):
    def __init__(self, events, case, *, mutate_case=False, extra_output=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.events = events
        self.case = case
        self.mutate_case = mutate_case
        self.extra_output = extra_output

    def forward(self, _domain):
        self.events.append("model")
        if self.mutate_case:
            self.case["target_measure"].add_(1.0)
        result = _FloatMapping(
            pressure=self.scale * torch.ones(3),
            wss=self.scale * torch.ones(3, 3),
        )
        if self.extra_output:
            result["extra"] = self.scale.reshape(())
        return result


def _callback_package(study, model):
    return study.TrajectoryPackage(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-3),
        {},
    )


def _callback_producer():
    producer = ModuleType("callback_producer")
    producer.TARGET_CONFIG = {"pressure": "scalar", "wss": "vector"}
    producer._tensor_raw_equal = lambda left, right: (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )
    return producer


def test_v3_forward_loss_callback_keeps_update_ownership_in_kernel(
    study,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    events = []
    _install_fake_loss(monkeypatch, events)
    case = _callback_case()
    runtime = _callback_runtime(events)
    producer = _callback_producer()
    model = _LegacyCallbackModel(events, case)
    callback = study._new_v3_forward_loss(
        producer,
        runtime,
        case,
        case_index=0,
        cohort_ordinal=0,
        case_id="run_118",
        historical_start=0,
        precision="bfloat16",
        geometry_path="legacy",
    )
    assert "producer" not in callback.__code__.co_freevars
    assert "runtime" not in callback.__code__.co_freevars
    outputs, loss = callback(_callback_package(study, model))
    assert events == [
        ("autocast_enter", "bfloat16"),
        "model",
        ("autocast_exit", "bfloat16"),
        "normalize",
        "loss",
    ]
    assert tuple(outputs) == ("pressure", "wss")
    assert loss.requires_grad
    assert model.scale.grad is None


def test_v3_forward_loss_callback_rejects_output_and_case_mutation(
    study,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    for mutate_case, extra_output, message in (
        (True, False, "mutated prepared-case"),
        (False, True, "prediction keys or order differ"),
    ):
        events = []
        _install_fake_loss(monkeypatch, events)
        case = _callback_case()
        runtime = _callback_runtime(events)
        producer = _callback_producer()
        model = _LegacyCallbackModel(
            events,
            case,
            mutate_case=mutate_case,
            extra_output=extra_output,
        )
        callback = study._new_v3_forward_loss(
            producer,
            runtime,
            case,
            case_index=0,
            cohort_ordinal=0,
            case_id="run_118",
            historical_start=0,
            precision="bfloat16",
            geometry_path="legacy",
        )
        with pytest.raises(ValueError, match=message):
            callback(_callback_package(study, model))


@pytest.mark.parametrize(
    "corrupt_field",
    ("points", "cells", "centroids", "areas", "normals", "center", "reference"),
)
def test_v3_canonical_callback_authenticates_every_geometry_field(
    study,
    monkeypatch,
    corrupt_field,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    events = []
    _install_fake_loss(monkeypatch, events)
    case = _callback_case()
    runtime = _callback_runtime(events)
    producer = _callback_producer()
    bundle = case["bundle"]
    geometry = SimpleNamespace(
        points=bundle.points,
        cells=bundle.cells,
        centroids=bundle.centroids,
        areas=bundle.areas,
        normals=bundle.normals,
        center=torch.zeros(3),
        reference_length=torch.ones(()),
    )
    producer._canonical_geometry_for_model = lambda *_args: geometry

    class CanonicalModel(_LegacyCallbackModel):
        def encode(self, _domain, *, canonical_source_geometry):
            source = _CallbackMesh(
                canonical_source_geometry.points,
                canonical_source_geometry.cells,
            )
            source.cell_centroids = canonical_source_geometry.centroids
            source.cell_areas = canonical_source_geometry.areas
            source.cell_normals = canonical_source_geometry.normals
            encoded = SimpleNamespace(
                source_mesh=source,
                center=canonical_source_geometry.center,
                reference_length=canonical_source_geometry.reference_length,
            )
            target = (
                "reference_length" if corrupt_field == "reference" else corrupt_field
            )
            if target in {"center", "reference_length"}:
                value = getattr(encoded, target).clone()
                value.reshape(-1)[0] += 1
                setattr(encoded, target, value)
            else:
                attribute = {
                    "centroids": "cell_centroids",
                    "areas": "cell_areas",
                    "normals": "cell_normals",
                }.get(target, target)
                value = getattr(source, attribute).clone()
                value.reshape(-1)[0] += 1
                setattr(source, attribute, value)
            return encoded

        def decode(self, _encoded, _query_mesh):
            return _FloatMapping(
                pressure=self.scale * torch.ones(3),
                wss=self.scale * torch.ones(3, 3),
            )

    model = CanonicalModel(events, case)
    callback = study._new_v3_forward_loss(
        producer,
        runtime,
        case,
        case_index=0,
        cohort_ordinal=0,
        case_id="run_118",
        historical_start=0,
        precision="bfloat16",
        geometry_path="canonical",
    )
    with pytest.raises(ValueError, match="not installed byte-exactly"):
        callback(_callback_package(study, model))


@pytest.mark.parametrize(
    ("value", "version", "message"),
    (
        (
            np.asarray([np.nan], dtype="<f4"),
            (2, 0),
            "payload is non-finite",
        ),
        (
            np.asarray([2], dtype=np.uint8).view(np.bool_),
            (2, 0),
            "Boolean payload is noncanonical",
        ),
        (
            np.arange(3, dtype="<f4"),
            (1, 0),
            "NPY header differs",
        ),
    ),
)
def test_raw_npz_reread_validator_rejects_semantically_invalid_npy_payloads(
    study,
    tmp_path,
    value,
    version,
    message,
) -> None:
    path = (tmp_path / f"invalid_member_{message.split()[0]}.npz").resolve()
    key = "adversarial_array"
    writer = study._RawNpzWriter(path)
    info = zipfile.ZipInfo(
        filename=f"{key}.npy",
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    with writer._archive.open(info, mode="w", force_zip64=True) as member:
        np.lib.format.write_array(
            member,
            value,
            version=version,
            allow_pickle=False,
        )
    writer.close()

    manifest = {
        key: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": value.nbytes,
            "sha256": study._raw_array_sha256(value),
        }
    }
    with pytest.raises(ValueError, match=message):
        study._validate_raw_npz(
            path,
            manifest=manifest,
            expected_order=(key,),
        )


def test_array_group_writes_exact_case_schema_without_partial_validation(
    study,
    tmp_path,
) -> None:
    schema = study._fixed_array_schema(
        "case_control",
        parameter_tensor_count=140,
        learning_rate_count=2,
    )
    values = {
        "selected_cell_ids_int64": np.arange(10_000, dtype="<i8"),
        "target_pressure_float32": np.linspace(
            -1.0,
            1.0,
            10_000,
            dtype="<f4",
        ),
        "target_wss_float32": np.zeros((10_000, 3), dtype="<f4"),
        "target_measure_float32": np.ones(10_000, dtype="<f4"),
    }
    path = (tmp_path / "case_group.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        keys = study._write_array_group(
            writer,
            prefix="case_c00_o00_run_118",
            values=values,
            schema=schema,
        )
        manifest = writer.manifest()
    assert keys == tuple(
        f"case_c00_o00_run_118_{field}" for field in study.CASE_CONTROL_ARRAY_FIELDS
    )
    assert tuple(manifest) == tuple(sorted(keys))
    with zipfile.ZipFile(path, "r") as archive:
        assert tuple(info.filename for info in archive.infolist()) == tuple(
            f"{key}.npy" for key in keys
        )

    bad_values = dict(values)
    bad_values["target_wss_float32"] = np.zeros((10_000, 2), dtype="<f4")
    rejected_path = (tmp_path / "rejected_group.npz").resolve()
    writer = study._RawNpzWriter(rejected_path)
    with pytest.raises(ValueError, match="dtype or shape differs"):
        study._write_array_group(
            writer,
            prefix="case_c00_o00_run_118",
            values=bad_values,
            schema=schema,
        )
    assert writer.manifest() == {}
    writer.close()


def _toy_evidence_payload(study):
    model, optimizer, generators = _package()
    state = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    parameter_names = tuple(name for name, _parameter in model.named_parameters())
    parameter_numels = tuple(
        parameter.numel() for _name, parameter in model.named_parameters()
    )
    model_structure_sha256 = study._model_evidence_structure_sha256(state["model"])
    state_hashes = study._complete_state_hashes(state)
    learning_rates = study._ordered_learning_rates(optimizer)
    record = {
        "schema_version": study.TRAJECTORY_RECORD_SCHEMA_VERSION,
        "step_index": 0,
        "parameter_names": parameter_names,
        "parameter_count": 8,
        "parameter_order_sha256": study.stable_sha256(parameter_names),
        "outputs": (
            {
                "name": "pressure",
                "value_float32": torch.arange(3, dtype=torch.float32),
            },
            {
                "name": "wss",
                "value_float32": torch.arange(
                    9,
                    dtype=torch.float32,
                ).reshape(3, 3),
            },
        ),
        "loss_float64": np.asarray([1.25], dtype="<f8"),
        "gradient_float32": torch.arange(8, dtype=torch.float32),
        "parameter_update_float32": torch.zeros(8, dtype=torch.float32),
        "learning_rates_pre": learning_rates,
        "learning_rates_post": learning_rates,
        "state_sha256": {
            "continuation": state_hashes,
            "pre": state_hashes,
            "post": state_hashes,
        },
        "rng_state": {
            "pre": copy.deepcopy(state["rng"]),
            "post": copy.deepcopy(state["rng"]),
        },
    }
    return (
        state,
        record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    )


def test_transition_and_checkpoint_evidence_stream_complete_states(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        state,
        record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    ) = _toy_evidence_payload(study)
    path = (tmp_path / "transition_evidence.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        transition = study._write_transition_evidence(
            writer,
            prefix="main_test",
            family="main",
            record=record,
            continuation_state=state,
            pre_state=state,
            post_state=state,
            expected_parameter_names=parameter_names,
            expected_parameter_numels=parameter_numels,
            expected_model_structure_sha256=model_structure_sha256,
            learning_rate_count=len(learning_rates),
        )
        checkpoint = study._write_checkpoint_evidence(
            writer,
            prefix="checkpoint_test",
            state=state,
            expected_parameter_names=parameter_names,
            expected_parameter_numels=parameter_numels,
            expected_model_structure_sha256=model_structure_sha256,
            learning_rate_count=len(learning_rates),
        )
        manifest = writer.manifest()
        member_order = writer.member_order()

    assert tuple(transition["array_keys"]) == study.MAIN_ARRAY_FIELDS
    assert transition["pre_state_tree"]["stable_sha256"] == study.stable_sha256(state)
    assert transition["post_state_tree"]["stable_sha256"] == study.stable_sha256(state)
    assert checkpoint["parameter_vector_array_key"].endswith("parameter_vector_float32")
    assert checkpoint["state_tree"]["stable_sha256"] == study.stable_sha256(state)
    validated = study._validate_raw_npz(
        path,
        manifest=manifest,
        expected_order=member_order,
    )
    assert validated["member_count"] == len(member_order)


@pytest.mark.parametrize(
    "defect",
    (
        "parameter_order",
        "boolean_step",
        "boolean_learning_rate_identity",
        "update_delta",
        "learning_rates",
        "continuation",
    ),
)
def test_transition_evidence_rejects_independent_state_binding_defects(
    study,
    tmp_path,
    monkeypatch,
    defect,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        state,
        record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    ) = _toy_evidence_payload(study)
    record = copy.deepcopy(record)
    if defect == "parameter_order":
        record["parameter_names"] = tuple(reversed(parameter_names))
        record["parameter_order_sha256"] = study.stable_sha256(
            record["parameter_names"]
        )
    elif defect == "boolean_step":
        record["step_index"] = False
    elif defect == "boolean_learning_rate_identity":
        record["learning_rates_pre"][0]["member_index"] = False
        record["learning_rates_pre"][0]["group_index"] = False
    elif defect == "update_delta":
        record["parameter_update_float32"].fill_(7.0)
    elif defect == "learning_rates":
        forged = copy.deepcopy(learning_rates)
        for index, item in enumerate(forged):
            item["value_float64"] = 91.0 + index
        record["learning_rates_pre"] = forged
    elif defect == "continuation":
        record["state_sha256"]["continuation"] = {
            key: "0" * 64 for key in ("complete", "model", "optimizer", "rng")
        }
    else:
        raise AssertionError(f"Unknown defect: {defect}")

    path = (tmp_path / f"rejected_transition_{defect}.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        with pytest.raises(ValueError):
            study._write_transition_evidence(
                writer,
                prefix=f"main_rejected_{defect}",
                family="main",
                record=record,
                continuation_state=state,
                pre_state=state,
                post_state=state,
                expected_parameter_names=parameter_names,
                expected_parameter_numels=parameter_numels,
                expected_model_structure_sha256=model_structure_sha256,
                learning_rate_count=len(learning_rates),
            )
        assert writer.manifest() == {}


@pytest.mark.parametrize("family", ("replay", "crossover"))
def test_nonmain_transition_evidence_rejects_arbitrary_learning_rate_metadata(
    study,
    tmp_path,
    monkeypatch,
    family,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        state,
        record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    ) = _toy_evidence_payload(study)
    record["learning_rates_pre"] = "not-learning-rates"
    record["learning_rates_post"] = {"also": "wrong"}
    path = (tmp_path / f"rejected_{family}_learning_rates.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        with pytest.raises(ValueError, match="learning rates differ"):
            study._write_transition_evidence(
                writer,
                prefix=f"{family}_rejected_learning_rates",
                family=family,
                record=record,
                continuation_state=state,
                pre_state=state,
                post_state=state,
                expected_parameter_names=parameter_names,
                expected_parameter_numels=parameter_numels,
                expected_model_structure_sha256=model_structure_sha256,
                learning_rate_count=len(learning_rates),
            )
        assert writer.manifest() == {}


@pytest.mark.parametrize(
    "defect",
    (
        "boolean_schema",
        "parameter_order",
        "missing_buffers",
        "parameter_numels",
    ),
)
def test_checkpoint_evidence_rejects_independent_schema_and_mapping_defects(
    study,
    tmp_path,
    monkeypatch,
    defect,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        state,
        _record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    ) = _toy_evidence_payload(study)
    state = copy.deepcopy(state)
    expected_numels = parameter_numels
    if defect == "boolean_schema":
        state["schema_version"] = True
    elif defect == "parameter_order":
        state["model"]["parameters"] = tuple(reversed(state["model"]["parameters"]))
    elif defect == "missing_buffers":
        del state["model"]["buffers"]
    elif defect == "parameter_numels":
        expected_numels = tuple(reversed(parameter_numels))
    else:
        raise AssertionError(f"Unknown defect: {defect}")

    path = (tmp_path / f"rejected_checkpoint_{defect}.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        with pytest.raises((TypeError, ValueError)):
            study._write_checkpoint_evidence(
                writer,
                prefix=f"checkpoint_rejected_{defect}",
                state=state,
                expected_parameter_names=parameter_names,
                expected_parameter_numels=expected_numels,
                expected_model_structure_sha256=model_structure_sha256,
                learning_rate_count=len(learning_rates),
            )
        assert writer.manifest() == {}


@pytest.mark.parametrize(
    "defect",
    (
        "empty_buffer_inventory",
        "parameter_nonvisible_device",
        "buffer_boolean_stride",
        "buffer_boolean_storage_offset",
        "optimizer_member_partition",
        "short_torch_cpu_rng",
        "nonvisible_explicit_generator",
        "unsupported_optimizer_tensor_dtype",
        "python_rng_tuple_subclass",
    ),
)
def test_checkpoint_evidence_rejects_forged_or_nonreplayable_state_before_write(
    study,
    tmp_path,
    monkeypatch,
    defect,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 3)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        state,
        _record,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rates,
    ) = _toy_evidence_payload(study)
    state = copy.deepcopy(state)
    if defect == "empty_buffer_inventory":
        state["model"]["buffers"] = ()
    elif defect == "parameter_nonvisible_device":
        state["model"]["parameters"][0]["device"] = "cuda:999"
    elif defect == "buffer_boolean_stride":
        state["model"]["buffers"][0]["stride"] = (True,)
    elif defect == "buffer_boolean_storage_offset":
        state["model"]["buffers"][0]["storage_offset"] = False
    elif defect == "optimizer_member_partition":
        members = state["optimizer"]["members"]
        left_name = members[0]["parameter_groups"][0]["parameters"][0]
        right_name = members[1]["parameter_groups"][0]["parameters"][0]
        members[0]["parameter_groups"][0]["parameters"] = (right_name,)
        members[0]["parameter_states"][0]["parameter"] = right_name
        members[1]["parameter_groups"][0]["parameters"] = (left_name,)
        members[1]["parameter_states"][0]["parameter"] = left_name
    elif defect == "short_torch_cpu_rng":
        state["rng"]["torch_cpu"] = torch.ones(1, dtype=torch.uint8)
    elif defect == "nonvisible_explicit_generator":
        state["rng"]["explicit_generators"][0]["device"] = "cuda:999"
    elif defect == "unsupported_optimizer_tensor_dtype":
        tensor_record = next(
            item
            for member in state["optimizer"]["members"]
            for parameter_state in member["parameter_states"]
            for item in parameter_state["values"].values()
            if type(item) is dict and item.get("kind") == study._TENSOR_RECORD_KIND
        )
        tensor_record["value"] = torch.ones(
            tensor_record["value"].shape,
            dtype=torch.uint16,
        )
    elif defect == "python_rng_tuple_subclass":

        class TupleSubclass(tuple):
            pass

        state["rng"]["python"] = TupleSubclass(state["rng"]["python"])
    else:
        raise AssertionError(f"Unknown defect: {defect}")

    path = (tmp_path / f"rejected_forged_state_{defect}.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        with pytest.raises((TypeError, ValueError)):
            study._write_checkpoint_evidence(
                writer,
                prefix=f"checkpoint_rejected_{defect}",
                state=state,
                expected_parameter_names=parameter_names,
                expected_parameter_numels=parameter_numels,
                expected_model_structure_sha256=model_structure_sha256,
                learning_rate_count=len(learning_rates),
            )
        assert writer.manifest() == {}


def test_regime_evidence_uses_exact_global_identities(
    study,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    model, optimizer, generators = _package()
    state = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    state_hashes = study._complete_state_hashes(state)
    parameter_names = tuple(name for name, _parameter in model.named_parameters())
    parameter_numels = tuple(
        parameter.numel() for _name, parameter in model.named_parameters()
    )
    model_structure_sha256 = study._model_evidence_structure_sha256(state["model"])
    main = {
        path: tuple(
            {
                "step": step,
                "case_index": study.STEP_CASE_INDICES[step],
                "transition": {"step_index": step},
                "pre_state": state,
                "post_state": state,
            }
            for step in range(study.TRAJECTORY_STEP_COUNT)
        )
        for path in study.GEOMETRY_PATHS
    }
    checkpoints = {
        path: tuple(
            {
                "schema_version": study.TRAJECTORY_RECORD_SCHEMA_VERSION,
                "step": step,
                "state": state,
                "hashes": state_hashes,
            }
            for step in study.TRAJECTORY_CHECKPOINT_STEPS
        )
        for path in study.GEOMETRY_PATHS
    }
    replays = {
        path: tuple(
            {
                "step": step,
                "case_index": study.STEP_CASE_INDICES[step],
                "transition": {"step_index": step},
                "pre_state": state,
                "post_state": state,
            }
            for step in study.TRAJECTORY_REPLAY_STEPS
        )
        for path in study.GEOMETRY_PATHS
    }
    crossovers = tuple(
        {
            "checkpoint_step": checkpoint_step,
            "case_index": case_index,
            "history_path": history_path,
            "evaluation_path": evaluation_path,
            "transition": {"step_index": checkpoint_step},
            "pre_state": state,
            "post_state": state,
        }
        for checkpoint_step in study.CROSSOVER_STEPS
        for case_index in range(len(study.FIXED_CASE_SPECS))
        for history_path in study.GEOMETRY_PATHS
        for evaluation_path in study.GEOMETRY_PATHS
    )
    writes = []

    def write_transition(_writer, **kwargs):
        writes.append((kwargs["family"], kwargs["prefix"]))
        return {"prefix": kwargs["prefix"]}

    def write_checkpoint(_writer, **kwargs):
        writes.append(("checkpoint", kwargs["prefix"]))
        return {"prefix": kwargs["prefix"]}

    monkeypatch.setattr(study, "_write_transition_evidence", write_transition)
    monkeypatch.setattr(study, "_write_checkpoint_evidence", write_checkpoint)
    monkeypatch.setattr(
        study,
        "_validate_regime_trace_for_evidence",
        lambda *_args, **_kwargs: None,
    )
    result = study._write_regime_evidence(
        object(),
        regime_index=1,
        trace={
            "step_count": study.TRAJECTORY_STEP_COUNT,
            "main": main,
            "checkpoints": checkpoints,
            "replays": replays,
            "crossovers": crossovers,
        },
        expected_parameter_names=parameter_names,
        expected_parameter_numels=parameter_numels,
        expected_model_structure_sha256=model_structure_sha256,
        learning_rate_count=2,
    )

    assert len(result["main_records"]) == 32
    assert len(result["checkpoint_records"]) == 12
    assert len(result["replay_records"]) == 10
    assert len(result["crossover_records"]) == 64
    assert result["main_records"][0]["record_ordinal"] == 32
    assert result["main_records"][-1]["record_ordinal"] == 63
    assert result["checkpoint_records"][0]["record_ordinal"] == 12
    assert result["replay_records"][0]["record_ordinal"] == 10
    assert result["crossover_records"][0]["record_ordinal"] == 64
    assert result["crossover_records"][-1]["record_ordinal"] == 127
    assert len(writes) == 118
    assert len({prefix for _family, prefix in writes}) == len(writes)


def _paired_evidence_trace(study):
    random.seed(3401)
    np.random.seed(3402)
    packages = {
        path: _trajectory_package(study, seed=3403, warmed=True)
        for path in study.GEOMETRY_PATHS
    }

    def package_factory():
        return _trajectory_package(study, seed=3403, warmed=True)

    callbacks = []
    canonical_callbacks = []
    for case_index in range(len(study.FIXED_CASE_SPECS)):
        base = _trajectory_callback(case_index)

        def callback(package, base=base):
            outputs, loss = base(package)
            return {
                "pressure": outputs["pressure"],
                "wss": torch.cat(
                    (outputs["wss"], outputs["pressure"].reshape(-1, 1)),
                    dim=1,
                ),
            }, loss

        callbacks.append(callback)

        def canonical_callback(package, callback=callback):
            outputs, loss = callback(package)
            regularizer = sum(
                parameter.square().sum() for parameter in package.model.parameters()
            )
            return outputs, loss + 1e-4 * regularizer

        canonical_callbacks.append(canonical_callback)
    trace = study._run_paired_microtrajectory(
        packages,
        package_factories={path: package_factory for path in study.GEOMETRY_PATHS},
        case_forward_losses={
            "legacy": tuple(callbacks),
            "canonical": tuple(canonical_callbacks),
        },
    )
    state = trace["checkpoints"]["legacy"][0]["state"]
    parameter_records = state["model"]["parameters"]
    parameter_names = tuple(record["name"] for record in parameter_records)
    parameter_numels = tuple(record["value"].numel() for record in parameter_records)
    parameter_shapes = tuple(
        tuple(record["value"].shape) for record in parameter_records
    )
    model_structure_sha256 = study._model_evidence_structure_sha256(state["model"])
    learning_rate_count = len(
        study._validate_evidence_optimizer_state(
            state["optimizer"],
            expected_parameter_names=parameter_names,
            expected_parameter_shapes=parameter_shapes,
            label="Test initial optimizer",
        )
    )
    return (
        trace,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rate_count,
    )


def test_regime_evidence_preflight_rejects_every_lineage_class(
    study,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 2)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        trace,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rate_count,
    ) = _paired_evidence_trace(study)
    kwargs = {
        "regime_index": 0,
        "expected_parameter_names": parameter_names,
        "expected_parameter_numels": parameter_numels,
        "expected_model_structure_sha256": model_structure_sha256,
        "learning_rate_count": learning_rate_count,
    }
    study._validate_regime_trace_for_evidence(trace, **kwargs)

    broken = copy.deepcopy(trace)
    broken["main"]["legacy"][1]["pre_state"] = copy.deepcopy(
        broken["checkpoints"]["legacy"][0]["state"]
    )
    with pytest.raises(ValueError):
        study._validate_regime_trace_for_evidence(broken, **kwargs)

    broken = copy.deepcopy(trace)
    checkpoint = broken["checkpoints"]["legacy"][3]
    checkpoint["state"] = copy.deepcopy(broken["checkpoints"]["legacy"][2]["state"])
    checkpoint["hashes"] = study._complete_state_hashes(checkpoint["state"])
    with pytest.raises(ValueError, match="checkpoint lineage differs"):
        study._validate_regime_trace_for_evidence(broken, **kwargs)

    broken = copy.deepcopy(trace)
    replay = broken["replays"]["legacy"][3]
    main_zero = broken["main"]["legacy"][0]
    replay["transition"] = copy.deepcopy(main_zero["transition"])
    replay["transition"]["step_index"] = 4
    replay["pre_state"] = copy.deepcopy(main_zero["pre_state"])
    replay["post_state"] = copy.deepcopy(main_zero["post_state"])
    with pytest.raises(ValueError):
        study._validate_regime_trace_for_evidence(broken, **kwargs)

    broken = copy.deepcopy(trace)
    legacy_cell = next(
        cell
        for cell in broken["crossovers"]
        if cell["checkpoint_step"] == 4
        and cell["case_index"] == 0
        and cell["history_path"] == "legacy"
        and cell["evaluation_path"] == "legacy"
    )
    canonical_cell = next(
        cell
        for cell in broken["crossovers"]
        if cell["checkpoint_step"] == 4
        and cell["case_index"] == 0
        and cell["history_path"] == "canonical"
        and cell["evaluation_path"] == "legacy"
    )
    for field in ("transition", "pre_state", "post_state"):
        legacy_cell[field] = copy.deepcopy(canonical_cell[field])
    with pytest.raises(ValueError):
        study._validate_regime_trace_for_evidence(broken, **kwargs)

    broken = copy.deepcopy(trace)
    canonical_t0 = broken["checkpoints"]["canonical"][0]
    canonical_t0["state"]["model"]["parameters"][0]["value"].add_(1.0)
    canonical_t0["hashes"] = study._complete_state_hashes(canonical_t0["state"])
    with pytest.raises(ValueError, match="t0 checkpoint states differ"):
        study._validate_regime_trace_for_evidence(broken, **kwargs)

    broken = copy.deepcopy(trace)
    broken["main"]["legacy"][0]["step"] = False
    with pytest.raises(ValueError, match="main identity differs"):
        study._validate_regime_trace_for_evidence(broken, **kwargs)


def test_regime_evidence_preflight_rejects_late_coordinate_and_shape_defects_before_write(
    study,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(study, "PANEL_RESOLUTION", 2)
    monkeypatch.setattr(study, "EXPECTED_PARAMETER_COUNT", 8)
    (
        trace,
        parameter_names,
        parameter_numels,
        model_structure_sha256,
        learning_rate_count,
    ) = _paired_evidence_trace(study)
    kwargs = {
        "regime_index": 0,
        "expected_parameter_names": parameter_names,
        "expected_parameter_numels": parameter_numels,
        "expected_model_structure_sha256": model_structure_sha256,
        "learning_rate_count": learning_rate_count,
    }
    for defect in ("late_nested_step", "late_gradient_shape"):
        broken = copy.deepcopy(trace)
        last_transition = broken["crossovers"][-1]["transition"]
        if defect == "late_nested_step":
            last_transition["step_index"] -= 1
        else:
            last_transition["gradient_float32"] = last_transition["gradient_float32"][
                :-1
            ].contiguous()
        path = (tmp_path / f"rejected_regime_{defect}.npz").resolve()
        with study._RawNpzWriter(path) as writer:
            with pytest.raises(ValueError):
                study._write_regime_evidence(
                    writer,
                    trace=broken,
                    **kwargs,
                )
            assert writer.manifest() == {}


def test_state_tree_round_trip_restores_a_complete_package(study, tmp_path) -> None:
    random.seed(1101)
    np.random.seed(1102)
    torch.manual_seed(1103)
    model, optimizer, generators = _package()
    state = study.capture_complete_state(
        model,
        optimizer,
        explicit_generators=generators,
    )
    expected_hash = study.stable_sha256(state)

    path = (tmp_path / "complete_state.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        tree = study._StateTreeEncoder(
            writer,
            prefix="checkpoint_000",
            state_kind="complete_state",
        ).encode(state)
        manifest = writer.manifest()
    json.dumps(tree, allow_nan=False, sort_keys=True)

    with np.load(path, allow_pickle=False) as archive:
        decoded, referenced = study._decode_state_tree(
            tree,
            archive,
            expected_prefix="checkpoint_000",
            expected_state_kind="complete_state",
            claimed_keys=set(),
        )
        assert referenced == frozenset(archive.files)
        assert referenced == frozenset(manifest)
    assert study.stable_sha256(decoded) == expected_hash

    hostile_model, hostile_optimizer, hostile_generators = _package()
    with torch.no_grad():
        hostile_model.linear.weight.fill_(91.0)
        hostile_model.persistent_counter.zero_()
        hostile_model.scratch.fill_(17.0)
    hostile_model.eval()
    hostile_model.activation.train()
    hostile_optimizer.optimizers[0].param_groups[0]["lr"] = 0.125
    next(iter(hostile_optimizer.optimizers[0].state.values()))[
        "momentum_buffer"
    ].zero_()
    hostile_generators["probe"].manual_seed(9999)

    study.restore_complete_state(
        hostile_model,
        hostile_optimizer,
        decoded,
        explicit_generators=hostile_generators,
    )
    restored = study.capture_complete_state(
        hostile_model,
        hostile_optimizer,
        explicit_generators=hostile_generators,
    )
    assert study.stable_sha256(restored) == expected_hash


def test_state_tree_preserves_raw_dtypes_containers_and_signed_zero(
    study,
    tmp_path,
) -> None:
    value = {
        "bfloat16": torch.tensor(
            [-0.0, 1.5],
            dtype=torch.bfloat16,
        ),
        "empty_complex": torch.empty((0, 3), dtype=torch.complex64),
        "numpy": np.asarray([-0.0, 2.5], dtype="<f8"),
        "containers": (b"\x00\xff", [None, True, -17, -0.0, "μ"]),
    }
    path = (tmp_path / "typed_state.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        tree = study._StateTreeEncoder(
            writer,
            prefix="typed",
            state_kind="complete_state",
        ).encode(value)

    with np.load(path, allow_pickle=False) as archive:
        decoded, referenced = study._decode_state_tree(
            tree,
            archive,
            expected_prefix="typed",
            expected_state_kind="complete_state",
            claimed_keys=set(),
        )
        assert referenced == frozenset(archive.files)

    assert study.stable_sha256(decoded) == study.stable_sha256(value)
    assert decoded["bfloat16"].dtype is torch.bfloat16
    assert decoded["empty_complex"].shape == (0, 3)
    assert decoded["empty_complex"].dtype is torch.complex64
    assert bool(torch.signbit(decoded["bfloat16"][0]))
    assert bool(np.signbit(decoded["numpy"][0]))
    assert decoded["containers"][1][3].hex() == "-0x0.0p+0"


def test_state_tree_rejects_unsupported_or_nonfinite_values(study, tmp_path) -> None:
    path = (tmp_path / "rejected_state.npz").resolve()
    writer = study._RawNpzWriter(path)
    encoder = study._StateTreeEncoder(
        writer,
        prefix="rejected",
        state_kind="complete_state",
    )

    with pytest.raises(ValueError, match="non-finite"):
        encoder.encode({"bad": torch.tensor(float("nan"))})
    with pytest.raises(ValueError, match="non-finite"):
        encoder.encode({"bad": np.asarray([np.inf], dtype="<f8")})
    with pytest.raises(TypeError, match="Unsupported"):
        encoder.encode({"bad": {1, 2}})

    class FancyTensor(torch.Tensor):
        pass

    subclass = torch.ones(2).as_subclass(FancyTensor)
    with pytest.raises(TypeError, match="Tensor subclass"):
        encoder.encode(subclass)
    with pytest.raises(TypeError, match="mapping keys"):
        encoder.encode({torch.tensor(float("nan")): "bad key"})
    with pytest.raises(ValueError, match="canonical primitive"):
        structured = np.zeros(
            1,
            dtype=[("integer", "<i4"), ("floating", "<f4")],
        )
        encoder.encode(structured)
    with pytest.raises(ValueError, match="canonical primitive"):
        study.stable_sha256(structured)
    writer.close()


def test_state_tree_rejects_noncanonical_strides_views_and_boolean_bytes(
    study,
    tmp_path,
) -> None:
    path = (tmp_path / "noncanonical_state.npz").resolve()
    writer = study._RawNpzWriter(path)
    encoder = study._StateTreeEncoder(
        writer,
        prefix="noncanonical",
        state_kind="complete_state",
    )

    for value in (
        torch.empty_strided((0, 3), (100, 7)),
        torch.empty_strided((1, 3), (99, 1)),
        torch.ones(2, dtype=torch.complex64).conj(),
        torch._neg_view(torch.ones(2)),
    ):
        with pytest.raises(ValueError, match="canonical contiguous"):
            encoder.encode(value)

    torch_bad_bool = torch.asarray([2], dtype=torch.uint8).view(torch.bool)
    with pytest.raises(ValueError, match="Boolean storage"):
        encoder.encode(torch_bad_bool)

    numpy_bad_bool = np.asarray([2], dtype=np.uint8).view(np.bool_)
    with pytest.raises(ValueError, match="Boolean storage"):
        encoder.encode(numpy_bad_bool)
    writer.close()


def test_state_tree_decoder_rejects_malformed_or_ambiguous_trees(
    study,
    tmp_path,
) -> None:
    path = (tmp_path / "malformed_state.npz").resolve()
    with study._RawNpzWriter(path) as writer:
        tree = study._StateTreeEncoder(
            writer,
            prefix="malformed",
            state_kind="rng_state",
        ).encode((torch.arange(2, dtype=torch.int16), np.ones(3, dtype="<f4")))

    with np.load(path, allow_pickle=False) as archive:
        missing_arrays = {
            key: archive[key]
            for key in archive.files
            if key != "malformed_leaf_000000_bytes"
        }
        with pytest.raises(ValueError, match="array is absent"):
            study._decode_state_tree(
                tree,
                missing_arrays,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        duplicate_reference = copy.deepcopy(tree)
        duplicate_reference["root"]["items"][1]["array_key"] = duplicate_reference[
            "root"
        ]["items"][0]["array_key"]
        with pytest.raises(ValueError, match="next contiguous leaf"):
            study._decode_state_tree(
                duplicate_reference,
                archive,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        wrong_shape = copy.deepcopy(tree)
        wrong_shape["root"]["items"][0]["shape"] = [3]
        with pytest.raises(ValueError, match="byte count differs"):
            study._decode_state_tree(
                wrong_shape,
                archive,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        malformed_dtype = copy.deepcopy(tree)
        malformed_dtype["root"]["items"][0]["dtype"] = []
        with pytest.raises(ValueError, match="Torch dtype is malformed"):
            study._decode_state_tree(
                malformed_dtype,
                archive,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        extra_field = copy.deepcopy(tree)
        extra_field["root"]["unexpected"] = True
        with pytest.raises(ValueError, match="tuple node keys differ"):
            study._decode_state_tree(
                extra_field,
                archive,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        with pytest.raises(ValueError, match="kind differs"):
            study._decode_state_tree(
                tree,
                archive,
                expected_prefix="malformed",
                expected_state_kind="complete_state",
                claimed_keys=set(),
            )

        with pytest.raises(ValueError, match="next contiguous leaf"):
            study._decode_state_tree(
                tree,
                archive,
                expected_prefix="wrong_prefix",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )

        claimed_keys: set[str] = set()
        study._decode_state_tree(
            tree,
            archive,
            expected_prefix="malformed",
            expected_state_kind="rng_state",
            claimed_keys=claimed_keys,
        )
        with pytest.raises(ValueError, match="already claimed"):
            study._decode_state_tree(
                tree,
                archive,
                expected_prefix="malformed",
                expected_state_kind="rng_state",
                claimed_keys=claimed_keys,
            )

    reversed_mapping = {
        "schema_version": 1,
        "state_kind": "rng_state",
        "stable_sha256": study.stable_sha256({"a": None, "b": None}),
        "root": {
            "kind": "mapping",
            "items": [
                {
                    "key": {"kind": "str", "value": "b"},
                    "value": {"kind": "none"},
                },
                {
                    "key": {"kind": "str", "value": "a"},
                    "value": {"kind": "none"},
                },
            ],
        },
    }
    with pytest.raises(ValueError, match="not strictly ordered"):
        study._decode_state_tree(
            reversed_mapping,
            {},
            expected_prefix="no_leaves",
            expected_state_kind="rng_state",
            claimed_keys=set(),
        )

    bad_bool_envelope = {
        "schema_version": 1,
        "state_kind": "rng_state",
        "stable_sha256": "0" * 64,
        "root": {
            "kind": "torch_tensor",
            "dtype": "torch.bool",
            "shape": [1],
            "array_key": "bad_bool_leaf_000000_bytes",
        },
    }
    with pytest.raises(ValueError, match="Boolean storage"):
        study._decode_state_tree(
            bad_bool_envelope,
            {"bad_bool_leaf_000000_bytes": np.asarray([2], dtype=np.uint8)},
            expected_prefix="bad_bool",
            expected_state_kind="rng_state",
            claimed_keys=set(),
        )

    duplicate_mapping_key = {
        "schema_version": 1,
        "state_kind": "rng_state",
        "stable_sha256": "0" * 64,
        "root": {
            "kind": "mapping",
            "items": [
                {
                    "key": {"kind": "str", "value": "same"},
                    "value": {"kind": "none"},
                },
                {
                    "key": {"kind": "str", "value": "same"},
                    "value": {"kind": "none"},
                },
            ],
        },
    }
    with pytest.raises(ValueError, match="mapping key is duplicated"):
        study._decode_state_tree(
            duplicate_mapping_key,
            {},
            expected_prefix="no_leaves",
            expected_state_kind="rng_state",
            claimed_keys=set(),
        )

    for malformed_scalar, message in (
        ({"kind": "int", "value_decimal": "01"}, "integer is not canonical"),
        ({"kind": "float", "value_hex": "0x1p+999999"}, "float is invalid"),
        ({"kind": "bool", "value": 1}, "Boolean is malformed"),
    ):
        envelope = {
            "schema_version": 1,
            "state_kind": "rng_state",
            "stable_sha256": "0" * 64,
            "root": malformed_scalar,
        }
        with pytest.raises(ValueError, match=message):
            study._decode_state_tree(
                envelope,
                {},
                expected_prefix="no_leaves",
                expected_state_kind="rng_state",
                claimed_keys=set(),
            )
