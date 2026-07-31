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

"""Numerical correctness for ``LossCalculator``.

These tests pin two things:

1. The total produced by ``LossCalculator`` for a flat-tensor input
   matches an explicit per-field reference formula
   (``sum(per_field_losses) / total_channels``, where per-field is mean
   Huber for scalars and per-component sum of mean Huber for vectors)
   when ``field_weights`` is ``None`` (or all 1.0).
2. ``field_weights`` correctly multiplies each per-field loss before
   summation; unknown field names raise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from loss import DEFAULT_HUBER_DELTA, LossCalculator
from output_normalize import split_concat_by_target
from tensordict import TensorDict
from utils import FieldType, field_dim


def _reference_total_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_config: dict[str, FieldType],
    n_spatial_dims: int = 3,
) -> torch.Tensor:
    """Per-field-and-summed Huber total for a concatenated ``(B, N, C)`` tensor.

    - Scalar fields: ``F.huber_loss(pred_field, target_field, reduction='mean')``
      with the channel dim squeezed off (so input is ``(B, N)``).
    - Vector fields: sum across components of per-component mean Huber.
    - Total: ``sum(per_field) / total_channels``.

    Used as the reference value that :class:`~loss.LossCalculator` is
    pinned to by :class:`TestReferenceTotal`.
    """
    total_channels = sum(field_dim(t, n_spatial_dims) for t in target_config.values())
    total = torch.zeros((), dtype=pred.dtype, device=pred.device)
    idx = 0
    for ftype in target_config.values():
        dim = field_dim(ftype, n_spatial_dims)
        p = pred[..., idx : idx + dim]
        t = target[..., idx : idx + dim]
        if ftype == "scalar":
            total = total + F.huber_loss(
                p.squeeze(-1),
                t.squeeze(-1),
                reduction="mean",
                delta=DEFAULT_HUBER_DELTA,
            )
        else:  # vector: per-component mean, summed
            for i in range(dim):
                total = total + F.huber_loss(
                    p[..., i], t[..., i], reduction="mean", delta=DEFAULT_HUBER_DELTA
                )
        idx += dim
    return total / total_channels


def _make_td(leaves: dict[str, torch.Tensor]) -> TensorDict:
    """Wrap a flat ``{name: tensor}`` dict in a TensorDict.

    Uses the leading dims shared by every leaf as ``batch_size`` -- i.e.,
    ``shape[:min_ndim]`` of any leaf. For the test fixtures here, scalars
    are ``(B, N)`` and vectors are ``(B, N, D)`` (or ``(N,)`` / ``(N, D)``
    in the mesh-style no-batch case), so the scalar's shape is always a
    prefix of every vector's.
    """
    min_ndim = min(v.ndim for v in leaves.values())
    first = next(iter(leaves.values()))
    return TensorDict(leaves, batch_size=first.shape[:min_ndim], device=first.device)


### ---------------------------------------------------------------------------
### Reference total (no field_weights)
### ---------------------------------------------------------------------------


class TestReferenceTotal:
    """``LossCalculator`` reproduces :func:`_reference_total_huber` for unit weights."""

    @pytest.mark.parametrize(
        "target_config",
        [
            {"pressure": "scalar", "wss": "vector"},
            {"velocity": "vector", "pressure": "scalar", "nut": "scalar"},
            {"phi": "scalar"},
            {"u": "vector"},
            {
                "pressure": "scalar",
                "temperature": "scalar",
                "density": "scalar",
                "velocity": "vector",
                "tau_wall": "vector",
            },
        ],
        ids=["pressure_wss", "vel_p_nut", "scalar_only", "vector_only", "highlift"],
    )
    def test_huber_total_matches_reference(self, target_config):
        """Total Huber matches the reference per-field-and-summed formula."""
        ### Pin RNG so the comparison is deterministic.
        torch.manual_seed(123)

        total_channels = sum(field_dim(t) for t in target_config.values())

        ### Concatenated (1, N, C) tensors -- the flat-tensor loss interface.
        pred = torch.randn(1, 50, total_channels)
        target = torch.randn(1, 50, total_channels)

        ref_total = _reference_total_huber(pred, target, target_config)

        ### TensorDict-based loss with default (no) weights.
        loss = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights=None,
        )
        pred_td = split_concat_by_target(pred, target_config)
        target_td = split_concat_by_target(target, target_config)
        total, loss_dict = loss(pred_td, target_td)

        ### Bit-exact (modulo floating-point reductions in the same order).
        assert torch.allclose(total, ref_total, atol=1e-7, rtol=1e-6), (
            f"got={float(total):.10f} reference={float(ref_total):.10f}"
        )
        assert "loss/total" in loss_dict
        assert torch.allclose(loss_dict["loss/total"], ref_total, atol=1e-7, rtol=1e-6)

    @pytest.mark.parametrize("loss_type", ["huber", "mse"])
    def test_per_field_keys_match_target_config(self, loss_type):
        """Per field keys match target config."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_td = _make_td(
            {"pressure": torch.randn(1, 30), "wss": torch.randn(1, 30, 3)}
        )
        target_td = _make_td(
            {"pressure": torch.randn(1, 30), "wss": torch.randn(1, 30, 3)}
        )
        lc = LossCalculator(target_config=target_config, loss_type=loss_type)
        _, ldict = lc(pred_td, target_td)
        ### Per-field entries plus loss/total. ``ldict`` is a 0-D TensorDict;
        ### iterating it directly raises StopIteration (no batch dim to walk),
        ### so we materialise the keys explicitly.
        assert set(ldict.keys()) == {"loss/pressure", "loss/wss", "loss/total"}


class TestTargetQuadratureMeasure:
    """Quadrature weighting is invariant to unequal local refinement."""

    @pytest.mark.parametrize(
        ("loss_type", "expected"),
        [
            ("huber", 0.75),
            ("mse", 2.0),
            ("relative_mse", 2.0),
            ("relative_l2", 2.0**0.5),
        ],
    )
    def test_scalar_loss_matches_analytic_refinement_invariant_value(
        self, loss_type, expected
    ):
        ### Two equal-area regions, then refine only the zero-error region
        ### into three cells. Its three 1/3 measures must not triple-count it.
        coarse_target = _make_td({"pressure": torch.ones(2)})
        coarse_pred = _make_td({"pressure": torch.tensor([1.0, 3.0])})
        coarse_measure = torch.tensor([1.0, 1.0])

        refined_target = _make_td({"pressure": torch.ones(4)})
        refined_pred = _make_td({"pressure": torch.tensor([1.0, 1.0, 1.0, 3.0])})
        refined_measure = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0])

        calculator = LossCalculator({"pressure": "scalar"}, loss_type=loss_type)
        coarse, _ = calculator(coarse_pred, coarse_target, coarse_measure)
        refined, _ = calculator(refined_pred, refined_target, refined_measure)

        assert coarse.item() == pytest.approx(expected)
        assert torch.allclose(refined, coarse)

    def test_vector_component_reductions_normalize_by_measure(self):
        coarse_target = TensorDict({"velocity": torch.ones(2, 2)}, batch_size=[])
        coarse_pred = TensorDict(
            {"velocity": torch.tensor([[1.0, 1.0], [3.0, 5.0]])},
            batch_size=[],
        )
        refined_target = TensorDict({"velocity": torch.ones(4, 2)}, batch_size=[])
        refined_pred = TensorDict(
            {
                "velocity": torch.tensor(
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [3.0, 5.0]]
                )
            },
            batch_size=[],
        )
        calculator = LossCalculator(
            {"velocity": "vector"}, loss_type="mse", n_spatial_dims=2
        )

        coarse, _ = calculator(coarse_pred, coarse_target, torch.tensor([1.0, 1.0]))
        refined, _ = calculator(
            refined_pred,
            refined_target,
            torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0]),
        )

        ### Per-component weighted MSEs are 2 and 8; channel normalization
        ### divides their sum by two.
        assert coarse.item() == pytest.approx(5.0)
        assert torch.allclose(refined, coarse)

    def test_explicit_none_is_bitwise_identical_to_legacy_call(self):
        torch.manual_seed(7)
        pred = _make_td({"pressure": torch.randn(1, 17), "wss": torch.randn(1, 17, 3)})
        target = _make_td(
            {"pressure": torch.randn(1, 17), "wss": torch.randn(1, 17, 3)}
        )
        calculator = LossCalculator(
            {"pressure": "scalar", "wss": "vector"}, loss_type="huber"
        )

        legacy_total, legacy_fields = calculator(pred, target)
        none_total, none_fields = calculator(pred, target, None)

        assert torch.equal(none_total, legacy_total)
        for key in legacy_fields.keys():
            assert torch.equal(none_fields[key], legacy_fields[key])

    def test_misaligned_measure_raises(self):
        pred = _make_td({"pressure": torch.ones(3)})
        calculator = LossCalculator({"pressure": "scalar"}, loss_type="mse")
        with pytest.raises(ValueError, match="one value per target query"):
            calculator(pred, pred, torch.ones(2))

    @pytest.mark.parametrize(
        "measure",
        [
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, -2.0]),
            torch.tensor([1.0, float("nan")]),
            torch.tensor([1.0, float("inf")]),
        ],
    )
    def test_nonpositive_or_nonfinite_measure_raises(self, measure):
        pred = _make_td({"pressure": torch.ones(2)})
        calculator = LossCalculator({"pressure": "scalar"}, loss_type="mse")
        with pytest.raises(ValueError, match="finite positive"):
            calculator(pred, pred, measure)

    @pytest.mark.parametrize("loss_type", ["relative_mse", "relative_l2"])
    @pytest.mark.parametrize(
        ("field_name", "field_type", "pred"),
        [
            ("pressure", "scalar", torch.tensor([1e-5, -1e-5])),
            (
                "wss",
                "vector",
                torch.tensor([[1e-5, -2e-5], [-1e-5, 2e-5]]),
            ),
        ],
    )
    def test_relative_loss_is_invariant_to_uniform_measure_scale_near_zero(
        self, loss_type, field_name, field_type, pred
    ):
        """A fixed epsilon must not turn measure units into a loss input."""
        target = torch.zeros_like(pred)
        calculator = LossCalculator(
            {field_name: field_type},
            loss_type=loss_type,
            n_spatial_dims=2,
        )
        pred_td = _make_td({field_name: pred})
        target_td = _make_td({field_name: target})

        base, _ = calculator(pred_td, target_td, torch.ones(2))
        rescaled, _ = calculator(pred_td, target_td, 880.0 * torch.ones(2))

        assert torch.equal(rescaled, base)


### ---------------------------------------------------------------------------
### field_weights correctness
### ---------------------------------------------------------------------------


class TestFieldWeights:
    """Tests for field weights."""

    def test_uniform_weights_match_no_weights(self):
        """field_weights={...: 1.0} is a no-op vs default (None)."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_td = _make_td(
            {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)}
        )
        target_td = _make_td(
            {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)}
        )
        no_weights = LossCalculator(target_config=target_config, loss_type="huber")
        unit_weights = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights={"pressure": 1.0, "wss": 1.0},
        )
        a, _ = no_weights(pred_td, target_td)
        b, _ = unit_weights(pred_td, target_td)
        assert torch.allclose(a, b, atol=1e-7)

    def test_single_field_weight_scales_linearly(self):
        """Weighting one field by k scales its per-field contribution by k."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_td = _make_td(
            {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)}
        )
        target_td = _make_td(
            {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)}
        )
        baseline = LossCalculator(target_config=target_config, loss_type="huber")
        boosted = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights={"pressure": 1.0, "wss": 100.0},
        )
        _, base_dict = baseline(pred_td, target_td)
        _, boost_dict = boosted(pred_td, target_td)
        ### Per-field loss values are weighted in the dict (loss_dict shows
        ### the WEIGHTED per-field loss).
        assert torch.allclose(
            boost_dict["loss/wss"], 100.0 * base_dict["loss/wss"], atol=1e-6
        )
        assert torch.allclose(boost_dict["loss/pressure"], base_dict["loss/pressure"])

    def test_total_weighted_matches_explicit_sum(self):
        """Total = sum(weighted per-field losses) / total_channels."""
        torch.manual_seed(42)
        target_config = {"a": "scalar", "b": "vector", "c": "scalar"}
        n_pts = 40
        pred_td = _make_td(
            {
                "a": torch.randn(1, n_pts),
                "b": torch.randn(1, n_pts, 3),
                "c": torch.randn(1, n_pts),
            }
        )
        target_td = pred_td.apply(torch.randn_like)
        weights = {"a": 0.5, "b": 2.0, "c": 1.5}
        lc = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights=weights,
        )
        total, dct = lc(pred_td, target_td)
        expected = (dct["loss/a"] + dct["loss/b"] + dct["loss/c"]) / lc.total_channels
        assert torch.allclose(total, expected, atol=1e-7)

    def test_unknown_field_name_raises(self):
        """Unknown field name raises."""
        with pytest.raises(ValueError, match="references unknown fields"):
            LossCalculator(
                target_config={"pressure": "scalar"},
                loss_type="huber",
                field_weights={"not_a_field": 2.0},
            )


### ---------------------------------------------------------------------------
### Shape-agnostic invariance: same loss for (1, N, C) and (N, C) inputs
### ---------------------------------------------------------------------------


class TestShapeAgnostic:
    """Tests for shape agnostic."""

    def test_with_or_without_batch_dim(self):
        """With or without batch dim."""
        torch.manual_seed(7)
        target_config = {"pressure": "scalar", "wss": "vector"}
        ### Mesh-input style (no batch dim) -- batch_size=[80].
        pred_no_batch = _make_td(
            {"pressure": torch.randn(80), "wss": torch.randn(80, 3)}
        )
        target_no_batch = _make_td(
            {"pressure": torch.randn(80), "wss": torch.randn(80, 3)}
        )
        ### Same data with a leading batch dim of 1 -- batch_size=[1, 80].
        ### TensorDict.unsqueeze grows the batch_size and every leaf in lock-step.
        pred_with_batch = pred_no_batch.unsqueeze(0)
        target_with_batch = target_no_batch.unsqueeze(0)

        lc = LossCalculator(target_config=target_config, loss_type="huber")
        loss_no_batch, _ = lc(pred_no_batch, target_no_batch)
        loss_with_batch, _ = lc(pred_with_batch, target_with_batch)
        assert torch.allclose(loss_no_batch, loss_with_batch, atol=1e-7)
