"""Tests for aphex.inputspec — single/multi input parsing and building."""
from __future__ import annotations

import pytest
import torch

from aphex.inputspec import InputSpec, TensorSpec

# ── parsing ───────────────────────────────────────────────────────────────────


def test_parse_single_plain_shape() -> None:
    spec = InputSpec.parse("3,224,224")
    assert spec.is_single
    assert spec.names == ["input"]
    assert spec.primary_shape == [3, 224, 224]
    assert spec.tensors[0].dtype == "float"


def test_parse_multi_input() -> None:
    spec = InputSpec.parse("input_ids:128;attention_mask:128")
    assert not spec.is_single
    assert spec.names == ["input_ids", "attention_mask"]
    assert spec.tensors[0].shape == (128,)


def test_parse_multi_with_dtype() -> None:
    spec = InputSpec.parse("input_ids:128:long;attention_mask:128:long")
    assert all(t.dtype == "long" for t in spec.tensors)


def test_parse_single_named_with_dims() -> None:
    spec = InputSpec.parse("pixel_values:3,224,224")
    assert spec.names == ["pixel_values"]
    assert spec.primary_shape == [3, 224, 224]


def test_parse_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        InputSpec.parse("ids:128:int4")


def test_parse_rejects_bad_dims() -> None:
    with pytest.raises(ValueError):
        InputSpec.parse("ids:abc:long")


def test_parse_empty_raises() -> None:
    with pytest.raises(ValueError):
        InputSpec.parse("   ")


# ── dynamic axes / names ──────────────────────────────────────────────────────


def test_dynamic_axes_cover_all_inputs_and_output() -> None:
    spec = InputSpec.parse("a:4;b:6")
    axes = spec.dynamic_axes()
    assert set(axes) == {"a", "b", "output"}
    assert all(ax == {0: "batch"} for ax in axes.values())


# ── building inputs ───────────────────────────────────────────────────────────


def test_build_single_returns_tensor() -> None:
    spec = InputSpec.single([4])
    out = spec.build(2)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4)


def test_build_multi_returns_tuple() -> None:
    spec = InputSpec.parse("a:4;b:6")
    out = spec.build(3)
    assert isinstance(out, tuple)
    assert [t.shape for t in out] == [(3, 4), (3, 6)]


def test_build_long_dtype_respected() -> None:
    spec = InputSpec.parse("ids:8:long")
    out = spec.build(2, vocab=50)
    assert out.dtype == torch.long
    assert int(out.max()) < 50


def test_build_single_tiles_representative_sample() -> None:
    spec = InputSpec.single([4])
    sample = torch.full((4,), 9.0)
    out = spec.build(3, samples=[sample])
    assert out.shape == (3, 4)
    assert bool((out == 9.0).all())


def test_from_shape_long_when_vocab() -> None:
    assert InputSpec.from_shape([8], vocab=100).tensors[0].dtype == "long"
    assert InputSpec.from_shape([8], vocab=None).tensors[0].dtype == "float"


def test_tensorspec_random_shapes() -> None:
    ts = TensorSpec("x", (5,), "float")
    t = ts.random(4, torch.float32, "cpu", None)
    assert t.shape == (4, 5)
    assert t.dtype == torch.float32
