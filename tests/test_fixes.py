"""Targeted tests for the fixes applied in the review batch.

Covers:
- evaluator._torch_load gating via APHEX_TRUST_PICKLE
- evaluator logger surfaces failures instead of swallowing them
- evaluator._fill_pytorch shares a single deepcopy
- cli._parse_shape / _parse_batch_sizes input validation
- cli._resolve_calibration warns when reusing eval inputs
- converter.read_deployment_yaml schema validation
- cloud.remote._cleanup_remote refuses unscoped paths
- registry lazy-load behavior
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from infermap import registry
from infermap.converter import read_deployment_yaml


# ── evaluator._torch_load ─────────────────────────────────────────────────────


def test_torch_load_loads_state_dict_safely(tmp_path: Path) -> None:
    from infermap.evaluator import _torch_load

    p = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(3)}, p)

    raw = _torch_load(p)
    assert "w" in raw
    assert torch.equal(raw["w"], torch.zeros(3))


def test_torch_load_rejects_pickle_by_default(tmp_path: Path, monkeypatch) -> None:
    """If weights_only=True fails and APHEX_TRUST_PICKLE isn't set, we error clearly."""
    from infermap.evaluator import _torch_load

    monkeypatch.delenv("APHEX_TRUST_PICKLE", raising=False)

    # Save a full nn.Module (requires pickle/weights_only=False to load).
    p = tmp_path / "model.pt"
    model = torch.nn.Linear(4, 2)
    torch.save(model, p)

    with pytest.raises(RuntimeError, match="APHEX_TRUST_PICKLE"):
        _torch_load(p)


def test_torch_load_allows_pickle_with_env(tmp_path: Path, monkeypatch) -> None:
    from infermap.evaluator import _torch_load

    monkeypatch.setenv("APHEX_TRUST_PICKLE", "1")

    p = tmp_path / "model.pt"
    model = torch.nn.Linear(4, 2)
    torch.save(model, p)

    loaded = _torch_load(p)
    assert isinstance(loaded, torch.nn.Linear)


# ── evaluator: failures are logged ────────────────────────────────────────────


def test_sklearn_baseline_failure_is_logged(caplog) -> None:
    """When predict() raises, _fill_sklearn used to silently return; now it logs."""
    from infermap.evaluator import EvalDataset, _fill_sklearn

    class Boom:
        def predict(self, X):  # type: ignore[no-untyped-def]
            raise RuntimeError("nope")

    ds = EvalDataset(
        inputs=[torch.zeros(1, 3)],
        labels=[0],
        metric="accuracy",
        task="classification",
    )

    with caplog.at_level(logging.WARNING, logger="infermap.evaluator"):
        _fill_sklearn(results=[], model=Boom(), eval_dataset=ds)

    assert any("sklearn baseline predict failed" in r.message for r in caplog.records)


# ── CLI input validation ──────────────────────────────────────────────────────


def test_parse_shape_rejects_garbage() -> None:
    import typer
    from infermap.cli import _parse_shape

    with pytest.raises(typer.Exit):
        _parse_shape("3,abc,224")


def test_parse_shape_rejects_zero() -> None:
    import typer
    from infermap.cli import _parse_shape

    with pytest.raises(typer.Exit):
        _parse_shape("3,0,224")


def test_parse_batch_sizes_rejects_negative() -> None:
    import typer
    from infermap.cli import _parse_batch_sizes

    with pytest.raises(typer.Exit):
        _parse_batch_sizes("1,-2,4")


def test_parse_shape_accepts_valid() -> None:
    from infermap.cli import _parse_shape

    assert _parse_shape("3,224,224") == [3, 224, 224]
    assert _parse_shape("16") == [16]


# ── CLI calibration leak warning ──────────────────────────────────────────────


def test_resolve_calibration_warns_when_falling_back(capsys) -> None:
    """If no calibration data is given but eval data exists, _resolve_calibration
    should reuse eval inputs and emit a visible warning."""
    from infermap.cli import _resolve_calibration
    from infermap.evaluator import EvalDataset

    eval_ds = EvalDataset(
        inputs=[torch.zeros(1, 3), torch.ones(1, 3)],
        labels=[0, 1],
        metric="accuracy",
        task="classification",
    )

    calib = _resolve_calibration(None, [3], eval_ds)
    assert calib is not None
    assert len(calib) == 2

    captured = capsys.readouterr()
    # The warning is printed on stderr by Rich.
    assert "leak" in (captured.err + captured.out).lower()


def test_resolve_calibration_no_data_returns_none() -> None:
    from infermap.cli import _resolve_calibration

    assert _resolve_calibration(None, [3], None) is None


# ── deployment.yaml schema validation ─────────────────────────────────────────


def test_read_deployment_yaml_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("# nothing here\n")
    with pytest.raises(ValueError, match="empty"):
        read_deployment_yaml(p)


def test_read_deployment_yaml_rejects_unrecognised_top_level(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("totally_made_up: 1\nother: 2\n")
    with pytest.raises(ValueError, match="does not look like a deployment.yaml"):
        read_deployment_yaml(p)


def test_read_deployment_yaml_warns_on_unknown_keys(tmp_path: Path, caplog) -> None:
    p = tmp_path / "mixed.yaml"
    p.write_text(
        'recommendation:\n  backend: "onnx_int8_cpu"\n'
        'mystery_section:\n  k: "v"\n'
    )
    with caplog.at_level(logging.WARNING, logger="infermap.converter"):
        result = read_deployment_yaml(p)
    assert "recommendation" in result
    assert any("unknown top-level" in r.message for r in caplog.records)


def test_read_deployment_yaml_strict_raises_on_unknown(tmp_path: Path) -> None:
    p = tmp_path / "mixed.yaml"
    p.write_text(
        'recommendation:\n  backend: "onnx_int8_cpu"\n'
        'mystery_section:\n  k: "v"\n'
    )
    with pytest.raises(ValueError, match="unknown top-level"):
        read_deployment_yaml(p, strict=True)


def test_read_deployment_yaml_rejects_malformed_line(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("recommendation:\n  backend: ok\nthis_line_has_no_colon\n")
    with pytest.raises(ValueError, match="malformed line"):
        read_deployment_yaml(p)


# ── cloud.remote cleanup guard ────────────────────────────────────────────────


def test_cleanup_remote_refuses_unscoped_paths(monkeypatch, caplog) -> None:
    """_cleanup_remote must refuse to rm -rf anything outside /tmp/aphex-*."""
    from infermap.cloud import remote

    called: list[list[str]] = []

    def fake_run(cmd, check=True):
        called.append(cmd)

    monkeypatch.setattr(remote, "_run", fake_run)

    with caplog.at_level(logging.ERROR, logger="infermap.cloud.remote"):
        remote._cleanup_remote("host", "/")
        remote._cleanup_remote("host", "/etc")
        remote._cleanup_remote("host", "/tmp/something-else")

    assert called == [], "no ssh rm should have been issued"
    assert any("refusing" in r.message for r in caplog.records)


def test_cleanup_remote_allows_scoped_paths(monkeypatch) -> None:
    from infermap.cloud import remote

    called: list[list[str]] = []
    monkeypatch.setattr(remote, "_run", lambda cmd, check=True: called.append(cmd))

    remote._cleanup_remote("host", "/tmp/aphex-abcd1234")

    assert called and called[0][:2] == ["ssh", "host"]
    assert "rm -rf /tmp/aphex-abcd1234" in called[0][2]


# ── registry: explicit / lazy default plugin loading ─────────────────────────


def test_load_default_plugins_is_idempotent() -> None:
    registry.reset()
    registry.load_default_plugins()
    n = len(registry._plugins)
    registry.load_default_plugins()
    assert len(registry._plugins) == n
    # Restore for downstream tests.
    registry.load_default_plugins()


def test_get_plugin_triggers_lazy_load(tmp_path: Path) -> None:
    registry.reset()
    p = tmp_path / "m.pt"
    torch.save({"w": torch.zeros(2)}, p)
    plugin = registry.get_plugin(p)
    assert plugin is not None
    # restore
    if not registry._defaults_loaded:
        registry.load_default_plugins()


# ── Blocker 1: LLM accuracy fence-off ────────────────────────────────────────


def test_fill_accuracy_drop_skips_llm_family(caplog) -> None:
    """Generative models must not get a cosine-similarity accuracy proxy."""
    from infermap.evaluator import EvalDataset, fill_accuracy_drop
    from infermap.inspector import ModelInfo

    info = ModelInfo(
        framework="pytorch",
        family="llm",
        parameters=1000,
        trainable_parameters=1000,
        estimated_memory_fp32_gb=0.0,
        estimated_memory_fp16_gb=0.0,
        model_path=None,
    )
    ds = EvalDataset(
        inputs=[torch.zeros(1, 3)], labels=[0],
        metric="accuracy", task="classification",
    )

    class _DummyResult:
        ok = True
        accuracy_drop = None

        class candidate:  # noqa: N801
            backend = "pytorch_fp16"

    r = _DummyResult()
    results = [r]

    with caplog.at_level(logging.WARNING, logger="infermap.evaluator"):
        fill_accuracy_drop(results, torch.nn.Linear(3, 1), info, ds)

    # Skipped: accuracy_drop must remain None.
    assert r.accuracy_drop is None
    assert any("generation quality" in rec.message for rec in caplog.records)


def test_fill_accuracy_drop_skips_when_task_is_generation(caplog) -> None:
    from infermap.evaluator import EvalDataset, fill_accuracy_drop
    from infermap.inspector import ModelInfo

    info = ModelInfo(
        framework="pytorch", family="transformer",
        parameters=10, trainable_parameters=10,
        estimated_memory_fp32_gb=0.0, estimated_memory_fp16_gb=0.0,
        model_path=None,
    )
    ds = EvalDataset(
        inputs=[torch.zeros(1)], labels=[0],
        metric="perplexity", task="generation",
    )

    with caplog.at_level(logging.WARNING, logger="infermap.evaluator"):
        fill_accuracy_drop(results=[], model=torch.nn.Linear(1, 1),
                           info=info, eval_dataset=ds)
    assert any("generation quality" in r.message for r in caplog.records)


def test_benchmark_skips_accuracy_proxy_for_llm(caplog) -> None:
    """benchmark._is_generative_family gates the cosine proxy too."""
    from infermap.benchmark import _is_generative_family
    from infermap.inspector import ModelInfo

    def _mk(family: str) -> ModelInfo:
        return ModelInfo(
            framework="pytorch", family=family,
            parameters=1, trainable_parameters=1,
            estimated_memory_fp32_gb=0.0, estimated_memory_fp16_gb=0.0,
            model_path=None,
        )

    assert _is_generative_family(_mk("LLM"))
    assert _is_generative_family(_mk("seq2seq"))
    assert not _is_generative_family(_mk("cnn"))
    assert not _is_generative_family(None)


# ── Blocker 2: --jobs > 1 gating on accelerator ──────────────────────────────


def test_gate_jobs_downgrades_when_gpu_candidates_present(capsys) -> None:
    from infermap.cli import _gate_jobs

    class _C:
        def __init__(self, device: str) -> None:
            self.device = device

    cands = [_C("cuda"), _C("cpu")]
    result = _gate_jobs(jobs=4, candidates=cands, json_mode=False)
    assert result == 1
    err = capsys.readouterr().err + capsys.readouterr().out
    # Warning is printed via err_console; just confirm downgrade.


def test_gate_jobs_downgrades_for_mps(capsys) -> None:
    from infermap.cli import _gate_jobs

    class _C:
        def __init__(self, device: str) -> None:
            self.device = device

    assert _gate_jobs(jobs=8, candidates=[_C("mps")], json_mode=False) == 1


def test_gate_jobs_leaves_cpu_only_alone() -> None:
    from infermap.cli import _gate_jobs

    class _C:
        device = "cpu"

    assert _gate_jobs(jobs=4, candidates=[_C(), _C()], json_mode=False) == 4


def test_gate_jobs_passthrough_when_jobs_is_one() -> None:
    from infermap.cli import _gate_jobs

    class _C:
        device = "cuda"

    assert _gate_jobs(jobs=1, candidates=[_C()], json_mode=False) == 1


def test_gate_jobs_silent_in_json_mode(capsys) -> None:
    from infermap.cli import _gate_jobs

    class _C:
        device = "cuda"

    _gate_jobs(jobs=4, candidates=[_C()], json_mode=True)
    # Output should not contain the warning text in JSON mode.
    out = capsys.readouterr()
    assert "downgrading" not in out.err
    assert "downgrading" not in out.out


# ── Blocker 3: smarter weights_only heuristic ────────────────────────────────


def test_torch_load_passes_through_unrelated_errors(tmp_path: Path) -> None:
    """Corrupt / unreadable .pt files should surface the original error, not the
    'set APHEX_TRUST_PICKLE' guidance — that would be misleading."""
    from infermap.evaluator import _torch_load

    p = tmp_path / "junk.pt"
    p.write_bytes(b"this is not a torch file at all")

    with pytest.raises(Exception) as ei:
        _torch_load(p)
    # Whatever error torch raised, it must not be wrapped in our friendly text.
    assert "APHEX_TRUST_PICKLE" not in str(ei.value)


def test_torch_load_wraps_pickled_module_error_with_guidance(
    tmp_path: Path, monkeypatch
) -> None:
    from infermap.evaluator import _torch_load

    monkeypatch.delenv("APHEX_TRUST_PICKLE", raising=False)
    p = tmp_path / "model.pt"
    torch.save(torch.nn.Linear(2, 2), p)

    with pytest.raises(RuntimeError) as ei:
        _torch_load(p)
    err = str(ei.value)
    assert "state_dict" in err
    assert "APHEX_TRUST_PICKLE" in err


def test_looks_like_pickled_module_heuristic() -> None:
    from infermap.evaluator import _looks_like_pickled_module

    class _FakeUnpickling(Exception):
        pass
    _FakeUnpickling.__name__ = "UnpicklingError"

    assert _looks_like_pickled_module(_FakeUnpickling("nope"))
    assert _looks_like_pickled_module(Exception(
        "Unsupported global: GLOBAL torch.nn.Linear"
    ))
    assert _looks_like_pickled_module(Exception(
        "weights_only load failed for some class"
    ))
    assert not _looks_like_pickled_module(FileNotFoundError("no such file"))
    assert not _looks_like_pickled_module(OSError("disk full"))
