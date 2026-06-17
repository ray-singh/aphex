"""Tests for infermap.cloud — storage backend, config, and registry operations."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from infermap.cloud.storage import MemoryBackend, get_backend


# ── MemoryBackend ─────────────────────────────────────────────────────────────


def test_memory_upload_download_roundtrip(tmp_path: Path) -> None:
    b = MemoryBackend()
    src = tmp_path / "model.onnx"
    src.write_bytes(b"fake-onnx")
    b.upload("mymodel/v1/model.onnx", src)

    dest = tmp_path / "out.onnx"
    b.download("mymodel/v1/model.onnx", dest)
    assert dest.read_bytes() == b"fake-onnx"


def test_memory_list_prefix(tmp_path: Path) -> None:
    b = MemoryBackend()
    for name in ["mymodel/v1/a.onnx", "mymodel/v1/dep.yaml", "other/v1/a.onnx"]:
        f = tmp_path / name.replace("/", "_")
        f.write_bytes(b"x")
        b.upload(name, f)

    keys = b.list("mymodel/")
    assert set(keys) == {"mymodel/v1/a.onnx", "mymodel/v1/dep.yaml"}


def test_memory_list_empty_prefix_returns_all(tmp_path: Path) -> None:
    b = MemoryBackend()
    for name in ["a/v1/f.pt", "b/v2/g.onnx"]:
        f = tmp_path / name.replace("/", "_")
        f.write_bytes(b"x")
        b.upload(name, f)
    assert len(b.list("")) == 2


def test_memory_exists_true(tmp_path: Path) -> None:
    b = MemoryBackend()
    f = tmp_path / "f.pt"
    f.write_bytes(b"x")
    b.upload("a/v1/f.pt", f)
    assert b.exists("a/v1/f.pt")


def test_memory_exists_false() -> None:
    b = MemoryBackend()
    assert not b.exists("a/v1/missing.pt")


def test_memory_download_missing_raises(tmp_path: Path) -> None:
    b = MemoryBackend()
    with pytest.raises(FileNotFoundError):
        b.download("missing/key", tmp_path / "out.pt")


def test_memory_read_write_text(tmp_path: Path) -> None:
    b = MemoryBackend()
    b.write_text("mymodel/latest", "v3")
    assert b.read_text("mymodel/latest") == "v3"


# ── get_backend ───────────────────────────────────────────────────────────────


def test_get_backend_s3_returns_s3_backend() -> None:
    pytest.importorskip("boto3")
    from infermap.cloud.storage import S3Backend
    backend = get_backend("s3://my-bucket/aphex")
    assert isinstance(backend, S3Backend)


def test_get_backend_gcs_returns_gcs_backend() -> None:
    pytest.importorskip("google.cloud.storage")
    from infermap.cloud.storage import GCSBackend
    backend = get_backend("gs://my-bucket/aphex")
    assert isinstance(backend, GCSBackend)


def test_get_backend_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported registry URI"):
        get_backend("ftp://bucket/prefix")


def test_get_backend_s3_no_boto3_raises(monkeypatch) -> None:
    import sys
    monkeypatch.setitem(sys.modules, "boto3", None)  # type: ignore[arg-type]
    with pytest.raises((ImportError, ValueError)):
        get_backend("s3://bucket/prefix")


# ── get_registry_uri ─────────────────────────────────────────────────────────


def test_get_registry_uri_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APHEX_REGISTRY", "s3://env-bucket/aphex")
    from infermap.cloud.config import get_registry_uri
    assert get_registry_uri() == "s3://env-bucket/aphex"


def test_get_registry_uri_from_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("APHEX_REGISTRY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[registry]\nuri = "gs://cfg-bucket/aphex"\n')

    import infermap.cloud.config as cloud_cfg
    monkeypatch.setattr(cloud_cfg, "_CONFIG_PATH", cfg)

    assert cloud_cfg.get_registry_uri() == "gs://cfg-bucket/aphex"


def test_get_registry_uri_raises_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("APHEX_REGISTRY", raising=False)
    import infermap.cloud.config as cloud_cfg
    monkeypatch.setattr(cloud_cfg, "_CONFIG_PATH", tmp_path / "nonexistent.toml")
    with pytest.raises(RuntimeError, match="No registry configured"):
        cloud_cfg.get_registry_uri()


# ── registry: push ────────────────────────────────────────────────────────────


def test_push_uploads_all_files(tmp_path: Path) -> None:
    from infermap.cloud.registry import push

    b = MemoryBackend()
    f1 = tmp_path / "model.onnx"
    f2 = tmp_path / "deployment.yaml"
    f1.write_bytes(b"onnx")
    f2.write_bytes(b"yaml")

    push("resnet50", "v1", [f1, f2], b)

    assert b.exists("resnet50/v1/model.onnx")
    assert b.exists("resnet50/v1/deployment.yaml")


def test_push_updates_latest_pointer(tmp_path: Path) -> None:
    from infermap.cloud.registry import push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")

    push("resnet50", "v2", [f], b)
    assert b.read_text("resnet50/latest") == "v2"


def test_push_overwrites_latest_on_second_push(tmp_path: Path) -> None:
    from infermap.cloud.registry import push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")

    push("resnet50", "v1", [f], b)
    push("resnet50", "v2", [f], b)
    assert b.read_text("resnet50/latest") == "v2"


# ── registry: pull ────────────────────────────────────────────────────────────


def test_pull_by_explicit_version(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull, push

    b = MemoryBackend()
    src = tmp_path / "model.onnx"
    src.write_bytes(b"onnx-content")
    push("resnet50", "v1", [src], b)

    out = tmp_path / "out"
    written = pull("resnet50@v1", out, b)

    assert len(written) == 1
    assert written[0].read_bytes() == b"onnx-content"


def test_pull_latest_resolves_correctly(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull, push

    b = MemoryBackend()
    f1 = tmp_path / "v1.onnx"
    f2 = tmp_path / "v2.onnx"
    f1.write_bytes(b"v1")
    f2.write_bytes(b"v2")
    push("resnet50", "v1", [f1], b)
    push("resnet50", "v2", [f2], b)

    out = tmp_path / "out"
    written = pull("resnet50", out, b)  # should get v2

    assert written[0].read_bytes() == b"v2"


def test_pull_unknown_model_raises(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull

    b = MemoryBackend()
    with pytest.raises(ValueError, match="No model named"):
        pull("ghost", tmp_path / "out", b)


def test_pull_unknown_version_raises(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull, push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")
    push("resnet50", "v1", [f], b)

    with pytest.raises(ValueError, match="not found in registry"):
        pull("resnet50@v99", tmp_path / "out", b)


def test_pull_creates_output_directory(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull, push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")
    push("resnet50", "v1", [f], b)

    out = tmp_path / "deep" / "nested" / "dir"
    assert not out.exists()
    pull("resnet50@v1", out, b)
    assert out.exists()


def test_pull_downloads_multiple_files(tmp_path: Path) -> None:
    from infermap.cloud.registry import pull, push

    b = MemoryBackend()
    f1 = tmp_path / "model.onnx"
    f2 = tmp_path / "deployment.yaml"
    f1.write_bytes(b"onnx")
    f2.write_bytes(b"yaml")
    push("resnet50", "v1", [f1, f2], b)

    out = tmp_path / "out"
    written = pull("resnet50@v1", out, b)
    assert len(written) == 2
    names = {p.name for p in written}
    assert names == {"model.onnx", "deployment.yaml"}


# ── registry: list ────────────────────────────────────────────────────────────


def test_list_models_returns_sorted_names(tmp_path: Path) -> None:
    from infermap.cloud.registry import list_models, push

    b = MemoryBackend()
    for model in ["resnet50", "bert-base", "yolov8"]:
        f = tmp_path / f"{model}.onnx"
        f.write_bytes(b"x")
        push(model, "v1", [f], b)

    assert list_models(b) == ["bert-base", "resnet50", "yolov8"]


def test_list_models_empty_registry() -> None:
    from infermap.cloud.registry import list_models

    b = MemoryBackend()
    assert list_models(b) == []


def test_list_versions_returns_sorted_newest_first(tmp_path: Path) -> None:
    from infermap.cloud.registry import list_versions, push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")
    for v in ["v1", "v2", "v3"]:
        push("resnet50", v, [f], b)

    versions = list_versions("resnet50", b)
    assert versions == ["v3", "v2", "v1"]


def test_list_versions_excludes_latest_pointer(tmp_path: Path) -> None:
    from infermap.cloud.registry import list_versions, push

    b = MemoryBackend()
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")
    push("resnet50", "v1", [f], b)

    versions = list_versions("resnet50", b)
    assert "latest" not in versions


def test_list_versions_unknown_model() -> None:
    from infermap.cloud.registry import list_versions

    b = MemoryBackend()
    assert list_versions("ghost", b) == []
