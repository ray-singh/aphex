"""Tests for infermap.cloud.remote — remote SSH execution helpers."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from infermap.cloud.remote import (
    _build_cmd,
    _quote,
    check_remote_aphex,
    run_remote_optimize,
)


# ── _quote ────────────────────────────────────────────────────────────────────


def test_quote_simple_string() -> None:
    assert _quote("hello") == "'hello'"


def test_quote_string_with_spaces() -> None:
    assert _quote("hello world") == "'hello world'"


def test_quote_string_with_single_quote() -> None:
    # internal ' → '\''
    assert _quote("it's") == "'it'\\''s'"


def test_quote_path_with_slashes() -> None:
    assert _quote("/tmp/aphex-abc/model.pt") == "'/tmp/aphex-abc/model.pt'"


def test_quote_empty_string() -> None:
    assert _quote("") == "''"


# ── _build_cmd ────────────────────────────────────────────────────────────────


def test_build_cmd_basic() -> None:
    cmd = _build_cmd("/tmp/s/model.pt", ["--input-shape", "3,224,224"], "/tmp/s/deployment.yaml")
    assert cmd.startswith("PATH=$HOME/.local/bin:$PATH aphex optimize")
    assert "'/tmp/s/model.pt'" in cmd
    assert "'--input-shape'" in cmd
    assert "'3,224,224'" in cmd
    assert "--output" in cmd
    assert "'/tmp/s/deployment.yaml'" in cmd


def test_build_cmd_extra_args_included() -> None:
    cmd = _build_cmd("/m.pt", ["--objective", "throughput", "--max-latency-ms", "50.0"], "/out.yaml")
    assert "'--objective'" in cmd
    assert "'throughput'" in cmd
    assert "'--max-latency-ms'" in cmd
    assert "'50.0'" in cmd


def test_build_cmd_output_always_appended() -> None:
    cmd = _build_cmd("/m.pt", [], "/custom/path/deployment.yaml")
    assert "--output '/custom/path/deployment.yaml'" in cmd


# ── check_remote_aphex ────────────────────────────────────────────────────────


def test_check_remote_aphex_returns_true_on_zero_exit() -> None:
    with patch("infermap.cloud.remote.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert check_remote_aphex("user@host") is True
        mock_run.assert_called_once_with(
            ["ssh", "user@host", "PATH=$HOME/.local/bin:$PATH command -v aphex"],
            capture_output=True,
            timeout=15,
        )


def test_check_remote_aphex_returns_false_on_nonzero_exit() -> None:
    with patch("infermap.cloud.remote.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert check_remote_aphex("user@host") is False


def test_check_remote_aphex_returns_false_on_timeout() -> None:
    import subprocess
    with patch("infermap.cloud.remote.subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 15)):
        with pytest.raises(Exception):
            check_remote_aphex("user@host")


# ── run_remote_optimize ───────────────────────────────────────────────────────


def _make_mock_run(exit_codes: list[int]):
    """Return a mock _run/_stream that yields the given exit codes in order."""
    codes = iter(exit_codes)

    def fake_run(cmd, check=True):
        result = MagicMock(returncode=next(codes, 0))
        if check and result.returncode != 0:
            import subprocess
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result

    return fake_run


def test_run_remote_creates_temp_dir(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=0):
        run_remote_optimize("user@host", model, ["--input-shape", "3,3"], output)

    # First call should be mkdir
    assert calls[0][0] == "ssh"
    assert "mkdir" in calls[0][-1]


def test_run_remote_uploads_model(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=0):
        run_remote_optimize("user@host", model, [], output)

    scp_up = next(c for c in calls if c[0] == "scp" and "model.pt" in str(c))
    assert "user@host" in str(scp_up)


def test_run_remote_streams_aphex_command(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    streamed = []

    def fake_stream(cmd):
        streamed.append(cmd)
        return 0

    with patch("infermap.cloud.remote._run", return_value=MagicMock(returncode=0)), \
         patch("infermap.cloud.remote._stream", side_effect=fake_stream):
        run_remote_optimize("user@host", model, ["--input-shape", "4"], output)

    assert len(streamed) == 1
    ssh_cmd = streamed[0]
    assert ssh_cmd[0] == "ssh"
    assert "aphex optimize" in ssh_cmd[-1]


def test_run_remote_pulls_result_on_success(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=0):
        run_remote_optimize("user@host", model, [], output)

    # download: user@host: appears in position 2 (source); upload has it in position 3 (dest)
    scp_down = [c for c in calls if c[0] == "scp" and c[2].startswith("user@host:")]
    assert len(scp_down) == 1
    assert str(output) in str(scp_down[0])


def test_run_remote_skips_pull_on_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=1):
        exit_code = run_remote_optimize("user@host", model, [], output)

    assert exit_code == 1
    scp_down = [c for c in calls if c[0] == "scp" and c[2].startswith("user@host:")]
    assert len(scp_down) == 0


def test_run_remote_cleans_up_on_success(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=0):
        run_remote_optimize("user@host", model, [], output)

    cleanup = [c for c in calls if "rm -rf" in str(c)]
    assert len(cleanup) == 1


def test_run_remote_cleans_up_on_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    output = tmp_path / "deployment.yaml"

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        if check and "mkdir" not in str(cmd) and "scp" not in str(cmd):
            pass
        return MagicMock(returncode=0)

    with patch("infermap.cloud.remote._run", side_effect=fake_run), \
         patch("infermap.cloud.remote._stream", return_value=1):
        run_remote_optimize("user@host", model, [], output)

    cleanup = [c for c in calls if "rm -rf" in str(c)]
    assert len(cleanup) == 1


def test_run_remote_returns_exit_code(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")

    with patch("infermap.cloud.remote._run", return_value=MagicMock(returncode=0)), \
         patch("infermap.cloud.remote._stream", return_value=42):
        code = run_remote_optimize("user@host", model, [], tmp_path / "out.yaml")

    assert code == 42


def test_run_remote_session_dirs_are_unique(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")

    dirs = []

    def fake_run(cmd, check=True):
        if isinstance(cmd[-1], str) and "mkdir" in cmd[-1]:
            dirs.append(cmd[-1])
        return MagicMock(returncode=0)

    for _ in range(3):
        with patch("infermap.cloud.remote._run", side_effect=fake_run), \
             patch("infermap.cloud.remote._stream", return_value=0):
            run_remote_optimize("user@host", model, [], tmp_path / "out.yaml")

    assert len(set(dirs)) == 3
