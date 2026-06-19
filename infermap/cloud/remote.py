"""Remote execution — SSH-based benchmarking on a cloud machine.

aphex shells out to the system ssh/scp binaries so it inherits the developer's
existing key configuration, SSH agent, and ~/.ssh/config entries. No extra
dependencies or credential management needed.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path


def run_remote_optimize(
    host: str,
    local_model: Path,
    aphex_args: list[str],
    local_output: Path,
    local_metrics: Path | None = None,
) -> int:
    """Run ``aphex optimize`` on *host* via SSH.

    1. Creates a temporary directory on the remote.
    2. Uploads *local_model* via scp.
    3. Runs ``aphex optimize <model> <aphex_args> --output deployment.yaml``
       on the remote, streaming output to the local terminal.
    4. Downloads the resulting deployment.yaml to *local_output*.
       If *local_metrics* is given, also downloads the metrics JSON.
    5. Cleans up the remote temp directory.

    Returns the exit code of the remote aphex command.
    """
    session = f"aphex-{uuid.uuid4().hex[:8]}"
    remote_dir = f"/tmp/{session}"
    remote_model = f"{remote_dir}/{local_model.name}"
    remote_output = f"{remote_dir}/deployment.yaml"
    remote_metrics = f"{remote_dir}/metrics.json" if local_metrics is not None else None

    try:
        _run(["ssh", host, f"mkdir -p {remote_dir}"])

        _run(["scp", "-q", str(local_model), f"{host}:{remote_model}"])

        remote_cmd = _build_cmd(remote_model, aphex_args, remote_output, remote_metrics)
        exit_code = _stream(["ssh"] + _tty_flag() + [host, remote_cmd])

        if exit_code == 0:
            _run(["scp", "-q", f"{host}:{remote_output}", str(local_output)])
            if remote_metrics and local_metrics is not None:
                _run(["scp", "-q", f"{host}:{remote_metrics}", str(local_metrics)], check=False)

        return exit_code

    finally:
        _run(["ssh", host, f"rm -rf {remote_dir}"], check=False)


def check_remote_aphex(host: str) -> bool:
    """Return True if aphex is installed and reachable on *host*."""
    result = subprocess.run(
        ["ssh", host, "PATH=$HOME/.local/bin:$PATH command -v aphex"],
        capture_output=True,
        timeout=15,
    )
    return result.returncode == 0


# ── internals ─────────────────────────────────────────────────────────────────


def _build_cmd(remote_model: str, aphex_args: list[str], remote_output: str, remote_metrics: str | None = None) -> str:
    parts = ["PATH=$HOME/.local/bin:$PATH", "aphex", "optimize", _quote(remote_model)]
    parts += [_quote(a) for a in aphex_args]
    parts += ["--output", _quote(remote_output)]
    if remote_metrics:
        parts += ["--metrics", _quote(remote_metrics)]
    return " ".join(parts)


def _tty_flag() -> list[str]:
    return ["-t"] if sys.stdout.isatty() else []


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def _stream(cmd: list[str]) -> int:
    """Run *cmd* inheriting the local terminal. Returns exit code."""
    return subprocess.run(cmd).returncode


def _quote(s: str) -> str:
    """Single-quote a shell argument, escaping any internal single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"
