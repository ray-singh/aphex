"""Remote execution — SSH-based benchmarking on a cloud machine.

aphex shells out to the system ssh/scp binaries so it inherits the developer's
existing key configuration, SSH agent, and ~/.ssh/config entries. No extra
dependencies or credential management needed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("infermap.cloud.remote")

_CLOUD_SCHEMES = ("s3://", "gs://", "az://")
_REMOTE_BASE = "/tmp"  # noqa: S108 — intentional; cleanup guard rejects paths outside /tmp/aphex-*


def _is_eval_uri(v: object) -> bool:
    return isinstance(v, str) and any(v.startswith(s) for s in _CLOUD_SCHEMES)


@dataclass(frozen=True)
class _RemoteLayout:
    """Path mapping for a single remote-optimize session."""

    session: str
    dir: str
    model: str
    output: str
    metrics: str | None
    eval: str | None

    @classmethod
    def for_session(
        cls,
        local_model: Path,
        local_eval: Path | str | None,
        want_metrics: bool,
    ) -> _RemoteLayout:
        session = f"aphex-{uuid.uuid4().hex[:8]}"
        directory = f"{_REMOTE_BASE}/{session}"
        if local_eval is None:
            remote_eval: str | None = None
        elif _is_eval_uri(local_eval):
            remote_eval = str(local_eval)
        else:
            remote_eval = f"{directory}/{Path(local_eval).name}"
        return cls(
            session=session,
            dir=directory,
            model=f"{directory}/{local_model.name}",
            output=f"{directory}/deployment.yaml",
            metrics=f"{directory}/metrics.json" if want_metrics else None,
            eval=remote_eval,
        )


def run_remote_optimize(
    host: str,
    local_model: Path,
    aphex_args: list[str],
    local_output: Path,
    local_metrics: Path | None = None,
    local_eval: Path | str | None = None,
) -> int:
    """Run ``aphex optimize`` on *host* via SSH.

    1. Creates a temporary directory on the remote.
    2. Uploads *local_model* via scp. If *local_eval* is a local path, uploads
       it too. Cloud URIs (s3://, gs://) are passed directly to the remote command.
    3. Runs ``aphex optimize <model> <aphex_args> --output deployment.yaml``
       on the remote, streaming output to the local terminal.
    4. Downloads the resulting deployment.yaml to *local_output*.
       If *local_metrics* is given, also downloads the metrics JSON.
    5. Cleans up the remote temp directory (best-effort, also on failure).

    Returns the exit code of the remote aphex command.
    """
    _reject_conflicting_args(aphex_args)
    layout = _RemoteLayout.for_session(local_model, local_eval, local_metrics is not None)

    try:
        _run(["ssh", host, f"mkdir -p {_quote(layout.dir)}"])

        _run(["scp", "-q", str(local_model), f"{host}:{layout.model}"])

        if local_eval is not None and not _is_eval_uri(local_eval):
            local_eval_path = Path(local_eval)
            scp_eval = ["scp", "-q"]
            if local_eval_path.is_dir():
                scp_eval.append("-r")
            scp_eval += [str(local_eval_path), f"{host}:{layout.eval}"]
            _run(scp_eval)

        extra_args = list(aphex_args)
        if layout.eval is not None:
            extra_args += ["--eval", layout.eval]

        remote_cmd = _build_cmd(layout.model, extra_args, layout.output, layout.metrics)
        exit_code = _stream(["ssh"] + _tty_flag() + [host, remote_cmd])

        if exit_code == 0:
            _run(["scp", "-q", f"{host}:{layout.output}", str(local_output)])
            if layout.metrics and local_metrics is not None:
                _run(
                    ["scp", "-q", f"{host}:{layout.metrics}", str(local_metrics)],
                    check=False,
                )

        return exit_code

    finally:
        _cleanup_remote(host, layout.dir)


def _cleanup_remote(host: str, remote_dir: str) -> None:
    """Best-effort cleanup of *remote_dir* on *host*.

    Only paths under _REMOTE_BASE are ever removed; anything else is rejected
    as a defensive guard against an accidentally-modified layout sending an
    unscoped path to ``rm -rf``.
    """
    if not remote_dir.startswith(f"{_REMOTE_BASE}/aphex-"):
        logger.error("refusing to cleanup unexpected remote dir: %r", remote_dir)
        return
    try:
        _run(["ssh", host, f"rm -rf {_quote(remote_dir)}"], check=False)
    except Exception as exc:
        logger.warning("remote cleanup of %s on %s failed: %s", remote_dir, host, exc)


def check_remote_aphex(host: str) -> bool:
    """Return True if aphex is installed and reachable on *host*.

    Returns False (rather than raising) on SSH timeout, connection refused,
    or any other OS-level failure — so the caller can show a clean install
    hint instead of a traceback.
    """
    try:
        result = subprocess.run(
            ["ssh", host, "PATH=$HOME/.local/bin:$PATH command -v aphex"],
            capture_output=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("ssh to %s failed during install check: %s", host, exc)
        return False
    return result.returncode == 0


# ── internals ─────────────────────────────────────────────────────────────────


def _reject_conflicting_args(aphex_args: list[str]) -> None:
    # We append --output (and optionally --metrics) on the remote side; a
    # second occurrence from the caller would make argparse error out after
    # we've already uploaded the model. Catch it up front with a clear message.
    for flag in ("--output", "--metrics", "--eval"):
        if flag in aphex_args:
            raise ValueError(
                f"{flag} cannot be passed through --remote; "
                "use the matching CLI flag on `aphex optimize --remote ...` instead."
            )


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
