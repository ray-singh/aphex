"""Registry operations: push, pull, list.

Path convention inside the registry root
-----------------------------------------
  <name>/<version>/<filename>   — artifact files (model + deployment.yaml)
  <name>/latest                 — plain text pointer to the most recent version
"""

from __future__ import annotations

from pathlib import Path

from infermap.cloud.storage import StorageBackend

_LATEST = "latest"


# ── write ─────────────────────────────────────────────────────────────────────


def push(
    name: str,
    version: str,
    files: list[Path],
    backend: StorageBackend,
) -> None:
    """Upload *files* to the registry under *name*/*version* and update latest."""
    for f in files:
        backend.upload(f"{name}/{version}/{f.name}", f)
    backend.write_text(f"{name}/{_LATEST}", version)


# ── read ──────────────────────────────────────────────────────────────────────


def pull(
    name_version: str,
    out_dir: Path,
    backend: StorageBackend,
) -> list[Path]:
    """Download all files for *name*[@*version*] into *out_dir*.

    If no version is given, the latest version is used.
    Returns the list of local paths written.
    """
    name, version = _resolve(name_version, backend)
    prefix = f"{name}/{version}/"
    keys = [k for k in backend.list(prefix) if not k.endswith("/")]

    if not keys:
        available = list_versions(name, backend)
        if available:
            raise ValueError(
                f"Version {version!r} of {name!r} not found in registry. "
                f"Available: {', '.join(available)}"
            )
        raise ValueError(
            f"No model named {name!r} in registry. "
            "Run 'aphex ls' to see available models."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key in keys:
        filename = key.rstrip("/").split("/")[-1]
        dest = out_dir / filename
        backend.download(key, dest)
        written.append(dest)
    return written


# ── list ──────────────────────────────────────────────────────────────────────


def list_models(backend: StorageBackend) -> list[str]:
    """Return a sorted list of all model names in the registry."""
    names: set[str] = set()
    for key in backend.list(""):
        parts = key.split("/")
        if parts and parts[0]:
            names.add(parts[0])
    return sorted(names)


def list_versions(name: str, backend: StorageBackend) -> list[str]:
    """Return versions of *name*, sorted newest-first. Excludes the 'latest' pointer."""
    versions: set[str] = set()
    for key in backend.list(f"{name}/"):
        parts = key.split("/")
        if len(parts) >= 2 and parts[1] and parts[1] != _LATEST:
            versions.add(parts[1])
    return sorted(versions, reverse=True)


# ── internal ──────────────────────────────────────────────────────────────────


def _resolve(name_version: str, backend: StorageBackend) -> tuple[str, str]:
    """Split 'name@version' or resolve 'name' → (name, version)."""
    if "@" in name_version:
        name, version = name_version.split("@", 1)
        return name, version

    name = name_version
    latest_key = f"{name}/{_LATEST}"
    if not backend.exists(latest_key):
        raise ValueError(
            f"No model named {name!r} in registry. "
            "Run 'aphex ls' to see available models."
        )
    version = backend.read_text(latest_key).strip()
    return name, version
