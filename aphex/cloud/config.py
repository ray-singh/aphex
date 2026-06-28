"""Registry configuration — reads APHEX_REGISTRY env var or ~/.aphex/config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_CONFIG_PATH = Path.home() / ".aphex" / "config.toml"

_SETUP_HINT = (
    "Set the APHEX_REGISTRY environment variable:\n\n"
    "  export APHEX_REGISTRY=s3://your-bucket/aphex\n\n"
    "Or create ~/.aphex/config.toml:\n\n"
    "  [registry]\n"
    '  uri = "s3://your-bucket/aphex"\n'
)


def get_registry_uri() -> str:
    """Return the configured registry URI.

    Priority: APHEX_REGISTRY env var → ~/.aphex/config.toml → error.
    """
    uri = os.environ.get("APHEX_REGISTRY")
    if uri:
        return uri

    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open("rb") as fh:
            cfg = tomllib.load(fh)
        uri = cfg.get("registry", {}).get("uri")
        if uri:
            return uri

    raise RuntimeError(f"No registry configured.\n\n{_SETUP_HINT}")
