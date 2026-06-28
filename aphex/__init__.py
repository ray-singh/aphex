"""aphex — hardware-aware ML deployment optimization framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aphex-ml")
except PackageNotFoundError:
    __version__ = "unknown"
