from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from infermap.deployment import DeploymentConfig


class ServingGenerator(ABC):
    name: str

    @abstractmethod
    def generate(self, config: DeploymentConfig, output_dir: Path) -> list[Path]:
        """Write serving config files to output_dir. Returns list of created paths."""
