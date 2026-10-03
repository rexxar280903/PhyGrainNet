from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KagglePaths:
    data_root: Path = Path("/kaggle/input/soil-grain-size-from-photos")
    work_root: Path = Path("/kaggle/working/phygrainnet")

    def ensure_work_root(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
