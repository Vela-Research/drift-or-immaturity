"""Where the experiments read and write data: `data/` at the repository root (see README, "Data")."""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = ["DATA_DIR"]

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data"
