"""Compatibility entry point for the project-local Qlib data updater."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_runner import run_data_update


if __name__ == "__main__":
    print(json.dumps(run_data_update(), ensure_ascii=False, indent=2))
