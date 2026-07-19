#!/usr/bin/env python3
"""CSI500 stage2 wrapper built on the shared walk-forward engine."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_practice_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scripts_root = Path(__file__).resolve().parents[1]
    module_path = scripts_root / "practice" / "run_stage2_walk_forward.py"
    spec = importlib.util.spec_from_file_location("run_stage2_walk_forward_practice", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_practice_module()
    root = Path(__file__).resolve().parents[2]
    template = root / "scripts" / "small" / "templates" / "workflow_config_lightgbm_Alpha158_csi500.yaml"

    module.MODEL_SPECS = [
        {
            "name": "csi500_lightgbm",
            "template": template,
            "model_mode": "robust",
            "route": "csi500",
            "universe_role": "target",
        },
    ]

    original_build_fold_dates = module._build_fold_dates

    def _build_fold_dates_limited(*args, **kwargs):
        folds = original_build_fold_dates(*args, **kwargs)
        raw_limit = str(os.environ.get("STAGE2_MAX_FOLDS", "0") or "0").strip()
        limit = int(raw_limit) if raw_limit else 0
        return folds[-limit:] if limit > 0 else folds

    module._build_fold_dates = _build_fold_dates_limited
    module.main()


if __name__ == "__main__":
    main()
