"""Compatibility entry point for the local Qlib training runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Qlib 本地训练执行器")
    parser.add_argument("--market", default="csi300", choices=["csi300", "csi1000"])
    parser.add_argument("--model-mode", default="robust", choices=["default", "robust"])
    parser.add_argument("--hold-num", type=int, default=20)
    args = parser.parse_args()

    result = run_training(
        market=args.market,
        model_mode=args.model_mode,
        hold_num=args.hold_num,
        lightgbm_only=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
