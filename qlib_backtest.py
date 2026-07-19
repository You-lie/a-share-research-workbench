"""Read-only parser for persisted Qlib walk-forward backtest artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_SUMMARY_METRICS = (
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "IC_mean",
    "ICIR",
    "monthly_win_rate",
)
_UNAVAILABLE_METRICS = (
    "turnover_mean",
    "slippage",
    "excess_return_with_cost_mean",
    "excess_return_without_cost_mean",
)


def resolve_model_dir(models_dir: Path, model_name: str) -> Path:
    if not MODEL_NAME_RE.fullmatch(model_name or ""):
        raise ValueError("模型名称不合法")
    root = models_dir.resolve()
    model_dir = (root / model_name).resolve()
    try:
        relative = model_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("模型路径不合法") from exc
    if len(relative.parts) != 1 or relative.name != model_name or not model_dir.is_dir():
        raise FileNotFoundError(f"模型不存在: {model_name}")
    return model_dir


def _number(value: str | None) -> Any:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(text) if re.fullmatch(r"-?\d+", text) else float(text)
    except ValueError:
        return text


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _market_info(model_name: str) -> tuple[str, str]:
    lowered = model_name.lower()
    if "csi1000" in lowered:
        return "中证 1000", "SH000852"
    if "csi500" in lowered:
        return "中证 500", "SH000905"
    if "csi300" in lowered:
        return "沪深 300", "SH000300"
    return "未知股票池", "未记录"


def _summary_from_folds(folds: list[dict]) -> dict:
    summary: dict[str, dict | None] = {}
    for metric in _SUMMARY_METRICS:
        values = [row.get(metric) for row in folds if isinstance(row.get(metric), (int, float))]
        if not values:
            summary[metric] = None
            continue
        mean = sum(values) / len(values)
        summary[metric] = {
            "mean": mean,
            "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
    return summary


def _yaml_scalar(path: Path, key: str) -> Any:
    if not path.is_file():
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*([^#\n]+)", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    return _number(match.group(1).strip().strip("'\"")) if match else None


def _full_cycle_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = {
        "mean_turnover": "Mean turnover",
        "total_turnover": "Total turnover",
        "annualized_return": "Annualized return",
        "max_drawdown": "Max drawdown",
        "sharpe_ratio": "Sharpe ratio",
        "monthly_win_rate": "Monthly win rate",
        "ic_mean": "IC mean",
        "icir": "ICIR",
    }
    result = {}
    for key, label in labels.items():
        match = re.search(rf"^{re.escape(label)}:\s*([-+.\d]+)", text, re.MULTILINE | re.IGNORECASE)
        if match:
            result[key] = _number(match.group(1))
    return result


def load_backtest(models_dir: Path, model_name: str) -> dict:
    """Return only existing Qlib outputs; absent values remain explicitly unavailable."""
    model_dir = resolve_model_dir(models_dir, model_name)
    output_dir = model_dir / "model_predict"
    folds = _read_csv(output_dir / "walk_forward_folds.csv")
    stored_summary = _read_csv(output_dir / "walk_forward_summary.csv")
    market, benchmark = _market_info(model_name)
    last_fold = folds[-1] if folds else {}
    fold_dir = output_dir / "walk_forward" / str(last_fold.get("signal_date") or "")
    config_path = fold_dir / "model_runs" / "lightgbm" / "workflow_config_practice.yaml"
    config_market = _yaml_scalar(config_path, "market")
    config_benchmark = _yaml_scalar(config_path, "benchmark")
    config_topk = _yaml_scalar(config_path, "topk")
    config_n_drop = _yaml_scalar(config_path, "n_drop")
    config_account = _yaml_scalar(config_path, "account")
    open_cost = _yaml_scalar(config_path, "open_cost")
    close_cost = _yaml_scalar(config_path, "close_cost")
    min_cost = _yaml_scalar(config_path, "min_cost")
    full_cycle = _full_cycle_metrics(model_dir / "report_of_backtest.txt")
    availability = {
        metric: any(row.get(metric) is not None for row in folds)
        for metric in _UNAVAILABLE_METRICS
    }
    availability["turnover_mean"] = full_cycle.get("mean_turnover") is not None
    return {
        "model": model_name,
        "available": bool(folds),
        "message": "" if folds else "该模型未找到逐折样本外回测文件",
        "metadata": {
            "universe": _market_info(str(config_market or model_name))[0] if config_market else market,
            "benchmark": config_benchmark or benchmark,
            "data_cutoff": last_fold.get("oos_end") or last_fold.get("test_end") or model_name[:10],
            "hold_num": config_topk or 20,
            "rebalance_frequency": "本模型未单独记录",
            "strategy_n_drop": config_n_drop,
            "initial_cash": config_account or 100000,
            "open_cost_rate": open_cost,
            "close_cost_rate": close_cost,
            "min_cost": min_cost,
            "cost_source": str(config_path.relative_to(model_dir)) if config_path.is_file() else "未找到该模型的工作流配置",
            "stamp_duty_assumption": "本模型未单独记录，无法从卖出总成本中拆分",
        },
        "folds": folds,
        "summary": _summary_from_folds(folds),
        "stored_summary": stored_summary,
        "stability": {
            "total_folds": len(folds),
            "profitable_folds": sum(1 for row in folds if (row.get("annualized_return") or 0) > 0),
            "negative_sharpe_folds": sum(1 for row in folds if (row.get("sharpe_ratio") or 0) < 0),
        },
        "availability": availability,
        "full_cycle": full_cycle,
        "disclaimer": "历史样本外结果，不代表实盘收益。费用、滑点和成交约束会影响实际表现。",
    }
