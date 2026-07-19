"""
Qlib 训练执行器 — 使用本地 Conda 环境运行 stage2 walk-forward 全量训练（含回测）。

用法:
    from train_runner import run_training
    result = run_training(market="csi300", progress_callback=cb)
    print(result["model_name"])
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from local_runtime import (
    ANALYSIS_OUTPUTS_DIR,
    MLRUNS_DIR,
    PROJECT_ROOT,
    QLIB_DATA_DIR,
    QLIB_PYTHON,
    build_runtime_env,
    python_command,
    run_streaming,
    validate_runtime,
)

# 训练超时 2 小时
TRAIN_TIMEOUT = 7200


def _safe_console_print(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_message, flush=True)


class _TrainingProgressTracker:
    def __init__(self) -> None:
        self.progress = 0.03
        self.phase = "初始化 Qlib"
        self.fold_current = 0
        self.fold_total = 0

    def update(self, line: str) -> dict:
        plan_match = re.search(r"\[(\d{4})/(\d{4})\]\s+signal=", line)
        if plan_match:
            self.fold_total = max(self.fold_total, int(plan_match.group(2)))

        if "Init data Done" in line:
            self.progress = max(self.progress, 0.12)
            self.phase = "加载 Alpha158 特征"
        elif "[Cache] Saving" in line:
            self.progress = max(self.progress, 0.22)
            self.phase = "保存特征缓存"
        elif "[Cache] Saved" in line or "feature cache is current" in line:
            self.progress = max(self.progress, 0.25)
            self.phase = "特征缓存已就绪"

        fold_match = re.search(r"\[(\d+)/(\d+)\]\s+Fold\s+", line)
        if fold_match:
            self.fold_current = int(fold_match.group(1))
            self.fold_total = int(fold_match.group(2))
            self.progress = max(
                self.progress,
                0.25 + ((self.fold_current - 1) / max(self.fold_total, 1)) * 0.63,
            )
            self.phase = f"训练第 {self.fold_current}/{self.fold_total} 折"
        elif "training lightgbm" in line.lower() and self.fold_current:
            self.progress = max(
                self.progress,
                0.25 + ((self.fold_current - 0.75) / max(self.fold_total, 1)) * 0.63,
            )
            self.phase = f"训练第 {self.fold_current}/{self.fold_total} 折 LightGBM"
        elif "validation IC weights" in line and self.fold_current:
            self.progress = max(
                self.progress,
                0.25 + ((self.fold_current - 0.1) / max(self.fold_total, 1)) * 0.63,
            )
            self.phase = f"整理第 {self.fold_current}/{self.fold_total} 折结果"
        elif "SKIPPED (resuming)" in line and self.fold_current:
            self.progress = max(
                self.progress,
                0.25 + (self.fold_current / max(self.fold_total, 1)) * 0.63,
            )
            self.phase = f"复用第 {self.fold_current}/{self.fold_total} 折缓存"
        elif "Latest fold copied to root output" in line:
            self.progress = max(self.progress, 0.90)
            self.phase = "聚合跨期预测"
        elif "Cross-Fold Factor IC Aggregation" in line:
            self.progress = max(self.progress, 0.93)
            self.phase = "汇总因子 IC"
        elif "Full-cycle backtest report" in line:
            self.progress = max(self.progress, 0.98)
            self.phase = "生成全周期回测报告"

        return {
            "progress": round(min(self.progress, 0.99), 4),
            "phase": self.phase,
            "fold_current": self.fold_current,
            "fold_total": self.fold_total,
        }


def _resolve_config(market: str) -> dict:
    """根据市场名称解析脚本 / YAML 模板."""
    if market == "csi1000":
        return {
            "market": "csi1000",
            "benchmark": "SH000852",
            "script": PROJECT_ROOT / "scripts" / "small" / "run_stage2_walk_forward_small.py",
            "template": PROJECT_ROOT / "scripts" / "small" / "templates" / "workflow_config_lightgbm_Alpha158_csi1000.yaml",
        }
    if market == "csi500":
        return {
            "market": "csi500",
            "benchmark": "SH000905",
            "script": PROJECT_ROOT / "scripts" / "small" / "run_stage2_walk_forward_csi500.py",
            "template": PROJECT_ROOT / "scripts" / "small" / "templates" / "workflow_config_lightgbm_Alpha158_csi500.yaml",
        }
    else:
        # 默认 CSI300
        return {
            "market": "csi300",
            "benchmark": "SH000300",
            "script": PROJECT_ROOT / "scripts" / "practice" / "run_stage2_walk_forward.py",
            "template": PROJECT_ROOT / "examples" / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml",
        }


def _get_last_trade_date() -> str:
    """从 qlib 日历获取最近交易日."""
    calendar_file = QLIB_DATA_DIR / "calendars" / "day.txt"
    if calendar_file.exists():
        dates = calendar_file.read_text().strip().splitlines()
        if dates:
            return dates[-1].strip()
    # fallback: 今天
    return datetime.now().strftime("%Y-%m-%d")


def _get_total_years() -> float:
    """从 qlib 日历获取总数据跨度（年），用于将 fold 数转换为 stride."""
    calendar_file = QLIB_DATA_DIR / "calendars" / "day.txt"
    if calendar_file.exists():
        dates = calendar_file.read_text().strip().splitlines()
        if len(dates) >= 2:
            first = datetime.strptime(dates[0].strip(), "%Y-%m-%d")
            last = datetime.strptime(dates[-1].strip(), "%Y-%m-%d")
            return (last - first).days / 365.25
    return 15.0  # fallback


def _ensure_mlruns_dir():
    """确保 mlruns 目录存在."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_backtest_metrics(analysis_root: Path, model_name: str) -> dict:
    """从训练输出中提取回测指标."""
    metrics = {}

    # 尝试读取 report_of_backtest.txt
    report_txt = analysis_root / "model_predict" / "walk_forward" / "full_backtest" / "report_of_backtest.txt"
    if not report_txt.exists():
        # 尝试其他路径
        alt_paths = [
            analysis_root / "model_predict" / "report_of_backtest.txt",
            analysis_root / "full_backtest" / "report_of_backtest.txt",
        ]
        for p in alt_paths:
            if p.exists():
                report_txt = p
                break

    if report_txt.exists():
        content = report_txt.read_text(encoding="utf-8")
        patterns = {
            "annualized_return": r"annualized_return[:\s]+([-.\d]+)",
            "max_drawdown": r"max_drawdown[:\s]+([-.\d]+)",
            "sharpe_ratio": r"sharpe_ratio[:\s]+([-.\d]+)",
            "information_ratio": r"information_ratio[:\s]+([-.\d]+)",
            "ic_mean": r"ic_mean[:\s]+([-.\d]+)",
            "icir": r"icir[:\s]+([-.\d]+)",
            "annualized_excess_return": r"annualized_excess_return[:\s]+([-.\d]+)",
            "turnover_mean": r"turnover_mean[:\s]+([-.\d]+)",
            "monthly_win_rate": r"monthly_win_rate[:\s]+([-.\d]+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                try:
                    metrics[key] = round(float(m.group(1)), 4)
                except ValueError:
                    pass
        metrics["raw_report"] = content[:2000]  # 截取前 2000 字符
    else:
        # 尝试从 metrics.csv 读取
        metrics_csv = analysis_root / "model_predict" / "metrics.csv"
        if metrics_csv.exists():
            with open(metrics_csv, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k, v in row.items():
                        try:
                            metrics[k] = round(float(v), 4)
                        except (ValueError, TypeError):
                            metrics[k] = v

    # 检查 scores.csv 是否存在（训练成功标志）
    scores_csv = analysis_root / "model_predict" / "scores.csv"
    metrics["has_scores"] = scores_csv.exists()
    if scores_csv.exists():
        with open(scores_csv, newline="", encoding="utf-8-sig") as f:
            metrics["score_count"] = sum(1 for _ in csv.DictReader(f))

    return metrics


def run_training(
    market: str = "csi300",
    model_mode: str = "robust",
    hold_num: int = 20,
    lightgbm_only: bool = True,
    train_years: int = 5,
    valid_val: int = 1,
    valid_unit: str = "year",
    test_val: int = 2,
    test_unit: str = "year",
    progress_callback: Callable | None = None,
    cancel_event=None,
) -> dict:
    """
    在本地 Qlib Conda 环境中运行完整的 walk-forward 训练（含回测）。

    Args:
        market: "csi300"、"csi500" 或 "csi1000"
        model_mode: "default" 或 "robust"
        hold_num: 持仓股票数量
        lightgbm_only: 仅训练 LightGBM（跳过 XGBoost）
        train_years: 训练窗口（年）
        valid_val: 验证窗口数值
        valid_unit: 验证窗口单位 ("year", "month", "week")
        test_val: 测试窗口数值
        test_unit: 测试窗口单位 ("year", "month", "week")
        progress_callback: 可选回调，接收 dict(message, progress, ...)

    Returns:
        {"success": True/False, "model_name": "...", "message": "...",
         "backtest_metrics": {...}}
    """

    def _log(msg: str, **extra):
        _safe_console_print(msg)
        if progress_callback:
            progress_callback({"message": msg, **extra})

    cfg = _resolve_config(market)
    last_trade_date = _get_last_trade_date()
    model_name = f"{last_trade_date}-{market}-alpha158"

    validate_runtime(require_data=True)

    # 确保 mlruns 目录存在
    _ensure_mlruns_dir()

    _log(f"市场: {cfg['market']}")
    _log(f"基准: {cfg['benchmark']}")
    _log(f"模型名称: {model_name}")
    _log(f"模型模式: {model_mode}")
    _log(f"持仓数量: {hold_num}")
    _log(f"LightGBM only: {lightgbm_only}")
    _log(f"Qlib Python: {QLIB_PYTHON}")
    _log(f"Qlib 数据: {QLIB_DATA_DIR}")

    analysis_root = ANALYSIS_OUTPUTS_DIR / model_name
    output_root = analysis_root / "model_predict"
    runtime_env = build_runtime_env(
        {
            "TARGET_MARKET": cfg["market"],
            "TARGET_BENCHMARK": cfg["benchmark"],
            "HOLD_NUM": str(hold_num),
            "CASH_TOTAL": "100000",
            "TX_FEE_RATE": "0.0001",
            "STAMP_DUTY_RATE": "0.0005",
            "FULL_BACKTEST_REBALANCE_FREQ": "yearly",
            "FULL_BACKTEST_STRATEGY": "master_analysis",
            "STAGE2_LIGHTGBM_ONLY": "1" if lightgbm_only else "0",
        }
    )

    cmd = python_command(
        cfg["script"],
        "--template", cfg["template"],
        "--output-root", output_root,
        "--analysis-root", analysis_root,
        "--experiment-name", model_name,
        "--uri-folder", MLRUNS_DIR,
        "--walk-forward-end", last_trade_date,
    )

    # ---- 自动计算 fold 数 (基于数据总跨度, ≤10) ----
    total_years = _get_total_years()
    _BASE = "2008-01-01"
    _eff_yrs = max(1.0, (datetime.strptime(last_trade_date, "%Y-%m-%d") - datetime.strptime(_BASE, "%Y-%m-%d")).days / 365.25)
    # 目标 stride ≈ 3 年/个, 不超过 10 个 fold
    fold_target = min(10, max(3, round(_eff_yrs / 3)))
    _log(f"数据跨度: {total_years:.1f}年 (有效: {_eff_yrs:.1f}年)  自动fold: {fold_target}")

    # ---- 数据划分参数 (根据单位动态构建) ----
    # valid: 根据单位传递对应参数
    valid_years_arg = valid_val if valid_unit == "year" else 0
    valid_months_arg = valid_val if valid_unit == "month" else 0
    valid_weeks_arg = valid_val if valid_unit == "week" else 0

    # test + step: 根据 test 单位确定 mode
    valid_desc = f"{valid_val}{valid_unit}"
    if test_unit == "year":
        # annual mode: stride = effective_years / fold_target
        stride_y = max(1, round(_eff_yrs / fold_target))
        test_years_arg, test_weeks_arg = test_val, 1  # test-weeks 必须 > 0
        step_years_final, step_weeks_final = stride_y, 0
        step_desc = f"{stride_y}y"
    else:
        # weekly mode: 部分 anchor 被跳过, 用 fold*2 补偿
        step_weeks_final = max(1, round(_eff_yrs * 52 / (fold_target * 2)))
        test_weeks_arg = test_val * 4 if test_unit == "month" else test_val
        test_years_arg = 1  # test-years 必须 > 0
        step_years_final = 0
        step_desc = f"{step_weeks_final}w"

    cmd += [
        "--step-years", str(step_years_final),
        "--step-weeks", str(step_weeks_final),
        "--train-years", str(train_years),
        "--valid-years", str(valid_years_arg),
        "--valid-months", str(valid_months_arg),
        "--valid-weeks", str(valid_weeks_arg),
        "--test-years", str(test_years_arg),
        "--test-weeks", str(test_weeks_arg),
        "--model-mode", model_mode,
        "--hold-num", str(hold_num),
    ]

    _log(f"数据划分: train={train_years}y  valid={valid_desc}  test={test_val}{test_unit}  stride={step_desc} (目标{fold_target}fold)")

    _log(f"启动本地 Qlib 训练 (超时 {TRAIN_TIMEOUT}s)...", progress=0.03, phase="初始化 Qlib")
    _log(f"脚本: {cfg['script']}")
    progress_tracker = _TrainingProgressTracker()

    def _handle_training_line(line: str) -> None:
        _log(f"[Local] {line[:300]}", **progress_tracker.update(line))

    run_streaming(
        cmd,
        _handle_training_line,
        timeout=TRAIN_TIMEOUT,
        cwd=PROJECT_ROOT,
        env=runtime_env,
        cancel_event=cancel_event,
    )
    _log("本地 Qlib 训练完成，正在提取结果...")

    # 提取结果
    backtest_metrics = _parse_backtest_metrics(analysis_root, model_name)

    # 构建结果消息
    msg_parts = [f"训练完成: {model_name}"]
    if backtest_metrics.get("sharpe_ratio") is not None:
        msg_parts.append(f"夏普比: {backtest_metrics['sharpe_ratio']}")
    if backtest_metrics.get("annualized_return") is not None:
        msg_parts.append(f"年化收益: {backtest_metrics['annualized_return']*100:.2f}%")
    if backtest_metrics.get("max_drawdown") is not None:
        msg_parts.append(f"最大回撤: {backtest_metrics['max_drawdown']*100:.2f}%")
    if backtest_metrics.get("ic_mean") is not None:
        msg_parts.append(f"IC均值: {backtest_metrics['ic_mean']}")
    msg = " | ".join(msg_parts)

    _log(msg, progress=1.0, status="completed", model_name=model_name)

    return {
        "success": True,
        "model_name": model_name,
        "message": msg,
        "backtest_metrics": backtest_metrics,
    }


def _print_progress(data):
    """CLI 进度回调."""
    msg = data.get("message", "")
    status = data.get("status", "")
    pct = data.get("progress", 0)
    if pct:
        print(f"[{pct*100:.0f}%] {msg}", file=sys.stderr)
    else:
        print(f"[train] {msg}", file=sys.stderr)


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Qlib 训练执行器")
    ap.add_argument("--market", default="csi300", choices=["csi300", "csi500", "csi1000"])
    ap.add_argument("--model-mode", default="robust", choices=["default", "robust"])
    ap.add_argument("--hold-num", type=int, default=20)
    ap.add_argument("--lightgbm-only", action="store_true", default=True)
    args = ap.parse_args()

    result = run_training(
        market=args.market,
        model_mode=args.model_mode,
        hold_num=args.hold_num,
        lightgbm_only=args.lightgbm_only,
        progress_callback=_print_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
