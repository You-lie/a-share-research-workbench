"""Local LLM usage tracking: daily token counters and balance snapshots.

All data stays in project-local JSON files; nothing is sent anywhere except the
provider's own balance endpoint when the UI asks for it.
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Keep runtime stats next to the other local data (paper_portfolio.db etc.)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USAGE_PATH = DATA_DIR / "llm_usage.json"
BALANCE_PATH = DATA_DIR / "llm_balance.json"
_RETENTION_DAYS = 90

_lock = threading.RLock()


def _today() -> str:
    return date.today().isoformat()


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning(f"用量记录写入失败: {exc}")


def _prune(data: Dict[str, Any]) -> Dict[str, Any]:
    cutoff = (date.today() - timedelta(days=_RETENTION_DAYS)).isoformat()
    return {key: value for key, value in data.items() if key >= cutoff}


def record_usage(prompt_tokens: Any, completion_tokens: Any) -> None:
    """Add one LLM call's token usage to today's counter."""
    try:
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return
    if prompt < 0 or completion < 0:
        return
    with _lock:
        data = _prune(_load(USAGE_PATH))
        day = data.setdefault(_today(), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        day["calls"] = int(day.get("calls", 0)) + 1
        day["prompt_tokens"] = int(day.get("prompt_tokens", 0)) + prompt
        day["completion_tokens"] = int(day.get("completion_tokens", 0)) + completion
        _save(USAGE_PATH, data)


def get_today_usage() -> Dict[str, int]:
    with _lock:
        data = _load(USAGE_PATH)
        day = data.get(_today()) or {}
        prompt = int(day.get("prompt_tokens", 0))
        completion = int(day.get("completion_tokens", 0))
        return {
            "date": _today(),
            "calls": int(day.get("calls", 0)),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


def record_balance_snapshot(balance: Optional[float]) -> None:
    """Store the first observed balance of each day for daily-spend estimation."""
    if balance is None:
        return
    try:
        value = float(balance)
    except (TypeError, ValueError):
        return
    with _lock:
        data = _prune(_load(BALANCE_PATH))
        day = _today()
        if day not in data:
            data[day] = value
            _save(BALANCE_PATH, data)


def get_daily_spend(current_balance: Optional[float]) -> Optional[float]:
    """Estimated spend today = first balance of today minus current balance.

    Returns None when unknown (no snapshot, fetch failure) or when the delta is
    negative (e.g. a top-up happened), so we never show a misleading number.
    """
    if current_balance is None:
        return None
    with _lock:
        data = _load(BALANCE_PATH)
        first = data.get(_today())
    if first is None:
        return None
    try:
        spend = float(first) - float(current_balance)
    except (TypeError, ValueError):
        return None
    if spend < 0:
        return None
    return round(spend, 4)
