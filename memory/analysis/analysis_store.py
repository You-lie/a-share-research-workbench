"""File-backed archive for completed stock analyses.

Each analysis is stored under ``memory/analysis/<symbol>/<timestamp>/``.  The
source package lives beside those runtime directories, so callers must ignore
only the generated symbol directories rather than the whole ``analysis``
package.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from memory import ANALYSIS_DIR


_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_STORE_LOCK = threading.RLock()


class AnalysisStore:
    """Persist immutable JSON snapshots of completed analyses."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or ANALYSIS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # The application creates short-lived AnalysisStore instances from
        # multiple task threads, so the lock must be shared across instances.
        self._lock = _STORE_LOCK

    def save(
        self,
        *,
        symbol: str,
        state: dict[str, Any],
        timestamp: str,
        employee_reports: Optional[list[Any]] = None,
        cio_decision: Optional[dict[str, Any]] = None,
        prediction: Optional[dict[str, Any]] = None,
    ) -> str:
        """Atomically save one analysis and return its snapshot directory."""
        normalized_symbol = self._normalize_symbol(symbol)
        snapshot_name = self._snapshot_name(timestamp)

        with self._lock:
            symbol_dir = self.base_dir / normalized_symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            final_dir = self._unique_snapshot_dir(symbol_dir, snapshot_name)
            staging_dir = symbol_dir / f".tmp-{uuid.uuid4().hex}"

            try:
                staging_dir.mkdir()
                self._write_json(staging_dir / "state.json", state)
                if employee_reports is not None:
                    self._write_json(staging_dir / "employee_reports.json", employee_reports)
                if cio_decision is not None:
                    self._write_json(staging_dir / "cio_decision.json", cio_decision)
                if prediction is not None:
                    self._write_json(staging_dir / "prediction.json", prediction)
                staging_dir.replace(final_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

        return str(final_dir)

    def list_all_symbols(self) -> list[str]:
        """Return symbols that contain at least one complete analysis snapshot."""
        if not self.base_dir.is_dir():
            return []

        symbols: list[str] = []
        with self._lock:
            for symbol_dir in self.base_dir.iterdir():
                if not symbol_dir.is_dir() or symbol_dir.name.startswith((".", "__")):
                    continue
                if any(
                    snapshot.is_dir() and (snapshot / "state.json").is_file()
                    for snapshot in symbol_dir.iterdir()
                ):
                    symbols.append(symbol_dir.name)
        return sorted(symbols)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        value = str(symbol or "").strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(value):
            raise ValueError("股票代码只能包含字母、数字、点、下划线或连字符")
        return value

    @staticmethod
    def _snapshot_name(timestamp: str) -> str:
        raw = str(timestamp or "").strip()
        if not raw:
            parsed = datetime.now()
        else:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("分析时间必须是 ISO 8601 格式") from exc
        return parsed.strftime("%Y%m%dT%H%M%S_%f")

    @staticmethod
    def _unique_snapshot_dir(symbol_dir: Path, snapshot_name: str) -> Path:
        candidate = symbol_dir / snapshot_name
        suffix = 2
        while candidate.exists():
            candidate = symbol_dir / f"{snapshot_name}-{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
