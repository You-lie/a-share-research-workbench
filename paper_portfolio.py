"""Local-only paper portfolio ledger. It never talks to a broker."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


DEFAULT_SETTINGS = {
    "initial_cash": 100000.0,
    "commission_rate": 0.0001,
    "sell_stamp_duty_rate": 0.0005,
}
BUY_ACTIONS = {"buy", "add"}
SELL_ACTIONS = {"reduce", "clear"}
ALL_ACTIONS = BUY_ACTIONS | SELL_ACTIONS


class PaperPortfolioStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Path(__file__).resolve().parent / "data" / "paper_portfolio.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission_rate REAL NOT NULL,
                    stamp_duty_rate REAL NOT NULL,
                    commission REAL NOT NULL,
                    stamp_duty REAL NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    analysis_snapshot TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    void_reason TEXT NOT NULL DEFAULT '',
                    voided_at TEXT,
                    correction_of INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quote_snapshots (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    price REAL NOT NULL,
                    quote_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(trades)")}
            if "voided_at" not in columns:
                connection.execute("ALTER TABLE trades ADD COLUMN voided_at TEXT")
            if "correction_of" not in columns:
                connection.execute("ALTER TABLE trades ADD COLUMN correction_of INTEGER")
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key, value, updated_at) VALUES (?, ?, ?)",
                    (key, str(value), datetime.now().isoformat()),
                )

    def get_settings(self) -> dict:
        with self._lock, self._connection() as connection:
            values = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM settings")}
        return {key: float(values.get(key, default)) for key, default in DEFAULT_SETTINGS.items()}

    def update_settings(self, payload: dict) -> dict:
        values = self.get_settings()
        for key in DEFAULT_SETTINGS:
            if key not in payload:
                continue
            try:
                parsed = float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} 必须是数字") from exc
            if parsed < 0:
                raise ValueError(f"{key} 不能小于 0")
            if key.endswith("rate") and parsed > 0.1:
                raise ValueError(f"{key} 不能超过 10%")
            values[key] = parsed
        now = datetime.now().isoformat()
        with self._lock, self._connection() as connection:
            for key, value in values.items():
                connection.execute(
                    "INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (key, str(value), now),
                )
        return values

    def _active_trades(self) -> list[dict]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM trades WHERE status = 'active' ORDER BY trade_at ASC, id ASC"
            ).fetchall()
        return [self._trade_row(row) for row in rows]

    @staticmethod
    def _trade_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        try:
            result["analysis_snapshot"] = json.loads(result.get("analysis_snapshot") or "{}")
        except json.JSONDecodeError:
            result["analysis_snapshot"] = {}
        return result

    @staticmethod
    def _ledger(trades: list[dict], initial_cash: float) -> tuple[float, dict[str, dict]]:
        cash = float(initial_cash)
        positions: dict[str, dict] = {}
        for trade in trades:
            symbol = trade["symbol"]
            position = positions.setdefault(symbol, {
                "symbol": symbol, "name": trade["name"], "quantity": 0,
                "cost_basis": 0.0, "realized_pnl": 0.0,
            })
            quantity = int(trade["quantity"])
            amount = quantity * float(trade["price"])
            costs = float(trade["commission"]) + float(trade["stamp_duty"])
            if trade["action"] in BUY_ACTIONS:
                cash -= amount + costs
                position["quantity"] += quantity
                position["cost_basis"] += amount + costs
            else:
                if quantity > position["quantity"]:
                    raise ValueError(f"交易流水异常：{symbol} 的卖出数量超过持仓")
                average_cost = position["cost_basis"] / position["quantity"] if position["quantity"] else 0.0
                disposed_cost = average_cost * quantity
                proceeds = amount - costs
                cash += proceeds
                position["realized_pnl"] += proceeds - disposed_cost
                position["quantity"] -= quantity
                position["cost_basis"] -= disposed_cost
        return cash, positions

    def _snapshot(self, symbol: str) -> Optional[dict]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM quote_snapshots WHERE symbol = ?", (symbol,)).fetchone()
        return dict(row) if row else None

    def _save_snapshot(self, symbol: str, name: str, price: float, quote_at: str, source: str) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO quote_snapshots(symbol, name, price, quote_at, source, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET name=excluded.name, price=excluded.price, quote_at=excluded.quote_at, source=excluded.source, updated_at=excluded.updated_at",
                (symbol, name, price, quote_at, source, now),
            )

    def _prepare_trade(self, payload: dict, active_trades: list[dict]) -> dict:
        action = str(payload.get("action") or "").strip().lower()
        if action not in ALL_ACTIONS:
            raise ValueError("操作只能是 buy、add、reduce 或 clear")
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("股票代码不能为空")
        try:
            price = float(payload.get("price"))
        except (TypeError, ValueError) as exc:
            raise ValueError("成交价必须是正数") from exc
        if price <= 0:
            raise ValueError("成交价必须是正数")
        settings = self.get_settings()
        cash, positions = self._ledger(active_trades, settings["initial_cash"])
        current_quantity = positions.get(symbol, {}).get("quantity", 0)
        raw_quantity = payload.get("quantity")
        if action == "clear" and raw_quantity in (None, "", 0, "0"):
            quantity = current_quantity
        else:
            try:
                parsed_quantity = float(raw_quantity)
            except (TypeError, ValueError) as exc:
                raise ValueError("成交数量必须是整数") from exc
            if not parsed_quantity.is_integer():
                raise ValueError("成交数量必须是整数")
            quantity = int(parsed_quantity)
        if quantity <= 0:
            raise ValueError("当前没有可清空的持仓" if action == "clear" else "成交数量必须大于 0")
        if action in BUY_ACTIONS and quantity % 100:
            raise ValueError("A 股买入和加仓数量必须是 100 股整数倍")
        if action in SELL_ACTIONS and quantity > current_quantity:
            raise ValueError("减持或清空数量不能超过当前纸面持仓")
        commission = round(price * quantity * settings["commission_rate"], 2)
        stamp_duty = round(price * quantity * settings["sell_stamp_duty_rate"], 2) if action in SELL_ACTIONS else 0.0
        if action in BUY_ACTIONS and price * quantity + commission > cash + 1e-8:
            raise ValueError("纸面可用资金不足，请调整成交价、数量或初始资金")
        analysis_snapshot = payload.get("analysis_snapshot") or {}
        if not isinstance(analysis_snapshot, dict):
            raise ValueError("分析快照格式不正确")
        trade_at = str(payload.get("trade_at") or datetime.now().strftime("%Y-%m-%dT%H:%M"))
        return {
            "trade_at": trade_at,
            "symbol": symbol,
            "name": str(payload.get("name") or ""),
            "action": action,
            "quantity": quantity,
            "price": price,
            "commission_rate": settings["commission_rate"],
            "stamp_duty_rate": settings["sell_stamp_duty_rate"],
            "commission": commission,
            "stamp_duty": stamp_duty,
            "note": str(payload.get("note") or "")[:2000],
            "analysis_snapshot": analysis_snapshot,
        }

    @staticmethod
    def _insert_trade(connection: sqlite3.Connection, prepared: dict, correction_of: Optional[int] = None) -> int:
        now = datetime.now().isoformat()
        cursor = connection.execute(
            """INSERT INTO trades (
                trade_at, symbol, name, action, quantity, price, commission_rate, stamp_duty_rate,
                commission, stamp_duty, note, analysis_snapshot, status, correction_of, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (
                prepared["trade_at"], prepared["symbol"], prepared["name"], prepared["action"],
                prepared["quantity"], prepared["price"], prepared["commission_rate"],
                prepared["stamp_duty_rate"], prepared["commission"], prepared["stamp_duty"],
                prepared["note"], json.dumps(prepared["analysis_snapshot"], ensure_ascii=False),
                correction_of, now,
            ),
        )
        return int(cursor.lastrowid)

    def create_trade(self, payload: dict) -> dict:
        prepared = self._prepare_trade(payload, self._active_trades())
        with self._lock, self._connection() as connection:
            trade_id = self._insert_trade(connection, prepared)
        return self.get_trade(trade_id)

    def get_trade(self, trade_id: int) -> dict:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            raise LookupError("交易记录不存在")
        return self._trade_row(row)

    def void_trade(self, trade_id: int, reason: str = "") -> dict:
        trade = self.get_trade(trade_id)
        if trade["status"] != "active":
            raise ValueError("该交易已经撤销")
        with self._lock, self._connection() as connection:
            later = connection.execute(
                "SELECT 1 FROM trades WHERE symbol = ? AND status = 'active' AND (trade_at > ? OR (trade_at = ? AND id > ?)) LIMIT 1",
                (trade["symbol"], trade["trade_at"], trade["trade_at"], trade_id),
            ).fetchone()
            if later:
                raise ValueError("只能撤销该股票最新一笔有效交易；请先更正后续记录")
            connection.execute(
                "UPDATE trades SET status = 'voided', void_reason = ?, voided_at = ? WHERE id = ?",
                (str(reason or "手工撤销")[:500], datetime.now().isoformat(), trade_id),
            )
        return self.get_trade(trade_id)

    def correct_trade(self, trade_id: int, payload: dict) -> dict:
        """Void the latest active record and append its corrected replacement atomically."""
        original = self.get_trade(trade_id)
        if original["status"] != "active":
            raise ValueError("只能更正有效交易记录")
        replacement_payload = dict(payload)
        replacement_payload.setdefault("analysis_snapshot", original["analysis_snapshot"])
        active_without_original = [
            trade for trade in self._active_trades() if trade["id"] != trade_id
        ]
        prepared = self._prepare_trade(replacement_payload, active_without_original)
        correction_reason = str(payload.get("correction_reason") or "已更正为后续交易记录")[:500]
        with self._lock, self._connection() as connection:
            later = connection.execute(
                "SELECT 1 FROM trades WHERE symbol = ? AND status = 'active' AND (trade_at > ? OR (trade_at = ? AND id > ?)) LIMIT 1",
                (original["symbol"], original["trade_at"], original["trade_at"], trade_id),
            ).fetchone()
            if later:
                raise ValueError("只能更正该股票最新一笔有效交易；请先更正后续记录")
            connection.execute(
                "UPDATE trades SET status = 'voided', void_reason = ?, voided_at = ? WHERE id = ?",
                (correction_reason, datetime.now().isoformat(), trade_id),
            )
            corrected_id = self._insert_trade(connection, prepared, correction_of=trade_id)
        return self.get_trade(corrected_id)

    def list_trades(self, limit: int = 200) -> list[dict]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT * FROM trades ORDER BY trade_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [self._trade_row(row) for row in rows]

    def overview(self, quote_fetcher: Optional[Callable[[str], Any]] = None) -> dict:
        settings = self.get_settings()
        cash, positions = self._ledger(self._active_trades(), settings["initial_cash"])
        displayed_positions = []
        total_market_value = 0.0
        has_unpriced_position = False
        for symbol, position in positions.items():
            if position["quantity"] <= 0:
                continue
            quote_data = None
            if quote_fetcher:
                try:
                    raw = quote_fetcher(symbol)
                    quote_data = raw.to_dict() if hasattr(raw, "to_dict") else raw
                except Exception:
                    quote_data = None
            if isinstance(quote_data, dict) and float(quote_data.get("price") or 0) > 0:
                price = float(quote_data["price"])
                quote_at = str(quote_data.get("timestamp") or datetime.now().isoformat())
                source = str(quote_data.get("source") or "")
                name = str(quote_data.get("name") or position["name"])
                self._save_snapshot(symbol, name, price, quote_at, source)
                market_status = "mock" if source.lower() in {"mock", "模拟数据"} else "fresh"
            else:
                snapshot = self._snapshot(symbol)
                price = float(snapshot["price"]) if snapshot else None
                quote_at = snapshot["quote_at"] if snapshot else ""
                source = snapshot["source"] if snapshot else ""
                name = snapshot["name"] if snapshot and snapshot["name"] else position["name"]
                market_status = "mock" if snapshot and source.lower() in {"mock", "模拟数据"} else "cached" if snapshot else "unavailable"
            market_value = price * position["quantity"] if price is not None else None
            if market_value is not None:
                total_market_value += market_value
            else:
                has_unpriced_position = True
            average_cost = position["cost_basis"] / position["quantity"]
            unrealized = market_value - position["cost_basis"] if market_value is not None else None
            displayed_positions.append({
                **position, "name": name, "average_cost": average_cost, "market_price": price,
                "market_value": market_value, "unrealized_pnl": unrealized,
                "quote_at": quote_at, "quote_source": source, "market_status": market_status,
            })
        realized = sum(position["realized_pnl"] for position in positions.values())
        unrealized = None if has_unpriced_position else sum(item["unrealized_pnl"] or 0 for item in displayed_positions)
        return {
            "settings": settings,
            "cash": cash,
            "market_value": None if has_unpriced_position else total_market_value,
            "total_assets": None if has_unpriced_position else cash + total_market_value,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "valuation_status": "unavailable" if has_unpriced_position else "cached" if any(item["market_status"] == "cached" for item in displayed_positions) else "mock" if any(item["market_status"] == "mock" for item in displayed_positions) else "fresh",
            "positions": displayed_positions,
            "updated_at": datetime.now().isoformat(),
        }
