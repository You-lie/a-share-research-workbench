"""Regression tests for time-ordered paper portfolio accounting."""
import unittest
from pathlib import Path

from paper_portfolio import PaperPortfolioStore


class PaperPortfolioTimelineTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).resolve().parent.parent / "data" / ".paper_portfolio_test.db"
        for suffix in ("", "-wal", "-shm"):
            (Path(f"{self.db_path}{suffix}")).unlink(missing_ok=True)
        self.store = PaperPortfolioStore(self.db_path)
        self.store.update_settings({"initial_cash": 1000})

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            (Path(f"{self.db_path}{suffix}")).unlink(missing_ok=True)

    def _trade(self, action, trade_at, symbol, quantity, price):
        return self.store.create_trade({
            "action": action,
            "trade_at": trade_at,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
        })

    def test_correcting_historical_trade_cannot_use_later_sale_proceeds(self):
        first_buy = self._trade("buy", "2026-01-01T09:30", "600001", 100, 5)
        self._trade("buy", "2026-01-02T09:30", "600002", 100, 4)
        self._trade("reduce", "2026-01-03T09:30", "600002", 100, 6)

        with self.assertRaisesRegex(ValueError, "可用资金不足"):
            self.store.correct_trade(first_buy["id"], {
                "action": "buy",
                "trade_at": "2026-01-01T09:30",
                "symbol": "600001",
                "quantity": 100,
                "price": 9,
            })

    def test_clear_always_records_the_full_current_position(self):
        self._trade("buy", "2026-01-01T09:30", "600001", 200, 3)
        cleared = self._trade("clear", "2026-01-02T09:30", "600001", 100, 4)

        self.assertEqual(cleared["quantity"], 200)
        overview = self.store.overview()
        self.assertEqual(overview["positions"], [])


if __name__ == "__main__":
    unittest.main()
