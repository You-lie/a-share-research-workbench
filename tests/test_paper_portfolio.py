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


class PaperPortfolioFeeTests(unittest.TestCase):
    """Realistic A-share cost model: min commission, transfer fee, sell-only stamp duty."""

    def setUp(self):
        self.db_path = Path(__file__).resolve().parent.parent / "data" / ".paper_fee_test.db"
        for suffix in ("", "-wal", "-shm"):
            (Path(f"{self.db_path}{suffix}")).unlink(missing_ok=True)
        self.store = PaperPortfolioStore(self.db_path)
        self.store.update_settings({"initial_cash": 500000})

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            (Path(f"{self.db_path}{suffix}")).unlink(missing_ok=True)

    def _trade(self, action, trade_at, symbol, quantity, price):
        return self.store.create_trade({
            "action": action, "trade_at": trade_at, "symbol": symbol,
            "quantity": quantity, "price": price,
        })

    def test_default_settings_use_realistic_cost_model(self):
        settings = self.store.get_settings()
        self.assertEqual(settings["min_commission"], 5.0)
        self.assertAlmostEqual(settings["commission_rate"], 0.00025)
        self.assertAlmostEqual(settings["transfer_fee_rate"], 0.00001)

    def test_small_buy_triggers_min_commission(self):
        trade = self._trade("buy", "2026-10-07T09:30", "300896", 100, 95)
        # 9500 × 0.025% = 2.375 -> bumped to 5.0; transfer fee 0.095 -> 0.10
        self.assertEqual(trade["commission"], 5.0)
        self.assertEqual(trade["stamp_duty"], 0.0, "buy must not pay stamp duty")
        self.assertAlmostEqual(trade["transfer_fee"], 0.1, places=2)

    def test_large_sell_pays_all_three_costs(self):
        self._trade("buy", "2026-10-07T09:31", "600519", 100, 1111)
        trade = self._trade("clear", "2026-10-07T09:32", "600519", 100, 1111)
        # 111100 × 0.025% = 27.775 -> 27.78; stamp 55.55; transfer 1.111 -> 1.11
        self.assertAlmostEqual(trade["commission"], 27.78, places=2)
        self.assertAlmostEqual(trade["stamp_duty"], 55.55, places=2)
        self.assertAlmostEqual(trade["transfer_fee"], 1.11, places=2)

    def test_cash_check_includes_new_costs(self):
        self.store.update_settings({"initial_cash": 9503.0})
        with self.assertRaisesRegex(ValueError, "可用资金不足"):
            # 9500 + 5.0 + 0.10 = 9505.10 > 9503
            self._trade("buy", "2026-10-07T09:30", "300896", 100, 95)


if __name__ == "__main__":
    unittest.main()
