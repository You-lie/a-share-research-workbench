"""Regression tests for history-PE provider selection in advanced mode."""
import unittest
from unittest.mock import patch

from market_data.provider_adapter import AdvancedBackend


class AdvancedHistoricalPeTests(unittest.TestCase):
    @patch("market_data.a_stock_provider.BaoStockBackend")
    @patch("market_data.tushare_provider.TushareBackend")
    def test_prefers_baostock_for_batch_safe_pe_history(self, tushare_backend, baostock_backend):
        baostock_backend.return_value.get_historical_pe.return_value = [12.0] * 30

        values = AdvancedBackend().get_historical_pe("600519", 365)

        self.assertEqual(values, [12.0] * 30)
        tushare_backend.assert_not_called()

    @patch("market_data.tushare_provider.TushareBackend")
    @patch("market_data.a_stock_provider.BaoStockBackend")
    def test_falls_back_to_tushare_when_baostock_has_too_few_samples(
        self, baostock_backend, tushare_backend
    ):
        baostock_backend.return_value.get_historical_pe.return_value = [8.0] * 29
        tushare_backend.return_value.get_historical_pe.return_value = [-1.0, 11.0, 12.0]

        values = AdvancedBackend().get_historical_pe("600519", 365)

        self.assertEqual(values, [11.0, 12.0])


if __name__ == "__main__":
    unittest.main()
