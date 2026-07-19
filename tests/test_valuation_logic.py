"""Regression tests for valuation references shown to investment research users."""
import unittest
from unittest.mock import patch

from analysis.agent import StockAnalysisAgent
from analysis.agents.valuation_agent import ValuationAgent
from analysis.batch_analyzer import BatchAnalyzer
from analysis.nodes.prediction_node import PredictionNode
from analysis.state.state import AnalysisState


class ValuationReferenceTests(unittest.TestCase):
    def setUp(self):
        # Avoid constructing network/data providers; these tests exercise only valuation rules.
        self.agent = StockAnalysisAgent.__new__(StockAnalysisAgent)
        self.agent._compute_long_term_pe = lambda *args: None

    @patch("analysis.agent.cache_manager.get_historical_pe")
    def test_negative_pe_has_no_valuation_reference(self, historical_pe):
        state = AnalysisState(symbol="000001", quote={"pe": -54.0, "price": 12.5})

        self.agent._compute_valuation("000001", state)

        historical_pe.assert_not_called()
        self.assertEqual(state.valuation_status, "unavailable")
        self.assertEqual(state.valuation_level, "PE不适用")
        self.assertIsNone(state.valuation_percentile)
        self.assertIsNone(state.suggested_buy_price)

    @patch("analysis.agent.cache_manager.get_historical_pe", return_value=[])
    def test_missing_history_has_no_valuation_reference(self, _historical_pe):
        state = AnalysisState(symbol="000001", quote={"pe": 20.0, "price": 100.0})

        self.agent._compute_valuation("000001", state)

        self.assertEqual(state.valuation_status, "insufficient")
        self.assertEqual(state.valuation_level, "估值数据不足")
        self.assertIsNone(state.suggested_buy_price)

    @patch("analysis.agent.cache_manager.get_historical_pe", return_value=list(range(10, 50)))
    def test_valid_pe_history_uses_formula_without_fixed_discount(self, _historical_pe):
        state = AnalysisState(symbol="000001", quote={"pe": 20.0, "price": 100.0})

        self.agent._compute_valuation("000001", state)

        self.assertEqual(state.valuation_status, "available")
        self.assertEqual(state.historical_pe_avg, 29.5)
        self.assertEqual(state.historical_pe_median, 29.5)
        self.assertEqual(state.suggested_buy_price, 147.5)

    @patch("analysis.agent.cache_manager.get_historical_pe", return_value=[10.0] * 39 + [500.0])
    def test_valuation_reference_uses_median_not_an_extreme_pe(self, _historical_pe):
        state = AnalysisState(symbol="000001", quote={"pe": 10.0, "price": 100.0})

        self.agent._compute_valuation("000001", state)

        self.assertEqual(state.historical_pe_avg, 22.25)
        self.assertEqual(state.historical_pe_median, 10.0)
        self.assertEqual(state.suggested_buy_price, 100.0)
        self.assertEqual(state.valuation_percentile, 48.8)
        self.assertEqual(state.valuation_level, "正常")


class ValuationFallbackTests(unittest.TestCase):
    def test_rule_prediction_does_not_fabricate_a_price_range(self):
        predictor = PredictionNode.__new__(PredictionNode)

        prediction = predictor._rule_predict({
            "signals": {"score": 3.0},
            "quote": {"price": 100.0},
        })

        self.assertEqual(prediction.price_target_current, 100.0)
        self.assertIsNone(prediction.price_target_low)
        self.assertIsNone(prediction.price_target_high)

    def test_valuation_agent_does_not_score_missing_pe_as_normal(self):
        agent = ValuationAgent.__new__(ValuationAgent)
        agent.employee_id = "valuation"
        agent.role = "估值分析师"
        agent.department = "研究部"

        report = agent._rule_analyze({
            "valuation_percentile": None,
            "valuation_note": "当前PE缺失，不能按PE估值",
            "quote": {"pb": 2.0},
        })

        self.assertEqual(report.score, 0.0)
        self.assertIn("当前PE缺失", report.key_points[0])

    def test_batch_fallback_does_not_invent_a_50_percentile_penalty(self):
        analyzer = BatchAnalyzer.__new__(BatchAnalyzer)
        results = [
            {
                "symbol": "000002",
                "data": {"stock_name": "有估值", "valuation_percentile": 50.0,
                         "score_breakdown": {"final": 1.2}},
            },
            {
                "symbol": "000001",
                "data": {"stock_name": "PE缺失", "valuation_percentile": None,
                         "score_breakdown": {"final": 1.0}},
            },
        ]

        pick = analyzer._fallback_pick(results)

        self.assertEqual(pick["best_stock"]["symbol"], "000001")
        self.assertEqual(pick["best_stock"]["suggested_action"], "观望")
        self.assertEqual(pick["best_stock"]["suggested_position_pct"], 0)


if __name__ == "__main__":
    unittest.main()
