"""Regression tests for decision safety, scoring availability, and financial provenance."""
import unittest

from analysis.agent import StockAnalysisAgent
from analysis.agents.base import CIODecision
from analysis.agents.cio import CIOAgent
from analysis.batch_analyzer import BatchAnalyzer
from analysis.nodes.prediction_node import PredictionNode
from analysis.scoring import ScoringEngine
from analysis.state.state import AnalysisState
from market_data.a_stock_provider import FinancialSummary
from market_data.provider_adapter import AdvancedBackend


class DecisionGuardrailTests(unittest.TestCase):
    def test_conflicting_buy_is_downgraded_to_observe(self):
        payload = {
            "outlook": "看多",
            "confidence": "高",
            "short_term": {"direction": "下跌", "change_pct": -3},
            "mid_term": {"direction": "下跌", "change_pct": -5},
            "long_term": {"direction": "震荡", "change_pct": 0},
            "suggested_action": {
                "action": "买入", "reason": "示例", "stop_loss": 93, "take_profit": 115,
            },
            "price_target_low": 93,
            "price_target_high": 110,
        }

        result = PredictionNode._apply_decision_guardrails(
            {"quote": {"price": 100}, "shares": 0}, payload
        )

        self.assertEqual(result["outlook"], "中性")
        self.assertEqual(result["confidence"], "低")
        self.assertEqual(result["suggested_action"]["action"], "观望")
        self.assertIsNone(result["suggested_action"]["stop_loss"])
        self.assertIsNone(result["suggested_action"]["take_profit"])

    def test_position_dependent_actions_need_a_position(self):
        result = PredictionNode._apply_decision_guardrails(
            {"quote": {"price": 100}, "shares": 0},
            {
                "outlook": "看多",
                "short_term": {"direction": "上涨", "change_pct": 2},
                "mid_term": {"direction": "上涨", "change_pct": 4},
                "suggested_action": {"action": "加仓"},
            },
        )

        self.assertEqual(result["suggested_action"]["action"], "观望")

    def test_buy_requires_a_usable_quote(self):
        result = PredictionNode._apply_decision_guardrails(
            {
                "quote": {},
                "data_provenance": {"sections": {"quote": {"status": "unavailable"}}},
            },
            {
                "outlook": "看多",
                "short_term": {"direction": "上涨", "change_pct": 2},
                "mid_term": {"direction": "上涨", "change_pct": 4},
                "suggested_action": {"action": "买入"},
            },
        )

        self.assertEqual(result["suggested_action"]["action"], "观望")

    def test_mock_quote_requires_observe(self):
        result = PredictionNode._apply_decision_guardrails(
            {
                "quote": {"price": 100, "source": "mock"},
                "data_provenance": {"sections": {"quote": {"status": "mock"}}},
            },
            {
                "outlook": "看多",
                "short_term": {"direction": "上涨", "change_pct": 2},
                "mid_term": {"direction": "上涨", "change_pct": 4},
                "suggested_action": {"action": "买入"},
            },
        )

        self.assertEqual(result["suggested_action"]["action"], "观望")

    def test_master_output_is_replaced_with_the_guarded_action(self):
        decision = CIODecision(
            short_term={"direction": "下跌", "change_pct": -2},
            mid_term={"direction": "下跌", "change_pct": -4},
            long_term={"direction": "震荡", "change_pct": 0},
            order={"action": "买入", "entry_conditions": "示例", "stop_loss": {"level": 93}},
            decision_quality={"confidence": "中"},
        )
        node = PredictionNode.__new__(PredictionNode)

        result = node._build_master_result({"quote": {"price": 100}, "shares": 0}, decision, [])

        self.assertEqual(result.suggested_action["action"], "观望")
        self.assertEqual(decision.order["action"], "观望")

    def test_master_position_is_limited_by_available_cash(self):
        decision = CIODecision(
            short_term={"direction": "上涨", "change_pct": 2},
            mid_term={"direction": "上涨", "change_pct": 4},
            long_term={"direction": "上涨", "change_pct": 6},
            order={"action": "买入", "position_size_pct": 100},
            decision_quality={"confidence": "中"},
        )
        node = PredictionNode.__new__(PredictionNode)

        node._build_master_result(
            {
                "quote": {"price": 100},
                "shares": 0,
                "total_assets": 1000,
                "available_cash": 200,
            },
            decision,
            [],
        )

        self.assertEqual(decision.order["position_size_pct"], 20.0)


class BatchGuardrailTests(unittest.TestCase):
    def test_batch_buy_cannot_override_single_stock_observe(self):
        analyzer = BatchAnalyzer.__new__(BatchAnalyzer)
        results = [{
            "symbol": "600001",
            "data": {
                "quote": {"price": 10, "source": "akshare"},
                "data_provenance": {"sections": {"quote": {"status": "fresh"}}},
                "prediction_summary": {
                    "outlook": "看空",
                    "short_term": {"direction": "下跌"},
                    "mid_term": {"direction": "下跌"},
                    "suggested_action": {"action": "观望"},
                },
                "shares": 0,
            },
        }]

        guarded = analyzer._guard_quality_pick(
            {
                "best_stock": {
                    "symbol": "600001", "suggested_action": "买入",
                    "suggested_position_pct": 80, "reasons": [],
                },
                "selection_rationale": "示例",
            },
            results,
            total_assets=1000,
            available_cash=500,
        )

        self.assertEqual(guarded["best_stock"]["suggested_action"], "观望")
        self.assertEqual(guarded["best_stock"]["suggested_position_pct"], 0)


class ScoringAvailabilityTests(unittest.TestCase):
    def test_missing_financial_fields_are_neutral_not_negative(self):
        state = AnalysisState(
            symbol="000001",
            quote={"price": 10, "prev_close": 10, "change_pct": 0},
            technical_indicators={"rsi_14": 50},
            financial_summary={"eps": None, "roe": None, "revenue": None},
        )

        result = ScoringEngine().compute(state)

        self.assertEqual(result.fundamental, 0.0)

    def test_missing_dividend_yield_is_not_inferred_from_roe(self):
        engine = ScoringEngine()
        self.assertEqual(engine._dividend_score({"eps": 3, "roe": 25, "debt_ratio": 20}), 0.0)
        self.assertEqual(engine._dividend_score({"dividend_yield": 6.5}), 0.5)


class FinancialProvenanceTests(unittest.TestCase):
    def test_same_period_fallback_keeps_field_sources(self):
        primary = FinancialSummary(symbol="000001", name="示例", eps=1.2, report_date="20251231")
        fallback = FinancialSummary(
            symbol="000001", name="示例", roe=12.5, revenue=100.0, report_date="20251231"
        )
        AdvancedBackend._tag_financial_fields(primary, "akshare")
        AdvancedBackend._tag_financial_fields(fallback, "tushare")

        merged = AdvancedBackend._merge_missing_financial_fields(primary, fallback, "tushare")

        self.assertEqual(set(merged), {"revenue", "roe"})
        self.assertEqual(primary.eps, 1.2)
        self.assertEqual(primary.roe, 12.5)
        self.assertEqual(primary.field_sources["eps"], "akshare")
        self.assertEqual(primary.field_sources["roe"], "tushare")

    def test_different_reporting_periods_are_not_merged(self):
        primary = FinancialSummary(symbol="000001", name="示例", report_date="20260331")
        fallback = FinancialSummary(symbol="000001", name="示例", roe=12.5, report_date="20251231")

        merged = AdvancedBackend._merge_missing_financial_fields(primary, fallback, "tushare")

        self.assertEqual(merged, [])
        self.assertIsNone(primary.roe)


class PePercentileTests(unittest.TestCase):
    def test_tied_pe_values_use_midpoint_rank(self):
        percentile = StockAnalysisAgent._pe_percentile([10.0] * 39 + [500.0], 10.0)

        self.assertAlmostEqual(percentile, 48.75)


class ScenarioProbabilityTests(unittest.TestCase):
    def test_scenario_probabilities_are_normalized(self):
        agent = CIOAgent.__new__(CIOAgent)
        decision = agent._parse_cio_result(
            {
                "base_case": {"probability": 0.6},
                "bull_case": {"probability": 0.6},
                "bear_case": {"probability": 0.3},
                "decision_quality": {"confidence": "高"},
            },
            {"name": "测试", "key": "test"},
        )

        total = sum(
            case["probability"]
            for case in (decision.base_case, decision.bull_case, decision.bear_case)
        )
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertEqual(decision.decision_quality["confidence"], "低")


if __name__ == "__main__":
    unittest.main()
