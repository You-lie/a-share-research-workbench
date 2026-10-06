"""Regression tests for the audit fixes:

- BaoStock quote field mapping (was fully misaligned and crashed).
- Batch fallback ranking/sorting with missing scores.
- Task dict pruning keeps running tasks and caps finished history.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BaoStockFieldMappingTests(unittest.TestCase):
    def test_row_field_indices_align_with_query(self):
        """The quote parser must read close at index 4, peTTM at 7, pbMRQ at 8."""
        source = Path(__file__).resolve().parent.parent / "market_data" / "a_stock_provider.py"
        text = source.read_text(encoding="utf-8")
        # Locate the BaoStock quote block and verify the corrected indices.
        start = text.index("date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn")
        block = text[start:start + 2600]
        self.assertIn("price = float(r[4])", block, "close price must come from r[4]")
        self.assertIn("pe = float(r[7])", block, "peTTM must come from r[7]")
        self.assertIn("pb = float(r[8])", block, "pbMRQ must come from r[8]")
        self.assertIn("turnover = float(r[9])", block, "turn must come from r[9]")
        self.assertIn("open_=float(r[1])", block, "open must come from r[1]")
        self.assertNotIn("open_=float(r[0])", block, "r[0] is the date, not open")


class FallbackRankingTests(unittest.TestCase):
    def _analyzer(self):
        from analysis.batch_analyzer import BatchAnalyzer
        return BatchAnalyzer.__new__(BatchAnalyzer)

    def test_missing_score_sorts_last_and_is_not_zero(self):
        analyzer = self._analyzer()
        results = [
            {'symbol': 'AAA', 'data': {'score_breakdown': {'final': 1.2}, 'valuation_percentile': 30}},
            {'symbol': 'BBB', 'data': {'score_breakdown': {}, 'valuation_percentile': None}},
            {'symbol': 'CCC', 'data': {'score_breakdown': {'final': -0.4}, 'valuation_percentile': 80}},
        ]
        summary = analyzer._fallback_summary(results)
        ranked = [item['symbol'] for item in summary['ranking']]
        self.assertEqual(ranked[0], 'AAA')
        self.assertEqual(ranked[1], 'CCC')
        self.assertEqual(ranked[2], 'BBB', 'missing score must sort last')
        bbb = next(item for item in summary['ranking'] if item['symbol'] == 'BBB')
        self.assertIsNone(bbb['score'], 'missing score must stay None, not 0')
        self.assertIn('N/A', bbb['reason'])

    def test_pick_skips_missing_scores_and_returns_none_when_all_missing(self):
        analyzer = self._analyzer()
        all_missing = [
            {'symbol': 'AAA', 'data': {'score_breakdown': {}}},
            {'symbol': 'BBB', 'data': {'score_breakdown': {}}},
        ]
        pick = analyzer._fallback_pick(all_missing)
        self.assertIsNone(pick['best_stock'])
        self.assertIn('缺少可用综合评分', pick['selection_rationale'])

    def test_pick_prefers_scored_stock(self):
        analyzer = self._analyzer()
        results = [
            {'symbol': 'AAA', 'data': {'score_breakdown': {}}},
            {'symbol': 'BBB', 'data': {'score_breakdown': {'final': 0.9}, 'valuation_percentile': 20}},
        ]
        pick = analyzer._fallback_pick(results)
        self.assertIsNotNone(pick['best_stock'])
        self.assertEqual(pick['best_stock']['symbol'], 'BBB')


class TaskPruningTests(unittest.TestCase):
    def test_prune_keeps_running_and_caps_finished(self):
        import app as app_module

        tasks = {}
        # 70 finished + 2 running => after prune it should be 62 total
        for i in range(70):
            tasks[f'done_{i}'] = {'status': 'completed', 'completed_at': f'2026-10-01T00:00:{i:02d}'}
        tasks['running_1'] = {'status': 'running'}
        tasks['running_2'] = {'status': 'pending'}

        app_module._prune_tasks_locked(tasks)
        self.assertIn('running_1', tasks)
        self.assertIn('running_2', tasks)
        self.assertEqual(len(tasks), app_module._TASK_HISTORY_LIMIT)
        # Oldest finished tasks are the ones removed
        self.assertNotIn('done_0', tasks)
        self.assertIn('done_69', tasks)

    def test_prune_noop_below_limit(self):
        import app as app_module
        tasks = {f't{i}': {'status': 'completed', 'completed_at': 'x'} for i in range(5)}
        app_module._prune_tasks_locked(tasks)
        self.assertEqual(len(tasks), 5)


if __name__ == '__main__':
    unittest.main()
