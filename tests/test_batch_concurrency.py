"""Batch concurrency behavior: parallel workers, input order, cancel, progress."""
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import batch_analyzer as batch_module
from analysis.batch_analyzer import BatchAnalyzer


def _fake_analyze(self, symbol, **kwargs):
    """Simulate analysis: variable duration so completion order differs from input order."""
    durations = {'AAA': 0.30, 'BBB': 0.05, 'CCC': 0.15, 'DDD': 0.02, 'EEE': 0.08}
    time.sleep(durations.get(symbol, 0.05))
    return {
        'status': 'complete',
        'symbol': symbol,
        'stock_name': f'股票{symbol}',
        'score_breakdown': {'final': 1.0},
    }


class BatchConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self._orig_analyze = batch_module.StockAnalysisAgent.analyze
        batch_module.StockAnalysisAgent.analyze = _fake_analyze
        # Batch summary/pick call the real LLM; stub them so tests stay offline.
        self._orig_summarize = BatchAnalyzer._summarize
        self._orig_pick = BatchAnalyzer._pick_best
        BatchAnalyzer._summarize = lambda self, *a, **k: {'summary_text': 'stub', 'ranking': []}
        BatchAnalyzer._pick_best = lambda self, *a, **k: {'best_stock': None, 'runner_up': None, 'selection_rationale': 'stub'}
        self._tmp = Path(__file__).resolve().parent / f'tmp_batch_{int(time.time()*1000)}'

    def tearDown(self):
        batch_module.StockAnalysisAgent.analyze = self._orig_analyze
        BatchAnalyzer._summarize = self._orig_summarize
        BatchAnalyzer._pick_best = self._orig_pick
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_analyzer(self, concurrency=5, cache_dir=None):
        analyzer = BatchAnalyzer(max_concurrency=concurrency)
        analyzer._cache_dir = cache_dir or self._tmp
        return analyzer

    def test_parallel_execution_is_faster_than_serial(self):
        analyzer = self._make_analyzer(concurrency=5)
        symbols = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
        started = time.time()
        result = analyzer.run_batch(
            symbols=symbols, cost_prices=[0]*5, shares_list=[0]*5,
            total_assets=0, available_cash=0, master='',
        )
        elapsed = time.time() - started
        self.assertEqual(result['success_count'], 5)
        # Serial would need >= 0.55s (sum of sleeps); 5 workers should be ~0.30s
        self.assertLess(elapsed, 0.55, f'expected parallel completion, took {elapsed:.2f}s')

    def test_result_order_matches_input_order(self):
        analyzer = self._make_analyzer(concurrency=5)
        symbols = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
        result = analyzer.run_batch(
            symbols=symbols, cost_prices=[0]*5, shares_list=[0]*5,
            total_assets=0, available_cash=0, master='',
        )
        returned = [r['symbol'] for r in result['results']]
        self.assertEqual(returned, symbols, 'results must preserve input order')

    def test_concurrency_actually_used(self):
        """Track peak concurrent workers inside the fake analysis."""
        peak = {'current': 0, 'max': 0}
        lock = threading.Lock()

        def counting_analyze(self, symbol, **kwargs):
            with lock:
                peak['current'] += 1
                peak['max'] = max(peak['max'], peak['current'])
            time.sleep(0.15)
            with lock:
                peak['current'] -= 1
            return {'status': 'complete', 'symbol': symbol, 'score_breakdown': {'final': 0}}

        batch_module.StockAnalysisAgent.analyze = counting_analyze
        try:
            analyzer = self._make_analyzer(concurrency=4)
            analyzer.run_batch(
                symbols=['AAA', 'BBB', 'CCC', 'DDD', 'EEE'], cost_prices=[0]*5,
                shares_list=[0]*5, total_assets=0, available_cash=0, master='',
            )
        finally:
            batch_module.StockAnalysisAgent.analyze = _fake_analyze
        self.assertGreaterEqual(peak['max'], 2, 'expected real concurrency')
        self.assertLessEqual(peak['max'], 4, 'must not exceed configured workers')

    def test_cancel_stops_promptly(self):
        cancel_event = threading.Event()

        def slow_analyze(self, symbol, **kwargs):
            for _ in range(50):
                if kwargs.get('cancel_event') and kwargs['cancel_event'].is_set():
                    return {'status': 'cancelled'}
                time.sleep(0.05)
            return {'status': 'complete', 'symbol': symbol}

        batch_module.StockAnalysisAgent.analyze = slow_analyze
        try:
            analyzer = self._make_analyzer(concurrency=2)
            events = []
            timer = threading.Timer(0.15, cancel_event.set)
            timer.start()
            started = time.time()
            result = analyzer.run_batch(
                symbols=['AAA', 'BBB', 'CCC', 'DDD'], cost_prices=[0]*4,
                shares_list=[0]*4, total_assets=0, available_cash=0, master='',
                cancel_event=cancel_event,
                progress_callback=lambda et, d: events.append(et),
            )
            elapsed = time.time() - started
            timer.cancel()
        finally:
            batch_module.StockAnalysisAgent.analyze = _fake_analyze
        self.assertEqual(result['status'], 'cancelled')
        self.assertLess(elapsed, 1.0, 'cancel must not wait for all workers')
        self.assertIn('cancelled', events)

    def test_error_in_one_stock_does_not_break_batch(self):
        def flaky_analyze(self, symbol, **kwargs):
            if symbol == 'BBB':
                raise RuntimeError('boom')
            return {'status': 'complete', 'symbol': symbol, 'score_breakdown': {'final': 0}}

        batch_module.StockAnalysisAgent.analyze = flaky_analyze
        try:
            analyzer = self._make_analyzer(concurrency=3)
            result = analyzer.run_batch(
                symbols=['AAA', 'BBB', 'CCC'], cost_prices=[0]*3,
                shares_list=[0]*3, total_assets=0, available_cash=0, master='',
            )
        finally:
            batch_module.StockAnalysisAgent.analyze = _fake_analyze
        statuses = {r['symbol']: r['status'] for r in result['results']}
        self.assertEqual(statuses['BBB'], 'error')
        self.assertEqual(statuses['AAA'], 'complete')
        self.assertEqual(statuses['CCC'], 'complete')

    def test_llm_semaphore_limits_concurrency(self):
        from analysis.llm_guard import llm_slot, LLM_SEMAPHORE
        limit = LLM_SEMAPHORE._initial_value
        peak = {'current': 0, 'max': 0}
        lock = threading.Lock()

        def worker():
            with llm_slot():
                with lock:
                    peak['current'] += 1
                    peak['max'] = max(peak['max'], peak['current'])
                time.sleep(0.05)
                with lock:
                    peak['current'] -= 1

        threads = [threading.Thread(target=worker) for _ in range(limit + 6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(peak['max'], limit)

    def test_baostock_calls_are_serialized(self):
        from market_data.baostock_guard import BAOSTOCK_LOCK
        concurrent = {'current': 0, 'max': 0}
        lock = threading.Lock()

        def worker():
            with BAOSTOCK_LOCK:
                with lock:
                    concurrent['current'] += 1
                    concurrent['max'] = max(concurrent['max'], concurrent['current'])
                time.sleep(0.03)
                with lock:
                    concurrent['current'] -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(concurrent['max'], 1, 'BaoStock access must be serialized')


if __name__ == '__main__':
    unittest.main()
