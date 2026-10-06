"""Regression tests for the regime label and news-filter fallback fixes."""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RegimeLabelTests(unittest.TestCase):
    def test_agent_builds_chinese_regime_label(self):
        source = Path(__file__).resolve().parent.parent / "analysis" / "agent.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("'regime_label'", text)
        self.assertIn("'trending_up': '上升趋势'", text)

    def test_frontend_maps_regime_enum(self):
        source = Path(__file__).resolve().parent.parent / "static" / "index.html"
        text = source.read_text(encoding="utf-8")
        self.assertIn("function regimeLabel(", text)
        self.assertIn("trending_up: '上升趋势'", text)
        # The signal header must render through the mapping helper.
        self.assertIn("regimeLabel(sb.regime, sb.regime_label)", text)


class NewsFilterFallbackTests(unittest.TestCase):
    def _service(self):
        from market_data.search.search_service import SearchService
        return SearchService(tavily_keys=None)

    def _response(self, dates):
        from market_data.search.search_service import SearchResponse, SearchResult
        results = [
            SearchResult(title=f"news {i}", snippet="", url=f"https://example.com/{i}",
                         source="test", published_date=d)
            for i, d in enumerate(dates)
        ]
        return SearchResponse(query="q", results=results, provider="test",
                              success=True, error_message="", search_time=0.1)

    def test_out_of_window_results_fall_back_to_newest_dated(self):
        svc = self._service()
        today = datetime.now().date()
        # All dated results are 30 days old -> outside the short window.
        old = (today - timedelta(days=30)).isoformat()
        older = (today - timedelta(days=40)).isoformat()
        response = self._response([older, old])
        filtered = svc._filter_news_response(
            response, search_days=3, max_results=10, log_scope="test"
        )
        self.assertTrue(filtered.results, "fallback must keep newest dated results")
        self.assertEqual(filtered.results[0].published_date, old,
                         "newest dated item should come first")
        self.assertEqual(len(filtered.results), 2)

    def test_window_results_used_without_fallback(self):
        svc = self._service()
        today = datetime.now().date()
        fresh = today.isoformat()
        response = self._response([fresh])
        filtered = svc._filter_news_response(
            response, search_days=3, max_results=10, log_scope="test"
        )
        self.assertEqual(len(filtered.results), 1)
        self.assertEqual(filtered.results[0].published_date, fresh)

    def test_undated_results_still_dropped(self):
        svc = self._service()
        response = self._response([None])
        filtered = svc._filter_news_response(
            response, search_days=3, max_results=10, log_scope="test"
        )
        self.assertEqual(len(filtered.results), 0,
                         "undated results must not be resurrected by the fallback")

    def test_future_results_not_resurrected(self):
        svc = self._service()
        today = datetime.now().date()
        future = (today + timedelta(days=30)).isoformat()
        old = (today - timedelta(days=30)).isoformat()
        response = self._response([future, old])
        filtered = svc._filter_news_response(
            response, search_days=3, max_results=10, log_scope="test"
        )
        dates = [item.published_date for item in filtered.results]
        self.assertNotIn(future, dates, "far-future dates must stay dropped")


if __name__ == '__main__':
    unittest.main()
