"""Regression tests for the committed analysis archive package."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory.analysis.analysis_store import AnalysisStore


class AnalysisStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name) / "analysis"
        self.store = AnalysisStore(self.base_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_writes_complete_windows_safe_snapshot(self):
        snapshot = Path(self.store.save(
            symbol="600519",
            state={"symbol": "600519", "status": "completed"},
            timestamp="2026-08-25T10:30:45.123456+08:00",
            employee_reports=[{"role": "risk"}],
            cio_decision={"action": "observe"},
            prediction={"outlook": "neutral"},
        ))

        self.assertNotIn(":", snapshot.name)
        self.assertEqual(json.loads((snapshot / "state.json").read_text(encoding="utf-8"))["status"], "completed")
        self.assertTrue((snapshot / "employee_reports.json").is_file())
        self.assertTrue((snapshot / "cio_decision.json").is_file())
        self.assertTrue((snapshot / "prediction.json").is_file())
        self.assertEqual(self.store.list_all_symbols(), ["600519"])

    def test_repeated_timestamp_does_not_overwrite_an_existing_snapshot(self):
        first = self.store.save(symbol="000001", state={"run": 1}, timestamp="2026-08-25T10:30:45")
        second = self.store.save(symbol="000001", state={"run": 2}, timestamp="2026-08-25T10:30:45")

        self.assertNotEqual(first, second)
        self.assertEqual(json.loads((Path(first) / "state.json").read_text(encoding="utf-8"))["run"], 1)
        self.assertEqual(json.loads((Path(second) / "state.json").read_text(encoding="utf-8"))["run"], 2)

    def test_symbol_cannot_escape_the_analysis_directory(self):
        with self.assertRaises(ValueError):
            self.store.save(symbol="../outside", state={}, timestamp="2026-08-25T10:30:45")

    def test_store_instances_do_not_collide_across_task_threads(self):
        def save_snapshot(run: int) -> str:
            return AnalysisStore(self.base_dir).save(
                symbol="600000",
                state={"run": run},
                timestamp="2026-08-25T10:30:45",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            snapshots = list(executor.map(save_snapshot, range(8)))

        self.assertEqual(len(set(snapshots)), 8)
        self.assertTrue(all((Path(snapshot) / "state.json").is_file() for snapshot in snapshots))


if __name__ == "__main__":
    unittest.main()
