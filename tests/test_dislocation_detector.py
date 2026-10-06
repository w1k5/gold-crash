import json
import unittest
from pathlib import Path

import pandas as pd

from scripts.dislocation_detector import (
    DEFAULT_FRED_SERIES, RunMeta, SignalResult, compute_dashboard,
    compute_signals, summarize_dislocation,
)


def meta():
    return RunMeta("2026-10-05", "2026-10-05T20:00:00+00:00", "2026-10-05T20:00:00+00:00", 0, False, [])


def signals(count):
    return [SignalResult(f"independent_{i}", True, {"score_0_1": 1.0}) for i in range(count)]


class PersistenceTests(unittest.TestCase):
    def test_one_then_two_cannot_enter(self):
        previous = summarize_dislocation(signals(1), meta())
        current = summarize_dislocation(signals(2), meta(), previous_summary=previous)
        self.assertEqual(current["status"], "stress_building")
        self.assertFalse(current["dislocation"])

    def test_confirmed_entry_holds_then_exits(self):
        state = summarize_dislocation(signals(3), meta())
        for _ in range(3):
            state = summarize_dislocation(signals(2), meta(), previous_summary=state)
            self.assertTrue(state["dislocation"])
            self.assertTrue(state["persistence_applied"])
        state = summarize_dislocation(signals(1), meta(), previous_summary=state)
        self.assertFalse(state["dislocation"])
        self.assertIsNone(state["last_confirmed_dislocation_session"])
        self.assertFalse(summarize_dislocation(signals(2), meta(), previous_summary=state)["dislocation"])

    def test_legacy_false_positive_is_reset(self):
        previous = {"status": "dislocation", "dislocation": True, "signals_triggered_count": 2}
        self.assertFalse(summarize_dislocation(signals(2), meta(), previous_summary=previous)["dislocation"])
        previous["signals_triggered_count"] = 3
        self.assertTrue(summarize_dislocation(signals(2), meta(), previous_summary=previous)["dislocation"])

    def test_k_one_does_not_hold_with_zero_signals(self):
        previous = summarize_dislocation(signals(1), meta(), k_required=1)
        self.assertFalse(summarize_dislocation([], meta(), k_required=1, previous_summary=previous)["dislocation"])
        with self.assertRaises(ValueError):
            summarize_dislocation([], meta(), k_required=0)

    def test_current_checked_in_snapshot_does_not_perpetuate(self):
        previous = json.loads(Path("dislocation.json").read_text())
        # Regression fixture captured from the audited state, independent of future refreshes.
        previous.update(status="dislocation", dislocation=True, signals_triggered_count=2)
        previous.pop("rule_version", None)
        self.assertFalse(summarize_dislocation(signals(2), meta(), previous_summary=previous)["dislocation"])

    def test_changed_threshold_requires_new_confirmation(self):
        previous = summarize_dislocation(signals(3), meta())
        self.assertFalse(summarize_dislocation(signals(3), meta(), k_required=4, previous_summary=previous)["dislocation"])

    def test_weekly_context_does_not_confirm_cross_market_dislocation(self):
        warning = SignalResult("gold_futures_deleveraging", True, {"score_0_1": 1.0})
        current = summarize_dislocation(signals(2) + [warning], meta())
        self.assertEqual(current["signals_triggered_count"], 2)
        self.assertFalse(current["dislocation"])
        self.assertEqual(current["dashboard"]["pillar_scores"]["gold_deleveraging"], 100)


class FeedQualityTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.bdate_range(end="2026-10-05", periods=700, tz="UTC")
        n = len(self.idx)
        self.market = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.}, index=self.idx)
        self.vix = self.market.copy()
        self.vix["Close"] = 15.
        self.fred = {key: pd.DataFrame({series: [2 + i / n for i in range(n)]}, index=self.idx)
                     for key, series in DEFAULT_FRED_SERIES.items()}

    def run_signals(self, fred=None):
        return compute_signals(self.market, self.market, self.market, vix=self.vix,
                               fred_map=self.fred if fred is None else fred)

    def test_three_calendar_years_produce_hy_z(self):
        found, run = self.run_signals()
        credit = next(s for s in found if s.name == "credit_spread_widening_fred")
        self.assertIsNotNone(credit.details["hy_oas_z"])
        self.assertEqual(credit.details["hy_oas_z_observations"], 504)
        json.dumps(summarize_dislocation(found, run), allow_nan=False)

    def test_insufficient_hy_history_is_explicit(self):
        self.fred["hy_oas"] = self.fred["hy_oas"].iloc[-100:]
        found, _ = self.run_signals()
        credit = next(s for s in found if s.name == "credit_spread_widening_fred")
        self.assertIsNone(credit.details["hy_oas_z"])
        self.assertEqual(credit.details["hy_oas_z_observations"], 100)

    def test_one_day_oas_lag_is_visible_and_reduces_confidence(self):
        found, run = self.run_signals()
        fresh_confidence = compute_dashboard(found, run)["confidence_score"]
        self.fred["hy_oas"] = self.fred["hy_oas"].iloc[:-1]
        found, run = self.run_signals()
        self.assertEqual(run.fred_inputs["hy_oas"]["source_date"], "2026-10-02")
        self.assertEqual(run.fred_inputs["hy_oas"]["stale_days"], 1)
        self.assertTrue(run.fred_inputs["hy_oas"]["eligible"])
        self.assertLess(compute_dashboard(found, run)["confidence_score"], fresh_confidence)

    def test_stale_credit_is_excluded_from_count_and_scores(self):
        self.fred["hy_oas"] = self.fred["hy_oas"].iloc[:-4].copy()
        self.fred["hy_oas"].iloc[-1, 0] = 8.0
        found, run = self.run_signals()
        output = summarize_dislocation(found, run)
        self.assertIn("credit_spread_widening_fred", output["signal_counting"]["excluded_from_threshold"])
        self.assertNotIn("credit_spread_widening_fred", output["signals_triggered"])
        self.assertEqual(output["dashboard"]["pillar_scores"]["credit_stress"], 0)
        self.assertEqual(output["status"], "data_stale")

    def test_future_fred_observation_does_not_leak(self):
        frame = self.fred["hy_oas"]
        frame.loc[pd.Timestamp("2026-10-06", tz="UTC")] = 10.
        found, run = self.run_signals()
        credit = next(s for s in found if s.name == "credit_spread_widening_fred")
        self.assertLess(credit.details["hy_oas_level"], 4)
        self.assertEqual(run.fred_inputs["hy_oas"]["source_date"], "2026-10-05")

    def test_missing_fred_series_are_visible(self):
        found, run = self.run_signals({})
        self.assertEqual(len(run.fred_inputs), len(DEFAULT_FRED_SERIES))
        self.assertIn("hy_oas_missing", run.stale_reasons)
        self.assertFalse(run.fred_inputs["hy_oas"]["eligible"])
        self.assertLess(compute_dashboard(found, run)["confidence_score"], 100)

    def test_monthly_jgb_feed_uses_monthly_tolerance(self):
        self.fred["jgb10"] = self.fred["jgb10"].iloc[:-25]
        _, run = self.run_signals()
        self.assertTrue(run.fred_inputs["jgb10"]["eligible"])
        self.assertNotIn("jgb10_stale", run.stale_reasons)


if __name__ == "__main__":
    unittest.main()
