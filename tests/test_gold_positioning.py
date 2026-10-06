import unittest
from unittest.mock import patch, Mock

import pandas as pd

from scripts.gold_positioning import compute_gold_deleveraging, fetch_gold_cot, parse_gold_cot


def cot_rows():
    # Matches the official disaggregated futures-only schema, not the legacy non-commercial categories.
    return [{
        "cftc_contract_market_code": "088691", "futonly_or_combined": "FutOnly",
        "report_date_as_yyyy_mm_dd": date, "open_interest_all": str(oi),
        "m_money_positions_long_all": str(long), "m_money_positions_short_all": "10000",
        "m_money_positions_spread": "30000",
    } for date, oi, long in [("2026-09-22", 400000, 140000), ("2026-09-29", 390000, 130000)]]


class GoldPositioningTests(unittest.TestCase):
    def setUp(self):
        self.cot = parse_gold_cot(cot_rows())
        idx = pd.bdate_range("2026-09-01", "2026-10-05", tz="UTC")
        self.prices = pd.Series(100., index=idx)
        self.prices.loc[self.prices.index >= "2026-09-29"] = 95.
        self.holdings = pd.Series(1000., index=idx)
        self.holdings.loc[self.holdings.index >= "2026-09-29"] = 1020.

    def compute(self, cot=None, holdings=None):
        return compute_gold_deleveraging(self.prices, self.cot if cot is None else cot,
                                        self.holdings if holdings is None else holdings)

    def test_deleveraging_and_etf_buying_divergence(self):
        details, quality = self.compute()
        self.assertTrue(details["triggered"])
        self.assertTrue(details["etf_buying_vs_futures_selling"])
        self.assertAlmostEqual(details["gld_report_interval_return_pct"], -5)
        self.assertAlmostEqual(details["gld_holdings_report_interval_change_pct"], 2)
        self.assertEqual(details["managed_money_net_change_contracts"], -10000)
        self.assertEqual(quality["cftc_gold"]["source_date"], "2026-09-29")

    def test_today_price_does_not_replace_matched_report_price(self):
        self.prices.iloc[-1] = 200.
        details, _ = self.compute()
        self.assertTrue(details["triggered"])
        self.assertAlmostEqual(details["gld_report_interval_return_pct"], -5)

    def test_each_leg_is_required(self):
        for column in ("open_interest", "managed_money_long"):
            cot = self.cot.copy()
            cot.iloc[-1, cot.columns.get_loc(column)] = cot.iloc[-2][column] + 1
            self.assertFalse(self.compute(cot=cot)[0]["triggered"])
        self.prices[:] = 100.
        self.assertFalse(self.compute()[0]["triggered"])

    def test_missing_holdings_preserves_positioning_but_not_divergence(self):
        details, _ = self.compute(holdings=pd.Series(dtype=float))
        self.assertTrue(details["triggered"])
        self.assertFalse(details["etf_buying_vs_futures_selling"])
        self.assertIsNone(details["gld_holdings_report_interval_change_pct"])

    def test_stale_weekly_report_is_ineligible(self):
        self.prices.loc[pd.Timestamp("2026-10-20", tz="UTC")] = 95.
        details, quality = self.compute()
        self.assertFalse(details["eligible"])
        self.assertFalse(details["triggered"])
        self.assertFalse(quality["cftc_gold"]["eligible"])

    def test_nonconsecutive_reports_are_ineligible(self):
        cot = self.cot.copy()
        cot.index = pd.to_datetime(["2026-09-15", "2026-09-29"], utc=True)
        self.assertFalse(self.compute(cot=cot)[0]["eligible"])

    def test_missing_or_single_report_is_unavailable(self):
        for cot in (pd.DataFrame(), self.cot.iloc[-1:]):
            details, _ = self.compute(cot=cot)
            self.assertFalse(details["available"])
            self.assertFalse(details["triggered"])

    def test_future_report_is_ignored(self):
        cot = self.cot.copy()
        cot.loc[pd.Timestamp("2026-10-06", tz="UTC")] = cot.iloc[-1]
        self.assertEqual(self.compute(cot=cot)[0]["report_date"], "2026-09-29")

    def test_parser_rejects_other_markets_and_combined_reports(self):
        for field, value in (("cftc_contract_market_code", "084691"), ("futonly_or_combined", "Combined")):
            rows = cot_rows()
            for row in rows:
                row[field] = value
            with self.assertRaises(ValueError):
                parse_gold_cot(rows)
        rows = cot_rows()
        rows[-1]["open_interest_all"] = "not a number"
        with self.assertRaises(ValueError):
            parse_gold_cot(rows)

    @patch("scripts.gold_positioning.requests.get")
    def test_fetch_uses_gold_futures_only_feed(self, get):
        get.return_value = Mock()
        get.return_value.json.return_value = cot_rows()[::-1]
        frame = fetch_gold_cot()
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertEqual(get.call_args.kwargs["params"]["cftc_contract_market_code"], "088691")
        get.return_value.raise_for_status.assert_called_once()

    def test_parser_rejects_invalid_payloads_and_nonfinite_counts(self):
        for rows in ({"error": "unavailable"}, ["invalid record"]):
            with self.assertRaises(ValueError):
                parse_gold_cot(rows)
        for value in ("NaN", "Infinity", "-1", "0"):
            rows = cot_rows()
            rows[-1]["open_interest_all"] = value
            with self.assertRaises(ValueError):
                parse_gold_cot(rows)


if __name__ == "__main__":
    unittest.main()
