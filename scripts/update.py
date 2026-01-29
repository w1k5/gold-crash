#!/usr/bin/env python3
"""Fetch data, compute metrics, and update public/data.json for Gold Risk Monitor."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Tuple
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo


FRED_SERIES_ID = "DFII10"
GLD_STOOQ_URL = "https://stooq.com/q/d/l/?s=gld.us&i=d"
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63


@dataclass
class Status:
    fetch_ok: bool
    flag: str | None
    issue_title: str | None
    issue_body: str | None
    error: str | None


class DataFetchError(RuntimeError):
    pass


def fetch_url(url: str, timeout: int = 20) -> str:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        raise DataFetchError(f"Failed to fetch {url}: {exc}") from exc


def parse_stooq_csv(csv_text: str) -> List[Tuple[datetime, Decimal]]:
    reader = csv.DictReader(csv_text.splitlines())
    rows: List[Tuple[datetime, Decimal]] = []
    for row in reader:
        date_str = row.get("Date")
        close_str = row.get("Close")
        if not date_str or not close_str:
            continue
        if close_str.lower() == "nan":
            continue
        rows.append((datetime.fromisoformat(date_str), Decimal(close_str)))
    if len(rows) < TRADING_DAYS_3M + 1:
        raise DataFetchError("Not enough GLD data points to compute 3M metrics")
    return rows


def fetch_gld_prices() -> List[Tuple[datetime, Decimal]]:
    csv_text = fetch_url(GLD_STOOQ_URL)
    return parse_stooq_csv(csv_text)


def fetch_fred_dfii10(api_key: str) -> List[Tuple[datetime, Decimal]]:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={FRED_SERIES_ID}"
        f"&api_key={api_key}"
        "&file_type=json"
        "&sort_order=asc"
        "&limit=200"
    )
    payload = fetch_url(url)
    data = json.loads(payload)
    observations = data.get("observations", [])
    rows: List[Tuple[datetime, Decimal]] = []
    for obs in observations:
        value = obs.get("value")
        date_str = obs.get("date")
        if not value or value == "." or not date_str:
            continue
        rows.append((datetime.fromisoformat(date_str), Decimal(value)))
    if len(rows) < TRADING_DAYS_3M + 1:
        raise DataFetchError("Not enough DFII10 observations to compute 3M metrics")
    return rows


def compute_return(prices: List[Decimal], days_ago: int) -> Decimal:
    if len(prices) < days_ago + 1:
        raise DataFetchError("Insufficient price history for return calculation")
    today = prices[-1]
    prior = prices[-(days_ago + 1)]
    return (today / prior) - Decimal("1")


def compute_max_drawdown(prices: List[Decimal]) -> Decimal:
    peak = prices[0]
    max_drawdown = Decimal("0")
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (price / peak) - Decimal("1")
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def classify_flag(
    gld_ret_3m: Decimal,
    gld_max_drawdown_3m: Decimal,
    real_yield_change_1m_bp: Decimal,
) -> str:
    if gld_ret_3m <= Decimal("-0.15"):
        return "RED"
    if gld_max_drawdown_3m <= Decimal("-0.18"):
        return "RED"
    if real_yield_change_1m_bp >= Decimal("50"):
        return "RED"
    if gld_ret_3m <= Decimal("-0.08"):
        return "YELLOW"
    if real_yield_change_1m_bp >= Decimal("25"):
        return "YELLOW"
    return "GREEN"


def build_issue_body(metrics: dict) -> str:
    lines = [
        "Latest metrics:",
        "",
        f"- GLD 1M return: {metrics['gld_ret_1m']:+.2%}",
        f"- GLD 3M return: {metrics['gld_ret_3m']:+.2%}",
        f"- GLD 3M max drawdown: {metrics['gld_max_drawdown_3m']:+.2%}",
        f"- DFII10 level: {metrics['real_yield_today']:.2f}%",
        f"- DFII10 1M change: {metrics['real_yield_change_1m_bp']:+.0f} bp",
        f"- DFII10 3M change: {metrics['real_yield_change_3m_bp']:+.0f} bp",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Gold Risk Monitor data")
    parser.add_argument("--output", default="public/data.json")
    parser.add_argument("--status-file", default="/tmp/monitor_status.json")
    args = parser.parse_args()

    status = Status(fetch_ok=False, flag=None, issue_title=None, issue_body=None, error=None)
    try:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise DataFetchError("FRED_API_KEY is not set")

        gld_rows = fetch_gld_prices()
        fred_rows = fetch_fred_dfii10(api_key)

        gld_prices = [row[1] for row in gld_rows]
        gld_ret_1m = compute_return(gld_prices, TRADING_DAYS_1M)
        gld_ret_3m = compute_return(gld_prices, TRADING_DAYS_3M)
        gld_window = gld_prices[-TRADING_DAYS_3M:]
        gld_max_drawdown_3m = compute_max_drawdown(gld_window)

        fred_values = [row[1] for row in fred_rows]
        real_yield_today = fred_values[-1]
        real_yield_change_1m_bp = (real_yield_today - fred_values[-(TRADING_DAYS_1M + 1)]) * Decimal("100")
        real_yield_change_3m_bp = (real_yield_today - fred_values[-(TRADING_DAYS_3M + 1)]) * Decimal("100")

        flag = classify_flag(gld_ret_3m, gld_max_drawdown_3m, real_yield_change_1m_bp)

        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

        data = {
            "updated_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "updated_et": now_et.isoformat(),
            "sources": {
                "gld_prices": GLD_STOOQ_URL,
                "dfii10": "https://fred.stlouisfed.org/series/DFII10",
            },
            "metrics": {
                "gld_price": float(gld_prices[-1]),
                "gld_ret_1m": float(gld_ret_1m),
                "gld_ret_3m": float(gld_ret_3m),
                "gld_max_drawdown_3m": float(gld_max_drawdown_3m),
                "real_yield_today": float(real_yield_today),
                "real_yield_change_1m_bp": float(real_yield_change_1m_bp),
                "real_yield_change_3m_bp": float(real_yield_change_3m_bp),
            },
            "flag": flag,
            "rules": {
                "red": "GLD 3M return <= -15% OR GLD 3M max drawdown <= -18% OR DFII10 1M change >= +50 bp",
                "yellow": "GLD 3M return <= -8% OR DFII10 1M change >= +25 bp",
                "green": "Otherwise",
            },
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

        status.fetch_ok = True
        status.flag = flag
        if flag == "RED":
            status.issue_title = "🚨 Gold Risk Monitor: RED flag"
            status.issue_body = build_issue_body(data["metrics"])
    except DataFetchError as exc:
        status.fetch_ok = False
        status.flag = None
        status.issue_title = "⚠️ Gold Risk Monitor: data fetch failed"
        status.issue_body = f"Data fetch failed: {exc}"
        status.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - keep any unexpected errors surfaced
        status.fetch_ok = False
        status.flag = None
        status.issue_title = "⚠️ Gold Risk Monitor: data fetch failed"
        status.issue_body = f"Unexpected error: {exc}"
        status.error = str(exc)

    with open(args.status_file, "w", encoding="utf-8") as handle:
        json.dump(status.__dict__, handle, indent=2)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
