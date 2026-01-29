#!/usr/bin/env python3
"""Fetch data, compute metrics, and update public/data.json for Gold Risk Monitor."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


FRED_SERIES_ID = "DFII10"
GLD_STOOQ_URL = "https://stooq.com/q/d/l/?s=gld.us&i=d"
GLD_HOLDINGS_URL = "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv"
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_1W = 5
TRADING_DAYS_200 = 200
HISTORY_YEARS = 5


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
        req = Request(url, headers={"User-Agent": "gold-risk-monitor"})
        with urlopen(req, timeout=timeout) as response:
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


def parse_holdings_csv(csv_text: str) -> List[Tuple[datetime, Decimal]]:
    reader = csv.DictReader(csv_text.splitlines())
    if not reader.fieldnames:
        raise DataFetchError("Holdings CSV missing header")
    date_key = None
    holdings_key = None
    for key in reader.fieldnames:
        if key and key.strip().lower() in {"date", "as of date", "asofdate"}:
            date_key = key
        if not key:
            continue
        normalized = key.strip().lower()
        if "tonnes" in normalized:
            holdings_key = key
        elif "holdings" in normalized and not holdings_key:
            holdings_key = key
    if not date_key:
        date_key = reader.fieldnames[0]
    if not holdings_key:
        raise DataFetchError("Holdings CSV missing holdings column")

    rows: List[Tuple[datetime, Decimal]] = []
    for row in reader:
        date_str = row.get(date_key, "").strip()
        holdings_str = row.get(holdings_key, "").strip()
        if not date_str or not holdings_str:
            continue
        if date_str.lower() == "holiday" or holdings_str.lower() == "holiday":
            continue
        parsed_date = parse_holdings_date(date_str)
        if not parsed_date:
            continue
        sanitized = holdings_str.replace(",", "").replace("t", "").strip()
        try:
            holdings = Decimal(sanitized)
        except Exception:
            continue
        rows.append((parsed_date, holdings))
    if len(rows) < TRADING_DAYS_1M + 1:
        raise DataFetchError("Not enough GLD holdings data points")
    rows.sort(key=lambda item: item[0])
    return rows


def parse_holdings_date(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def fetch_gld_holdings() -> List[Tuple[datetime, Decimal]]:
    csv_text = fetch_url(GLD_HOLDINGS_URL)
    return parse_holdings_csv(csv_text)


def fetch_fred_dfii10(api_key: str, reference_date: datetime) -> List[Tuple[datetime, Decimal]]:
    observation_start = (reference_date - timedelta(days=HISTORY_YEARS * 365 + 3 * 365)).date()
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={FRED_SERIES_ID}"
        f"&api_key={api_key}"
        "&file_type=json"
        "&sort_order=asc"
        f"&observation_start={observation_start.isoformat()}"
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


def compute_simple_moving_average(prices: List[Decimal], window: int) -> Decimal:
    if len(prices) < window:
        raise DataFetchError("Insufficient price history for moving average calculation")
    window_prices = prices[-window:]
    return sum(window_prices, Decimal("0")) / Decimal(window)


def compute_percentile(value: Decimal, history: List[Decimal]) -> float | None:
    if not history:
        return None
    sorted_history = sorted(history)
    count = 0
    for item in sorted_history:
        if item <= value:
            count += 1
        else:
            break
    return (count / len(sorted_history)) * 100


def percentile_history_cutoff(reference_date: datetime) -> datetime:
    return reference_date - timedelta(days=HISTORY_YEARS * 365)


def compute_return_series(
    dates: List[datetime],
    values: List[Decimal],
    days_ago: int,
    cutoff: datetime,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(days_ago, len(values)):
        if dates[idx] < cutoff:
            continue
        history.append((values[idx] / values[idx - days_ago]) - Decimal("1"))
    return history


def compute_drawdown_series(
    dates: List[datetime],
    prices: List[Decimal],
    window: int,
    cutoff: datetime,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(window - 1, len(prices)):
        if dates[idx] < cutoff:
            continue
        window_prices = prices[idx - window + 1 : idx + 1]
        history.append(compute_max_drawdown(window_prices))
    return history


def compute_change_series(
    dates: List[datetime],
    values: List[Decimal],
    days_ago: int,
    cutoff: datetime,
    multiplier: Decimal = Decimal("1"),
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(days_ago, len(values)):
        if dates[idx] < cutoff:
            continue
        history.append((values[idx] - values[idx - days_ago]) * multiplier)
    return history


def compute_pct_above_ma_series(
    dates: List[datetime],
    prices: List[Decimal],
    window: int,
    cutoff: datetime,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(window - 1, len(prices)):
        if dates[idx] < cutoff:
            continue
        window_prices = prices[idx - window + 1 : idx + 1]
        moving_avg = sum(window_prices, Decimal("0")) / Decimal(window)
        history.append((prices[idx] / moving_avg) - Decimal("1"))
    return history


def classify_flag(
    gld_ret_1m: Decimal,
    gld_ret_3m: Decimal,
    gld_max_drawdown_3m: Decimal,
    gld_ret_3m_pctile_5y: float | None,
    gld_max_drawdown_pctile_5y: float | None,
    gld_pct_above_200dma_pctile_5y: float | None,
    real_yield_change_1m_bp: Decimal,
    gld_holdings_change_21d_pct: Decimal,
    gld_holdings_change_5d_pct: Decimal,
    previous_flag: str | None,
    red_streak: int,
    no_red_streak: int,
) -> tuple[str, List[str], int, int]:
    extension_score = 0
    extension_triggers = []
    if gld_ret_3m_pctile_5y is not None and gld_ret_3m_pctile_5y >= 90:
        extension_score += 1
        extension_triggers.append("GLD 3M return ≥ 90th percentile")
    if gld_pct_above_200dma_pctile_5y is not None and gld_pct_above_200dma_pctile_5y >= 90:
        extension_score += 1
        extension_triggers.append("GLD % above 200DMA ≥ 90th percentile")
    if gld_max_drawdown_pctile_5y is not None and gld_max_drawdown_pctile_5y >= 70:
        extension_score += 1
        extension_triggers.append("GLD 3M drawdown ≥ 70th percentile (smooth)")

    deterioration_score = 0
    deterioration_triggers = []
    if gld_holdings_change_21d_pct <= Decimal("-0.015") and gld_ret_1m > Decimal("0"):
        deterioration_score += 1
        deterioration_triggers.append("Holdings 21D ≤ -1.5% with positive 1M return (divergence)")
    if real_yield_change_1m_bp >= Decimal("25"):
        deterioration_score += 1
        deterioration_triggers.append("Real yield change 1M ≥ +25 bp")
    if gld_ret_1m <= Decimal("0") or gld_max_drawdown_3m <= Decimal("-0.08"):
        deterioration_score += 1
        deterioration_triggers.append("Price crack: 1M return ≤ 0% or 3M drawdown ≤ -8%")

    red_primary = (
        gld_ret_3m <= Decimal("-0.15")
        or gld_max_drawdown_3m <= Decimal("-0.18")
        or real_yield_change_1m_bp >= Decimal("50")
    )
    red_primary_triggers = []
    if gld_ret_3m <= Decimal("-0.15"):
        red_primary_triggers.append("GLD 3M return ≤ -15%")
    if gld_max_drawdown_3m <= Decimal("-0.18"):
        red_primary_triggers.append("GLD 3M drawdown ≤ -18%")
    if real_yield_change_1m_bp >= Decimal("50"):
        red_primary_triggers.append("Real yield change 1M ≥ +50 bp")

    composite_conditions = [
        ("GLD 3M return ≤ -8%", gld_ret_3m <= Decimal("-0.08")),
        ("Holdings 21D ≤ -2.0%", gld_holdings_change_21d_pct <= Decimal("-0.02")),
        ("Holdings 5D ≤ -1.0%", gld_holdings_change_5d_pct <= Decimal("-0.01")),
        ("Real yield change 1M ≥ +25 bp", real_yield_change_1m_bp >= Decimal("25")),
    ]
    composite_hits = [label for label, hit in composite_conditions if hit]
    red_composite = len(composite_hits) >= 2

    candidate_red = red_primary or red_composite
    candidate_orange = extension_score >= 2 and deterioration_score >= 1
    candidate_blue = extension_score >= 2

    if candidate_red:
        red_streak += 1
        no_red_streak = 0
    else:
        red_streak = 0
        no_red_streak += 1

    red_persisted = False
    if red_primary:
        red_persisted = True
    elif candidate_red and red_streak >= 2:
        red_persisted = True
    elif previous_flag == "RED" and no_red_streak < 5:
        red_persisted = True

    if red_persisted:
        triggers = []
        if red_primary_triggers:
            triggers.extend(red_primary_triggers)
        if red_composite:
            triggers.append(f"Composite stress (2 of 4): {', '.join(composite_hits)}")
        if not candidate_red and previous_flag == "RED":
            triggers.append("RED persistence: cooldown period (needs 5 clear runs)")
        if not triggers:
            triggers.append("RED: persistence without active triggers")
        return "RED", triggers, red_streak, no_red_streak
    if candidate_orange:
        return "ORANGE", extension_triggers + deterioration_triggers, red_streak, no_red_streak
    if candidate_blue:
        return "BLUE", extension_triggers, red_streak, no_red_streak
    return "GREEN", [], red_streak, no_red_streak


def build_issue_body(metrics: dict) -> str:
    lines = [
        "Latest metrics:",
        "",
        f"- GLD 1M return: {metrics['gld_ret_1m']:+.2%}",
        f"- GLD 3M return: {metrics['gld_ret_3m']:+.2%}",
        f"- GLD 3M max drawdown: {metrics['gld_max_drawdown_3m']:+.2%}",
        f"- GLD % above 200DMA: {metrics['gld_pct_above_200dma']:+.2%}",
        f"- DFII10 level: {metrics['real_yield_today']:.2f}%",
        f"- DFII10 1M change: {metrics['real_yield_change_1m_bp']:+.0f} bp",
        f"- DFII10 3M change: {metrics['real_yield_change_3m_bp']:+.0f} bp",
        f"- GLD holdings today: {metrics['gld_holdings_today']:.1f} t",
        f"- GLD holdings 5D change: {metrics['gld_holdings_change_5d_pct']:+.2%}",
        f"- GLD holdings 21D change: {metrics['gld_holdings_change_21d_pct']:+.2%}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Gold Risk Monitor data")
    parser.add_argument("--output", default="data.json")
    parser.add_argument("--status-file", default="/tmp/monitor_status.json")
    args = parser.parse_args()

    status = Status(fetch_ok=False, flag=None, issue_title=None, issue_body=None, error=None)
    previous_state = {}
    previous_flag = None
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as handle:
                previous_data = json.load(handle)
                previous_state = previous_data.get("state", {}) if isinstance(previous_data, dict) else {}
                previous_flag = previous_data.get("flag") if isinstance(previous_data, dict) else None
        except (OSError, json.JSONDecodeError):
            previous_state = {}
            previous_flag = None
    try:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise DataFetchError("FRED_API_KEY is not set")

        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
        now_utc_naive = now_utc.replace(tzinfo=None)

        gld_rows = fetch_gld_prices()
        fred_rows = fetch_fred_dfii10(api_key, now_utc_naive)
        holdings_rows = fetch_gld_holdings()

        gld_dates = [row[0] for row in gld_rows]
        gld_prices = [row[1] for row in gld_rows]
        gld_ret_1m = compute_return(gld_prices, TRADING_DAYS_1M)
        gld_ret_3m = compute_return(gld_prices, TRADING_DAYS_3M)
        gld_window = gld_prices[-TRADING_DAYS_3M:]
        gld_max_drawdown_3m = compute_max_drawdown(gld_window)
        gld_200dma = compute_simple_moving_average(gld_prices, TRADING_DAYS_200)
        gld_pct_above_200dma = (gld_prices[-1] / gld_200dma) - Decimal("1")

        fred_dates = [row[0] for row in fred_rows]
        fred_values = [row[1] for row in fred_rows]
        real_yield_today = fred_values[-1]
        real_yield_change_1m_bp = (real_yield_today - fred_values[-(TRADING_DAYS_1M + 1)]) * Decimal("100")
        real_yield_change_3m_bp = (real_yield_today - fred_values[-(TRADING_DAYS_3M + 1)]) * Decimal("100")

        holdings_dates = [row[0] for row in holdings_rows]
        holdings_values = [row[1] for row in holdings_rows]
        gld_holdings_today = holdings_values[-1]
        gld_holdings_change_5d_pct = compute_return(holdings_values, TRADING_DAYS_1W)
        gld_holdings_change_21d_pct = compute_return(holdings_values, TRADING_DAYS_1M)

        cutoff_date = percentile_history_cutoff(now_utc_naive)
        gld_ret_1m_history = compute_return_series(gld_dates, gld_prices, TRADING_DAYS_1M, cutoff_date)
        gld_ret_3m_history = compute_return_series(gld_dates, gld_prices, TRADING_DAYS_3M, cutoff_date)
        gld_drawdown_history = compute_drawdown_series(gld_dates, gld_prices, TRADING_DAYS_3M, cutoff_date)
        gld_pct_above_200dma_history = compute_pct_above_ma_series(
            gld_dates,
            gld_prices,
            TRADING_DAYS_200,
            cutoff_date,
        )
        real_yield_change_1m_history = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_1M,
            cutoff_date,
            Decimal("100"),
        )
        real_yield_change_3m_history = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_3M,
            cutoff_date,
            Decimal("100"),
        )
        holdings_change_5d_history = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1W,
            cutoff_date,
        )
        holdings_change_21d_history = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1M,
            cutoff_date,
        )

        gld_ret_1m_pctile_5y = compute_percentile(gld_ret_1m, gld_ret_1m_history)
        gld_ret_3m_pctile_5y = compute_percentile(gld_ret_3m, gld_ret_3m_history)
        gld_drawdown_pctile_5y = compute_percentile(gld_max_drawdown_3m, gld_drawdown_history)
        gld_pct_above_200dma_pctile_5y = compute_percentile(
            gld_pct_above_200dma,
            gld_pct_above_200dma_history,
        )
        real_yield_change_1m_pctile_5y = compute_percentile(
            real_yield_change_1m_bp,
            real_yield_change_1m_history,
        )
        real_yield_change_3m_pctile_5y = compute_percentile(
            real_yield_change_3m_bp,
            real_yield_change_3m_history,
        )
        holdings_change_5d_pctile_5y = compute_percentile(
            gld_holdings_change_5d_pct,
            holdings_change_5d_history,
        )
        holdings_change_21d_pctile_5y = compute_percentile(
            gld_holdings_change_21d_pct,
            holdings_change_21d_history,
        )

        red_streak = int(previous_state.get("red_streak", 0) or 0)
        no_red_streak = int(previous_state.get("no_red_streak", 0) or 0)
        flag, flag_triggers, red_streak, no_red_streak = classify_flag(
            gld_ret_1m,
            gld_ret_3m,
            gld_max_drawdown_3m,
            gld_ret_3m_pctile_5y,
            gld_drawdown_pctile_5y,
            gld_pct_above_200dma_pctile_5y,
            real_yield_change_1m_bp,
            gld_holdings_change_21d_pct,
            gld_holdings_change_5d_pct,
            previous_flag,
            red_streak,
            no_red_streak,
        )

        repository = None
        repo_env = os.environ.get("GITHUB_REPOSITORY")
        if repo_env:
            repository = f"https://github.com/{repo_env}"

        data = {
            "updated_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "updated_et": now_et.isoformat(),
            "sources": {
                "gld_prices": GLD_STOOQ_URL,
                "gld_holdings": GLD_HOLDINGS_URL,
                "dfii10": "https://fred.stlouisfed.org/series/DFII10",
            },
            "repository": repository,
            "metrics": {
                "gld_price": float(gld_prices[-1]),
                "gld_holdings_today": float(gld_holdings_today),
                "gld_holdings_change_5d_pct": float(gld_holdings_change_5d_pct),
                "gld_holdings_change_21d_pct": float(gld_holdings_change_21d_pct),
                "gld_holdings_change_5d_pct_pctile_5y": holdings_change_5d_pctile_5y,
                "gld_holdings_change_21d_pct_pctile_5y": holdings_change_21d_pctile_5y,
                "gld_ret_1m": float(gld_ret_1m),
                "gld_ret_3m": float(gld_ret_3m),
                "gld_max_drawdown_3m": float(gld_max_drawdown_3m),
                "gld_200dma": float(gld_200dma),
                "gld_pct_above_200dma": float(gld_pct_above_200dma),
                "gld_ret_1m_pctile_5y": gld_ret_1m_pctile_5y,
                "gld_ret_3m_pctile_5y": gld_ret_3m_pctile_5y,
                "gld_max_drawdown_3m_pctile_5y": gld_drawdown_pctile_5y,
                "gld_pct_above_200dma_pctile_5y": gld_pct_above_200dma_pctile_5y,
                "real_yield_today": float(real_yield_today),
                "real_yield_change_1m_bp": float(real_yield_change_1m_bp),
                "real_yield_change_3m_bp": float(real_yield_change_3m_bp),
                "real_yield_change_1m_bp_pctile_5y": real_yield_change_1m_pctile_5y,
                "real_yield_change_3m_bp_pctile_5y": real_yield_change_3m_pctile_5y,
            },
            "flag": flag,
            "flag_triggers": flag_triggers,
            "state": {
                "red_streak": red_streak,
                "no_red_streak": no_red_streak,
            },
            "rules": {
                "red": (
                    "Primary: 3M return <= -15% OR 3M drawdown <= -18% OR DFII10 1M change >= +50 bp. "
                    "Composite: 2 of 4 (3M return <= -8%, holdings 21D <= -2%, "
                    "holdings 5D <= -1%, DFII10 1M change >= +25 bp). "
                    "Enter RED after 2 consecutive composite runs; exit after 5 clean runs."
                ),
                "orange": (
                    "Extension score >= 2 AND deterioration score >= 1. "
                    "Deterioration signals: flow divergence (holdings 21D <= -1.5% with 1M return > 0), "
                    "real yield 1M change >= +25 bp, or price crack (1M return <= 0 or 3M drawdown <= -8%)."
                ),
                "blue": (
                    "Extension score >= 2. +1 each: 3M return percentile >= 90, "
                    "% above 200DMA percentile >= 90, 3M drawdown percentile >= 70."
                ),
                "green": "No extension or deterioration signals.",
            },
        }

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
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
