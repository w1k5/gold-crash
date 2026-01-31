#!/usr/bin/env python3
"""Fetch data, compute metrics, and update public/data.json for Gold Risk Monitor."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


FRED_SERIES = {
    "real10": "DFII10",
    "nom10": "DGS10",
    "nom2": "DGS2",
    "be10": "T10YIE",
}
GLD_STOOQ_URL = "https://stooq.com/q/d/l/?s=gld.us&i=d"
GLD_HOLDINGS_URL = "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv"
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_1Y = 252
TRADING_DAYS_3Y = 756
TRADING_DAYS_5Y = 1260
TRADING_DAYS_1W = 5
TRADING_DAYS_200 = 200
HISTORY_YEARS = 5
HORIZON_WINDOWS = {
    "3M": TRADING_DAYS_3M,
    "6M": TRADING_DAYS_6M,
    "1Y": TRADING_DAYS_1Y,
    "3Y": TRADING_DAYS_3Y,
    "5Y": TRADING_DAYS_5Y,
}


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
    rows.sort(key=lambda item: item[0])
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


def fetch_fred_series(series_id: str, api_key: str, reference_date: datetime) -> List[Tuple[datetime, Decimal]]:
    observation_start = (reference_date - timedelta(days=HISTORY_YEARS * 365 + 3 * 365)).date()
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
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
    return rows


def compute_bp_change(values: List[Decimal], days_ago: int) -> Decimal | None:
    if len(values) < days_ago + 1:
        return None
    return (values[-1] - values[-(days_ago + 1)]) * Decimal("100")


def compute_return(prices: List[Decimal], days_ago: int) -> Decimal | None:
    if len(prices) < days_ago + 1:
        return None
    today = prices[-1]
    prior = prices[-(days_ago + 1)]
    return (today / prior) - Decimal("1")


def compute_max_drawdown(prices: List[Decimal]) -> Decimal | None:
    if not prices:
        return None
    peak = prices[0]
    max_drawdown = Decimal("0")
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (price / peak) - Decimal("1")
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def compute_simple_moving_average(prices: List[Decimal], window: int) -> Decimal | None:
    if len(prices) < window:
        return None
    window_prices = prices[-window:]
    return sum(window_prices, Decimal("0")) / Decimal(window)


def compute_percentile(value: Decimal, history: List[Decimal]) -> int | None:
    if not history:
        return None
    sorted_history = sorted(history)
    count = 0
    for item in sorted_history:
        if item <= value:
            count += 1
        else:
            break
    percentile = (count / len(sorted_history)) * 100
    rounded = int(round(percentile))
    return max(0, min(100, rounded))


def percentile_history_cutoff(reference_date: datetime) -> datetime:
    return reference_date - timedelta(days=HISTORY_YEARS * 365)


def compute_return_series(
    dates: List[datetime],
    values: List[Decimal],
    days_ago: int,
    cutoff: datetime | None,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(days_ago, len(values)):
        if cutoff:
            if dates[idx] < cutoff or dates[idx - days_ago] < cutoff:
                continue
        history.append((values[idx] / values[idx - days_ago]) - Decimal("1"))
    return history


def compute_drawdown_series(
    dates: List[datetime],
    prices: List[Decimal],
    window: int,
    cutoff: datetime | None,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(window - 1, len(prices)):
        if cutoff:
            if dates[idx] < cutoff or dates[idx - window + 1] < cutoff:
                continue
        window_prices = prices[idx - window + 1 : idx + 1]
        drawdown = compute_max_drawdown(window_prices)
        if drawdown is not None:
            history.append(drawdown)
    return history


def compute_change_series(
    dates: List[datetime],
    values: List[Decimal],
    days_ago: int,
    cutoff: datetime | None,
    multiplier: Decimal = Decimal("1"),
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(days_ago, len(values)):
        if cutoff and dates[idx] < cutoff:
            continue
        history.append((values[idx] - values[idx - days_ago]) * multiplier)
    return history


def compute_pct_above_ma_series(
    dates: List[datetime],
    prices: List[Decimal],
    window: int,
    cutoff: datetime | None,
) -> List[Decimal]:
    history: List[Decimal] = []
    for idx in range(window - 1, len(prices)):
        if cutoff and dates[idx] < cutoff:
            continue
        window_prices = prices[idx - window + 1 : idx + 1]
        moving_avg = sum(window_prices, Decimal("0")) / Decimal(window)
        history.append((prices[idx] / moving_avg) - Decimal("1"))
    return history


def align_series_by_date(
    series_a: List[Tuple[datetime, Decimal]],
    series_b: List[Tuple[datetime, Decimal]],
) -> Tuple[List[datetime], List[Decimal], List[Decimal]]:
    a_map: Dict[datetime, Decimal] = {d.date(): v for d, v in series_a}
    b_map: Dict[datetime, Decimal] = {d.date(): v for d, v in series_b}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    dates = [datetime.combine(d, datetime.min.time()) for d in common]
    a_vals = [a_map[d] for d in common]
    b_vals = [b_map[d] for d in common]
    return dates, a_vals, b_vals


def compute_daily_return_series(values: List[Decimal]) -> List[Decimal]:
    out: List[Decimal] = []
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        cur = values[idx]
        if prev == 0:
            out.append(Decimal("0"))
        else:
            out.append((cur / prev) - Decimal("1"))
    return out


def compute_daily_change_bp_series(values: List[Decimal]) -> List[Decimal]:
    out: List[Decimal] = []
    for idx in range(1, len(values)):
        out.append((values[idx] - values[idx - 1]) * Decimal("100"))
    return out


def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    for idx in range(n):
        dx = x[idx] - mx
        dy = y[idx] - my
        sx += dx * dx
        sy += dy * dy
        sxy += dx * dy
    if sx <= 0.0 or sy <= 0.0:
        return None
    return sxy / math.sqrt(sx * sy)


def rolling_corr(
    dates: List[datetime],
    x: List[Decimal],
    y: List[Decimal],
    window: int,
) -> List[Tuple[datetime, Optional[float]]]:
    out: List[Tuple[datetime, Optional[float]]] = []
    if len(x) != len(y) or len(x) != len(dates):
        return out
    if len(x) < window:
        return out
    xf = [float(v) for v in x]
    yf = [float(v) for v in y]
    for idx in range(window - 1, len(x)):
        xs = xf[idx - window + 1 : idx + 1]
        ys = yf[idx - window + 1 : idx + 1]
        out.append((dates[idx], pearson_corr(xs, ys)))
    return out


def compute_corr_history(
    corr_series: List[Tuple[datetime, Optional[float]]],
    cutoff: datetime | None,
) -> List[Decimal]:
    history: List[Decimal] = []
    for date_value, corr_value in corr_series:
        if corr_value is None:
            continue
        if cutoff and date_value < cutoff:
            continue
        history.append(Decimal(str(corr_value)))
    return history


def percentile_with_fallback(
    value: Decimal | None,
    history: List[Decimal],
    fallback_history: List[Decimal],
) -> tuple[int | None, str | None, str | None, int, str]:
    if value is None:
        return None, "insufficient history", None, 0, "none"
    if history:
        return compute_percentile(value, history), None, None, len(history), "5y"
    if fallback_history:
        return (
            compute_percentile(value, fallback_history),
            "insufficient 5y history (used available history)",
            "used available history",
            len(fallback_history),
            "available",
        )
    return None, "insufficient history", None, 0, "none"


def classify_regime(
    primary_ret: Decimal | None,
    primary_ret_pctile: int | None,
    primary_drawdown: Decimal | None,
    primary_drawdown_pctile: int | None,
    ret_1m: Decimal | None,
    pct_above_200dma_pctile: int | None,
    holdings_change_21d_pct: Decimal | None,
    holdings_change_5d_pct: Decimal | None,
    holdings_change_21d_pct_pctile: int | None,
    real_yield_change_1m_bp: Decimal | None,
    real_yield_change_1m_bp_pctile: int | None,
    corr20: float | None,
    cut_label: str | None,
    cuts_priced: bool | None,
    previous_state: str | None,
    red_enter_streak: int,
    red_exit_streak: int,
) -> tuple[str, List[str], int, int]:
    if primary_ret is None or primary_drawdown is None:
        fallback_state = previous_state if previous_state in {"GREEN", "BLUE", "ORANGE", "RED"} else "GREEN"
        return fallback_state, ["insufficient_primary_history"], red_enter_streak, red_exit_streak

    extension_score = 0
    extension_triggers: List[str] = []
    if primary_ret_pctile is not None and primary_ret_pctile >= 90:
        extension_score += 1
        extension_triggers.append("extension_ret_pctile_90")
    if pct_above_200dma_pctile is not None and pct_above_200dma_pctile >= 90:
        extension_score += 1
        extension_triggers.append("extension_200dma_pctile_90")
    if primary_drawdown_pctile is not None and primary_drawdown_pctile >= 70:
        extension_score += 1
        extension_triggers.append("extension_drawdown_pctile_70")

    green_conditions = (
        primary_ret_pctile is not None
        and primary_ret_pctile < 80
        and pct_above_200dma_pctile is not None
        and pct_above_200dma_pctile < 80
        and holdings_change_21d_pct_pctile is not None
        and holdings_change_21d_pct_pctile > 20
        and real_yield_change_1m_bp_pctile is not None
        and real_yield_change_1m_bp_pctile < 80
    )

    ret_1m_positive = True if ret_1m is None else ret_1m > Decimal("0")
    ret_1m_nonpositive = False if ret_1m is None else ret_1m <= Decimal("0")

    deterioration_triggers: List[str] = []
    flow_divergence = (
        holdings_change_21d_pct is not None
        and holdings_change_21d_pct <= Decimal("-0.015")
        and primary_ret > Decimal("0")
        and ret_1m_positive
    )
    if flow_divergence:
        deterioration_triggers.append("deterioration_flow_divergence")
    if cut_label == "CREDIBILITY_CUT":
        macro_threshold = Decimal("15")
    elif cut_label == "STIMULUS_CUT":
        macro_threshold = Decimal("35")
    else:
        macro_threshold = Decimal("25")
    macro_turn = real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= macro_threshold
    if macro_turn:
        deterioration_triggers.append("deterioration_macro_turn")
    policy_conflict = (
        cuts_priced is True
        and cut_label == "CREDIBILITY_CUT"
        and real_yield_change_1m_bp is not None
        and real_yield_change_1m_bp >= Decimal("10")
    )
    if policy_conflict:
        deterioration_triggers.append("deterioration_policy_conflict")
    price_crack = ret_1m_nonpositive or primary_drawdown <= Decimal("-0.08")
    if price_crack:
        deterioration_triggers.append("deterioration_price_crack")
    follow_through = (
        corr20 is not None
        and corr20 <= -0.40
        and real_yield_change_1m_bp is not None
        and real_yield_change_1m_bp >= Decimal("10")
    )
    if follow_through:
        deterioration_triggers.append("deterioration_followthrough")

    deterioration_present = any([flow_divergence, macro_turn, price_crack, policy_conflict, follow_through])

    red_primary_triggers: List[str] = []
    red_primary = False
    if primary_ret <= Decimal("-0.15"):
        red_primary = True
        red_primary_triggers.append("primary_3m_return")
    if primary_drawdown <= Decimal("-0.18"):
        red_primary = True
        red_primary_triggers.append("primary_3m_drawdown")
    if real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= Decimal("50"):
        red_primary = True
        red_primary_triggers.append("primary_real_yield_1m")

    composite_conditions = [
        ("composite_3m_return", primary_ret <= Decimal("-0.08")),
        (
            "composite_holdings_21d",
            holdings_change_21d_pct is not None and holdings_change_21d_pct <= Decimal("-0.02"),
        ),
        (
            "composite_holdings_5d",
            holdings_change_5d_pct is not None and holdings_change_5d_pct <= Decimal("-0.01"),
        ),
        (
            "composite_real_yield_1m",
            real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= Decimal("25"),
        ),
        ("composite_followthrough", follow_through),
    ]
    composite_hits = [label for label, hit in composite_conditions if hit]
    red_composite = len(composite_hits) >= 2

    candidate_red = red_primary or red_composite
    candidate_blue = not green_conditions and extension_score >= 2
    candidate_orange = candidate_blue and deterioration_present

    if candidate_red:
        red_enter_streak += 1
        red_exit_streak = 0
    else:
        red_enter_streak = 0
        red_exit_streak += 1

    red_active = False
    if red_primary:
        red_active = True
    elif candidate_red and red_enter_streak >= 2:
        red_active = True
    elif previous_state == "RED" and red_exit_streak < 5:
        red_active = True

    if red_active:
        reasons = []
        reasons.extend(red_primary_triggers)
        if red_composite:
            reasons.extend(composite_hits)
        if not candidate_red and previous_state == "RED":
            reasons.append("red_persistence_cooldown")
        if not reasons:
            reasons.append("red_persistence")
        return "RED", reasons, red_enter_streak, red_exit_streak
    if candidate_orange:
        return "ORANGE", extension_triggers + deterioration_triggers, red_enter_streak, red_exit_streak
    if candidate_blue:
        return "BLUE", extension_triggers, red_enter_streak, red_exit_streak
    return "GREEN", ["green_conditions_met"] if green_conditions else [], red_enter_streak, red_exit_streak


def build_transition_row(
    *,
    name: str,
    threshold_label: str,
    current: Decimal | int | None,
    threshold_value: Decimal | int | None,
    fired: bool,
    unit: str,
    note: str | None = None,
) -> dict:
    if current is None:
        current_value = None
        fired_value = None
    else:
        current_value = float(current)
        fired_value = fired

    if threshold_value is None or current is None:
        distance = None
    else:
        if fired:
            distance = Decimal("0")
        else:
            distance = Decimal(str(threshold_value)) - Decimal(str(current))

    return {
        "trigger": name,
        "threshold": threshold_label,
        "current": current_value,
        "fired": fired_value,
        "distance": float(distance) if distance is not None else None,
        "unit": unit,
        "note": note,
    }


def build_regime_transitions(
    *,
    primary_ret: Decimal | None,
    primary_drawdown: Decimal | None,
    ret_1m: Decimal | None,
    primary_ret_pctile: int | None,
    pct_above_200dma_pctile: int | None,
    holdings_change_21d_pct: Decimal | None,
    holdings_change_5d_pct: Decimal | None,
    holdings_change_21d_pct_pctile: int | None,
    real_yield_change_1m_bp: Decimal | None,
    real_yield_change_1m_bp_pctile: int | None,
    corr20: float | None,
    cut_label: str | None = None,
) -> dict:
    flow_threshold = Decimal("-0.015")
    if cut_label == "CREDIBILITY_CUT":
        macro_threshold = Decimal("15")
    elif cut_label == "STIMULUS_CUT":
        macro_threshold = Decimal("35")
    else:
        macro_threshold = Decimal("25")
    price_crack_return_threshold = Decimal("0")
    price_crack_drawdown_threshold = Decimal("-0.08")
    red_primary_return_threshold = Decimal("-0.15")
    red_primary_drawdown_threshold = Decimal("-0.18")
    red_primary_real_yield_threshold = Decimal("50")
    composite_return_threshold = Decimal("-0.08")
    composite_holdings_21d_threshold = Decimal("-0.02")
    composite_holdings_5d_threshold = Decimal("-0.01")
    composite_real_yield_threshold = Decimal("25")
    follow_through_corr_threshold = -0.40
    follow_through_real_yield_threshold = Decimal("10")

    ret_1m_positive = True if ret_1m is None else ret_1m > Decimal("0")
    ret_1m_nonpositive = False if ret_1m is None else ret_1m <= Decimal("0")
    flow_divergence = (
        holdings_change_21d_pct is not None
        and holdings_change_21d_pct <= flow_threshold
        and primary_ret is not None
        and primary_ret > Decimal("0")
        and ret_1m_positive
    )
    macro_turn = real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= macro_threshold
    price_crack = ret_1m_nonpositive or (
        primary_drawdown is not None and primary_drawdown <= price_crack_drawdown_threshold
    )
    follow_through = (
        corr20 is not None
        and corr20 <= follow_through_corr_threshold
        and real_yield_change_1m_bp is not None
        and real_yield_change_1m_bp >= follow_through_real_yield_threshold
    )

    deescalate_to_blue = [
        build_transition_row(
            name="Price crack clears (1M return)",
            threshold_label="1M return > 0%",
            current=ret_1m,
            threshold_value=price_crack_return_threshold,
            fired=True if ret_1m is None else ret_1m > price_crack_return_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="Price crack clears (3M drawdown)",
            threshold_label="3M drawdown > -8%",
            current=primary_drawdown,
            threshold_value=price_crack_drawdown_threshold,
            fired=primary_drawdown is not None and primary_drawdown > price_crack_drawdown_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="Flow divergence clears",
            threshold_label="Holdings 21D > -1.5%",
            current=holdings_change_21d_pct,
            threshold_value=flow_threshold,
            fired=not flow_divergence,
            unit="percent",
            note="Flow divergence only evaluated when 3M return > 0 and 1M return > 0.",
        ),
        build_transition_row(
            name="Macro turn clears",
            threshold_label=f"Real yield 1M change < +{macro_threshold} bp",
            current=real_yield_change_1m_bp,
            threshold_value=macro_threshold,
            fired=real_yield_change_1m_bp is not None and real_yield_change_1m_bp < macro_threshold,
            unit="bp",
        ),
    ]

    escalate_to_red_primary = [
        build_transition_row(
            name="RED primary: 3M return",
            threshold_label="≤ -15%",
            current=primary_ret,
            threshold_value=red_primary_return_threshold,
            fired=primary_ret is not None and primary_ret <= red_primary_return_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="RED primary: 3M drawdown",
            threshold_label="≤ -18%",
            current=primary_drawdown,
            threshold_value=red_primary_drawdown_threshold,
            fired=primary_drawdown is not None and primary_drawdown <= red_primary_drawdown_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="RED primary: real yield 1M",
            threshold_label="≥ +50 bp",
            current=real_yield_change_1m_bp,
            threshold_value=red_primary_real_yield_threshold,
            fired=real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= red_primary_real_yield_threshold,
            unit="bp",
        ),
    ]

    escalate_to_red_composite = [
        build_transition_row(
            name="RED composite: 3M return",
            threshold_label="≤ -8%",
            current=primary_ret,
            threshold_value=composite_return_threshold,
            fired=primary_ret is not None and primary_ret <= composite_return_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="RED composite: holdings 21D",
            threshold_label="≤ -2%",
            current=holdings_change_21d_pct,
            threshold_value=composite_holdings_21d_threshold,
            fired=holdings_change_21d_pct is not None and holdings_change_21d_pct <= composite_holdings_21d_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="RED composite: holdings 5D",
            threshold_label="≤ -1%",
            current=holdings_change_5d_pct,
            threshold_value=composite_holdings_5d_threshold,
            fired=holdings_change_5d_pct is not None and holdings_change_5d_pct <= composite_holdings_5d_threshold,
            unit="percent",
        ),
        build_transition_row(
            name="RED composite: real yield 1M",
            threshold_label="≥ +25 bp",
            current=real_yield_change_1m_bp,
            threshold_value=composite_real_yield_threshold,
            fired=real_yield_change_1m_bp is not None and real_yield_change_1m_bp >= composite_real_yield_threshold,
            unit="bp",
        ),
        build_transition_row(
            name="RED composite: macro follow-through",
            threshold_label="Corr ≤ -0.40 & 1M real yield ≥ +10 bp",
            current=real_yield_change_1m_bp,
            threshold_value=follow_through_real_yield_threshold,
            fired=follow_through,
            unit="bp",
            note=f"20D corr: {corr20:.2f}" if corr20 is not None else "20D corr unavailable.",
        ),
    ]

    normalize_to_green = [
        build_transition_row(
            name="Return percentile normalizes",
            threshold_label="3M return percentile < 80",
            current=primary_ret_pctile,
            threshold_value=80,
            fired=primary_ret_pctile is not None and primary_ret_pctile < 80,
            unit="pctile",
        ),
        build_transition_row(
            name="% above 200DMA normalizes",
            threshold_label="% above 200DMA percentile < 80",
            current=pct_above_200dma_pctile,
            threshold_value=80,
            fired=pct_above_200dma_pctile is not None and pct_above_200dma_pctile < 80,
            unit="pctile",
        ),
        build_transition_row(
            name="Flows percentile supportive",
            threshold_label="Holdings 21D flow percentile > 20",
            current=holdings_change_21d_pct_pctile,
            threshold_value=20,
            fired=holdings_change_21d_pct_pctile is not None and holdings_change_21d_pct_pctile > 20,
            unit="pctile",
        ),
        build_transition_row(
            name="Real yield percentile benign",
            threshold_label="Real yield change percentile < 80",
            current=real_yield_change_1m_bp_pctile,
            threshold_value=80,
            fired=real_yield_change_1m_bp_pctile is not None and real_yield_change_1m_bp_pctile < 80,
            unit="pctile",
        ),
    ]

    return {
        "deescalate_to_blue": deescalate_to_blue,
        "escalate_to_red_primary": escalate_to_red_primary,
        "escalate_to_red_composite": escalate_to_red_composite,
        "normalize_to_green": normalize_to_green,
        "flags": {
            "flow_divergence": flow_divergence,
            "macro_turn": macro_turn,
            "price_crack": price_crack,
            "macro_followthrough": follow_through,
        },
    }


def build_issue_body(metrics: dict, regime: dict, transition: str) -> str:
    reasons = regime.get("reasons") or []
    reasons_text = "\n".join(f"- {reason}" for reason in reasons) if reasons else "- None"
    lines = [
        f"Regime transition: {transition}",
        "",
        "Latest metrics:",
        "",
        f"- GLD price: {metrics['gld_price']:.2f}" if metrics.get("gld_price") is not None else "- GLD price: n/a",
        f"- GLD 3M return: {metrics['horizons']['3M']['ret']:+.2%}"
        if metrics.get("horizons", {}).get("3M", {}).get("ret") is not None
        else "- GLD 3M return: n/a",
        f"- GLD 3M max drawdown: {metrics['horizons']['3M']['max_drawdown']:+.2%}"
        if metrics.get("horizons", {}).get("3M", {}).get("max_drawdown") is not None
        else "- GLD 3M max drawdown: n/a",
        f"- GLD % above 200DMA: {metrics['pct_above_200dma']:+.2%}"
        if metrics.get("pct_above_200dma") is not None
        else "- GLD % above 200DMA: n/a",
        f"- DFII10 level: {metrics['macro']['real_yield_today']:.2f}%"
        if metrics.get("macro", {}).get("real_yield_today") is not None
        else "- DFII10 level: n/a",
        f"- DFII10 1M change: {metrics['macro']['real_yield_change_1m_bp']:+.0f} bp"
        if metrics.get("macro", {}).get("real_yield_change_1m_bp") is not None
        else "- DFII10 1M change: n/a",
        f"- GLD holdings today: {metrics['flows']['holdings_today_tonnes']:.1f} t"
        if metrics.get("flows", {}).get("holdings_today_tonnes") is not None
        else "- GLD holdings today: n/a",
        f"- GLD holdings 5D change: {metrics['flows']['holdings_change_5d_pct']:+.2%}"
        if metrics.get("flows", {}).get("holdings_change_5d_pct") is not None
        else "- GLD holdings 5D change: n/a",
        f"- GLD holdings 21D change: {metrics['flows']['holdings_change_21d_pct']:+.2%}"
        if metrics.get("flows", {}).get("holdings_change_21d_pct") is not None
        else "- GLD holdings 21D change: n/a",
        "",
        "Triggers:",
        reasons_text,
    ]
    return "\n".join(lines)


def classify_cut_style(
    *,
    dgs2_1m_bp: Decimal | None,
    dgs2_3m_bp: Decimal | None,
    dgs10_1m_bp: Decimal | None,
    be10_1m_bp: Decimal | None,
    real10_1m_bp: Decimal | None,
) -> dict:
    if (
        dgs2_1m_bp is None
        or dgs10_1m_bp is None
        or be10_1m_bp is None
        or real10_1m_bp is None
    ):
        return {
            "label": "UNKNOWN",
            "score": None,
            "cuts_priced": None,
            "explain": ["insufficient_data"],
            "inputs": {},
        }

    cuts_priced = dgs2_1m_bp <= Decimal("-15") or (
        dgs2_3m_bp is not None and dgs2_3m_bp <= Decimal("-25")
    )

    denom = max(abs(dgs10_1m_bp), Decimal("1"))
    disinflation_ratio = abs(be10_1m_bp) / denom

    be_falling = be10_1m_bp <= Decimal("-10")
    real_rising = real10_1m_bp >= Decimal("10")
    ratio_cred = disinflation_ratio >= Decimal("1.2")

    be_rising = be10_1m_bp >= Decimal("5")
    real_falling = real10_1m_bp <= Decimal("-10")
    ratio_stim = disinflation_ratio <= Decimal("0.8")

    explain = [
        f"cuts_priced={cuts_priced}",
        f"dgs2_1m_bp={dgs2_1m_bp}",
        f"dgs10_1m_bp={dgs10_1m_bp}",
        f"be10_1m_bp={be10_1m_bp}",
        f"real10_1m_bp={real10_1m_bp}",
        f"disinflation_ratio={float(disinflation_ratio):.2f}",
    ]

    if cuts_priced and ((be_falling and real_rising) or ratio_cred):
        score = 2
        if be_falling:
            score += 1
        if real_rising:
            score += 1
        if ratio_cred:
            score += 1
        label = "CREDIBILITY_CUT"
    elif cuts_priced and (be_rising or (real_falling and ratio_stim)):
        score = -2
        if be_rising:
            score -= 1
        if real_falling:
            score -= 1
        if ratio_stim:
            score -= 1
        label = "STIMULUS_CUT"
    elif cuts_priced:
        score = 0
        label = "MIXED_CUT"
    else:
        score = 0
        label = "NO_CUTS_PRICED"

    return {
        "label": label,
        "score": score,
        "cuts_priced": bool(cuts_priced),
        "explain": explain,
        "inputs": {
            "nominal_2y_change_1m_bp": float(dgs2_1m_bp),
            "nominal_2y_change_3m_bp": float(dgs2_3m_bp) if dgs2_3m_bp is not None else None,
            "nominal_10y_change_1m_bp": float(dgs10_1m_bp),
            "breakeven_10y_change_1m_bp": float(be10_1m_bp),
            "real_10y_change_1m_bp": float(real10_1m_bp),
            "disinflation_ratio": float(disinflation_ratio),
        },
    }


def build_horizon_metrics(
    label: str,
    days: int,
    dates: List[datetime],
    prices: List[Decimal],
    cutoff: datetime,
    percentile_notes: dict,
) -> dict:
    available = len(prices) >= days + 1
    if not available:
        return {
            "days": days,
            "available": False,
            "ret": None,
            "ret_pctile_5y": None,
            "ret_pctile_5y_explain": "insufficient history",
            "ret_pctile_basis": "none",
            "ret_pctile_n": 0,
            "max_drawdown": None,
            "max_drawdown_pctile_5y": None,
            "max_drawdown_pctile_5y_explain": "insufficient history",
            "max_drawdown_pctile_basis": "none",
            "max_drawdown_pctile_n": 0,
        }

    ret = compute_return(prices, days)
    window_prices = prices[-days:]
    max_drawdown = compute_max_drawdown(window_prices)

    ret_history = compute_return_series(dates, prices, days, cutoff)
    ret_history_full = compute_return_series(dates, prices, days, None)
    ret_pctile, ret_explain, ret_note, ret_n, ret_basis = percentile_with_fallback(
        ret,
        ret_history,
        ret_history_full,
    )
    if ret_note:
        percentile_notes[label] = ret_note

    drawdown_history = compute_drawdown_series(dates, prices, days, cutoff)
    drawdown_history_full = compute_drawdown_series(dates, prices, days, None)
    draw_pctile, draw_explain, draw_note, draw_n, draw_basis = percentile_with_fallback(
        max_drawdown,
        drawdown_history,
        drawdown_history_full,
    )
    if draw_note:
        if label in percentile_notes:
            percentile_notes[label] = f"{percentile_notes[label]} {draw_note}"
        else:
            percentile_notes[label] = draw_note

    return {
        "days": days,
        "available": True,
        "ret": float(ret) if ret is not None else None,
        "ret_pctile_5y": ret_pctile,
        "ret_pctile_5y_explain": ret_explain,
        "ret_pctile_basis": ret_basis,
        "ret_pctile_n": ret_n,
        "max_drawdown": float(max_drawdown) if max_drawdown is not None else None,
        "max_drawdown_pctile_5y": draw_pctile,
        "max_drawdown_pctile_5y_explain": draw_explain,
        "max_drawdown_pctile_basis": draw_basis,
        "max_drawdown_pctile_n": draw_n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Gold Risk Monitor data")
    parser.add_argument("--output", default="data.json")
    parser.add_argument("--status-file", default="/tmp/monitor_status.json")
    args = parser.parse_args()

    status = Status(fetch_ok=False, flag=None, issue_title=None, issue_body=None, error=None)
    previous_state = {}
    previous_regime = {}
    previous_regime_state = None
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as handle:
                previous_data = json.load(handle)
                if isinstance(previous_data, dict):
                    previous_regime = previous_data.get("regime", {}) or {}
                    previous_state = previous_data.get("state", {}) or {}
                    previous_regime_state = previous_regime.get("state") or previous_data.get("flag")
        except (OSError, json.JSONDecodeError):
            previous_state = {}
            previous_regime = {}
            previous_regime_state = None
    try:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise DataFetchError("FRED_API_KEY is not set")

        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
        now_utc_naive = now_utc.replace(tzinfo=None)

        gld_rows = fetch_gld_prices()
        fred_rows = fetch_fred_series(FRED_SERIES["real10"], api_key, now_utc_naive)
        fred_nom10_rows = fetch_fred_series(FRED_SERIES["nom10"], api_key, now_utc_naive)
        fred_nom2_rows = fetch_fred_series(FRED_SERIES["nom2"], api_key, now_utc_naive)
        fred_be10_rows = fetch_fred_series(FRED_SERIES["be10"], api_key, now_utc_naive)
        holdings_rows = fetch_gld_holdings()

        gld_dates = [row[0] for row in gld_rows]
        gld_prices = [row[1] for row in gld_rows]
        if not gld_prices:
            raise DataFetchError("No GLD price data available")
        gld_ret_1m = compute_return(gld_prices, TRADING_DAYS_1M)
        gld_200dma = compute_simple_moving_average(gld_prices, TRADING_DAYS_200)
        gld_pct_above_200dma = (
            (gld_prices[-1] / gld_200dma) - Decimal("1") if gld_200dma is not None else None
        )

        fred_dates = [row[0] for row in fred_rows]
        fred_values = [row[1] for row in fred_rows]
        real_yield_today = fred_values[-1] if fred_values else None
        real_yield_change_1m_bp = compute_bp_change(fred_values, TRADING_DAYS_1M)
        real_yield_change_3m_bp = compute_bp_change(fred_values, TRADING_DAYS_3M)

        aligned_dates, aligned_prices, aligned_real_yields = align_series_by_date(gld_rows, fred_rows)
        corr_dates = aligned_dates[1:]
        gld_daily_ret = compute_daily_return_series(aligned_prices)
        real_daily_chg_bp = compute_daily_change_bp_series(aligned_real_yields)
        corr20_series = rolling_corr(corr_dates, gld_daily_ret, real_daily_chg_bp, window=20)
        corr20_today = corr20_series[-1][1] if corr20_series else None

        nom10_values = [row[1] for row in fred_nom10_rows]
        nom2_values = [row[1] for row in fred_nom2_rows]
        be10_values = [row[1] for row in fred_be10_rows]

        nom10_1m_bp = compute_bp_change(nom10_values, TRADING_DAYS_1M)
        nom2_1m_bp = compute_bp_change(nom2_values, TRADING_DAYS_1M)
        nom2_3m_bp = compute_bp_change(nom2_values, TRADING_DAYS_3M)
        be10_1m_bp = compute_bp_change(be10_values, TRADING_DAYS_1M)

        cut_classifier = classify_cut_style(
            dgs2_1m_bp=nom2_1m_bp,
            dgs2_3m_bp=nom2_3m_bp,
            dgs10_1m_bp=nom10_1m_bp,
            be10_1m_bp=be10_1m_bp,
            real10_1m_bp=real_yield_change_1m_bp,
        )
        cut_inputs = cut_classifier.get("inputs", {})
        cut_inputs.update(
            {
                "nominal_2y_today": float(nom2_values[-1]) if nom2_values else None,
                "nominal_10y_today": float(nom10_values[-1]) if nom10_values else None,
                "breakeven_10y_today": float(be10_values[-1]) if be10_values else None,
            }
        )

        holdings_dates = [row[0] for row in holdings_rows]
        holdings_values = [row[1] for row in holdings_rows]
        gld_holdings_today = holdings_values[-1] if holdings_values else None
        gld_holdings_change_5d_pct = compute_return(holdings_values, TRADING_DAYS_1W)
        gld_holdings_change_21d_pct = compute_return(holdings_values, TRADING_DAYS_1M)

        cutoff_date = percentile_history_cutoff(now_utc_naive)
        gld_pct_above_200dma_history = compute_pct_above_ma_series(
            gld_dates,
            gld_prices,
            TRADING_DAYS_200,
            cutoff_date,
        )
        gld_pct_above_200dma_history_full = compute_pct_above_ma_series(
            gld_dates,
            gld_prices,
            TRADING_DAYS_200,
            None,
        )
        real_yield_change_1m_history = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_1M,
            cutoff_date,
            Decimal("100"),
        )
        real_yield_change_1m_history_full = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_1M,
            None,
            Decimal("100"),
        )
        real_yield_change_3m_history = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_3M,
            cutoff_date,
            Decimal("100"),
        )
        real_yield_change_3m_history_full = compute_change_series(
            fred_dates,
            fred_values,
            TRADING_DAYS_3M,
            None,
            Decimal("100"),
        )
        corr20_history_5y = compute_corr_history(corr20_series, cutoff_date)
        corr20_history_full = compute_corr_history(corr20_series, None)
        holdings_change_5d_history = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1W,
            cutoff_date,
        )
        holdings_change_5d_history_full = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1W,
            None,
        )
        holdings_change_21d_history = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1M,
            cutoff_date,
        )
        holdings_change_21d_history_full = compute_return_series(
            holdings_dates,
            holdings_values,
            TRADING_DAYS_1M,
            None,
        )

        (
            gld_pct_above_200dma_pctile_5y,
            gld_pct_above_200dma_pctile_explain,
            gld_pct_above_200dma_pctile_note,
            gld_pct_above_200dma_pctile_n,
            gld_pct_above_200dma_pctile_basis,
        ) = percentile_with_fallback(
            gld_pct_above_200dma,
            gld_pct_above_200dma_history,
            gld_pct_above_200dma_history_full,
        )
        (
            real_yield_change_1m_pctile_5y,
            real_yield_change_1m_pctile_explain,
            real_yield_change_1m_pctile_note,
            real_yield_change_1m_pctile_n,
            real_yield_change_1m_pctile_basis,
        ) = percentile_with_fallback(
            real_yield_change_1m_bp,
            real_yield_change_1m_history,
            real_yield_change_1m_history_full,
        )
        (
            real_yield_change_3m_pctile_5y,
            real_yield_change_3m_pctile_explain,
            real_yield_change_3m_pctile_note,
            real_yield_change_3m_pctile_n,
            real_yield_change_3m_pctile_basis,
        ) = percentile_with_fallback(
            real_yield_change_3m_bp,
            real_yield_change_3m_history,
            real_yield_change_3m_history_full,
        )
        (
            holdings_change_5d_pctile_5y,
            holdings_change_5d_pctile_explain,
            holdings_change_5d_pctile_note,
            holdings_change_5d_pctile_n,
            holdings_change_5d_pctile_basis,
        ) = percentile_with_fallback(
            gld_holdings_change_5d_pct,
            holdings_change_5d_history,
            holdings_change_5d_history_full,
        )
        (
            holdings_change_21d_pctile_5y,
            holdings_change_21d_pctile_explain,
            holdings_change_21d_pctile_note,
            holdings_change_21d_pctile_n,
            holdings_change_21d_pctile_basis,
        ) = percentile_with_fallback(
            gld_holdings_change_21d_pct,
            holdings_change_21d_history,
            holdings_change_21d_history_full,
        )
        (
            corr20_pctile_5y,
            corr20_pctile_explain,
            corr20_pctile_note,
            corr20_pctile_n,
            corr20_pctile_basis,
        ) = percentile_with_fallback(
            Decimal(str(corr20_today)) if corr20_today is not None else None,
            corr20_history_5y,
            corr20_history_full,
        )

        percentile_notes: dict[str, str] = {}
        for note in (
            gld_pct_above_200dma_pctile_note,
            real_yield_change_1m_pctile_note,
            real_yield_change_3m_pctile_note,
            holdings_change_5d_pctile_note,
            holdings_change_21d_pctile_note,
            corr20_pctile_note,
        ):
            if note:
                percentile_notes["short_term"] = note
        horizons = {
            label: build_horizon_metrics(label, days, gld_dates, gld_prices, cutoff_date, percentile_notes)
            for label, days in HORIZON_WINDOWS.items()
        }

        persistence = previous_regime.get("persistence", {}) if isinstance(previous_regime, dict) else {}
        red_enter_streak = int(
            persistence.get("red_enter_streak", previous_state.get("red_streak", 0) or 0) or 0
        )
        red_exit_streak = int(
            persistence.get("red_exit_streak", previous_state.get("no_red_streak", 0) or 0) or 0
        )

        primary_horizon = horizons.get("3M", {})
        primary_ret_value = (
            Decimal(str(primary_horizon.get("ret"))) if primary_horizon.get("ret") is not None else None
        )
        primary_drawdown_value = (
            Decimal(str(primary_horizon.get("max_drawdown")))
            if primary_horizon.get("max_drawdown") is not None
            else None
        )
        regime_state, regime_reasons, red_enter_streak, red_exit_streak = classify_regime(
            primary_ret_value,
            primary_horizon.get("ret_pctile_5y"),
            primary_drawdown_value,
            primary_horizon.get("max_drawdown_pctile_5y"),
            gld_ret_1m,
            gld_pct_above_200dma_pctile_5y,
            gld_holdings_change_21d_pct,
            gld_holdings_change_5d_pct,
            holdings_change_21d_pctile_5y,
            real_yield_change_1m_bp,
            real_yield_change_1m_pctile_5y,
            corr20_today,
            cut_classifier.get("label"),
            cut_classifier.get("cuts_priced"),
            previous_regime_state,
            red_enter_streak,
            red_exit_streak,
        )
        transitions = build_regime_transitions(
            primary_ret=primary_ret_value,
            primary_drawdown=primary_drawdown_value,
            ret_1m=gld_ret_1m,
            primary_ret_pctile=primary_horizon.get("ret_pctile_5y"),
            pct_above_200dma_pctile=gld_pct_above_200dma_pctile_5y,
            holdings_change_21d_pct=gld_holdings_change_21d_pct,
            holdings_change_5d_pct=gld_holdings_change_5d_pct,
            holdings_change_21d_pct_pctile=holdings_change_21d_pctile_5y,
            real_yield_change_1m_bp=real_yield_change_1m_bp,
            real_yield_change_1m_bp_pctile=real_yield_change_1m_pctile_5y,
            corr20=corr20_today,
            cut_label=cut_classifier.get("label"),
        )

        repository = None
        repo_env = os.environ.get("GITHUB_REPOSITORY")
        if repo_env:
            repository = f"https://github.com/{repo_env}"

        metrics = {
            "gld_price": float(gld_prices[-1]) if gld_prices else None,
            "ma_200": float(gld_200dma) if gld_200dma is not None else None,
            "pct_above_200dma": float(gld_pct_above_200dma) if gld_pct_above_200dma is not None else None,
            "pct_above_200dma_pctile_5y": gld_pct_above_200dma_pctile_5y,
            "pct_above_200dma_pctile_5y_explain": gld_pct_above_200dma_pctile_explain,
            "pct_above_200dma_pctile_basis": gld_pct_above_200dma_pctile_basis,
            "pct_above_200dma_pctile_n": gld_pct_above_200dma_pctile_n,
            "flows": {
                "holdings_today_tonnes": float(gld_holdings_today) if gld_holdings_today is not None else None,
                "holdings_change_5d_pct": float(gld_holdings_change_5d_pct)
                if gld_holdings_change_5d_pct is not None
                else None,
                "holdings_change_21d_pct": float(gld_holdings_change_21d_pct)
                if gld_holdings_change_21d_pct is not None
                else None,
                "holdings_change_5d_pct_pctile_5y": holdings_change_5d_pctile_5y,
                "holdings_change_5d_pct_pctile_5y_explain": holdings_change_5d_pctile_explain,
                "holdings_change_5d_pctile_basis": holdings_change_5d_pctile_basis,
                "holdings_change_5d_pctile_n": holdings_change_5d_pctile_n,
                "holdings_change_21d_pct_pctile_5y": holdings_change_21d_pctile_5y,
                "holdings_change_21d_pct_pctile_5y_explain": holdings_change_21d_pctile_explain,
                "holdings_change_21d_pctile_basis": holdings_change_21d_pctile_basis,
                "holdings_change_21d_pctile_n": holdings_change_21d_pctile_n,
            },
            "macro": {
                "real_yield_today": float(real_yield_today) if real_yield_today is not None else None,
                "real_yield_change_1m_bp": float(real_yield_change_1m_bp)
                if real_yield_change_1m_bp is not None
                else None,
                "real_yield_change_3m_bp": float(real_yield_change_3m_bp)
                if real_yield_change_3m_bp is not None
                else None,
                "corr_gld_ret_vs_real_yield_chg_20d": corr20_today,
                "corr_gld_ret_vs_real_yield_chg_20d_pctile_5y": corr20_pctile_5y,
                "corr_gld_ret_vs_real_yield_chg_20d_pctile_5y_explain": corr20_pctile_explain,
                "corr_gld_ret_vs_real_yield_chg_20d_pctile_basis": corr20_pctile_basis,
                "corr_gld_ret_vs_real_yield_chg_20d_pctile_n": corr20_pctile_n,
                "real_yield_change_1m_bp_pctile_5y": real_yield_change_1m_pctile_5y,
                "real_yield_change_1m_bp_pctile_5y_explain": real_yield_change_1m_pctile_explain,
                "real_yield_change_1m_bp_pctile_basis": real_yield_change_1m_pctile_basis,
                "real_yield_change_1m_bp_pctile_n": real_yield_change_1m_pctile_n,
                "real_yield_change_3m_bp_pctile_5y": real_yield_change_3m_pctile_5y,
                "real_yield_change_3m_bp_pctile_5y_explain": real_yield_change_3m_pctile_explain,
                "real_yield_change_3m_bp_pctile_basis": real_yield_change_3m_pctile_basis,
                "real_yield_change_3m_bp_pctile_n": real_yield_change_3m_pctile_n,
                "nominal_10y_today": float(nom10_values[-1]) if nom10_values else None,
                "nominal_2y_today": float(nom2_values[-1]) if nom2_values else None,
                "breakeven_10y_today": float(be10_values[-1]) if be10_values else None,
                "nominal_10y_change_1m_bp": float(nom10_1m_bp) if nom10_1m_bp is not None else None,
                "nominal_2y_change_1m_bp": float(nom2_1m_bp) if nom2_1m_bp is not None else None,
                "nominal_2y_change_3m_bp": float(nom2_3m_bp) if nom2_3m_bp is not None else None,
                "breakeven_10y_change_1m_bp": float(be10_1m_bp) if be10_1m_bp is not None else None,
                "cut_style_classifier": {
                    "label": cut_classifier.get("label"),
                    "score": cut_classifier.get("score"),
                    "explain": cut_classifier.get("explain"),
                    "inputs": cut_inputs,
                },
            },
            "horizons": horizons,
        }

        data = {
            "updated_at": now_et.isoformat(),
            "sources": {
                "gld_prices": GLD_STOOQ_URL,
                "gld_holdings": GLD_HOLDINGS_URL,
                "dfii10": "https://fred.stlouisfed.org/series/DFII10",
                "dgs10": "https://fred.stlouisfed.org/series/DGS10",
                "dgs2": "https://fred.stlouisfed.org/series/DGS2",
                "t10yie": "https://fred.stlouisfed.org/series/T10YIE",
            },
            "repository": repository,
            "regime": {
                "state": regime_state,
                "primary_horizon": "3M",
                "reasons": regime_reasons,
                "persistence": {
                    "red_enter_streak": red_enter_streak,
                    "red_exit_streak": red_exit_streak,
                },
                "flags": transitions.get("flags", {}),
                "transitions": transitions,
            },
            "metrics": metrics,
            "notes": {
                "percentile_notes": percentile_notes,
                "methodology": "Descriptive market regime; no implied actions.",
            },
        }

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

        status.fetch_ok = True
        status.flag = regime_state
        if previous_regime_state != "RED" and regime_state == "RED":
            status.issue_title = "🚨 Gold Risk Monitor: RED regime"
            status.issue_body = build_issue_body(data["metrics"], data["regime"], "entered RED")
        elif previous_regime_state == "RED" and regime_state != "RED":
            status.issue_title = "🚨 Gold Risk Monitor: RED regime"
            status.issue_body = build_issue_body(data["metrics"], data["regime"], "exited RED")
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
