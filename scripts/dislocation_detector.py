#!/usr/bin/env python3
"""Dislocation detector (daily, low-churn) using free data (Stooq) + optional FRED."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests


STOOQ_DAILY = "https://stooq.com/q/d/l/?s={symbol}&i=d"
DEFAULT_TICKERS = {
    "equity_core": "spy.us",
    "equity_broad": "vti.us",
    "gold": "gld.us",
    "credit_hy": "hyg.us",
    "vix": "vi.c",
}


def fetch_stooq_daily(symbol: str, require_volume: bool = True) -> pd.DataFrame:
    url = STOOQ_DAILY.format(symbol=symbol)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    if "<html" in response.text.lower():
        raise ValueError(f"Stooq returned HTML for symbol: {symbol}")
    try:
        df = pd.read_csv(StringIO(response.text))
    except pd.errors.ParserError as exc:
        raise ValueError(f"Unable to parse Stooq CSV for symbol: {symbol}") from exc
    normalized = {column.lower(): column for column in df.columns}
    if "date" not in normalized:
        raise ValueError(f"Stooq CSV missing Date column for symbol: {symbol}")
    df = df.rename(columns={normalized["date"]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").set_index("Date")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    required = {"Open", "High", "Low", "Close"}
    if require_volume:
        required.add("Volume")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Stooq CSV missing columns for symbol {symbol}: {sorted(missing)}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df.dropna(subset=["Close"])


def fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    """Pull a FRED series via their JSON API (optional)."""
    import urllib.parse

    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()
    rows = []
    for observation in data.get("observations", []):
        dt = pd.to_datetime(observation["date"], utc=True)
        value = observation["value"]
        try:
            value = float(value)
        except ValueError:
            value = None
        rows.append((dt, value))
    df = pd.DataFrame(rows, columns=["Date", series_id]).dropna()
    return df.set_index("Date").sort_index()


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods) * 100.0


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, pd.NA)


def intraday_range_pct(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]) / df["Close"] * 100.0


def safe_last(series: pd.Series) -> float:
    return float(series.dropna().iloc[-1])


@dataclass
class SignalResult:
    name: str
    triggered: bool
    details: Dict[str, object]


@dataclass
class RunMeta:
    equities_data_date_utc: Optional[str]
    vix_data_date_utc: Optional[str]
    vix_stale_days: Optional[int]


def compute_signals(
    spy: pd.DataFrame,
    gld: pd.DataFrame,
    hyg: pd.DataFrame,
    vix: Optional[pd.DataFrame] = None,
    fred_hy_spread: Optional[pd.DataFrame] = None,
    lookback: int = 252,
) -> tuple[List[SignalResult], RunMeta]:
    idx = spy.index.intersection(gld.index).intersection(hyg.index)
    if vix is not None:
        idx = idx.intersection(vix.index)
    spy = spy.loc[idx]
    gld = gld.loc[idx]
    hyg = hyg.loc[idx]
    if vix is not None:
        vix = vix.loc[idx]

    equities_data_date = idx.max() if len(idx) else None
    vix_data_date = None
    vix_stale_days = None
    if vix is not None and not vix.empty:
        vix_data_date = vix.index.max()
        if equities_data_date is not None:
            vix_stale_days = (equities_data_date - vix_data_date).days

    spy_ret = pct_change(spy["Close"], 1)
    gld_ret = pct_change(gld["Close"], 1)
    hyg_ret = pct_change(hyg["Close"], 1)

    spy_rng = intraday_range_pct(spy)
    spy_rng_z = rolling_z(spy_rng, lookback)

    spy_vol = spy["Volume"].replace(0, pd.NA)
    spy_vol_z = rolling_z(spy_vol, lookback)

    vix_level = None
    vix_ret = None
    vix_level_z = None
    vix_ret_z = None
    if vix is not None:
        vix_level = vix["Close"]
        vix_ret = pct_change(vix_level, 1)
        vix_level_z = rolling_z(vix_level, lookback)
        vix_ret_z = rolling_z(vix_ret, lookback)

    big_down = spy_ret <= -2.5
    whipsaw = (spy_rng >= 2.5) | (spy_rng_z >= 2.0)
    sig1 = big_down & whipsaw

    liq_proxy = (spy_rng_z >= 2.0) & (spy_vol_z >= 2.0)

    if vix is None:
        vol_spike = pd.Series(False, index=idx)
    else:
        vol_spike = ((vix_level >= 30) | (vix_level_z >= 1.5)) & ((vix_ret >= 20) | (vix_ret_z >= 2.0))

    credit_stress = (hyg_ret - spy_ret) <= -1.5

    fred_flag = pd.Series(False, index=idx)
    fred_details: Dict[str, float] = {}
    if fred_hy_spread is not None and not fred_hy_spread.empty:
        series_id = fred_hy_spread.columns[0]
        tmp = fred_hy_spread.reindex(idx, method="ffill")[series_id]
        change_5d = tmp.diff(5)
        spread_z = rolling_z(tmp, lookback)
        fred_flag = (change_5d >= 0.50) | (spread_z >= 2.0)
        fred_details = {
            "hy_oas_level": safe_last(tmp),
            "hy_oas_5d_change": float(change_5d.dropna().iloc[-1]),
            "hy_oas_z": float(spread_z.dropna().iloc[-1]),
        }

    everything_sells = (spy_ret <= -2.0) & (gld_ret <= -1.0)
    forced_flow_proxy = big_down & credit_stress & vol_spike

    def latest_bool(series: pd.Series) -> bool:
        cleaned = series.dropna()
        return bool(cleaned.iloc[-1]) if not cleaned.empty else False

    results: List[SignalResult] = []

    results.append(SignalResult(
        name="big_down_and_whipsaw",
        triggered=latest_bool(sig1),
        details={
            "spy_1d_return_pct": safe_last(spy_ret),
            "spy_intraday_range_pct": safe_last(spy_rng),
            "spy_range_z": safe_last(spy_rng_z),
        },
    ))

    results.append(SignalResult(
        name="liquidity_degraded_proxy",
        triggered=latest_bool(liq_proxy),
        details={
            "spy_volume_z": safe_last(spy_vol_z),
            "spy_range_z": safe_last(spy_rng_z),
        },
    ))

    if vix is None:
        results.append(SignalResult(
            name="volatility_spike",
            triggered=False,
            details={"note": "VIX data unavailable from Stooq."},
        ))
    else:
        results.append(SignalResult(
            name="volatility_spike",
            triggered=latest_bool(vol_spike),
            details={
                "vix_level": safe_last(vix_level),
                "vix_1d_change_pct": safe_last(vix_ret),
                "vix_level_z": safe_last(vix_level_z),
                "vix_change_z": safe_last(vix_ret_z),
            },
        ))

    results.append(SignalResult(
        name="credit_stress_proxy",
        triggered=latest_bool(credit_stress),
        details={
            "hyg_1d_return_pct": safe_last(hyg_ret),
            "spy_1d_return_pct": safe_last(spy_ret),
            "hyg_minus_spy_pct": safe_last(hyg_ret - spy_ret),
        },
    ))

    if fred_hy_spread is not None and not fred_hy_spread.empty:
        results.append(SignalResult(
            name="credit_spread_widening_fred",
            triggered=latest_bool(fred_flag),
            details=fred_details,
        ))

    results.append(SignalResult(
        name="everything_sells_together",
        triggered=latest_bool(everything_sells),
        details={
            "spy_1d_return_pct": safe_last(spy_ret),
            "gld_1d_return_pct": safe_last(gld_ret),
        },
    ))

    if vix is None:
        results.append(SignalResult(
            name="forced_flow_proxy_combo",
            triggered=False,
            details={
                "spy_1d_return_pct": safe_last(spy_ret),
                "hyg_minus_spy_pct": safe_last(hyg_ret - spy_ret),
                "note": "VIX data unavailable from Stooq.",
            },
        ))
    else:
        results.append(SignalResult(
            name="forced_flow_proxy_combo",
            triggered=latest_bool(forced_flow_proxy),
            details={
                "spy_1d_return_pct": safe_last(spy_ret),
                "hyg_minus_spy_pct": safe_last(hyg_ret - spy_ret),
                "vix_level": safe_last(vix_level),
                "vix_1d_change_pct": safe_last(vix_ret),
            },
        ))

    meta = RunMeta(
        equities_data_date_utc=equities_data_date.isoformat() if equities_data_date is not None else None,
        vix_data_date_utc=vix_data_date.isoformat() if vix_data_date is not None else None,
        vix_stale_days=int(vix_stale_days) if vix_stale_days is not None else None,
    )
    return results, meta


def derive_status(count: int, dislocation: bool) -> str:
    if dislocation:
        return "dislocation"
    if count >= 1:
        return "stress_building"
    return "normal"


def summarize_dislocation(
    signals: List[SignalResult],
    meta: RunMeta,
    k_required: int = 3,
    previous_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    triggered = [signal for signal in signals if signal.triggered]
    count = len(triggered)
    yesterday_count = 0
    previous_status = None
    if previous_summary:
        try:
            yesterday_count = int(previous_summary.get("signals_triggered_count", 0) or 0)
        except (TypeError, ValueError):
            yesterday_count = 0
        previous_status = previous_summary.get("status")
        if not previous_status:
            previous_dislocation = bool(previous_summary.get("dislocation", False))
            previous_status = derive_status(yesterday_count, previous_dislocation)

    carry_forward = count == (k_required - 1) and yesterday_count >= 1
    dislocation = bool(count >= k_required or carry_forward)
    watch = bool((not dislocation) and count >= 1)
    status = derive_status(count, dislocation)
    transitions = []
    if previous_summary and isinstance(previous_summary.get("transitions"), list):
        transitions = list(previous_summary["transitions"])
    if previous_status and previous_status != status:
        transitions.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "from": previous_status,
            "to": status,
        })

    return {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "dislocation": dislocation,
        "watch": watch,
        "status": status,
        "signals_triggered_count": count,
        "signals_triggered_count_yesterday": yesterday_count,
        "signals_triggered": [signal.name for signal in triggered],
        "signals": [
            {"name": signal.name, "triggered": signal.triggered, "details": signal.details}
            for signal in signals
        ],
        "data_dates": {
            "equities_close_utc": meta.equities_data_date_utc,
            "vix_close_utc": meta.vix_data_date_utc,
            "vix_stale_days": meta.vix_stale_days,
        },
        "transitions": transitions,
        "rule": {
            "k_required": k_required,
            "notes": "Designed to be low-churn: multiple independent stress signals must agree.",
            "persistence": {
                "enabled": True,
                "today_minimum": max(k_required - 1, 1),
                "yesterday_minimum": 1,
                "description": "Dislocation holds if today is k-1 and yesterday had >=1 signal.",
            },
        },
    }


def load_previous_summary(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dislocation.json", help="Output JSON path")
    parser.add_argument("--k", type=int, default=3, help="Signals required to flag dislocation")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window for z-scores")
    parser.add_argument("--vix-symbol", default=DEFAULT_TICKERS["vix"], help="Stooq symbol for VIX (try vi.c, vix, or ^vix)")
    parser.add_argument("--spy-symbol", default=DEFAULT_TICKERS["equity_core"], help="Stooq symbol for SPY proxy")
    parser.add_argument("--gld-symbol", default=DEFAULT_TICKERS["gold"], help="Stooq symbol for GLD")
    parser.add_argument("--hyg-symbol", default=DEFAULT_TICKERS["credit_hy"], help="Stooq symbol for HYG")
    parser.add_argument("--fred-series", default="", help="Optional FRED series id (e.g., BAMLH0A0HYM2)")
    args = parser.parse_args()

    spy = fetch_stooq_daily(args.spy_symbol)
    gld = fetch_stooq_daily(args.gld_symbol)
    hyg = fetch_stooq_daily(args.hyg_symbol)

    vix = None
    tried = []
    for candidate in [args.vix_symbol, "vi.c", "^vix", "vix"]:
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            vix = fetch_stooq_daily(candidate, require_volume=False)
            break
        except ValueError:
            continue
    if vix is None:
        print(f"Warning: Unable to fetch VIX data from Stooq. Tried: {tried}")

    fred_df = None
    if args.fred_series:
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if api_key:
            fred_df = fetch_fred_series(args.fred_series, api_key)

    previous_summary = load_previous_summary(args.output)
    signals, meta = compute_signals(
        spy=spy,
        gld=gld,
        hyg=hyg,
        vix=vix,
        fred_hy_spread=fred_df,
        lookback=args.lookback,
    )
    summary = summarize_dislocation(signals, meta, k_required=args.k, previous_summary=previous_summary)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({
        "wrote": args.output,
        "dislocation": summary["dislocation"],
        "count": summary["signals_triggered_count"],
        "status": summary["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
