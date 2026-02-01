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
    "vix": "vix",
}


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = STOOQ_DAILY.format(symbol=symbol)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").set_index("Date")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
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
    details: Dict[str, float]


def compute_signals(
    spy: pd.DataFrame,
    gld: pd.DataFrame,
    hyg: pd.DataFrame,
    vix: pd.DataFrame,
    fred_hy_spread: Optional[pd.DataFrame] = None,
    lookback: int = 252,
) -> List[SignalResult]:
    idx = spy.index.intersection(gld.index).intersection(hyg.index).intersection(vix.index)
    spy = spy.loc[idx]
    gld = gld.loc[idx]
    hyg = hyg.loc[idx]
    vix = vix.loc[idx]

    spy_ret = pct_change(spy["Close"], 1)
    gld_ret = pct_change(gld["Close"], 1)
    hyg_ret = pct_change(hyg["Close"], 1)

    spy_rng = intraday_range_pct(spy)
    spy_rng_z = rolling_z(spy_rng, lookback)

    spy_vol = spy["Volume"].replace(0, pd.NA)
    spy_vol_z = rolling_z(spy_vol, lookback)

    vix_level = vix["Close"]
    vix_ret = pct_change(vix_level, 1)
    vix_level_z = rolling_z(vix_level, lookback)
    vix_ret_z = rolling_z(vix_ret, lookback)

    big_down = spy_ret <= -2.5
    whipsaw = (spy_rng >= 2.5) | (spy_rng_z >= 2.0)
    sig1 = big_down & whipsaw

    liq_proxy = (spy_rng_z >= 2.0) & (spy_vol_z >= 2.0)

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

    return results


def summarize_dislocation(signals: List[SignalResult], k_required: int = 3) -> Dict[str, object]:
    triggered = [signal for signal in signals if signal.triggered]
    count = len(triggered)

    return {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "dislocation": bool(count >= k_required),
        "signals_triggered_count": count,
        "signals_triggered": [signal.name for signal in triggered],
        "signals": [
            {"name": signal.name, "triggered": signal.triggered, "details": signal.details}
            for signal in signals
        ],
        "rule": {
            "k_required": k_required,
            "notes": "Designed to be low-churn: multiple independent stress signals must agree.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dislocation.json", help="Output JSON path")
    parser.add_argument("--k", type=int, default=3, help="Signals required to flag dislocation")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window for z-scores")
    parser.add_argument("--vix-symbol", default=DEFAULT_TICKERS["vix"], help="Stooq symbol for VIX (try vix or ^vix)")
    parser.add_argument("--spy-symbol", default=DEFAULT_TICKERS["equity_core"], help="Stooq symbol for SPY proxy")
    parser.add_argument("--gld-symbol", default=DEFAULT_TICKERS["gold"], help="Stooq symbol for GLD")
    parser.add_argument("--hyg-symbol", default=DEFAULT_TICKERS["credit_hy"], help="Stooq symbol for HYG")
    parser.add_argument("--fred-series", default="", help="Optional FRED series id (e.g., BAMLH0A0HYM2)")
    args = parser.parse_args()

    spy = fetch_stooq_daily(args.spy_symbol)
    gld = fetch_stooq_daily(args.gld_symbol)
    hyg = fetch_stooq_daily(args.hyg_symbol)

    try:
        vix = fetch_stooq_daily(args.vix_symbol)
    except Exception:
        if args.vix_symbol != "^vix":
            vix = fetch_stooq_daily("^vix")
        else:
            raise

    fred_df = None
    if args.fred_series:
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if api_key:
            fred_df = fetch_fred_series(args.fred_series, api_key)

    signals = compute_signals(
        spy=spy,
        gld=gld,
        hyg=hyg,
        vix=vix,
        fred_hy_spread=fred_df,
        lookback=args.lookback,
    )
    summary = summarize_dislocation(signals, k_required=args.k)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({
        "wrote": args.output,
        "dislocation": summary["dislocation"],
        "count": summary["signals_triggered_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
