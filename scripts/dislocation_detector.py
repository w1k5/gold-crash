#!/usr/bin/env python3
"""Dislocation detector (daily, low-churn) using free data (Stooq) + optional FRED."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from statistics import median
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
    "jgb_etf": "2561.jp",
    "equity_equal_weight": "rsp.us",
}
DEFAULT_FRED_SERIES = {
    "hy_oas": "BAMLH0A0HYM2",
    "sofr": "SOFR",
    "iorb": "IORB",
    "tgcr_rate": "TGCRRATE",
    "tgcr_volume": "TGCRVOLUME",
    "dgs10": "DGS10",
    "usdjpy": "DEXJPUS",
    "jgb10": "IRLTLT01JPM156N",
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
        try:
            value = float(observation["value"])
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


def safe_last(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def safe_bool(series: pd.Series) -> bool:
    cleaned = series.dropna()
    return bool(cleaned.iloc[-1]) if not cleaned.empty else False


def clamp_0_1(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, value)))


def clamp_0_100(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(100.0, value)))


def score_to_severity(score: Optional[float]) -> float:
    return clamp_0_100((score or 0.0) * 100.0)


def score_ge(value: Optional[float], threshold: float, ramp: float) -> tuple[float, Optional[float]]:
    """Score how close value is to meeting a >= threshold rule."""
    if value is None or ramp <= 0:
        return 0.0, None
    score = (value - (threshold - ramp)) / ramp
    return clamp_0_1(score), float(value - threshold)


def score_le(value: Optional[float], threshold: float, ramp: float) -> tuple[float, Optional[float]]:
    """Score how close value is to meeting a <= threshold rule."""
    if value is None or ramp <= 0:
        return 0.0, None
    score = (threshold - value + ramp) / ramp
    return clamp_0_1(score), float(threshold - value)


def combine_or(*scores: float) -> float:
    return clamp_0_1(max(scores) if scores else 0.0)


def combine_and(*scores: float) -> float:
    return clamp_0_1(min(scores) if scores else 0.0)


def bday_lag_days(base_index: pd.Index, other_index: Optional[pd.Index]) -> Optional[int]:
    if base_index.empty or other_index is None or len(other_index) == 0:
        return None
    base_day = pd.Timestamp(base_index.max()).normalize().tz_localize(None)
    other_day = pd.Timestamp(other_index.max()).normalize().tz_localize(None)
    return int(max(pd.bdate_range(other_day, base_day).size - 1, 0))


@dataclass
class SignalResult:
    name: str
    triggered: bool
    details: Dict[str, object]


@dataclass
class RunMeta:
    equity_session_date: Optional[str]
    equities_data_date_utc: Optional[str]
    vix_data_date_utc: Optional[str]
    vix_stale_days: Optional[int]
    data_stale: bool
    stale_reasons: List[str]


def compute_signals(
    spy: pd.DataFrame,
    gld: pd.DataFrame,
    hyg: pd.DataFrame,
    rsp: Optional[pd.DataFrame] = None,
    vix: Optional[pd.DataFrame] = None,
    jgb_etf: Optional[pd.DataFrame] = None,
    fred_map: Optional[Dict[str, pd.DataFrame]] = None,
    lookback: int = 252,
) -> tuple[List[SignalResult], RunMeta]:
    idx = spy.index.intersection(gld.index).intersection(hyg.index)
    spy = spy.loc[idx]
    gld = gld.loc[idx]
    hyg = hyg.loc[idx]

    equities_data_date = idx.max() if len(idx) else None
    session_date = equities_data_date.date().isoformat() if equities_data_date is not None else None
    vix_data_date = vix.index.max() if vix is not None and not vix.empty else None
    vix_stale_days = bday_lag_days(idx, vix.index if vix is not None else None)

    spy_ret = pct_change(spy["Close"], 1)
    gld_ret = pct_change(gld["Close"], 1)
    hyg_ret = pct_change(hyg["Close"], 1)

    spy_rng = intraday_range_pct(spy)
    spy_rng_z = rolling_z(spy_rng, lookback)

    spy_dollar_vol = (spy["Close"] * spy["Volume"]).replace(0, pd.NA)
    spy_dollar_vol_z = rolling_z(spy_dollar_vol, lookback)

    vix_level = None
    vix_ret = None
    vix_level_z = None
    vix_ret_z = None
    if vix is not None:
        raw_vix_level = vix["Close"].where(vix["Close"] > 0)
        raw_vix_ret = pct_change(raw_vix_level, 1)
        valid_equity_move = spy_ret.abs().reindex(raw_vix_ret.index).fillna(0) >= 0.25
        cleaned_vix_ret = raw_vix_ret.where((raw_vix_ret.abs() <= 60) | valid_equity_move)

        vix_level = raw_vix_level.reindex(idx, method="ffill")
        vix_ret = cleaned_vix_ret.reindex(idx, method="ffill")
        vix_level_z = rolling_z(raw_vix_level, lookback).reindex(idx, method="ffill")
        vix_ret_z = rolling_z(cleaned_vix_ret, lookback).reindex(idx, method="ffill")

    last_spy_ret = safe_last(spy_ret)
    last_gld_ret = safe_last(gld_ret)
    last_hyg_ret = safe_last(hyg_ret)
    last_hyg_minus_spy = safe_last(hyg_ret - spy_ret)
    last_spy_rng = safe_last(spy_rng)
    last_spy_rng_z = safe_last(spy_rng_z)
    last_spy_dollar_vol_z = safe_last(spy_dollar_vol_z)
    last_vix_level = safe_last(vix_level)
    last_vix_ret = safe_last(vix_ret)
    last_vix_level_z = safe_last(vix_level_z)
    last_vix_ret_z = safe_last(vix_ret_z)

    big_down = spy_ret <= -2.5
    whipsaw = (spy_rng >= 2.5) | (spy_rng_z >= 2.0)
    sig1 = big_down & whipsaw

    score_big_down, _ = score_le(last_spy_ret, -2.5, 2.5)
    score_whipsaw = combine_or(
        score_ge(last_spy_rng, 2.5, 2.5)[0],
        score_ge(last_spy_rng_z, 2.0, 2.0)[0],
    )
    score_sig1 = combine_and(score_big_down, score_whipsaw)

    liq_proxy = (spy_rng_z >= 2.0) & (spy_dollar_vol_z >= 2.0)
    score_liq = combine_and(
        score_ge(last_spy_rng_z, 2.0, 2.0)[0],
        score_ge(last_spy_dollar_vol_z, 2.0, 2.0)[0],
    )

    if vix is None:
        vol_spike = pd.Series(False, index=idx)
        score_vol_spike = 0.0
    else:
        vol_spike = ((vix_level >= 30) | (vix_level_z >= 1.5)) & ((vix_ret >= 20) | (vix_ret_z >= 2.0))
        score_vol_level = combine_or(
            score_ge(last_vix_level, 30.0, 10.0)[0],
            score_ge(last_vix_level_z, 1.5, 1.5)[0],
        )
        score_vol_change = combine_or(
            score_ge(last_vix_ret, 20.0, 20.0)[0],
            score_ge(last_vix_ret_z, 2.0, 2.0)[0],
        )
        score_vol_spike = combine_and(score_vol_level, score_vol_change)

    credit_stress = (hyg_ret - spy_ret) <= -1.5
    score_credit, margin_credit = score_le(last_hyg_minus_spy, -1.5, 1.5)

    everything_sells = (spy_ret <= -2.0) & (gld_ret <= -1.0)
    score_everything = combine_and(
        score_le(last_spy_ret, -2.0, 2.0)[0],
        score_le(last_gld_ret, -1.0, 1.0)[0],
    )
    forced_flow_proxy = big_down & credit_stress & vol_spike
    score_forced_flow = combine_and(score_big_down, score_credit, score_vol_spike)

    results: List[SignalResult] = []
    results.append(SignalResult(
        name="big_down_and_whipsaw",
        triggered=safe_bool(sig1),
        details={
            "spy_1d_return_pct": last_spy_ret,
            "spy_intraday_range_pct": last_spy_rng,
            "spy_range_z": last_spy_rng_z,
            "score_0_1": score_sig1,
        },
    ))
    results.append(SignalResult(
        name="liquidity_degraded_proxy",
        triggered=safe_bool(liq_proxy),
        details={
            "spy_dollar_volume_z": last_spy_dollar_vol_z,
            "spy_range_z": last_spy_rng_z,
            "volume_interpretation": "High dollar volume contributes to stress only when paired with elevated range.",
            "score_0_1": score_liq,
        },
    ))

    if vix is None:
        results.append(SignalResult(
            name="volatility_spike",
            triggered=False,
            details={
                "note": "VIX data unavailable from Stooq.",
                "score_0_1": 0.0,
            },
        ))
    else:
        results.append(SignalResult(
            name="volatility_spike",
            triggered=safe_bool(vol_spike),
            details={
                "vix_level": last_vix_level,
                "vix_1d_change_pct": last_vix_ret,
                "vix_level_z": last_vix_level_z,
                "vix_change_z": last_vix_ret_z,
                "vix_eligible": bool(vix_stale_days is not None and vix_stale_days <= 1),
                "score_0_1": score_vol_spike,
            },
        ))

    results.append(SignalResult(
        name="credit_stress_proxy",
        triggered=safe_bool(credit_stress),
        details={
            "hyg_1d_return_pct": last_hyg_ret,
            "spy_1d_return_pct": last_spy_ret,
            "hyg_minus_spy_pct": last_hyg_minus_spy,
            "score_0_1": score_credit,
            "margin": margin_credit,
            "margin_unit": "%",
        },
    ))

    if fred_map:
        def col(df: Optional[pd.DataFrame], key: str) -> pd.Series:
            if df is None or df.empty or key not in df.columns:
                return pd.Series(dtype=float)
            return df[key]

        hy = fred_map.get("hy_oas")
        if hy is not None and not hy.empty:
            oas = hy[DEFAULT_FRED_SERIES["hy_oas"]].reindex(idx, method="ffill")
            oas_chg_5 = oas.diff(5)
            oas_chg_10 = oas.diff(10)
            oas_z = rolling_z(oas, 756)
            trig = (oas >= 6.5) | (oas_chg_5 >= 0.50) | (oas_chg_10 >= 0.40) | ((oas_z >= 2.0) & (oas_chg_10 >= 0.60))
            last_oas = safe_last(oas)
            last_oas_5 = safe_last(oas_chg_5)
            last_oas_10 = safe_last(oas_chg_10)
            last_oas_z = safe_last(oas_z)
            score_oas = combine_or(
                score_ge(last_oas, 6.5, 2.0)[0],
                score_ge(last_oas_5, 0.50, 0.50)[0],
                score_ge(last_oas_10, 0.40, 0.40)[0],
                combine_and(score_ge(last_oas_z, 2.0, 2.0)[0], score_ge(last_oas_10, 0.60, 0.60)[0]),
            )
            results.append(SignalResult(
                name="credit_spread_widening_fred",
                triggered=safe_bool(trig),
                details={
                    "hy_oas_level": last_oas,
                    "hy_oas_5d_change": last_oas_5,
                    "hy_oas_10d_change": last_oas_10,
                    "hy_oas_z": last_oas_z,
                    "score_0_1": score_oas,
                },
            ))

        sofr = fred_map.get("sofr")
        iorb = fred_map.get("iorb")
        if sofr is not None and iorb is not None and not sofr.empty and not iorb.empty:
            spread_bp = (sofr[DEFAULT_FRED_SERIES["sofr"]].reindex(idx, method="ffill") - iorb[DEFAULT_FRED_SERIES["iorb"]].reindex(idx, method="ffill")) * 100
            spread_z = rolling_z(spread_bp, 756)
            spread_5 = spread_bp.diff(5)
            persistent = spread_bp.rolling(3).apply(lambda x: int((x >= 15).sum()), raw=True) >= 2
            trig = (spread_bp >= 25) | ((spread_z >= 2.5) & (spread_5 >= 10)) | persistent
            last_spread = safe_last(spread_bp)
            last_spread_5 = safe_last(spread_5)
            last_spread_z = safe_last(spread_z)
            score_spread = combine_or(
                score_ge(last_spread, 25.0, 25.0)[0],
                combine_and(score_ge(last_spread_z, 2.5, 2.5)[0], score_ge(last_spread_5, 10.0, 10.0)[0]),
                score_ge(last_spread, 15.0, 10.0)[0],
            )
            results.append(SignalResult(
                name="funding_stress_sofr_iorb",
                triggered=safe_bool(trig),
                details={
                    "sofr_iorb_spread_bp": last_spread,
                    "sofr_iorb_spread_5d_bp": last_spread_5,
                    "sofr_iorb_spread_z": last_spread_z,
                    "score_0_1": score_spread,
                },
            ))

        tgcr_rate = fred_map.get("tgcr_rate")
        tgcr_volume = fred_map.get("tgcr_volume")
        if tgcr_rate is not None and tgcr_volume is not None and not tgcr_rate.empty and not tgcr_volume.empty:
            rate = tgcr_rate[DEFAULT_FRED_SERIES["tgcr_rate"]].reindex(idx, method="ffill")
            vol = tgcr_volume[DEFAULT_FRED_SERIES["tgcr_volume"]].reindex(idx, method="ffill")
            rate_z = rolling_z(rate, 756)
            vol_z = rolling_z(vol, 756)
            rate_5 = rate.diff(5)
            vol_5_pct = vol.pct_change(5) * 100
            trig = ((rate_z >= 2.5) | (rate_5 >= 0.15)) & ((vol_z <= -2.0) | (vol_5_pct <= -20))
            last_rate = safe_last(rate)
            last_rate_5 = safe_last(rate_5)
            last_rate_z = safe_last(rate_z)
            last_vol_z = safe_last(vol_z)
            last_vol_5_pct = safe_last(vol_5_pct)
            score_repo = combine_and(
                combine_or(score_ge(last_rate_z, 2.5, 2.5)[0], score_ge(last_rate_5, 0.15, 0.15)[0]),
                combine_or(score_le(last_vol_z, -2.0, 2.0)[0], score_le(last_vol_5_pct, -20.0, 20.0)[0]),
            )
            results.append(SignalResult(
                name="repo_rate_volume_dislocation",
                triggered=safe_bool(trig),
                details={
                    "tgcr_rate": last_rate,
                    "tgcr_rate_5d_change_pct": last_rate_5,
                    "tgcr_rate_z": last_rate_z,
                    "tgcr_volume_z": last_vol_z,
                    "tgcr_volume_5d_change_pct": last_vol_5_pct,
                    "score_0_1": score_repo,
                },
            ))

        dgs10 = fred_map.get("dgs10")
        if dgs10 is not None and not dgs10.empty:
            yld = dgs10[DEFAULT_FRED_SERIES["dgs10"]].reindex(idx, method="ffill")
            dy_bp = yld.diff() * 100
            rv5 = dy_bp.rolling(5).std(ddof=0)
            rv5_z = rolling_z(rv5, 756)
            rv5_5 = rv5.diff(5)
            yld_5 = yld.diff(5) * 100
            trig = (rv5 >= 12) | ((rv5_z >= 2.5) & (rv5_5 >= 4)) | ((yld_5.abs() >= 25) & (rv5 >= 8))
            last_rv5 = safe_last(rv5)
            last_rv5_z = safe_last(rv5_z)
            last_rv5_5 = safe_last(rv5_5)
            last_yld_5 = safe_last(yld_5)
            score_rates = combine_or(
                score_ge(last_rv5, 12.0, 12.0)[0],
                combine_and(score_ge(last_rv5_z, 2.5, 2.5)[0], score_ge(last_rv5_5, 4.0, 4.0)[0]),
                combine_and(score_ge(abs(last_yld_5) if last_yld_5 is not None else None, 25.0, 25.0)[0], score_ge(last_rv5, 8.0, 8.0)[0]),
            )
            results.append(SignalResult(
                name="rates_volatility_shock",
                triggered=safe_bool(trig),
                details={
                    "dgs10_rv5_bp": last_rv5,
                    "dgs10_rv5_z": last_rv5_z,
                    "dgs10_rv5_5d_change_bp": last_rv5_5,
                    "dgs10_5d_change_bp": last_yld_5,
                    "score_0_1": score_rates,
                },
            ))

        usdjpy_df = fred_map.get("usdjpy")
        jgb10_df = fred_map.get("jgb10")

        if usdjpy_df is not None and not usdjpy_df.empty:
            fx = col(usdjpy_df, DEFAULT_FRED_SERIES["usdjpy"]).reindex(idx, method="ffill")
            fx_5d = fx.pct_change(5) * 100
            fx_20d = fx.pct_change(20) * 100
            fx_z = rolling_z(fx_5d, 756)
            trig_fx = (fx_5d <= -2.0) | (fx_20d <= -4.0) | ((fx_z <= -2.5) & (fx_5d <= -1.5))

            last_fx = safe_last(fx)
            last_fx_5d = safe_last(fx_5d)
            last_fx_20d = safe_last(fx_20d)
            last_fx_z = safe_last(fx_z)
            score_fx, margin_fx = score_le(last_fx_5d, -2.0, 2.0)

            results.append(SignalResult(
                name="yen_strengthening_fast",
                triggered=safe_bool(trig_fx),
                details={
                    "usdjpy": last_fx,
                    "usdjpy_5d_change_pct": last_fx_5d,
                    "usdjpy_20d_change_pct": last_fx_20d,
                    "usdjpy_5d_change_z": last_fx_z,
                    "score_0_1": score_fx,
                    "margin": margin_fx,
                    "margin_unit": "% (5D)",
                },
            ))
        else:
            results.append(SignalResult(
                name="yen_strengthening_fast",
                triggered=False,
                details={
                    "note": "USD/JPY series unavailable (FRED usdjpy not loaded).",
                    "score_0_1": 0.0,
                },
            ))

        if jgb_etf is not None and not jgb_etf.empty:
            jgb_px = jgb_etf["Close"]
            jgb_5d = jgb_px.pct_change(5) * 100
            jgb_20d = jgb_px.pct_change(20) * 100
            jgb_5d_z = rolling_z(jgb_5d, 756)
            th_jgb_5d = -1.0
            th_jgb_20d = -2.5
            th_jgb_z = -2.5
            th_jgb_z_5d = -0.75

            trig_jgb_5d = jgb_5d <= th_jgb_5d
            trig_jgb_20d = jgb_20d <= th_jgb_20d
            trig_jgb_z = (jgb_5d_z <= th_jgb_z) & (jgb_5d <= th_jgb_z_5d)
            trig_jgb = trig_jgb_5d | trig_jgb_20d | trig_jgb_z

            last_jgb_px = safe_last(jgb_px)
            last_jgb_5d = safe_last(jgb_5d)
            last_jgb_20d = safe_last(jgb_20d)
            last_jgb_5d_z = safe_last(jgb_5d_z)
            score_jgb_5d, margin_jgb_5d = score_le(last_jgb_5d, th_jgb_5d, abs(th_jgb_5d))
            score_jgb_20d, margin_jgb_20d = score_le(last_jgb_20d, th_jgb_20d, abs(th_jgb_20d))
            score_jgb_z, _ = score_le(last_jgb_5d_z, th_jgb_z, abs(th_jgb_z))
            score_jgb = combine_or(score_jgb_5d, score_jgb_20d, score_jgb_z)

            winner = "5d"
            if score_jgb_20d > score_jgb_5d and score_jgb_20d >= score_jgb_z:
                winner = "20d"
            elif score_jgb_z > score_jgb_5d and score_jgb_z > score_jgb_20d:
                winner = "z"

            if winner == "20d":
                margin_jgb = margin_jgb_20d
                margin_unit = "% (20D)"
            elif winner == "z":
                margin_jgb = None
                margin_unit = "z (5D return)"
            else:
                margin_jgb = margin_jgb_5d
                margin_unit = "% (5D)"

            results.append(SignalResult(
                name="jgb_price_drop_fast",
                triggered=safe_bool(trig_jgb),
                details={
                    "jgb_etf_close": last_jgb_px,
                    "jgb_etf_5d_return_pct": last_jgb_5d,
                    "jgb_etf_20d_return_pct": last_jgb_20d,
                    "jgb_etf_5d_return_z": last_jgb_5d_z,
                    "trigger_reason": winner,
                    "score_0_1": score_jgb,
                    "margin": margin_jgb,
                    "margin_unit": margin_unit,
                },
            ))
        else:
            results.append(SignalResult(
                name="jgb_price_drop_fast",
                triggered=False,
                details={
                    "note": "Japan bond ETF price series unavailable (Stooq 2561.jp not loaded).",
                    "score_0_1": 0.0,
                },
            ))

        if dgs10 is not None and not dgs10.empty and jgb10_df is not None and not jgb10_df.empty:
            us10 = dgs10[DEFAULT_FRED_SERIES["dgs10"]].reindex(idx, method="ffill")
            jgb = col(jgb10_df, DEFAULT_FRED_SERIES["jgb10"]).reindex(idx, method="ffill")
            spread = us10 - jgb
            spread_20d_bp = spread.diff(20) * 100
            spread_z = rolling_z(spread_20d_bp, 756)
            trig_spread = (spread_20d_bp <= -35) | ((spread_z <= -2.5) & (spread_20d_bp <= -25))

            last_spread = safe_last(spread)
            last_spread_20 = safe_last(spread_20d_bp)
            last_spread_z = safe_last(spread_z)
            score_spread, margin_spread = score_le(last_spread_20, -35.0, 35.0)

            results.append(SignalResult(
                name="us_jp_spread_compression",
                triggered=safe_bool(trig_spread),
                details={
                    "us_jp_10y_spread_pct": last_spread,
                    "us_jp_10y_spread_20d_change_bp": last_spread_20,
                    "us_jp_10y_spread_20d_change_z": last_spread_z,
                    "score_0_1": score_spread,
                    "margin": margin_spread,
                    "margin_unit": "bp (20D)",
                },
            ))
        else:
            results.append(SignalResult(
                name="us_jp_spread_compression",
                triggered=False,
                details={
                    "note": "US10/JGB10 spread unavailable (missing DGS10 or JGB10).",
                    "score_0_1": 0.0,
                },
            ))

        def trig_by_name(name: str) -> bool:
            for signal in results:
                if signal.name == name:
                    return bool(signal.triggered)
            return False

        score_combo = combine_and(
            next((s.details.get("score_0_1", 0.0) for s in results if s.name == "yen_strengthening_fast"), 0.0),
            combine_or(
                next((s.details.get("score_0_1", 0.0) for s in results if s.name == "jgb_price_drop_fast"), 0.0),
                next((s.details.get("score_0_1", 0.0) for s in results if s.name == "us_jp_spread_compression"), 0.0),
            ),
        )
        combo = trig_by_name("yen_strengthening_fast") and (
            trig_by_name("jgb_price_drop_fast") or trig_by_name("us_jp_spread_compression")
        )

        results.append(SignalResult(
            name="carry_trade_unwind_combo",
            triggered=bool(combo),
            details={
                "yen_strengthening_fast": trig_by_name("yen_strengthening_fast"),
                "jgb_price_drop_fast": trig_by_name("jgb_price_drop_fast"),
                "us_jp_spread_compression": trig_by_name("us_jp_spread_compression"),
                "score_0_1": score_combo,
            },
        ))

    results.append(SignalResult(
        name="everything_sells_together",
        triggered=safe_bool(everything_sells),
        details={
            "spy_1d_return_pct": last_spy_ret,
            "gld_1d_return_pct": last_gld_ret,
            "score_0_1": score_everything,
        },
    ))

    if vix is None:
        results.append(SignalResult(
            name="forced_flow_proxy_combo",
            triggered=False,
            details={
                "spy_1d_return_pct": last_spy_ret,
                "hyg_minus_spy_pct": last_hyg_minus_spy,
                "note": "VIX data unavailable from Stooq.",
                "score_0_1": 0.0,
            },
        ))
    else:
        results.append(SignalResult(
            name="forced_flow_proxy_combo",
            triggered=safe_bool(forced_flow_proxy),
            details={
                "spy_1d_return_pct": last_spy_ret,
                "hyg_minus_spy_pct": last_hyg_minus_spy,
                "vix_level": last_vix_level,
                "vix_1d_change_pct": last_vix_ret,
                "score_0_1": score_forced_flow,
            },
        ))

    if rsp is not None and not rsp.empty:
        rsp_close = rsp["Close"].reindex(idx, method="ffill")
        spy_close = spy["Close"]
        ratio = (rsp_close / spy_close).replace(0, pd.NA)
        ratio_ret_20d = ratio.pct_change(20) * 100
        ratio_ret_60d = ratio.pct_change(60) * 100
        ratio_ret_20d_z = rolling_z(ratio_ret_20d, 756)
        trig_breadth = (ratio_ret_20d <= -2.0) | (ratio_ret_60d <= -4.0) | ((ratio_ret_20d_z <= -2.0) & (ratio_ret_20d <= -1.5))

        last_ratio = safe_last(ratio)
        last_ratio_20d = safe_last(ratio_ret_20d)
        last_ratio_60d = safe_last(ratio_ret_60d)
        last_ratio_20d_z = safe_last(ratio_ret_20d_z)
        score_breadth = combine_or(
            score_le(last_ratio_20d, -2.0, 2.0)[0],
            score_le(last_ratio_60d, -4.0, 4.0)[0],
            combine_and(score_le(last_ratio_20d_z, -2.0, 2.0)[0], score_le(last_ratio_20d, -1.5, 1.5)[0]),
        )

        results.append(SignalResult(
            name="breadth_deterioration_rsp_spy",
            triggered=safe_bool(trig_breadth),
            details={
                "rsp_spy_ratio": last_ratio,
                "rsp_spy_20d_return_pct": last_ratio_20d,
                "rsp_spy_60d_return_pct": last_ratio_60d,
                "rsp_spy_20d_return_z": last_ratio_20d_z,
                "score_0_1": score_breadth,
            },
        ))
    else:
        results.append(SignalResult(
            name="breadth_deterioration_rsp_spy",
            triggered=False,
            details={
                "note": "RSP series unavailable from Stooq.",
                "score_0_1": 0.0,
            },
        ))
    stale_reasons: List[str] = []
    if vix_stale_days is not None and vix_stale_days > 1:
        stale_reasons.append("vix_stale")

    meta = RunMeta(
        equity_session_date=session_date,
        equities_data_date_utc=equities_data_date.isoformat() if equities_data_date is not None else None,
        vix_data_date_utc=vix_data_date.isoformat() if vix_data_date is not None else None,
        vix_stale_days=vix_stale_days,
        data_stale=bool(stale_reasons),
        stale_reasons=stale_reasons,
    )
    return results, meta


def derive_status(count: int, dislocation: bool, data_stale: bool = False) -> str:
    if data_stale:
        return "data_stale"
    if dislocation:
        return "dislocation"
    if count >= 1:
        return "stress_building"
    return "normal"




def compute_dashboard(signals: List[SignalResult], meta: RunMeta) -> Dict[str, object]:
    pillar_map = {
        "liquidity_funding": [
            "liquidity_degraded_proxy",
            "funding_stress_sofr_iorb",
            "repo_rate_volume_dislocation",
        ],
        "credit_stress": [
            "credit_stress_proxy",
            "credit_spread_widening_fred",
        ],
        "volatility_convexity": [
            "volatility_spike",
            "rates_volatility_shock",
        ],
        "correlation_liquidation": [
            "big_down_and_whipsaw",
            "everything_sells_together",
            "forced_flow_proxy_combo",
            "carry_trade_unwind_combo",
        ],
        "structure_fragility": [
            "breadth_deterioration_rsp_spy",
        ],
    }

    severities: Dict[str, float] = {}
    for signal in signals:
        severities[signal.name] = score_to_severity(signal.details.get("score_0_1"))

    pillar_scores: Dict[str, float] = {}
    for pillar, names in pillar_map.items():
        values = [severities.get(name, 0.0) for name in names]
        pillar_scores[pillar] = clamp_0_100(median(values) if values else 0.0)

    crash_base = (
        0.30 * pillar_scores["liquidity_funding"]
        + 0.25 * pillar_scores["credit_stress"]
        + 0.25 * pillar_scores["volatility_convexity"]
        + 0.20 * pillar_scores["correlation_liquidation"]
    )

    core_pillars = [
        pillar_scores["liquidity_funding"],
        pillar_scores["credit_stress"],
        pillar_scores["volatility_convexity"],
        pillar_scores["correlation_liquidation"],
    ]

    bonus = 0.0
    if sum(1 for score in core_pillars if score > 70) >= 3:
        bonus += 10.0
    if pillar_scores["liquidity_funding"] > 80 and pillar_scores["credit_stress"] > 70:
        bonus += 10.0

    crash_risk = clamp_0_100(crash_base + bonus)

    fragility = clamp_0_100(
        0.45 * pillar_scores["structure_fragility"]
        + 0.20 * pillar_scores["credit_stress"]
        + 0.20 * pillar_scores["volatility_convexity"]
        + 0.15 * pillar_scores["liquidity_funding"]
    )

    liq = pillar_scores["liquidity_funding"]
    if liq >= 70:
        liquidity_regime = "Stressed"
    elif liq >= 40:
        liquidity_regime = "Tightening"
    else:
        liquidity_regime = "Normal"

    freshness_penalty = 0
    if meta.vix_stale_days is not None:
        freshness_penalty += min(meta.vix_stale_days * 20, 60)
    freshness_penalty += max(0, len(meta.stale_reasons) - 1) * 20
    confidence = clamp_0_100(100 - freshness_penalty)

    top_drivers = sorted(
        [{"name": signal.name, "severity_0_100": severities.get(signal.name, 0.0)} for signal in signals],
        key=lambda x: x["severity_0_100"],
        reverse=True,
    )[:3]

    return {
        "crash_risk_score_1_5d": crash_risk,
        "fragility_score_1_3m": fragility,
        "liquidity_regime_label": liquidity_regime,
        "confidence_score": confidence,
        "pillar_scores": pillar_scores,
        "top_drivers": top_drivers,
    }


def summarize_dislocation(
    signals: List[SignalResult],
    meta: RunMeta,
    k_required: int = 3,
    previous_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    stale_vix = bool(meta.vix_stale_days is not None and meta.vix_stale_days > 0)
    excluded_from_threshold = ["forced_flow_proxy_combo", "carry_trade_unwind_combo"]
    if stale_vix:
        excluded_from_threshold.append("volatility_spike")

    countable_signals = [s for s in signals if s.name not in set(excluded_from_threshold)]
    triggered = [signal for signal in countable_signals if signal.triggered]
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
    status = derive_status(count, dislocation, data_stale=meta.data_stale)
    transitions = []
    if previous_summary and isinstance(previous_summary.get("transitions"), list):
        transitions = list(previous_summary["transitions"])
    if previous_status and previous_status != status:
        transitions.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "from": previous_status,
            "to": status,
        })

    excluded_set = set(excluded_from_threshold)

    return {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "dislocation": dislocation,
        "watch": watch,
        "status": status,
        "signals_triggered_count": count,
        "signals_triggered_count_yesterday": yesterday_count,
        "signals_triggered": [signal.name for signal in triggered],
        "signal_counting": {
            "excluded_from_threshold": excluded_from_threshold,
            "notes": "Stale VIX data (>0 business day lag) is excluded from K-of-N counting.",
        },
        "signals": [
            {
                "name": signal.name,
                "triggered": signal.triggered,
                "details": {
                    **signal.details,
                    "counted_toward_threshold": signal.name not in excluded_set,
                    "severity_0_100": score_to_severity(signal.details.get("score_0_1")),
                },
            }
            for signal in signals
        ],
        "dashboard": compute_dashboard(signals, meta),
        "data_dates": {
            "equity_session_date": meta.equity_session_date,
            "equities_close_utc": meta.equities_data_date_utc,
            "vix_close_utc": meta.vix_data_date_utc,
            "vix_stale_days": meta.vix_stale_days,
            "data_stale": meta.data_stale,
            "stale_reasons": meta.stale_reasons,
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
    parser.add_argument("--fred-series", default="", help="Optional legacy FRED series id (e.g., BAMLH0A0HYM2)")
    args = parser.parse_args()

    spy = fetch_stooq_daily(args.spy_symbol)
    gld = fetch_stooq_daily(args.gld_symbol)
    hyg = fetch_stooq_daily(args.hyg_symbol)
    jgb_etf = fetch_stooq_daily(DEFAULT_TICKERS["jgb_etf"], require_volume=True)

    rsp = None
    try:
        rsp = fetch_stooq_daily(DEFAULT_TICKERS["equity_equal_weight"], require_volume=True)
    except ValueError:
        pass

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

    fred_map: Dict[str, pd.DataFrame] = {}
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if api_key:
        for key, series_id in DEFAULT_FRED_SERIES.items():
            try:
                fred_map[key] = fetch_fred_series(series_id, api_key)
            except requests.RequestException:
                continue
        if args.fred_series and args.fred_series not in DEFAULT_FRED_SERIES.values():
            try:
                fred_map["legacy"] = fetch_fred_series(args.fred_series, api_key)
            except requests.RequestException:
                pass

    previous_summary = load_previous_summary(args.output)
    signals, meta = compute_signals(
        spy=spy,
        gld=gld,
        hyg=hyg,
        rsp=rsp,
        vix=vix,
        jgb_etf=jgb_etf,
        fred_map=fred_map,
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
