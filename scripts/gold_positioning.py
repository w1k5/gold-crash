"""Weekly COMEX positioning context, matched to GLD prices and holdings dates."""
from __future__ import annotations

import math

import pandas as pd
import requests

CFTC_GOLD_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CFTC_GOLD_CODE = "088691"
COT_COLUMNS = {
    "open_interest_all": "open_interest",
    "m_money_positions_long_all": "managed_money_long",
    "m_money_positions_short_all": "managed_money_short",
    "m_money_positions_spread": "managed_money_spreading",
}


def parse_gold_cot(rows):
    if not isinstance(rows, list):
        raise ValueError("CFTC response must be a list of report records")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid CFTC report record")
        if row.get("cftc_contract_market_code") != CFTC_GOLD_CODE or row.get("futonly_or_combined") != "FutOnly":
            continue
        date = pd.to_datetime(row["report_date_as_yyyy_mm_dd"], utc=True)
        if date is None or pd.isna(date):
            raise ValueError("Missing CFTC report date")
        record = {"date": date}
        for source, target in COT_COLUMNS.items():
            record[target] = float(row[source])
            if not math.isfinite(record[target]) or record[target] < 0:
                raise ValueError("Invalid CFTC position count")
        if record["open_interest"] <= 0:
            raise ValueError("CFTC gold open interest must be positive")
        records.append(record)
    if not records:
        raise ValueError("CFTC response contains no COMEX gold futures-only records")
    frame = pd.DataFrame(records).set_index("date").sort_index()
    if frame.isna().any().any() or (frame[list(COT_COLUMNS.values())] < 0).any().any():
        raise ValueError("Invalid CFTC position counts")
    return frame.loc[~frame.index.duplicated(keep="last")]


def fetch_gold_cot():
    response = requests.get(CFTC_GOLD_URL, params={
        "cftc_contract_market_code": CFTC_GOLD_CODE,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 12,
    }, timeout=20)
    response.raise_for_status()
    return parse_gold_cot(response.json())


def fetch_gold_holdings():
    # Reuse the existing SPDR CSV/archive parser and fallback.
    if __package__:
        from .update import fetch_gld_holdings
    else:
        from update import fetch_gld_holdings
    rows = fetch_gld_holdings()
    return pd.Series([float(value) for _, value in rows],
                     index=pd.to_datetime([date for date, _ in rows], utc=True), dtype=float)


def _daily(series):
    if series is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    series = series.dropna().sort_index().copy()
    series.index = pd.to_datetime(series.index, utc=True).normalize()
    return series.loc[~series.index.duplicated(keep="last")]


def _quality(series, session, tolerance, cadence="daily"):
    lag = None
    if not series.empty and session is not None:
        lag = max(len(pd.bdate_range(series.index[-1], session)) - 1, 0)
    return {
        "source_date": series.index[-1].date().isoformat() if not series.empty else None,
        "stale_days": lag,
        "lag_tolerance_bdays": tolerance,
        "cadence": cadence,
        "available": not series.empty,
        "eligible": lag is not None and lag <= tolerance,
    }


def compute_gold_deleveraging(prices, cot=None, holdings=None):
    prices, holdings = _daily(prices), _daily(holdings)
    session = prices.index[-1] if not prices.empty else None
    if session is not None:
        holdings = holdings.loc[holdings.index <= session]
    cot = cot.copy().sort_index() if cot is not None else pd.DataFrame()
    if not cot.empty and session is not None:
        cot.index = pd.to_datetime(cot.index, utc=True).normalize()
        cot = cot.loc[cot.index <= session]
    quality = {
        "cftc_gold": _quality(cot, session, 7, "weekly"),
        "gld_holdings": _quality(holdings, session, 1),
    }
    details = {
        "triggered": False, "eligible": False, "available": False, "score_0_1": 0.0,
        "source_inputs": quality, "source_url": CFTC_GOLD_URL,
        "note": "Weekly positioning proxy; does not establish forced liquidation or investor motivation.",
        "holdings_scope": "GLD only; not aggregate gold ETF or physical demand.",
    }
    if len(cot) < 2 or prices.empty:
        details["note"] = "Need two CFTC gold reports and matching GLD prices; positioning unavailable."
        return details, quality
    previous, current = cot.iloc[-2], cot.iloc[-1]
    start, end = cot.index[-2:]
    gap = (end - start).days
    details.update({"report_date": end.date().isoformat(), "previous_report_date": start.date().isoformat(),
                    "report_interval_days": gap})
    # Compare the same reporting interval, never today's price to last week's positions.
    def value_on(series, day):
        prior = series.loc[series.index <= day]
        if prior.empty or (day - prior.index[-1]).days > 4:
            return None
        return float(prior.iloc[-1])
    p0, p1 = value_on(prices, start), value_on(prices, end)
    h0, h1 = value_on(holdings, start), value_on(holdings, end)
    oi_change = float(current.open_interest - previous.open_interest)
    net_change = float((current.managed_money_long - current.managed_money_short)
                       - (previous.managed_money_long - previous.managed_money_short))
    price_change = (p1 / p0 - 1) * 100 if p0 and p1 else None
    holdings_change = (h1 / h0 - 1) * 100 if h0 and h1 else None
    eligible = bool(quality["cftc_gold"]["eligible"] and 5 <= gap <= 9 and price_change is not None)
    triggered = eligible and oi_change < 0 and net_change < 0 and price_change < 0
    # Scale net-position changes by total OI (net length can cross zero).
    oi_pct = oi_change / previous.open_interest * 100 if previous.open_interest > 0 else None
    net_pct_oi = net_change / previous.open_interest * 100 if previous.open_interest > 0 else None
    score = min(max(0.0, min(1.0, -(value or 0.0) / 2.0))
                for value in (oi_pct, net_pct_oi, price_change)) if eligible else 0.0
    details.update({
        "available": price_change is not None, "eligible": eligible, "triggered": bool(triggered),
        "score_0_1": score, "open_interest_contracts": float(current.open_interest),
        "open_interest_change_contracts": oi_change, "open_interest_change_pct": oi_pct,
        "managed_money_net_contracts": float(current.managed_money_long - current.managed_money_short),
        "managed_money_net_change_contracts": net_change, "managed_money_net_change_pct_of_prior_oi": net_pct_oi,
        "managed_money_spreading_change_contracts": float(current.managed_money_spreading - previous.managed_money_spreading),
        "gld_report_interval_return_pct": price_change, "gld_holdings_report_interval_change_pct": holdings_change,
        "etf_buying_vs_futures_selling": bool(triggered and holdings_change is not None and holdings_change > 0),
        "thresholds": "Open interest change < 0 AND managed-money net change < 0 AND matched GLD return < 0",
    })
    return details, quality
