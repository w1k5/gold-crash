#!/usr/bin/env python3
"""Generate silver.json using Stooq + FRED with stdlib only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from statistics import pstdev
from typing import Dict, List, Tuple

STOOQ_URL = "https://stooq.com/q/d/l/?s=slv.us&i=d"
FRED_SERIES = {
    "dfii10": "DFII10",
    "t10yie": "T10YIE",
    "hy_spread": "BAMLH0A0HYM2",
}


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "silver-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_stooq_slv() -> List[Tuple[datetime, float]]:
    text = fetch_url(STOOQ_URL)
    rows = list(csv.DictReader(text.splitlines()))
    out: List[Tuple[datetime, float]] = []
    for r in rows:
        try:
            d = datetime.fromisoformat(r["Date"]).replace(tzinfo=timezone.utc)
            c = float(r["Close"])
        except Exception:
            continue
        out.append((d, c))
    out.sort(key=lambda x: x[0])
    if len(out) < 260:
        raise RuntimeError("Insufficient SLV history")
    return out


def fetch_fred(series_id: str, api_key: str) -> List[Tuple[datetime, float]]:
    start = (datetime.now(timezone.utc) - timedelta(days=365 * 8)).date().isoformat()
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": start,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    payload = json.loads(fetch_url(url))
    out: List[Tuple[datetime, float]] = []
    for obs in payload.get("observations", []):
        v = obs.get("value")
        d = obs.get("date")
        if not v or v == "." or not d:
            continue
        out.append((datetime.fromisoformat(d).replace(tzinfo=timezone.utc), float(v)))
    out.sort(key=lambda x: x[0])
    return out


def rolling_mean(values: List[float], w: int) -> List[float | None]:
    out: List[float | None] = [None] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= w:
            s -= values[i - w]
        if i >= w - 1:
            out[i] = s / w
    return out


def pct_rank(history: List[float], value: float) -> int:
    if not history:
        return 50
    le = sum(1 for x in history if x <= value)
    return max(0, min(100, round((le / len(history)) * 100)))


def pct_rank_window(series: List[float], i: int, w: int) -> int:
    lo = max(0, i - w + 1)
    return pct_rank(series[lo:i + 1], series[i])


def spark(vals: List[float]) -> str:
    if not vals:
        return "—"
    blocks = "▁▂▃▄▅▆▇"
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return blocks[0] * len(vals)
    return "".join(blocks[min(6, int((v - lo) / (hi - lo) * 6))] for v in vals)


def fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"




def compute_stability_label(streak: int, score: int) -> str:
    if streak >= 3:
        return "stabilized"
    if streak == 2:
        return "stabilizing"
    if streak == 1 and score >= 60:
        return "early_stabilizing"
    return "unstable"


def compute_stability_silver(
    *,
    shock_hist: List[int],
    risk_hist: List[int],
    dd_hist: List[float],
    absret_hist: List[float],
    prev_streak: int,
) -> Dict:
    shock_falling = len(shock_hist) >= 6 and shock_hist[-1] < shock_hist[-6]
    dd_not_worsening = len(dd_hist) >= 10 and min(dd_hist[-5:]) >= min(dd_hist[-10:-5])
    dd_improving = len(dd_hist) >= 6 and dd_hist[-1] > dd_hist[-6]
    risk_cooling = len(risk_hist) >= 6 and risk_hist[-1] < risk_hist[-6]

    range_contracting = False
    if len(absret_hist) >= 20:
        range_contracting = pstdev(absret_hist[-10:]) < pstdev(absret_hist[-20:-10])

    score = 0
    reasons: List[str] = []
    if shock_falling:
        score += 35
        reasons.append("shock_falling")
    if dd_not_worsening or dd_improving:
        score += 30
        reasons.append("drawdown_stabilizing")
    if range_contracting:
        score += 20
        reasons.append("range_contracting")
    if risk_cooling:
        score += 15
        reasons.append("credit_stress_cooling")

    is_stabilizing_today = score >= 60
    streak = prev_streak + 1 if is_stabilizing_today else 0
    return {
        "score_0_100": score,
        "is_stabilizing": streak >= 2,
        "label": compute_stability_label(streak, score),
        "streak": streak,
        "reasons": reasons,
    }

def build_payload(
    slv_rows: List[Tuple[datetime, float]],
    fred_rows: Dict[str, List[Tuple[datetime, float]]],
    previous_stability_streak: int = 0,
) -> Dict:
    dates = [d for d, _ in slv_rows]
    closes = [c for _, c in slv_rows]
    ma200 = rolling_mean(closes, 200)

    dist200: List[float] = []
    ret20: List[float] = []
    dd63: List[float] = []
    ret1: List[float] = []
    rv20: List[float] = []

    for i, c in enumerate(closes):
        d200 = ((c / ma200[i]) - 1) * 100 if ma200[i] else 0.0
        dist200.append(d200)
        r20 = ((c / closes[i - 20]) - 1) * 100 if i >= 20 else 0.0
        ret20.append(r20)
        peak63 = max(closes[max(0, i - 62): i + 1])
        dd63.append(((c / peak63) - 1) * 100)
        r1 = ((c / closes[i - 1]) - 1) * 100 if i >= 1 else 0.0
        ret1.append(r1)
        window = ret1[max(0, i - 19): i + 1]
        rv = pstdev(window) * (252 ** 0.5) if len(window) >= 5 else 0.0
        rv20.append(rv)

    lookback = min(1260, len(closes))
    idx0 = len(closes) - lookback
    stability_lookback = min(756, len(closes))

    latest = len(closes) - 1
    dist_p = pct_rank(dist200[idx0:], dist200[latest])
    ret20_p = pct_rank(ret20[idx0:], ret20[latest])
    dd_p = pct_rank([-x for x in dd63[idx0:]], -dd63[latest])
    trend_score = round((dist_p + ret20_p + dd_p) / 3)

    rv_p = pct_rank(rv20[idx0:], rv20[latest])
    down_sev = [max(0.0, -r) for r in ret1]
    down_hist = [x for x in down_sev[idx0:] if x > 0]
    if down_sev[latest] > 0 and down_hist:
        drop_p = pct_rank(down_hist, down_sev[latest])
    else:
        drop_p = 50
    whipsaw = (ret1[latest] <= -2.5 and ret1[latest - 1] >= 2.0) if latest >= 1 else False
    whipsaw_bonus = 15 if whipsaw else 0
    shock_score = round(min(100, (rv_p + drop_p) / 2 + whipsaw_bonus))

    def fred_level_change(name: str) -> Tuple[float | None, float | None, int]:
        rows = fred_rows.get(name, [])
        if len(rows) < 2:
            return None, None, 50
        vals = [v for _, v in rows]
        lvl = vals[-1]
        ch = vals[-1] - vals[-22] if len(vals) > 22 else vals[-1] - vals[0]
        changes = [vals[i] - vals[max(0, i - 22)] for i in range(1, len(vals))]
        p = pct_rank(changes, ch)
        return lvl, ch, p

    dfii_lvl, dfii_ch, dfii_p = fred_level_change("dfii10")
    t10_lvl, t10_ch, t10_p = fred_level_change("t10yie")
    hy_lvl, hy_ch, hy_p = fred_level_change("hy_spread")

    dfii_headwind_p = 100 - dfii_p
    macro_score = round((dfii_headwind_p + t10_p) / 2)
    risk_score = hy_p
    fragility = round((trend_score + shock_score + macro_score + risk_score) / 4)

    deterioration = shock_score >= 65 or risk_score >= 65 or macro_score >= 70
    severe = shock_score >= 80 or risk_score >= 80 or dd63[latest] <= -12

    regime_triggers: List[str] = []
    if trend_score >= 70:
        regime_triggers.append("high_extension")
    if dd63[latest] <= -12:
        regime_triggers.append("deep_drawdown")
    if shock_score >= 80:
        regime_triggers.append("high_shock")
    if risk_score >= 80:
        regime_triggers.append("credit_stress")
    if macro_score >= 70:
        regime_triggers.append("macro_headwind")

    if severe:
        regime, action, conf = "RED", "Defensive", "High"
        reason_labels = {
            "deep_drawdown": "deep drawdown",
            "high_shock": "high shock",
            "credit_stress": "credit stress",
            "macro_headwind": "macro headwinds",
            "high_extension": "high extension",
        }
        reasons = [reason_labels[r] for r in regime_triggers if r in {"deep_drawdown", "high_shock", "credit_stress", "macro_headwind"}] or ["stress conditions"]
        desc = f"Breakdown / stress regime driven by {', '.join(reasons)}."
    elif trend_score >= 70 and deterioration:
        regime, action, conf = "ORANGE", "Trim / tighten risk", "Medium"
        desc = "Extended + early deterioration: one or more stress pillars are active."
    elif trend_score >= 70:
        regime, action, conf = "BLUE", "Hold / add on pullbacks", "Medium"
        desc = "Extended, but intact: upside trend remains in place without confirmed breakdown."
    else:
        regime, action, conf = "GREEN", "Accumulate / hold core", "Medium"
        desc = "Normal / constructive: trend is healthy and stress signals are mostly quiet."

    risk_appetite = "Risk-off" if risk_score >= 70 else ("Mixed" if risk_score >= 45 else "Risk-on")

    sigs = [
        ("Distance above 200DMA", dist200, dist200[latest], dist_p),
        ("20D momentum", ret20, ret20[latest], ret20_p),
        ("20D realized volatility", rv20, rv20[latest], rv_p),
        ("1D down move severity", down_sev, down_sev[latest], drop_p),
    ]
    if dfii_ch is not None:
        sigs.append(("DFII10 1M change", [r[1] for r in fred_rows.get("dfii10", [])], dfii_ch, dfii_p))
    if t10_ch is not None:
        sigs.append(("T10YIE 1M change", [r[1] for r in fred_rows.get("t10yie", [])], t10_ch, t10_p))
    if hy_lvl is not None:
        sigs.append(("HY spread level", [r[1] for r in fred_rows.get("hy_spread", [])], hy_lvl, hy_p))

    cards = []
    for n, series, last, p in sigs:
        if n == "1D down move severity" and last <= 0:
            state = "Quiet"
        else:
            state = "Triggered" if p >= 80 else "Watch" if p >= 65 else "Quiet"

        if n == "1D down move severity":
            last_str = f"{last:.2f}%"
        else:
            last_str = f"{last:+.2f}" + ("%" if "volatility" in n.lower() or "momentum" in n.lower() or "Distance" in n or "severity" in n.lower() else "")

        cards.append({
            "name": n,
            "state": state,
            "last": last_str,
            "percentile": ordinal(p),
            "sparkline": spark(series[-7:]),
        })

    shock_hist: List[int] = []
    for i in range(idx0, latest + 1):
        rv_pi = pct_rank_window(rv20, i, stability_lookback)
        lo = max(0, i - stability_lookback + 1)
        down_win = [x for x in down_sev[lo:i + 1] if x > 0] or [0.0]
        drop_pi = pct_rank(down_win, down_sev[i])
        whipsaw_i = (ret1[i] <= -2.5 and ret1[i - 1] >= 2.0) if i >= 1 else False
        whipsaw_bonus_i = 15 if whipsaw_i else 0
        shock_hist.append(round(min(100, (rv_pi + drop_pi) / 2 + whipsaw_bonus_i)))

    hy_vals = [v for _, v in fred_rows.get("hy_spread", [])]
    risk_hist: List[int] = []
    for i in range(len(hy_vals)):
        if i < 22:
            risk_hist.append(50)
            continue
        hy_ch = hy_vals[i] - hy_vals[i - 22]
        lo = max(22, i - stability_lookback + 1)
        changes = [hy_vals[j] - hy_vals[j - 22] for j in range(lo, i + 1)]
        risk_hist.append(pct_rank(changes, hy_ch))

    if len(risk_hist) >= len(shock_hist):
        risk_hist_tail = risk_hist[-len(shock_hist):]
    else:
        risk_hist_tail = [risk_score] * len(shock_hist)

    stability = compute_stability_silver(
        shock_hist=shock_hist,
        risk_hist=risk_hist_tail,
        dd_hist=dd63[idx0:],
        absret_hist=[abs(x) for x in ret1[idx0:]],
        prev_streak=previous_stability_streak,
    )

    stability_candidates = {
        "shock_falling": len(shock_hist) >= 6 and shock_hist[-1] < shock_hist[-6],
        "drawdown_stabilizing": (len(dd63[idx0:]) >= 10 and min(dd63[idx0:][-5:]) >= min(dd63[idx0:][-10:-5])) or (len(dd63[idx0:]) >= 6 and dd63[idx0:][-1] > dd63[idx0:][-6]),
        "range_contracting": len(ret1[idx0:]) >= 20 and pstdev([abs(x) for x in ret1[idx0:]][-10:]) < pstdev([abs(x) for x in ret1[idx0:]][-20:-10]),
        "credit_stress_cooling": len(risk_hist_tail) >= 6 and risk_hist_tail[-1] < risk_hist_tail[-6],
    }
    stability["missing"] = [k for k, ok in stability_candidates.items() if not ok]

    last_slv = dates[-1].date().isoformat()
    fred_last_dates = [rows[-1][0].date().isoformat() for rows in fred_rows.values() if rows]
    fred_last = max(fred_last_dates) if fred_last_dates else "n/a"

    return {
        "title": "Silver Risk Monitor (SLV)",
        "subtitle": "Outcome first: regime, drivers, and what changes next.",
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "data_dates": f"SLV through {last_slv} · FRED through {fred_last}",
        "regime": regime,
        "stability": stability,
        "action_bias": action,
        "confidence": conf,
        "regime_description": desc,
        "top_drivers": [
            f"Trend/extension score is {trend_score}/100 (distance vs 200DMA: {fmt_pct(dist200[latest])})",
            f"Vol/shock score is {shock_score}/100 (20D realized vol percentile: {ordinal(rv_p)})",
            f"Risk appetite score is {risk_score}/100" + (f" (HY spread level: {hy_lvl:.2f})" if hy_lvl is not None else ""),
        ],
        "triggers": regime_triggers,
        "what_changes_next": {
            "escalate": [
                "If extension stays high and shock flips to Triggered, BLUE can escalate to ORANGE.",
                "If drawdown deepens and credit stress jumps, ORANGE can escalate to RED.",
            ],
            "deescalate": [
                "If shock and credit flags turn off while extension cools, BLUE can normalize to GREEN.",
                "If breakdown flags clear and trend stabilizes for multiple sessions, RED can step back to ORANGE.",
            ],
        },
        "now_cards": {
            "event_risk_1_5d": shock_score,
            "fragility_1_3m": fragility,
            "liquidity_regime": risk_appetite,
        },
        "pillars": [
            {"name": "Trend / Extension", "score": trend_score, "metrics": [f"Distance vs 200DMA: {fmt_pct(dist200[latest])}", f"20D momentum: {fmt_pct(ret20[latest])}", f"63D drawdown: {dd63[latest]:.2f}%"]},
            {"name": "Vol / Shock", "score": shock_score, "metrics": [f"Realized vol (20D): {rv20[latest]:.2f}%", f"1D drop percentile: {ordinal(drop_p)}", f"Whipsaw flag: {'triggered' if whipsaw else 'quiet'}"]},
            {"name": "Macro (headwinds)", "score": macro_score, "metrics": [f"DFII10: {dfii_lvl:.2f}" if dfii_lvl is not None else "DFII10: n/a", f"DFII10 1M change: {dfii_ch:+.2f}" if dfii_ch is not None else "DFII10 1M change: n/a", f"T10YIE 1M change: {t10_ch:+.2f}" if t10_ch is not None else "T10YIE 1M change: n/a"]},
            {"name": "Risk appetite", "score": risk_score, "metrics": [f"HY spread level: {hy_lvl:.2f}" if hy_lvl is not None else "HY spread level: n/a", f"HY spread 1M change: {hy_ch:+.2f}" if hy_ch is not None else "HY spread 1M change: n/a", f"Regime: {risk_appetite}"]},
        ],
        "signals": cards,
        "rules": [
            {"regime": "GREEN", "definition": "Trend positive-to-neutral, shock low, credit stress quiet.", "action_bias": "Accumulate on dips / hold core", "escalation": "Extension score rises above ~70th percentile while shock stays quiet.", "deescalation": "N/A"},
            {"regime": "BLUE", "definition": "Extended above trend, but no confirmed deterioration.", "action_bias": "Hold / add only on pullbacks", "escalation": "Extension high + shock Watch/Triggered or credit starts widening.", "deescalation": "Extension cools and deterioration flags remain off."},
            {"regime": "ORANGE", "definition": "Extended + at least one deterioration pillar active.", "action_bias": "Stop adding, trim rips, tighten risk", "escalation": "Breakdown pattern appears (vol spike + drawdown / credit jump).", "deescalation": "Shock clears and price stabilizes above key moving averages."},
            {"regime": "RED", "definition": "Breakdown / stress regime (triggered by deep drawdown, high shock, or credit stress).", "action_bias": "Defensive posture", "escalation": "N/A", "deescalation": "Shock flags turn off and trend repair persists for multiple sessions."},
        ],
        "sources": {
            "slv": "https://stooq.com/q/d/l/?s=slv.us&i=d",
            "dfii10": "https://fred.stlouisfed.org/series/DFII10",
            "t10yie": "https://fred.stlouisfed.org/series/T10YIE",
            "hy_spread": "https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="silver.json")
    args = parser.parse_args()

    prev_stability_streak = 0
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                previous = json.load(f)
            if isinstance(previous, dict):
                prev_stability_streak = int((previous.get("stability") or {}).get("streak", 0) or 0)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            prev_stability_streak = 0

    try:
        slv = fetch_stooq_slv()
        api_key = os.getenv("FRED_API_KEY", "").strip()
        fred: Dict[str, List[Tuple[datetime, float]]] = {}
        for k, series in FRED_SERIES.items():
            if not api_key:
                fred[k] = []
                continue
            try:
                fred[k] = fetch_fred(series, api_key)
            except Exception:
                fred[k] = []
        payload = build_payload(slv, fred, previous_stability_streak=prev_stability_streak)
    except Exception as exc:
        payload = {
            "title": "Silver Risk Monitor (SLV)",
            "subtitle": "Outcome first: regime, drivers, and what changes next.",
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "data_dates": "Data fetch unavailable in current environment",
            "regime": "BLUE",
            "stability": {
                "score_0_100": 0,
                "is_stabilizing": False,
                "label": "unstable",
                "streak": 0,
                "reasons": ["insufficient_data"],
            },
            "action_bias": "Hold / add on pullbacks",
            "confidence": "Low",
            "regime_description": "Fallback payload due to data-fetch failure.",
            "top_drivers": [f"Data fetch error: {exc}"],
            "what_changes_next": {"escalate": ["Awaiting live data"], "deescalate": ["Awaiting live data"]},
            "now_cards": {"event_risk_1_5d": 50, "fragility_1_3m": 50, "liquidity_regime": "Mixed"},
            "pillars": [],
            "signals": [],
            "rules": [],
            "sources": {
                "slv": "https://stooq.com/q/d/l/?s=slv.us&i=d",
                "dfii10": "https://fred.stlouisfed.org/series/DFII10",
                "t10yie": "https://fred.stlouisfed.org/series/T10YIE",
                "hy_spread": "https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
            },
            "notes": {"fetch_ok": False}
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
