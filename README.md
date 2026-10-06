# Gold Risk Monitor

A fully automated, $0-hosted dashboard that **classifies gold market regimes** for GLD (with contextual relevance for IAU / SLV / SIVR / PICK holders).
It runs daily on GitHub Actions, publishes a static dashboard via GitHub Pages, and conditionally raises GitHub Issues when **risk regimes or data integrity change**.

This project is designed to **describe regimes and transitions**, not to forecast prices or provide trade signals.

## What it does

* Fetches:

  * GLD daily prices (Stooq CSV; no API key)
  * GLD holdings (SPDR CSV)
  * US real yields and rates data from FRED (API key required)
* Computes:

  * Multi-horizon GLD returns and drawdowns (3M / 6M / 1Y / 3Y / 5Y)
  * Price extension vs 200-day moving average
  * Short-term flow changes from GLD holdings
  * Short-term macro pressure from **real-yield changes**
  * Rolling correlations between GLD returns and real-yield changes
* Classifies a **single regime color** (GREEN / BLUE / ORANGE / RED) based on:

  * Short-term price extension
  * Momentum and drawdown behavior
  * Flow confirmation or divergence
  * Macro pressure and follow-through
* Publishes:

  * `data.json` (machine-readable state)
  * A static dashboard (`index.html`) with:

    * “At a glance” cards
    * Regime drivers
    * Explicit transition thresholds (“what would change the color”)
    * Horizon Snapshot tables
    * Extension, Flow, and Macro detail sections
* GitHub Issues:

  * Opens or updates **“🚨 Gold Risk Monitor: RED regime”** only when the regime **enters or remains in RED**
  * Updates the same issue when RED conditions clear
  * Opens or updates **“⚠️ Gold Risk Monitor: data fetch failed”** when inputs fail, without overwriting the last valid dashboard

## Conceptual model

The monitor treats gold as moving through **market regimes**, not discrete buy/sell states:

* **GREEN** → Typical conditions
* **BLUE** → Extended rally without deterioration
* **ORANGE** → Extension plus early deterioration signals
* **RED** → Breakdown-style stress (price, flow, or macro)

The **3-month horizon is primary** for regime classification. Longer horizons are shown for context only.

## Time horizons

| Label | Trading days | Purpose                   |
| ----- | ------------ | ------------------------- |
| 3M    | 63           | **Primary regime driver** |
| 6M    | 126          | Medium-term context       |
| 1Y    | 252          | Trend context             |
| 3Y    | 756          | Cycle context             |
| 5Y    | 1260         | Structural context        |

## Metrics (definitions)

* `ret_H = (price_today / price_H_days_ago) - 1`
* `max_drawdown_H = worst peak-to-trough decline in last H days`
* `ma_200 = 200-day simple moving average`
* `pct_above_200dma = (price_today / ma_200) - 1`

**Flows**

* `holdings_today_tonnes`
* `holdings_change_5d_pct`
* `holdings_change_21d_pct`

**Macro**

* `real_yield_today = DFII10`
* `real_yield_change_1m_bp`
* `real_yield_change_3m_bp`
* `corr_gld_ret_vs_real_yield_chg_20d`

All percentile metrics are evaluated relative to a **rolling 5-year window when possible**.

## Percentiles & data quality

* Percentiles are contextual, not judgments.
* If a full 5-year window is unavailable:

  * The calculation uses available history
  * The dashboard flags this with an explanatory note
* Missing or insufficient data is surfaced explicitly (never silently filled).

## Regime logic (high-level)

### GREEN (typical)

* No meaningful extension
* No deterioration in flows
* No macro pressure

### BLUE (extended)

Triggered when **price extension conditions** are present (e.g. extreme 3M returns, distance above 200DMA, shallow drawdowns), **without** deterioration signals.

### ORANGE (early stress)

BLUE **plus at least one deterioration signal**, such as:

* Flow divergence (price rising while holdings fall)
* Rapid rise in real yields
* Short-term price weakness (“price crack”)

### RED (breakdown risk)

Triggered by either:

* **Primary stress** (large drawdowns or sharp real-yield spikes), or
* **Composite stress** (multiple moderate signals across price, flows, and macro)

The dashboard shows **exact thresholds and distances** for:

* De-escalation
* Escalation
* Normalization

## Persistence rules

* **RED entry**:

  * Requires confirmation across runs unless a primary trigger fires
* **RED exit**:

  * Requires multiple consecutive runs without RED conditions

This avoids single-day noise flipping regimes.

## Cut-style classifier (macro context)

The dashboard includes a **cut-style classifier** to explain *why* real yields are moving:

* **Credibility cut** → inflation expectations falling faster than nominal yields
* **Stimulus cut** → reflation impulse (breakevens rising)
* **Mixed / No cuts priced** → ambiguous or neutral macro signal

This is explanatory context only; it does not directly set the regime.

## Automation

* Runs daily via GitHub Actions
* Scheduled for **07:05 America/New_York**

  * Cron uses 12:05 UTC (07:05 EST; 08:05 EDT)
* On each run:

  * Updates `data.json`
  * Updates the dashboard
  * Evaluates regime transitions
  * Conditionally updates GitHub Issues

## Setup

### 1) Create the repo and push

```bash
git init
git add .
git commit -m "Initial Gold Risk Monitor"
git branch -M main
git remote add origin git@github.com:<OWNER>/<REPO>.git
git push -u origin main
```

### 2) Add the FRED API key (required)

1. Create or rotate a key at
   [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. GitHub → **Settings → Secrets and variables → Actions**
3. Add a repository secret:

   * **Name:** `FRED_API_KEY`
   * **Value:** your key

### 3) Enable GitHub Pages

1. **Settings → Pages**
2. Deploy from `main` → `/(root)`
3. Dashboard will be live at:
   `https://<OWNER>.github.io/<REPO>/`

### 4) (Optional) Local run

```bash
export FRED_API_KEY=your_key
python scripts/update.py --output data.json --status-file /tmp/monitor_status.json
```

## Data sources

* GLD prices (Stooq):
  [https://stooq.com/q/d/l/?s=gld.us&i=d](https://stooq.com/q/d/l/?s=gld.us&i=d)
* GLD holdings (SPDR):
  [https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv](https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv)
* Real yields (FRED DFII10):
  [https://fred.stlouisfed.org/series/DFII10](https://fred.stlouisfed.org/series/DFII10)

## Dislocation detector

`scripts/dislocation_detector.py` powers `/dislocation/` and the mirrored
`public/dislocation/` page. It describes cross-market stress separately from the
gold monitor's GREEN/BLUE/ORANGE/RED regime.

* Enter **DISLOCATION** only at `K` eligible checks (default 3). Hold at
  `max(K−1, 1)` only when the previous status was dislocation and there is a
  recorded confirmed entry. A one-check run followed by two checks stays
  **STRESS BUILDING**. Legacy two-check dislocations are reset rather than
  assumed to have been confirmed; changing K also requires a new confirmation.
* Every configured FRED input includes its actual `source_date`, business-day
  `stale_days`, cadence and eligibility. Daily inputs tolerate one business day
  for signal counting; the monthly JGB yield tolerates 35. Weekends are excluded
  from lag calculations; exchange/bank holidays are not separately modeled.
  Observations after the equity session are discarded. Missing or out-of-tolerance
  FRED feeds set **DATA STALE** and their dependent checks are excluded from
  both the threshold and dashboard scores. The UI gives DATA STALE precedence.
* Confidence is an input-freshness heuristic, **not a probability**. Each daily
  FRED/holdings input loses 5 points per lagged business day (maximum 60);
  missing inputs lose 20. VIX retains its 20-points-per-day penalty (maximum 60).
  Weekly CFTC and monthly JGB observations are penalized only beyond their
  cadence tolerance. JSON exposes each contribution in `confidence_penalties`.
* HY OAS uses a 504-observation z-score window with at least 252 valid source
  observations. Changes and z-scores are calculated before alignment to equity
  sessions, so forward-filled rows do not invent extra OAS observations.
  Insufficient history leaves the z-score null and reports the sample size.
* **Gold futures deleveraging** is a sixth, contextual pillar. It requires falling
  COMEX gold open interest, falling managed-money net length (long minus short),
  and a negative GLD return over the same two consecutive CFTC report dates.
  GLD holdings rising over that interval adds **ETF buying / futures selling**
  context; missing holdings leave that comparison unavailable. Holdings cover
  GLD only, not all ETFs or physical demand. Positioning is weekly and generally
  released after the observation date, not an intraday liquidation feed. Reports
  more than 7 business days behind the equity session, nonconsecutive reports,
  and unmatched price intervals cannot trigger the warning.
* The gold positioning check and existing combo checks are excluded from K-of-N.
  The new pillar does not change crash/fragility score weights; it adds a distinct
  gold warning without implying a broad funding crisis. Its severity uses a
  2% decline ramp for each leg (managed-money change scaled by prior total OI),
  and is descriptive rather than calibrated to crash probabilities.

CFTC source: [Disaggregated futures-only reports](https://publicreporting.cftc.gov/stories/s/Disaggregated-Futures-Only/ubmb-6exi/),
dataset `72hh-3qpy`, COMEX gold contract-market code `088691`.
GLD holdings reuse the existing SPDR CSV/archive fetcher. Source outages are
reported as unavailable; they do not silently replace weekly positions with zeros.

The daily job refreshes both `dislocation.json` copies after merge. Market-data
snapshots are not manually reclassified as part of code changes.

Run the deterministic regression suite (no API keys or network required):

```bash
pip install pandas requests
python -m unittest discover -s tests -v
node tests/test_dislocation_dashboard.js
```

PR checks run this suite separately from the production data-refresh job, which
only runs on main, schedule, or manual dispatch.

## What you must do manually

* Create the GitHub repository
* Add the `FRED_API_KEY` secret
* Enable GitHub Pages
* Rotate exposed API keys if needed

## License

MIT
