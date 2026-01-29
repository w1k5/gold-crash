# Gold Risk Monitor

A fully automated, $0-hosted dashboard that monitors gold risk signals for GLD/IAU/SLV/SIVR/PICK holders. It runs daily on GitHub Actions, publishes a static dashboard via GitHub Pages, and raises a GitHub Issue when a RED flag is triggered.

## What it does

- Fetches GLD daily prices (no API key), GLD holdings (SPDR CSV), and DFII10 real yield data from FRED (API key required).
- Computes multi-horizon GLD returns/drawdowns (3M/6M/1Y/3Y/5Y), a 200DMA extension metric, and short-term flow + macro changes.
- Assigns a single GREEN/BLUE/ORANGE/RED regime based on the **primary 3M horizon** plus extension, flow, and macro signals.
- Publishes `data.json` and a static dashboard (`index.html`) with a Horizon Snapshot table for context.
- Opens or updates a GitHub Issue titled **“🚨 Gold Risk Monitor: RED regime”** only when the regime transitions into RED, and updates the same issue when it exits RED.
- Opens or updates a GitHub Issue titled **“⚠️ Gold Risk Monitor: data fetch failed”** when data fetch fails, without overwriting the last known good dashboard.

## Regime logic (exact)

### Time horizons

- 3M = 63 trading days (primary)
- 6M = 126 trading days
- 1Y = 252 trading days
- 3Y = 756 trading days
- 5Y = 1260 trading days

### Metrics

- `ret_H = (price_today / price_H_days_ago) - 1`
- `max_drawdown_H = max peak-to-trough decline within last H days`
- `ma_200 = 200-day simple moving average of GLD`
- `pct_above_200dma = (price_today / ma_200) - 1`
- `real_yield_today = DFII10 latest value (percent)`
- `real_yield_change_1m_bp = (today - 21d_ago) * 100`
- `real_yield_change_3m_bp = (today - 63d_ago) * 100`
- `holdings_today_tonnes = latest GLD holdings in tonnes`
- `holdings_change_5d_pct = (holdings_today / holdings_5d_ago) - 1`
- `holdings_change_21d_pct = (holdings_today / holdings_21d_ago) - 1`
- `*_pctile_5y = percentile for the corresponding metric over the last 5 years`

### Percentiles

- Percentiles use a rolling 5-year context window when possible.
- If a 5-year window is insufficient (e.g., for 5Y horizon metrics), the percentile is computed on available history and flagged with a note, or set to `null` with an explanation.
- Each percentile includes its basis (`5y` or `available`) and the sample count (`n`) to show how much history backed the calculation.

### Regime rules (one state; primary horizon = 3M)

- **GREEN (normal)** if all are true:
  - 3M return percentile < 80
  - % above 200DMA percentile < 80
  - Holdings 21D change percentile > 20
  - Real yield 1M change percentile < 80
- **BLUE (overheated)** if not GREEN and extension score >= 2, where extension score counts:
  - 3M return percentile >= 90
  - % above 200DMA percentile >= 90
  - 3M drawdown percentile >= 70
- **ORANGE (topping risk)** if BLUE and deterioration is present:
  - Flow divergence: holdings 21D <= -1.5% AND 3M return > 0 AND 1M return > 0 (if 1M exists)
  - Macro turn: real yield 1M change >= +25 bp
  - Price crack: 1M return <= 0 OR 3M drawdown <= -8%
- **RED (breakdown risk)** if any primary trigger OR composite stress:
  - Primary triggers: 3M return <= -15%, 3M drawdown <= -18%, real yield 1M change >= +50 bp
  - Composite stress (2 of 4): 3M return <= -8%, holdings 21D <= -2%, holdings 5D <= -1%, real yield 1M change >= +25 bp

### Persistence

- Entering RED requires 2 consecutive runs meeting RED unless a primary trigger is true.
- Exiting RED requires 5 consecutive runs not meeting RED.

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

1. Create or rotate a FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
2. In GitHub, open **Settings → Secrets and variables → Actions**.
3. Add **New repository secret**:
   - Name: `FRED_API_KEY`
   - Value: your API key

If you believe a key has been exposed, revoke it in FRED and replace it in GitHub Secrets.

### 3) Enable GitHub Pages

1. Go to **Settings → Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Set **Branch** to `main` and **/(root)** as the folder.
4. Save; your dashboard will be available at `https://<OWNER>.github.io/<REPO>/`.

### 4) (Optional) Local test

```bash
export FRED_API_KEY=your_key
python scripts/update.py --output data.json --status-file /tmp/monitor_status.json
```

## Automation

The scheduled workflow runs daily at **07:05 America/New_York**. GitHub Actions cron is set to 12:05 UTC, which matches 07:05 EST; during daylight time the run occurs at 08:05 EDT. Adjust the cron time if you need a strict 07:05 EDT/EST schedule.

## Data sources

- GLD prices: Stooq CSV (no key required): https://stooq.com/q/d/l/?s=gld.us&i=d
- GLD holdings: SPDR Gold Shares CSV: https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv
- Real yields: FRED DFII10: https://fred.stlouisfed.org/series/DFII10

## What you must do by hand

- Create the GitHub repository and push the code.
- Add the `FRED_API_KEY` secret.
- Enable GitHub Pages from `/(root)`.
- Rotate any compromised API keys.

## License

MIT
