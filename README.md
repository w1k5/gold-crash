# Gold Risk Monitor

A fully automated, $0-hosted dashboard that monitors gold risk signals for GLD/IAU/SLV/SIVR/PICK holders. It runs daily on GitHub Actions, publishes a static dashboard via GitHub Pages, and raises a GitHub Issue when a RED flag is triggered.

## What it does

- Fetches GLD daily prices (no API key), GLD holdings (SPDR CSV), and DFII10 real yield data from FRED (API key required).
- Computes 1M/3M returns, 3M max drawdown, 200DMA extension, real-yield changes, and GLD holdings flow changes.
- Assigns a GREEN/BLUE/ORANGE/RED flag based on explicit rules.
- Publishes `data.json` and a static dashboard (`index.html`).
- Opens or updates a GitHub Issue titled **“🚨 Gold Risk Monitor: RED flag”** when RED is triggered.
- Opens or updates a GitHub Issue titled **“⚠️ Gold Risk Monitor: data fetch failed”** when data fetch fails, without overwriting the last known good dashboard.

## Flag logic (exact)

- 1M = 21 trading days
- 3M = 63 trading days

Metrics:
- `gld_ret_1m = (price_today / price_21d_ago) - 1`
- `gld_ret_3m = (price_today / price_63d_ago) - 1`
- `gld_max_drawdown_3m = max peak-to-trough decline within last 63 days`
- `gld_200dma = 200-day simple moving average of GLD`
- `gld_pct_above_200dma = (price_today / gld_200dma) - 1`
- `real_yield_today = DFII10 latest value (percent)`
- `real_yield_change_1m_bp = (today - 21d_ago) * 100`
- `real_yield_change_3m_bp = (today - 63d_ago) * 100`
- `gld_holdings_today = latest GLD holdings in tonnes`
- `gld_holdings_change_5d_pct = (holdings_today / holdings_5d_ago) - 1`
- `gld_holdings_change_21d_pct = (holdings_today / holdings_21d_ago) - 1`
- `*_pctile_5y = 5-year percentile for the corresponding metric`

Flags (extension → deterioration → breakdown):
- **BLUE (extension score >= 2)**: +1 each
  - 3M return percentile >= 90
  - % above 200DMA percentile >= 90
  - 3M drawdown percentile >= 70 (smooth rally)
- **ORANGE (extension + deterioration)**: extension score >= 2 AND deterioration score >= 1
  - Flow divergence: holdings 21D <= -1.5% and 1M return > 0
  - Macro turn: DFII10 1M change >= +25 bp
  - Price crack: 1M return <= 0 or 3M drawdown <= -8%
- **RED (breakdown)**
  - Primary triggers (immediate): 3M return <= -15% OR 3M drawdown <= -18% OR DFII10 1M change >= +50 bp
  - Composite stress (2 of 4): 3M return <= -8%, holdings 21D <= -2%, holdings 5D <= -1%, DFII10 1M change >= +25 bp
  - Enter RED after 2 consecutive composite runs; exit after 5 clean runs
- **GREEN** otherwise

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
