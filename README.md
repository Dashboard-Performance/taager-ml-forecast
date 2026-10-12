# Taager ML Forecast — shadow experiment

A real machine-learning demand forecast that **learns from every past month** and
**re-trains itself automatically each month** via GitHub Actions.

> ⚠️ **Shadow mode — it does NOT touch the live dashboard.** It only produces a
> results view so you can see *where the ML wins* before anything is wired into
> the Forecast Model. Wiring into the live forecast is a separate, later step you
> approve.

## What it does
1. Reads the public **Forecast History** tab of the dashboard sheet (same data the
   dashboard reads — no export, no credentials).
2. Builds leakage-safe features (lags, rolling stats, placed/delivered) and runs a
   **walk-forward backtest**: train on the months before each test month, predict it,
   compare to what actually happened.
3. Trains a global gradient-boosted model on **all** months and predicts next month
   per product.
4. Writes to `output/`:
   - `results.html` — the **View**: measured accuracy (1 − WAPE) per segment,
     ML vs naive vs 3-month-average. Open it to see where ML wins.
   - `accuracy_by_segment.csv`
   - `predictions.csv` — next-month ML forecast per product.

## Setup (one time)
1. Create a new GitHub repo and upload these files.
2. Actions → enable workflows. It runs monthly (2nd of the month) and can be run
   manually via **Run workflow**.
3. (Optional) Settings → Pages → serve from the repo so `output/results.html` opens
   in the browser. Or just download it from the repo.

## Run it yourself locally
```
pip install -r requirements.txt
python src/train.py                 # fetch live from the sheet
python src/train.py --local hist.csv  # or use a local pivot CSV
```

## Measured result (Oct 2026, same data as the dashboard)
Compared head-to-head against the dashboard's Forecast Model on the same 1,818
single SKUs (bundles split) and the same months (May-Sep 2026), accuracy = 1 - WAPE:

| Segment | Dashboard | This ML | Last month (naive) |
|---|---|---|---|
| All | 53.8% | 38.6% | 41.2% |
| A (100+) | 60.4% | 43.9% | 47.9% |
| B (30-99) | 26.4% | 15.0% | 8.8% |
| C (<30) | 9.4% | 4.2% | -0.2% |

The ML beats "last month" on B and C but is behind the dashboard everywhere, and
averaging the two (50.8%) is worse than the dashboard alone. Keep it in shadow mode.

> Note: the history tab is `Data Jan-July` (gid 1433599782, the dashboard's
> FORECAST_HISTORY_GID). Google serves the *first* sheet when a gid does not exist,
> so `train.py` now checks the column names and stops with an error instead of
> silently training on the wrong tab. It also drops a partial last month.
