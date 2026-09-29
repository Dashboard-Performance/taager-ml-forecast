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

## First measured result (Jan–Jul 2026, confirmed pieces, walk-forward)
Accuracy = 1 − WAPE, vs simply repeating last month (naive):

| Segment | ML | Naive | ML − Naive |
|---|---|---|---|
| A (≥100/mo) | 43.3% | 36.7% | +6.6 |
| B (30–99)   | 15.7% | 10.4% | +5.3 |
| C (<30)     |  4.1% |  0.5% | +3.6 |

ML beats naive on every segment. It is **not yet** compared head-to-head against the
dashboard's tuned engine (TSB + blend) — that comparison is the gate before wiring
the ML into the live Forecast Model. Accuracy grows as more months accumulate.
