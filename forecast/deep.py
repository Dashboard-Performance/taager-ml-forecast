"""v2.3 - NHITS: a deep global model trained from scratch on the daily series of every SKU (it reads the daily shape, not the monthly features,
so its mistakes differ from the GBMs' and from the Forecast Model recipe's - averaging the three cancels part of each one's error).

Measured on Jul-Oct 2026 (month-end forecast of the next month, 2025 + 2026 history):
  Forecast Model recipe 53.4% | GBM ensemble 50.2% | NHITS 52.6% | the three in equal parts ("merge3") 55.9%.

Leakage: the training window ends on the origin day (index o) - nothing after it is read (tests/test_all.py scrambles the future and checks).
Optional: without the neuralforecast package (or with ML_DEEP=0) every call returns None and merge3 falls back to Forecast Model + ensemble."""
import os
import numpy as np
import pandas as pd

WINDOW_DAYS = 365        # the last year of daily sales the model trains on
INPUT_DAYS = 56          # how far back one forecast looks
STEPS = int(os.environ.get("ML_DEEP_STEPS", "300"))


def available():
    if os.environ.get("ML_DEEP", "1") == "0": return False
    try:
        import neuralforecast  # noqa: F401
        return True
    except Exception:
        return False


def training_frame(panel, o):
    """(long DataFrame for training, SKU indexes) - only days <= o, only SKUs that sold in the 90 days up to o; log1p pieces."""
    lo = max(0, o - WINDOW_DAYS + 1)
    ids = np.flatnonzero(panel.Q[:, max(0, o - 89):o + 1].sum(axis=1) > 0)
    dates = panel.dates[lo:o + 1]
    Y = np.log1p(panel.Q[ids, lo:o + 1])
    return pd.DataFrame({"unique_id": np.repeat(np.asarray(panel.skus)[ids], len(dates)), "ds": np.tile(dates, len(ids)), "y": Y.ravel()}), ids


def next_month(panel, o, info, seed=7):
    """Pieces per SKU for the month after the origin's month (info from features.origin_info). None when the deep model is off / fails."""
    if not available(): return None
    try:
        import logging, warnings
        warnings.filterwarnings("ignore")
        for n in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"): logging.getLogger(n).setLevel(logging.ERROR)
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
        from neuralforecast.losses.pytorch import MAE
        df, ids = training_frame(panel, o)
        if not len(ids): return np.zeros(len(panel.skus))
        rest = int((info["c_last"] - panel.dates[o]).days)            # days left in the origin's month (0 on its last day)
        H = rest + info["len_m"]
        mdl = NHITS(h=H, input_size=INPUT_DAYS, loss=MAE(), max_steps=STEPS, batch_size=64, random_seed=seed, scaler_type=None,
                    enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator="cpu", devices=1)
        nf = NeuralForecast(models=[mdl], freq="D")
        nf.fit(df=df)
        fc = nf.predict().reset_index()
        col = [c for c in fc.columns if c not in ("unique_id", "ds", "index")][0]
        fc = fc[fc["ds"] >= info["m_first"]]                            # only the days of the target month
        tot = fc.assign(v=np.expm1(fc[col]).clip(lower=0)).groupby("unique_id")["v"].sum()
        out = np.zeros(len(panel.skus))
        out[ids] = pd.Series(np.asarray(panel.skus)[ids]).map(tot).fillna(0.0).to_numpy()
        return out
    except Exception as e:
        print("NHITS skipped:", e)
        return None
