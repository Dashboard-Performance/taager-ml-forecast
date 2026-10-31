"""Baselines + gradient-boosted models + honest uncertainty bands.

Nothing here is tuned by hand to look good: every candidate (the simple baselines included) is scored on real past months, and the
live forecast uses whichever candidate has been most accurate for that volume segment (see engine.champions).
"""
import numpy as np
import pandas as pd

from .features import FEATURES

BASELINES = ["naive", "ma3", "pace14", "pace30", "pace_mtd"]
GBMS = ["gbm_l1", "gbm_tw", "gbm_log"]
CANDIDATES = BASELINES + GBMS + ["ens"]
PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=7, n_jobs=2, verbose=-1, deterministic=True, force_row_wise=True)
MIN_TRAIN_ROWS = 300


def baselines(df, offset):
    """offset 1 = next month's total, 0 = the current month's total (nowcast)."""
    m = df[["m1", "m2", "m3"]].to_numpy()
    cnt = (~np.isnan(m)).sum(axis=1)
    ma3 = np.where(cnt > 0, np.nansum(m, axis=1) / np.maximum(cnt, 1), df["r30"].to_numpy() * 30.0)
    naive = np.where(np.isnan(df["m1"].to_numpy()), ma3, df["m1"].to_numpy())
    lm, lc, d = df["len_m"].to_numpy(), df["len_c"].to_numpy(), df["d"].to_numpy()
    if offset == 1:
        return {"naive": naive, "ma3": ma3, "pace14": df["r14"].to_numpy() * lm, "pace30": df["r30"].to_numpy() * lm, "pace_mtd": df["mtd_rr"].to_numpy() / lc * lm}
    rest = lc - d
    mtd = df["mtd"].to_numpy()
    return {"naive": naive, "ma3": ma3, "pace14": mtd + df["r14"].to_numpy() * rest, "pace30": mtd + df["r30"].to_numpy() * rest, "pace_mtd": df["mtd_rr"].to_numpy()}


def fit(X, y, scale, which=GBMS):
    import lightgbm as lgb
    if len(X) < MIN_TRAIN_ROWS: return {}
    out = {}
    if "gbm_l1" in which:       # minimises the absolute error in pieces (= what WAPE measures): ratio target, weight = scale
        m = lgb.LGBMRegressor(objective="l1", **PARAMS); m.fit(X, y / scale, sample_weight=scale); out["gbm_l1"] = m
    if "gbm_tw" in which:       # count-like data with many zeros
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **PARAMS); m.fit(X, y); out["gbm_tw"] = m
    if "gbm_log" in which:      # multiplicative errors
        m = lgb.LGBMRegressor(objective="regression", **PARAMS); m.fit(X, np.log1p(y)); out["gbm_log"] = m
    return out


def predict(models, X, scale):
    out = {}
    for k, m in models.items():
        p = m.predict(X)
        out[k] = np.maximum(0.0, p * scale) if k == "gbm_l1" else np.maximum(0.0, np.expm1(p)) if k == "gbm_log" else np.maximum(0.0, p)
    if len(out) >= 2: out["ens"] = np.mean([out[k] for k in GBMS if k in out], axis=0)
    return out


def accuracy(y, f):
    y = np.asarray(y, float); f = np.asarray(f, float); s = y.sum()
    return float((1.0 - np.abs(y - f).sum() / s) * 100.0) if s > 0 else float("nan")


def bias(y, f):
    y = np.asarray(y, float); f = np.asarray(f, float); s = y.sum()
    return float((f.sum() / s - 1.0) * 100.0) if s > 0 else float("nan")


def band_quantiles(y, f, lo=0.10, hi=0.90):
    """Scaled residuals z = (y - f) / (f + 5): relative for big SKUs, absolute for tiny ones. Returns (z_lo, z_hi)."""
    z = (np.asarray(y, float) - np.asarray(f, float)) / (np.asarray(f, float) + 5.0)
    if len(z) < 30: return None
    return float(np.quantile(z, lo)), float(np.quantile(z, hi))


def apply_band(f, zq):
    f = np.asarray(f, float)
    if zq is None: return f.copy(), f.copy()
    return np.maximum(0.0, f + zq[0] * (f + 5.0)), np.maximum(f, f + zq[1] * (f + 5.0))
