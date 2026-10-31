"""Features for one forecast origin: everything the model may know on the evening of day `o` (the last observed day).

Leakage rule: a feature may only read columns <= o. tests/test_features.py proves it by scrambling every day after o and checking nothing moves.
"""
import numpy as np
import pandas as pd

EPS = 1e-9
FEATURES = [
    "m1", "m2", "m3", "m4", "m5", "m6", "mtd", "mtd_rr", "d", "len_c", "frac_c", "len_m", "dow_units_m",
    "r3", "r7", "r14", "r30", "r60", "r90", "r7_r30", "r14_r60", "r30_r90", "mtdrr_m1", "r14m_m1",
    "slope30", "cv30", "zero30", "maxshare30", "cr14", "dr_mature", "age_days", "since_sale", "nz_months6", "scale",
    "p_chg30", "p_chg90", "cat",
]


def month_bounds(dates):
    """(period array, first index of each day's month, last index of each day's month)"""
    per = dates.to_period("M")
    codes = per.asi8
    first = np.zeros(len(dates), int); last = np.zeros(len(dates), int)
    s = 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or codes[i] != codes[i - 1]:
            first[s:i] = s; last[s:i] = i - 1; s = i
    return per, first, last


def _dow_idx(dates, Q, o):
    tot = Q[:, :o + 1].sum(axis=0)
    wd = dates[:o + 1].dayofweek.to_numpy()
    if len(tot) < 28 or tot.sum() <= 0: return np.ones(7)
    m = tot.mean()
    return np.array([tot[wd == k].mean() / m if (wd == k).any() and m > 0 else 1.0 for k in range(7)])


def _win(cs, o, w):
    a = max(o - w + 1, 0)
    return cs[:, o + 1] - cs[:, a], o - a + 1


def origin_info(dates, o):
    """Calendar facts about origin o: current month C (periods), days observed d, length of C, target month M and its first/last day."""
    day = dates[o]
    C = day.to_period("M"); M = C + 1
    return {"C": C, "M": M, "d": int(day.day), "len_c": int(C.days_in_month), "len_m": int(M.days_in_month),
            "c_first": C.start_time.normalize(), "c_last": C.end_time.normalize(), "m_first": M.start_time.normalize(), "m_last": M.end_time.normalize()}


def build_features(panel, o, cat_codes=None):
    dates, Q, P, R = panel.dates, panel.Q, panel.P, panel.R
    S = Q.shape[0]
    info = origin_info(dates, o)
    per, first, last = month_bounds(dates)
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(Q, axis=1)], axis=1)
    csP = np.concatenate([np.zeros((S, 1)), np.cumsum(P, axis=1)], axis=1)
    csR = np.concatenate([np.zeros((S, 1)), np.cumsum(R, axis=1)], axis=1)
    f = {}
    # --- last complete months before C (NaN when that month is not fully inside the data)
    c_start_idx = int(first[o])
    for k in range(1, 7):
        pm = info["C"] - k
        s = pm.start_time.normalize(); e = pm.end_time.normalize()
        if s < dates[0] or e > dates[o]: f["m%d" % k] = np.full(S, np.nan); continue
        si = int((s - dates[0]).days); ei = int((e - dates[0]).days)
        f["m%d" % k] = cs[:, ei + 1] - cs[:, si]
    # --- month to date
    d = info["d"]
    mtd = cs[:, o + 1] - cs[:, c_start_idx]
    f["mtd"] = mtd; f["mtd_rr"] = mtd / d * info["len_c"]
    f["d"] = np.full(S, float(d)); f["len_c"] = np.full(S, float(info["len_c"])); f["frac_c"] = np.full(S, d / info["len_c"]); f["len_m"] = np.full(S, float(info["len_m"]))
    # --- weekday-weighted trading days of the target month (pattern learned from the past only)
    di = _dow_idx(dates, Q, o)
    mdays = pd.date_range(info["m_first"], info["m_last"], freq="D")
    f["dow_units_m"] = np.full(S, float(di[mdays.dayofweek.to_numpy()].sum()))
    # --- recent windows: daily rates and how they move
    rate = {}
    for w in (3, 7, 14, 30, 60, 90):
        s_, n_ = _win(cs, o, w); rate[w] = s_ / max(n_, 1); f["r%d" % w] = rate[w]
    f["r7_r30"] = rate[7] / (rate[30] + EPS); f["r14_r60"] = rate[14] / (rate[60] + EPS); f["r30_r90"] = rate[30] / (rate[90] + EPS)
    m1 = f["m1"]; m1f = np.nan_to_num(m1, nan=0.0)
    f["mtdrr_m1"] = f["mtd_rr"] / (m1f + 1.0); f["r14m_m1"] = rate[14] * info["len_m"] / (m1f + 1.0)
    # --- shape of the last 30 days
    a = max(o - 29, 0); X = Q[:, a:o + 1]; n = X.shape[1]
    mu = X.mean(axis=1)
    if n >= 5:
        t = np.arange(n) - (n - 1) / 2.0
        slope = (X * t).sum(axis=1) / (t ** 2).sum()
        f["slope30"] = slope / (mu + EPS)
    else:
        f["slope30"] = np.zeros(S)
    sd = X.std(axis=1); f["cv30"] = sd / (mu + EPS); f["zero30"] = (X == 0).mean(axis=1)
    f["maxshare30"] = X.max(axis=1) / (X.sum(axis=1) + EPS)
    # --- funnel
    s14, _ = _win(cs, o, 14); p14, _ = _win(csP, o, 14)
    f["cr14"] = np.where(p14 > 0, s14 / (p14 + EPS), np.nan)
    a2, b2 = max(o - 44, 0), max(o - 14, 0)
    if b2 > a2:
        cm = cs[:, b2 + 1] - cs[:, a2]; dm = csR[:, b2 + 1] - csR[:, a2]
        f["dr_mature"] = np.where(cm > 0, dm / (cm + EPS), np.nan)
    else:
        f["dr_mature"] = np.full(S, np.nan)
    # --- how long the SKU has been selling
    nz = Q[:, :o + 1] > 0
    anyv = nz.any(axis=1)
    firstnz = np.where(anyv, nz.argmax(axis=1), o)
    lastnz = np.where(anyv, o - (nz[:, ::-1].argmax(axis=1)), -1)
    f["age_days"] = np.where(anyv, np.minimum(o - firstnz, 365), 0).astype(float)
    f["since_sale"] = np.where(anyv, o - lastnz, 999).astype(float)
    nzm = np.zeros(S)
    for k in range(1, 7):
        v = f["m%d" % k]; nzm += np.nan_to_num(v, nan=0.0) > 0
    f["nz_months6"] = nzm
    # --- scale used by the ratio models: a blend of what the SKU has been doing (always >= 1)
    parts = np.vstack([m1f, rate[30] * info["len_m"], rate[14] * info["len_m"]])
    f["scale"] = np.maximum(1.0, parts.mean(axis=0))
    # --- price (optional)
    if panel.price is not None:
        pn = panel.price[:, o]
        for k, w in (("p_chg30", 30), ("p_chg90", 90)):
            pp = panel.price[:, max(o - w, 0)]
            f[k] = np.where((pn > 0) & (pp > 0), pn / pp - 1.0, np.nan)
    else:
        f["p_chg30"] = np.full(S, np.nan); f["p_chg90"] = np.full(S, np.nan)
    f["cat"] = cat_codes.astype(float) if cat_codes is not None else np.full(S, np.nan)
    df = pd.DataFrame({k: f[k] for k in FEATURES})
    df["alive"] = (cs[:, o + 1] - cs[:, max(o - 89, 0)]) > 0           # sold anything in the last 90 days
    return df, info


def level(df):
    """Volume level used for the A / B / C segments: mean of the complete months we have (last 3), else the 30-day pace. A >= 100, B 30-99, C < 30 pieces a month."""
    m = df[["m1", "m2", "m3"]].to_numpy()
    mean3 = np.nanmean(np.where(np.isnan(m), np.nan, m), axis=1) if m.shape[1] else np.zeros(len(df))
    out = np.where(np.isnan(mean3), df["r30"].to_numpy() * 30.0, mean3)
    return out


def segment(df, a=100.0, b=30.0):
    lv = level(df)
    return np.where(lv >= a, "A", np.where(lv >= b, "B", "C"))


def target_total(panel, start, end):
    """Confirmed pieces per SKU between two dates (inclusive) — NaN when that range is not fully inside the data."""
    if start < panel.dates[0] or end > panel.dates[-1]: return None
    si = int((start - panel.dates[0]).days); ei = int((end - panel.dates[0]).days)
    return panel.Q[:, si:ei + 1].sum(axis=1)


def category_codes(panel):
    if panel.cat is None: return None
    return pd.Categorical(panel.cat).codes.astype(float)
