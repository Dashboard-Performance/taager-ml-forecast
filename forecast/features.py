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
    "nm1", "nm2", "nm_chg", "top1_1", "hhi_1", "ret_1", "nm_mtd", "nm14", "top1_mtd", "new_mtd",
    "inb30", "inb90", "inb_mtd", "since_inb", "inb30_m1", "bi_c", "bi_cover", "avail_pace",
    "inb_n180", "inb_n365", "inb_gap_med", "inb_due", "inb_mon6", "inb_buy_sell180", "inb_last_cover", "inb_first_age",
    "miss_pace1", "miss_pace2", "miss_ma3_1",
    "tsb_b30", "tsb_c30", "dash_ens", "dash_rate",
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



# ---------------------------------------------------------------- the Forecast Model's recipe (recent-rate rules + TSB), as features and as a forecast
def tsb_series(panel, alpha, beta):
    """TSB (Teunter-Syntetos-Babai) daily rate at every day, one causal pass: the value at day o reads days <= o only."""
    key = ("tsb", alpha, beta)
    if key in panel._cache: return panel._cache[key]
    Q = panel.Q; S, T = Q.shape
    pre = np.zeros((S, T)); size = np.zeros(S); prob = np.zeros(S); started = np.zeros(S, bool)
    for i in range(T):
        d = Q[:, i]; pos = d > 0
        size = np.where(pos & ~started, d, np.where(pos, alpha * d + (1 - alpha) * size, size))
        prob = np.where(pos, beta + (1 - beta) * prob, (1 - beta) * prob)
        started = started | pos
        pre[:, i] = np.where(started, prob * size, 0.0)
    panel._cache[key] = pre
    return pre


def dash_rule_features(panel, o, cs):
    """What the dashboard's Forecast Model computes from the last 30 days, per 30 days: the average of its recent-rate rules (segment A: 70% of it) and TSB (the other 30%)."""
    def w(n):
        a = max(o - n + 1, 0); return cs[:, o + 1] - cs[:, a]
    tot, l7, l10, l14, l15, l21 = w(30), w(7), w(10), w(14), w(15), w(21)
    cap = lambda v: np.maximum(tot * 0.6, np.minimum(tot * 1.4, v))
    rules = [tot, cap(l7 * 30 / 7), cap(l10 * 3), l14 * 30 / 14, cap(l15 * 2), l21 * 30 / 21, 0.5 * (l14 * 30 / 14) + 0.5 * tot]
    ens = np.mean([np.maximum(0.0, r) for r in rules], axis=0)
    tb = tsb_series(panel, 0.2, 0.2)[:, o] * 30.0; tc = tsb_series(panel, 0.5, 0.1)[:, o] * 30.0
    return {"tsb_b30": tb, "tsb_c30": tc, "dash_ens": ens, "dash_rate": 0.7 * ens + 0.3 * tc}


# ---------------------------------------------------------------- merchants (who sells the SKU)
def _mrows(panel, a, b):
    """merchant rows for day indexes a..b inclusive -> (sku_idx, merchant_code, pieces)"""
    di, si, mc, q = panel.merch
    lo = int(np.searchsorted(di, a, "left")); hi = int(np.searchsorted(di, b, "right"))
    return si[lo:hi], mc[lo:hi], q[lo:hi]


def _per_sku_merchant(panel, a, b):
    """pieces per (sku, merchant) between day indexes a..b -> (keys, pieces) with key = sku * 1e7 + merchant"""
    key = ("pm", a, b)
    if key in panel._cache: return panel._cache[key]
    si, mc, q = _mrows(panel, a, b)
    if len(si) == 0:
        res = (np.array([], dtype=np.int64), np.array([]))
    else:
        k = si * 10_000_000 + mc
        u, inv = np.unique(k, return_inverse=True)
        res = (u, np.bincount(inv, weights=q))
    panel._cache[key] = res
    return res


def _month_merchants(panel, pm, S):
    """for one complete month: number of merchants, share of the biggest one, concentration (HHI) per SKU"""
    keys, pcs = _per_sku_merchant(panel, *pm)
    n = np.zeros(S); top = np.zeros(S); tot = np.zeros(S); hhi = np.zeros(S)
    if len(keys):
        sk = keys // 10_000_000
        pos = pcs > 0
        sk, pc = sk[pos], pcs[pos]
        n = np.bincount(sk, minlength=S).astype(float)
        tot = np.bincount(sk, weights=pc, minlength=S)
        order = np.lexsort((pc, sk)); sk_o, pc_o = sk[order], pc[order]
        last = np.r_[sk_o[1:] != sk_o[:-1], True]
        top[sk_o[last]] = pc_o[last]
        hhi = np.bincount(sk, weights=(pc / np.maximum(tot[sk], EPS)) ** 2, minlength=S)
    share = np.where(tot > 0, top / np.maximum(tot, EPS), np.nan)
    return n, share, np.where(tot > 0, hhi, np.nan), tot


def merchant_features(panel, o, info, S):
    f = {k: np.full(S, np.nan) for k in ("nm1", "nm2", "nm_chg", "top1_1", "hhi_1", "ret_1", "nm_mtd", "nm14", "top1_mtd", "new_mtd")}
    if panel.merch is None: return f
    dates = panel.dates
    def bounds(k):
        pm = info["C"] - k; s_, e_ = pm.start_time.normalize(), pm.end_time.normalize()
        if s_ < dates[0] or e_ > dates[o]: return None
        return int((s_ - dates[0]).days), int((e_ - dates[0]).days)
    b1, b2 = bounds(1), bounds(2)
    if b1:
        n1, sh1, hh1, tot1 = _month_merchants(panel, b1, S)
        f["nm1"], f["top1_1"], f["hhi_1"] = n1, sh1, hh1
        if b2:
            n2, _, _, _ = _month_merchants(panel, b2, S)
            f["nm2"] = n2; f["nm_chg"] = np.where(n2 > 0, n1 / np.maximum(n2, 1), np.nan)
            k1, p1 = _per_sku_merchant(panel, *b1); k2, _ = _per_sku_merchant(panel, *b2)
            if len(k1):
                kept = np.isin(k1, k2)
                ret = np.bincount((k1 // 10_000_000)[kept], weights=p1[kept], minlength=S)
                f["ret_1"] = np.where(tot1 > 0, ret / np.maximum(tot1, EPS), np.nan)     # share of last month's pieces that came from merchants who were also selling the month before
    c0 = int((info["c_first"] - dates[0]).days); c0 = max(c0, 0)
    km, pm_ = _per_sku_merchant(panel, c0, o)
    if len(km):
        sk = km // 10_000_000
        f["nm_mtd"] = np.bincount(sk[pm_ > 0], minlength=S).astype(float)
        mtd_tot = np.bincount(sk, weights=pm_, minlength=S)
        if b1:
            k1, p1 = _per_sku_merchant(panel, *b1)
            # last month's biggest merchant of each SKU: how much of this month's pieces still comes from it; and how much comes from merchants who were not selling it last month
            top_key = {}
            if len(k1):
                sk1 = k1 // 10_000_000; order = np.lexsort((p1, sk1)); last = np.r_[sk1[order][1:] != sk1[order][:-1], True]
                tk = k1[order][last]
                is_top = np.isin(km, tk)
                top_p = np.bincount(sk[is_top], weights=pm_[is_top], minlength=S)
                f["top1_mtd"] = np.where(mtd_tot > 0, top_p / np.maximum(mtd_tot, EPS), np.nan)
            newm = ~np.isin(km, k1)
            new_p = np.bincount(sk[newm], weights=pm_[newm], minlength=S)
            f["new_mtd"] = np.where(mtd_tot > 0, new_p / np.maximum(mtd_tot, EPS), np.nan)
    a14 = max(o - 13, 0)
    k14, p14 = _per_sku_merchant(panel, a14, o)
    if len(k14): f["nm14"] = np.bincount((k14 // 10_000_000)[p14 > 0], minlength=S).astype(float)
    return f


def _inbound_history(panel):
    """(H, off): daily inbound per SKU over the WHOLE Inbound sheet (back to 2025), and how many days H starts before panel.dates[0]."""
    H = getattr(panel, "INBH", None)
    if H is not None: return H, int(getattr(panel, "inbh_off", 0) or 0)
    return panel.INB, 0


def purchase_features(panel, o, info, S):
    """v2.2 — what the buying team has been doing for this SKU (known at the origin: only inbound that already arrived). There is no
    inbound plan for next month, so the model learns from the PATTERN: how often we buy it, whether a buy is due, how much we buy vs sell."""
    keys = ("inb_n180", "inb_n365", "inb_gap_med", "inb_due", "inb_mon6", "inb_buy_sell180", "inb_last_cover", "inb_first_age")
    H, off = _inbound_history(panel)
    out = {k: np.full(S, np.nan) for k in keys}
    if H is None:
        return out, np.full(S, np.nan)
    t = o + off                                         # the origin in H's day index
    Hc = H[:, :t + 1]
    csH = np.concatenate([np.zeros((S, 1)), np.cumsum(Hc, axis=1)], axis=1)
    csQ = np.concatenate([np.zeros((S, 1)), np.cumsum(panel.Q[:, :o + 1], axis=1)], axis=1)
    sales180 = csQ[:, o + 1] - csQ[:, max(0, o + 1 - 180)]
    r30 = (csQ[:, o + 1] - csQ[:, max(0, o - 29)]) / 30.0
    out["inb_buy_sell180"] = (csH[:, t + 1] - csH[:, max(0, t + 1 - 180)]) / (sales180 + 10.0)
    since_h = np.full(S, 9999.0)
    mstart = [(info["C"] - k) for k in range(0, 6)]
    h0 = panel.dates[0] - pd.Timedelta(days=off)
    mon_lo = [int((m.start_time.normalize() - h0).days) for m in mstart]
    mon_hi = [int((m.end_time.normalize() - h0).days) for m in mstart]
    for s in range(S):
        e = np.flatnonzero(Hc[s] > 0)
        if not len(e):
            out["inb_n180"][s] = 0.0; out["inb_n365"][s] = 0.0; out["inb_mon6"][s] = 0.0; continue
        out["inb_n180"][s] = float((e > t - 180).sum()); out["inb_n365"][s] = float((e > t - 365).sum())
        out["inb_first_age"][s] = float(t - e[0])
        last = e[-1]; since_h[s] = float(t - last)
        gaps = np.diff(e[e > t - 365]); gaps = gaps[gaps > 3]          # one delivery spread over a few days counts once
        if len(gaps):
            gm = float(np.median(gaps)); out["inb_gap_med"][s] = gm; out["inb_due"][s] = since_h[s] / gm
        out["inb_last_cover"][s] = Hc[s, last] / (r30[s] + 0.1)
        out["inb_mon6"][s] = float(sum(((e >= lo) & (e <= hi)).any() for lo, hi in zip(mon_lo, mon_hi)))
    return out, since_h


def stock_now(panel, o, info):
    """Stock on hand at the origin: the Stock Archive (daily, from 23 Aug 2026) if it has that day (or one of the 3 before);
    at a month-end origin before the archive existed, the next month's beginning inventory (= stock at the end of this month). NaN = unknown."""
    S = panel.Q.shape[0]; now = np.full(S, np.nan)
    STK = getattr(panel, "STK", None)
    if STK is not None:
        for k in range(0, 4):
            if o - k >= 0: now = np.where(np.isfinite(now), now, STK[:, o - k])
    if info["d"] >= info["len_c"] and panel.BI:
        nb = panel.BI.get(str(info["C"] + 1))
        if nb is not None: now = np.where(np.isfinite(now), now, nb)
    return now


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
    # --- merchants: how many sell it, how concentrated, who stayed, who is new
    f.update(merchant_features(panel, o, info, S))
    # --- inbound that ALREADY happened (nothing about next month: there is no inbound plan for it)
    if panel.INB is not None:
        csI = np.concatenate([np.zeros((S, 1)), np.cumsum(panel.INB, axis=1)], axis=1)
        i30, _ = _win(csI, o, 30); i90, _ = _win(csI, o, 90)
        f["inb30"], f["inb90"] = i30, i90
        f["inb_mtd"] = csI[:, o + 1] - csI[:, c_start_idx]
        nzi = panel.INB[:, :o + 1] > 0
        anyi = nzi.any(axis=1)
        f["since_inb"] = np.where(anyi, nzi[:, ::-1].argmax(axis=1), 999).astype(float)
        f["inb30_m1"] = i30 / (m1f + 1.0)
    else:
        for k in ("inb30", "inb90", "inb_mtd", "since_inb", "inb30_m1"): f[k] = np.full(S, np.nan)
    # --- stock at the start of this month (known) vs how fast it is selling
    bi = (panel.BI or {}).get(str(info["C"]))
    if bi is not None:
        f["bi_c"] = bi
        f["bi_cover"] = bi / (rate[30] * 30.0 + 1.0)
        f["avail_pace"] = (np.nan_to_num(bi, nan=0.0) + np.nan_to_num(f["inb_mtd"], nan=0.0)) / (f["mtd_rr"] + 1.0)
    else:
        for k in ("bi_c", "bi_cover", "avail_pace"): f[k] = np.full(S, np.nan)
    # --- how wrong the simple forecasts were for this SKU in the last two months (learning from past misses)
    def pace_miss(k):
        pm = info["C"] - k - 1                      # the month whose total we compare ...
        s_, e_ = pm.start_time.normalize(), pm.end_time.normalize()
        pe = pm - 1                                 # ... with the 14-day pace known at the end of the month before it
        pe_end = pe.end_time.normalize()
        if s_ < dates[0] or e_ > dates[o] or pe_end < dates[13]: return np.full(S, np.nan)
        ei = int((pe_end - dates[0]).days); si_ = int((s_ - dates[0]).days); ee = int((e_ - dates[0]).days)
        r14 = (cs[:, ei + 1] - cs[:, ei - 13]) / 14.0
        actual = cs[:, ee + 1] - cs[:, si_]
        return np.log1p(actual) - np.log1p(r14 * pm.days_in_month)
    f["miss_pace1"] = pace_miss(0); f["miss_pace2"] = pace_miss(1)
    mm = np.vstack([f["m2"], f["m3"], f["m4"]])
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        ma_prev = np.nanmean(mm, axis=0)
    f["miss_ma3_1"] = np.where(np.isnan(f["m1"]) | np.isnan(ma_prev), np.nan, np.log1p(np.nan_to_num(f["m1"])) - np.log1p(np.nan_to_num(ma_prev)))
    f.update(dash_rule_features(panel, o, cs))
    f["cat"] = cat_codes.astype(float) if cat_codes is not None else np.full(S, np.nan)
    pf, since_h = purchase_features(panel, o, info, S); f.update(pf)
    info["since_inb_h"] = since_h; info["stock_now"] = stock_now(panel, o, info)
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
