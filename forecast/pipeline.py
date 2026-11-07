"""One daily run -> the payload the dashboard shows."""
import datetime as _dt
import numpy as np
import pandas as pd

from . import __version__
from .engine import BUCKETS, Engine, SEGS
from .features import target_total
from .models import accuracy, bias


def _r1(x): return float(np.round(x, 1))


def _rows(panel, f, lo, hi, seg, extra=None):
    out = {}
    for i, s in enumerate(panel.skus):
        if f[i] <= 0 and (extra is None or extra[0][i] <= 0): continue
        row = [_r1(f[i]), _r1(lo[i]), _r1(hi[i]), seg[i]]
        if extra is not None: row += [_r1(e[i]) for e in extra]
        out[s] = row
    return out


def month_table(bt):
    """Every backtest month, as it would have been forecast on day 7 / 14 / 21 / month end: actual total, forecast totals and accuracy per candidate."""
    out = {}
    if bt is None or not len(bt): return out
    for b, g in bt.groupby("bucket"):
        rows = []
        for m, x in g.groupby("month"):
            r = {"month": m, "actual": _r1(x["y"].sum()), "n": int(len(x))}
            for k in ("ens", "ens_cal", "gbm_l1", "gbm_l1_cal", "pace14", "naive"):
                if k in x: r[k] = _r1(accuracy(x["y"], x[k])); r[k + "F"] = _r1(x[k].sum())
            rows.append(r)
        out[str(b)] = sorted(rows, key=lambda r: r["month"])
    return out


SPLIT_KEYS = ["under_inbound", "under_demand", "over_demand", "empty_restocked", "empty_no_restock", "stockout", "no_stock_data"]


def error_split(panel, bt, eng):
    """v2.2 — where the month-end forecast error comes from, for every scored month the Stock Archive covers (>= 20 days of it).
    The forecast used for a month is the champion chosen from the 3 months BEFORE it (what the model would really have shown)."""
    STK = getattr(panel, "STK", None)
    if STK is None or bt is None or not len(bt): return None
    g = bt[bt["bucket"] == "end"]; months = sorted(g["month"].unique()); out = []
    d0 = panel.dates[0]
    for m in months:
        M = pd.Period(m, "M"); a = int((M.start_time - d0).days); b = min(int((M.end_time.normalize() - d0).days) + 1, len(panel.dates))
        if a < 1 or b - a < 20: continue
        W = STK[:, a:b]; known = np.isfinite(W).sum(axis=1)
        if np.isfinite(W).any(axis=0).sum() < 20: continue
        prev = g[g["month"] < m]
        if not len(prev): continue
        ch = eng.champions(prev, "end")
        x = g[g["month"] == m]; f = np.zeros(len(x))
        for sg in SEGS:
            idx = (x["seg"] == sg).to_numpy(); k = ch.get(sg)
            if k in x: f[idx] = x[k].to_numpy()[idx]
        sk = x["sku"].to_numpy(); y = x["y"].to_numpy()
        so = STK[:, a - 1]
        for k_ in (2, 3, 4):
            if a - k_ >= 0: so = np.where(np.isfinite(so), so, STK[:, a - k_])
        so = so[sk]; oos = np.where(known[sk] > 0, (W[sk] == 0).sum(axis=1) / np.maximum(known[sk], 1), np.nan)
        inb = panel.INB[sk, a:b].sum(axis=1) if panel.INB is not None else np.zeros(len(sk))
        cls = np.where(~np.isfinite(so), "no_stock_data",
              np.where((oos >= 0.3) & (f > y), "stockout",
              np.where((so <= 0) & (inb > 0), "empty_restocked",
              np.where(so <= 0, "empty_no_restock",
              np.where((y > f) & (inb > 0), "under_inbound",
              np.where(y > f, "under_demand", "over_demand"))))))
        ae = np.abs(y - f); tot = ae.sum()
        rows = {}
        for c in SPLIT_KEYS:
            i = cls == c
            if i.any(): rows[c] = {"n": int(i.sum()), "actual": _r1(y[i].sum()), "forecast": _r1(f[i].sum()), "error": _r1(ae[i].sum()), "share": _r1(ae[i].sum() / tot * 100) if tot > 0 else 0.0}
        out.append({"month": m, "accuracy": _r1(accuracy(y, f)), "classes": rows})
    return out or None


def is_last_day_of_month(ts): return ts.normalize() == ts.to_period("M").end_time.normalize()


def build(panel, fast=False, checks=None, locked_by_month=None):
    """Returns (payload, lock_candidate_or_None). `locked_by_month` = {'2026-10': locked payload} already stored in the Worker."""
    eng = Engine(panel, fast=fast)
    bt = eng.backtest(1)
    bt_now = eng.backtest(0)
    fc = eng.forecast_now(bt)
    info = fc["info"]; asof = panel.dates[-1]
    nc = eng.nowcast_now(bt_now)
    base = fc["preds"]
    payload = {
        "version": __version__, "generatedAt": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "asOf": str(asof.date()),
        "currentMonth": str(info["C"]), "dayOfMonth": info["d"], "daysInMonth": info["len_c"], "targetMonth": str(info["M"]),
        "data": panel.meta | (checks or {}),
        "features": len(__import__("forecast.features", fromlist=["FEATURES"]).FEATURES),
        "importance": [[n, round(v, 1)] for n, v in (fc.get("importance") or [])],
        "live": {"month": str(info["M"]), "bucket": fc["bucket"], "champion": fc["champion"], "train": fc["train"],
                 "rows": _rows(panel, fc["f"], fc["lo"], fc["hi"], fc["seg"], [base["naive"], base["ma3"], base["pace14"]])},
        "accuracy": {"next": Engine.summarize(bt), "now": Engine.summarize(bt_now),
                     "months": sorted(bt["month"].unique()) if len(bt) else [],
                     "perMonth": {"end": {k: Engine.per_month(bt, "end", k) for k in ("naive", "ma3", fc["champion"].get("A", "ma3"), "ens") if len(bt) and k in bt}} if len(bt) else {},
                     "byMonth": month_table(bt), "coverageP10P90": eng.coverage(bt, "end", eng.champions(bt, "end")) if len(bt) else None,
                     "champions": {str(b): eng.champions(bt, b) for b in BUCKETS},
                     "errorSplit": error_split(panel, bt, eng)},
        "stockCap": {"noInboundDays": __import__("forecast.engine", fromlist=["x"]).CAP_NO_INBOUND_DAYS, "stockX": __import__("forecast.engine", fromlist=["x"]).CAP_STOCK_X,
                     "skus": int(Engine.cap_mask(info)[0].sum()) if Engine.cap_mask(info)[0] is not None else 0},
    }
    if nc is not None:
        payload["nowcast"] = {"month": str(info["C"]), "champion": nc["champion"], "rows": {s: [_r1(nc["mtd"][i]), _r1(nc["f"][i])] for i, s in enumerate(panel.skus) if nc["f"][i] > 0}}
    lock = None
    if is_last_day_of_month(asof):
        lock = {"month": str(info["M"]), "frozenAt": str(asof.date()), "version": __version__, "champion": fc["champion"], "train": fc["train"],
                "rows": _rows(panel, fc["f"], fc["lo"], fc["hi"], fc["seg"], [base["naive"], base["ma3"], base["pace14"]])}
    # locked numbers that were frozen earlier: this month's (to track against) + how every finished one turned out
    lm = locked_by_month or {}
    cur = str(info["C"])
    if cur in lm: payload["locked"] = lm[cur]
    payload["lockedAccuracy"] = [score_locked(panel, p) for m, p in sorted(lm.items()) if score_locked(panel, p)]
    return payload, lock


def score_locked(panel, locked):
    """How a frozen forecast did once its month finished (None while the month is still running)."""
    m = pd.Period(locked["month"], "M")
    y = target_total(panel, m.start_time.normalize(), m.end_time.normalize())
    if y is None: return None
    idx = {s: i for i, s in enumerate(panel.skus)}
    f = np.zeros(len(y)); nv = f.copy(); ma = f.copy(); p14 = f.copy()
    for s, r in locked["rows"].items():
        i = idx.get(s)
        if i is None: continue
        f[i] = r[0]
        if len(r) >= 7: nv[i], ma[i], p14[i] = r[4], r[5], r[6]
    out = {"month": locked["month"], "frozenAt": locked.get("frozenAt"), "n": int(len(y)), "actual": _r1(y.sum()), "forecast": _r1(f.sum()),
           "ml": _r1(accuracy(y, f)), "bias": _r1(bias(y, f))}
    if nv.sum() > 0: out["naive"] = _r1(accuracy(y, nv)); out["ma3"] = _r1(accuracy(y, ma)); out["pace14"] = _r1(accuracy(y, p14))
    return out
