"""Walk-forward backtest, champion selection, and the daily run (live next-month forecast, the locked number, current-month nowcast)."""
import numpy as np
import pandas as pd

from .features import FEATURES, build_features, category_codes, origin_info, segment, target_total
from .models import (BASELINES, CAL_BASES, CANDIDATES, GBMS, accuracy, apply_band, band_quantiles, baselines, bias, fit, predict)

BUCKETS = [7, 14, 21, "end"]
SEGS = ["A", "B", "C"]


class Engine:
    def __init__(self, panel, stride=3, min_train_months=2, fast=False):
        self.p, self.stride, self.min_train_months, self.fast = panel, stride, min_train_months, fast
        self.dates = panel.dates
        self.cc = category_codes(panel)
        self._feat = {}
        self.origins = self._origins()

    # ---- origins
    def _origins(self):
        out = []
        per = self.dates.to_period("M")
        for C in pd.period_range(per[0] + 0, per[-1], freq="M"):
            s = C.start_time.normalize(); e = C.end_time.normalize()
            for d in sorted(set(list(range(3, C.days_in_month, self.stride)) + [7, 14, 21, C.days_in_month])):
                day = s + pd.Timedelta(days=d - 1)
                if day > self.dates[-1] or day < self.dates[0]: continue
                if d == C.days_in_month and day != e: continue
                out.append(int((day - self.dates[0]).days))
        return sorted(set(out))

    def feat(self, o):
        if o not in self._feat:
            self._feat[o] = build_features(self.p, o, self.cc)
        return self._feat[o]

    def bucket_of(self, info):
        if info["d"] >= info["len_c"]: return "end"
        return min([7, 14, 21], key=lambda b: abs(b - info["d"]))

    # ---- training sets
    def _train_rows(self, o_f, offset, d_filter=None):
        """Every earlier origin whose target month had already finished by day o_f — never anything the forecast could not have known."""
        Xs, ys, sc = [], [], []
        months = set()
        for o in self.origins:
            if o >= o_f: break
            df, info = self.feat(o)
            if offset == 1: s, e = info["m_first"], info["m_last"]
            else: s, e = info["c_first"], info["c_last"]
            if e > self.dates[o_f]: continue
            y = target_total(self.p, s, e)
            if y is None: continue
            keep = df["alive"].to_numpy()
            Xs.append(df.loc[keep, FEATURES]); ys.append(y[keep]); sc.append(df.loc[keep, "scale"].to_numpy()); months.add(str(s.to_period("M")))
        if not Xs: return None, None, None, 0
        return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(sc), len(months)

    def _models(self, o_f, offset, cache):
        # training depends only on which target months are already finished, so a whole month of origins shares one fit
        info = origin_info(self.dates, o_f)
        key = (offset, str(info["C"]) if info["d"] < info["len_c"] else str(info["C"]) + "e")
        if key in cache: return cache[key]
        X, y, sc, nm = self._train_rows(o_f, offset)
        res = {"models": {}, "months": nm, "rows": 0 if X is None else len(X)}
        if X is not None and nm >= self.min_train_months:
            res["models"] = fit(X, y, sc, ("gbm_l1",) if (offset == 0 or self.fast) else GBMS)
        cache[key] = res
        return res

    # ---- predictions at one origin
    def predict_at(self, o_f, offset, cache):
        df, info = self.feat(o_f)
        base = baselines(df, offset)
        tm = self._models(o_f, offset, cache)
        preds = dict(base)
        alive = df["alive"].to_numpy()
        if tm["models"]:
            g = predict(tm["models"], df[FEATURES], df["scale"].to_numpy())
            for k, v in g.items(): preds[k] = np.where(alive, v, 0.0)
        for k in BASELINES: preds[k] = np.where(alive, preds[k], 0.0)
        return preds, df, info, tm

    # ---- the backtest
    def backtest(self, offset=1, buckets=None):
        buckets = buckets or (BUCKETS if offset == 1 else [7, 14, 21])
        cache, rows = {}, []
        for o in self.origins:
            df0, info0 = self.feat(o)
            b = self.bucket_of(info0)
            if b not in buckets or self.bucket_exact(info0) is None: continue
            s, e = (info0["m_first"], info0["m_last"]) if offset == 1 else (info0["c_first"], info0["c_last"])
            y = target_total(self.p, s, e)
            if y is None: continue
            preds, df, info, tm = self.predict_at(o, offset, cache)
            if not tm["models"]: continue
            seg = segment(df)
            r = pd.DataFrame({"month": str(info["C"] + offset), "bucket": str(b), "sku": np.arange(len(y)), "seg": seg, "y": y})
            for k, v in preds.items(): r[k] = v
            rows.append(r)
        bt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return self.add_calibration(bt) if offset == 1 else bt

    @staticmethod
    def cal_lag(bucket):
        """how many months back the newest FINISHED target month is: at month end the running month is done (1); mid-month it is not (2)"""
        return 1 if str(bucket) == "end" else 2

    @staticmethod
    def cal_factor(g, base, month, lag, last_k=3, min_prior=2):
        """what the model got wrong, learned: total actual / total forecast over the last K target months that had finished by then (1.0 if fewer than 2)"""
        per = pd.Period(month, "M")
        prior = sorted([m for m in g["month"].unique() if pd.Period(m, "M") <= per - lag])[-last_k:]
        if len(prior) < min_prior: return 1.0
        gp = g[g["month"].isin(prior)]
        f = gp[base].sum()
        return float(gp["y"].sum() / f) if f > 0 else 1.0

    def add_calibration(self, bt):
        if bt is None or not len(bt): return bt
        for base in CAL_BASES:
            if base not in bt: continue
            bt[base + "_cal"] = bt[base]
            for (b, s), g in bt.groupby(["bucket", "seg"]):
                lag = self.cal_lag(b)
                for m in g["month"].unique():
                    fac = self.cal_factor(g, base, m, lag)
                    idx = g.index[g["month"] == m]
                    bt.loc[idx, base + "_cal"] = bt.loc[idx, base].to_numpy() * fac
        return bt

    @staticmethod
    def bucket_exact(info):
        return info["d"] if info["d"] in (7, 14, 21) or info["d"] == info["len_c"] else None

    # ---- summary + champions
    @staticmethod
    def summarize(bt):
        if bt is None or not len(bt): return {}
        out = {}
        for b, g in bt.groupby("bucket"):
            segs = {}
            for sname, gg in [("ALL", g)] + [(s, g[g["seg"] == s]) for s in SEGS]:
                segs[sname] = {"n": int(len(gg)), **{k: round(accuracy(gg["y"], gg[k]), 1) for k in CANDIDATES if k in gg}}
                segs[sname]["bias"] = {k: round(bias(gg["y"], gg[k]), 1) for k in ("naive", "gbm_l1", "ens") if k in gg}
            out[b] = segs
        return out

    @staticmethod
    def per_month(bt, bucket, model):
        g = bt[bt["bucket"] == bucket]
        return {m: round(accuracy(x["y"], x[model]), 1) for m, x in g.groupby("month") if model in x}

    @staticmethod
    def champions(bt, bucket, last_k=3):
        """Per volume segment: the candidate with the best accuracy over the last K finished months (ties -> the simpler one)."""
        out = {}
        if bt is None or not len(bt): return {s: "ma3" for s in SEGS}
        g = bt[bt["bucket"] == str(bucket)]
        months = sorted(g["month"].unique())[-last_k:]
        g = g[g["month"].isin(months)]
        for s in SEGS:
            gs = g[g["seg"] == s]
            if not len(gs): out[s] = "ma3"; continue
            best, ba = "ma3", -1e9
            for k in CANDIDATES:      # CANDIDATES lists the simple ones first
                if k not in gs: continue
                a = accuracy(gs["y"], gs[k])
                if not np.isnan(a) and a > ba + 1e-9: best, ba = k, a
            out[s] = best
        return out

    def bands(self, bt, bucket, champ, last_k=6):
        """Residual quantiles per segment from the champion's own past errors -> P10 / P90."""
        out = {}
        if bt is None or not len(bt): return {s: None for s in SEGS}
        g = bt[bt["bucket"] == str(bucket)]
        months = sorted(g["month"].unique())[-last_k:]
        g = g[g["month"].isin(months)]
        for s in SEGS:
            gs = g[g["seg"] == s]; k = champ.get(s)
            out[s] = band_quantiles(gs["y"], gs[k]) if k in gs and len(gs) else None
        return out

    def coverage(self, bt, bucket, champ):
        """Out-of-sample check of the band: calibrate on the earlier months, count how many of the last month's SKUs fall inside P10..P90 (should be ~80%)."""
        g = bt[bt["bucket"] == str(bucket)]
        months = sorted(g["month"].unique())
        if len(months) < 3: return None
        last, earlier = months[-1], g[g["month"] != months[-1]]
        tot = hit = 0
        for s in SEGS:
            k = champ.get(s); ge = earlier[earlier["seg"] == s]; gl = g[(g["month"] == last) & (g["seg"] == s)]
            if k not in ge or not len(gl): continue
            zq = band_quantiles(ge["y"], ge[k]);
            if zq is None: continue
            lo, hi = apply_band(gl[k].to_numpy(), zq)
            y = gl["y"].to_numpy(); tot += len(y); hit += int(((y >= lo - 1e-9) & (y <= hi + 1e-9)).sum())
        return round(100.0 * hit / tot, 1) if tot else None

    # ---- today's numbers
    def forecast_now(self, bt):
        """Live forecast for next month, made with everything known on the last observed day."""
        o = len(self.dates) - 1
        cache = {}
        preds, df, info, tm = self.predict_at(o, 1, cache)
        b = self.bucket_of(info)
        seg = segment(df)
        # the same "learn from the misses" correction, from the model's own recent finished months, per segment
        if bt is not None and len(bt):
            gb = bt[bt["bucket"] == str(b)]
            for base in CAL_BASES:
                if base in preds:
                    fac = np.ones(len(df))
                    for s_ in SEGS:
                        gs = gb[gb["seg"] == s_]
                        if len(gs): fac[seg == s_] = self.cal_factor(gs, base, str(info["M"]), self.cal_lag(b))
                    preds[base + "_cal"] = preds[base] * fac
        champ = self.champions(bt, b); zq = self.bands(bt, b, champ)
        f = np.zeros(len(df)); lo = np.zeros(len(df)); hi = np.zeros(len(df))
        for s in SEGS:
            idx = seg == s; k = champ[s] if champ[s] in preds else "ma3"
            f[idx] = preds[k][idx]; lo[idx], hi[idx] = apply_band(f[idx], zq[s])
        alive = df["alive"].to_numpy()
        lo[~alive] = 0; hi[~alive] = 0
        imp = None
        gm = tm["models"].get("gbm_l1")
        if gm is not None:
            gain = gm.booster_.feature_importance("gain"); tot = gain.sum()
            if tot > 0: imp = sorted([(n, float(g_ / tot * 100)) for n, g_ in zip(FEATURES, gain)], key=lambda x: -x[1])[:15]
        return {"f": f, "lo": lo, "hi": hi, "seg": seg, "importance": imp, "champion": champ, "info": info, "bucket": str(b), "df": df, "preds": preds, "train": {"months": tm["months"], "rows": tm["rows"]}}

    def nowcast_now(self, bt_now):
        """Where the CURRENT month is heading (end-of-month total), from the month-to-date."""
        o = len(self.dates) - 1
        df, info = self.feat(o)
        if info["d"] >= info["len_c"]: return None
        cache = {}
        preds, df, info, tm = self.predict_at(o, 0, cache)
        b = self.bucket_of(info)
        champ = self.champions(bt_now, b) if bt_now is not None and len(bt_now) else {s: "pace_mtd" for s in SEGS}
        seg = segment(df); f = np.zeros(len(df))
        for s in SEGS:
            k = champ[s] if champ[s] in preds else "pace_mtd"
            f[seg == s] = preds[k][seg == s]
        f = np.maximum(f, df["mtd"].to_numpy())          # the month cannot end below what is already confirmed
        return {"f": f, "mtd": df["mtd"].to_numpy(), "champion": champ, "info": info}
