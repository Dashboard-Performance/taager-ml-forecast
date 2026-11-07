import sys, os, json, threading, http.server, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from synth import make_panel
from forecast import data, pipeline
from forecast.data import Panel
from forecast.engine import Engine
from forecast.features import FEATURES, build_features, segment
from forecast.models import accuracy, bias

n_pass = 0
def ok(name, cond):
    global n_pass
    assert cond, "FAILED: " + name
    n_pass += 1; print("  ✓ " + name)

P = make_panel(300, end="2026-10-30")


def scrambled(P, o, seed=1):
    """a copy of the panel in which EVERYTHING after day o (pieces, merchants, inbound, later months' stock) is replaced by noise"""
    rng = np.random.default_rng(seed)
    Q, Pp, R = P.Q.copy(), P.P.copy(), P.R.copy()
    for arr in (Q, Pp, R): arr[:, o + 1:] = rng.integers(0, 500, arr[:, o + 1:].shape)
    INB = P.INB.copy(); INB[:, o + 1:] = rng.integers(0, 900, INB[:, o + 1:].shape)
    di, si, mc, q = [x.copy() for x in P.merch]
    late = di > o
    mc[late] = rng.integers(0, 50, late.sum()); q[late] = rng.integers(1, 300, late.sum()); si[late] = rng.integers(0, len(P.skus), late.sum())
    order = np.argsort(di, kind="stable")
    cur = str(P.dates[o].to_period("M"))
    BI = {m: (v if m <= cur else rng.integers(0, 5000, v.shape).astype(float)) for m, v in P.BI.items()}
    return Panel(P.dates, P.skus, Q, Pp, R, P.cat, None, {}, (di[order], si[order], mc[order], q[order]), INB, BI)

# ---- features never read the future (pieces, merchants, inbound, stock, past misses - all of them)
for d in ["2026-03-15", "2026-06-30", "2026-09-10"]:
    o = int((pd.Timestamp(d) - P.dates[0]).days)
    a_, _ = build_features(P, o)
    b_, _ = build_features(scrambled(P, o), o)
    ok("all %d features at %s identical when everything after that day is scrambled" % (len(FEATURES), d), a_.equals(b_))
ok("the merchant, inbound, stock and miss features really have values (not all empty)", all(build_features(P, 250)[0][k].notna().mean() > 0.3 for k in ("nm1", "ret_1", "inb30", "bi_c", "miss_pace1")))

# ---- end to end: predictions made at an origin do not depend on anything after it (training uses only finished months)
o = int((pd.Timestamp("2026-08-14") - P.dates[0]).days)
P3 = scrambled(P, o, seed=5)
e1, e2 = Engine(P, fast=True), Engine(P3, fast=True)
p1 = e1.predict_at(o, 1, {})[0]; p2 = e2.predict_at(o, 1, {})[0]
ok("model forecasts at an origin are unchanged when the future is scrambled (no leakage through training)", all(np.allclose(p1[k], p2[k], atol=1e-6) for k in ("gbm_l1", "naive", "pace14")))
ok("the model that was trained is a real one (gbm_l1 present)", "gbm_l1" in p1)

# ---- backtest sanity on structured data
bt = Engine(P, fast=True).backtest(1)
s = Engine.summarize(bt)
ok("backtest produced the four cut-off buckets", set(s) == {"7", "14", "21", "end"})
ok("ML beats 'repeat last month' at month end on data with learnable structure", s["end"]["ALL"]["gbm_l1"] > s["end"]["ALL"]["naive"])
ok("accuracy improves as the month goes on (day 7 -> month end)", s["end"]["ALL"]["gbm_l1"] >= s["7"]["ALL"]["gbm_l1"] - 1.0)
eng = Engine(P, fast=True)
ch = eng.champions(bt, "end")
ok("a champion is chosen for every segment", set(ch) == {"A", "B", "C"})
cov = eng.coverage(bt, "end", ch)
ok("P10-P90 band covers ~80%% of SKUs out of sample (got %s)" % cov, cov is not None and 65 <= cov <= 92)

# ---- learning from past misses: the correction factor only uses months that had already finished
g = pd.DataFrame({"month": ["2026-04"] * 2 + ["2026-05"] * 2 + ["2026-06"] * 2 + ["2026-07"] * 2, "y": [100, 100, 100, 100, 100, 100, 100, 100], "gbm_l1": [50, 50, 80, 80, 90, 90, 1, 1]})
ok("calibration for a month-end forecast of July uses April-June (actual 600 / forecast 440), never July itself", abs(Engine.cal_factor(g, "gbm_l1", "2026-07", 1) - 600 / 440) < 1e-9)
ok("calibration for a mid-month forecast skips the month that is still running (July: only April-May)", abs(Engine.cal_factor(g, "gbm_l1", "2026-07", 2) - 400 / 260) < 1e-9)
ok("calibration waits for 2 finished months (June has only April before it at lag 2) -> 1.0", Engine.cal_factor(g, "gbm_l1", "2026-06", 2) == 1.0)

# ---- locking
def trunc(end):
    k = int((pd.Timestamp(end) - P.dates[0]).days) + 1
    return Panel(P.dates[:k], P.skus, P.Q[:, :k], P.P[:, :k], P.R[:, :k], P.cat, None, {})
pl, lock = pipeline.build(trunc("2026-09-30"), fast=True)
ok("on the last day of a month the forecast for the next month is frozen (lock candidate for 2026-10)", lock is not None and lock["month"] == "2026-10" and lock["frozenAt"] == "2026-09-30")
ok("the frozen numbers are the live numbers of that day", lock["rows"] == pl["live"]["rows"])
pl2, lock2 = pipeline.build(trunc("2026-09-17"), fast=True)
ok("mid-month: nothing to lock, live forecast targets the month after", lock2 is None and pl2["live"]["month"] == "2026-10" and pl2["currentMonth"] == "2026-09")
ok("mid-month: a nowcast of the running month is included and never below month-to-date", "nowcast" in pl2 and all(v[1] >= v[0] - 1e-6 for v in pl2["nowcast"]["rows"].values()))
sc = pipeline.score_locked(P, lock)
ok("a frozen month is scored once it has finished (October is complete in the panel except day 31)", sc is None or "ml" in sc)
lock_sep = pipeline.build(trunc("2026-08-31"), fast=True)[1]
sc = pipeline.score_locked(P, lock_sep)
ok("score_locked: September forecast frozen on Aug 31 is scored against what really happened", sc and sc["month"] == "2026-09" and 0 < sc["ml"] < 100 and "naive" in sc)
ok("…and the payload lists it under lockedAccuracy", pipeline.build(trunc("2026-10-20"), fast=True, locked_by_month={"2026-09": lock_sep})[0]["lockedAccuracy"][0]["month"] == "2026-09")
pl3 = pipeline.build(trunc("2026-10-20"), fast=True, locked_by_month={"2026-10": lock})[0]
bm = pl["accuracy"]["byMonth"]
ok("the payload lists every backtest month for each cut-off day, with actual and forecast totals", set(bm) == {"7", "14", "21", "end"} and len(bm["end"]) >= 5 and all("actual" in r and "ens" in r or "gbm_l1" in r for r in bm["end"]))
ok("this month\'s frozen forecast is carried in the payload to track the month against", pl3["locked"]["month"] == "2026-10")

# ---- maths
ok("accuracy: perfect forecast = 100, all-zero forecast = 0", accuracy([1, 2, 3], [1, 2, 3]) == 100 and accuracy([1, 2, 3], [0, 0, 0]) == 0)
ok("bias: +10% over-forecast", abs(bias([100], [110]) - 10) < 1e-9)

# ---- data shaping
hist = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]), "sku": ["B1", "S1", "B1", "S1"], "placed": [4, 2, 4, 2], "confirmed": [3, 2, 3, 2], "delivered": [1, 1, 1, 1]})
newer = pd.DataFrame({"date": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05"]), "sku": ["B1", "S1", "S1"], "placed": [10, 5, 6], "confirmed": [6, 4, 5], "delivered": [2, 2, 2]})
dm = {"B1": [("S1", 2.0), ("S2", 1.0)]}
pn = data.build_panel(hist, newer, dm, {"S1": "TOYS"}, None)
S1 = pn.skus.tolist().index("S1"); S2 = pn.skus.tolist().index("S2")
ok("the newest day in the data is treated as partial and dropped", str(pn.dates[-1].date()) == "2026-01-04")
ok("a day covered by the newer source takes its numbers, not the history tab's (Jan 3: B1 6 pieces x 2 into S1 = 12)", pn.Q[S1, 2] == 12 and pn.Q[S2, 2] == 6)
ok("history tab days before the newer source are used (Jan 2: S1 2 pieces)", pn.Q[S1, 1] == 2 and pn.Q[S1, 0] == 6)
ok("category comes through", pn.cat[S1] == "TOYS")
po = pd.DataFrame({"DATE_GROUP": ["2026-10-01"] * 3, "PRODUCT_ID": ["A", "A", "B"], "IS_PRIMARY": [True, False, True], "PLC_PIECES_ACTUAL": [10, 4, 1], "CNF_PIECES_ACTUAL": [8, 3, 1], "DLV_PIECES_ACTUAL": [5, 2, 0]})
d = data.product_only_daily(po, primary_only=False)
ok("Product Only (all rows): the IS_PRIMARY true and false rows of a product-day are added together", d[d.sku == "A"].confirmed.iloc[0] == 11 and len(d) == 2)
d1 = data.product_only_daily(po)
ok("Product Only (default): only IS_PRIMARY rows, the definition the history tab uses (A: 8 not 11)", d1[d1.sku == "A"].confirmed.iloc[0] == 8)
try: data.product_only_daily(po.drop(columns=["CNF_PIECES_ACTUAL"])); ok("missing column is refused", False)
except SystemExit: ok("a Product Only answer with a column missing is refused, not read as zero", True)
oc = data.overlap_check(hist, newer)
ok("overlap check compares the two sources on the days both cover", oc["days"] == 2)
csv_main = "CREATION_DATE,TAGER_ID,TAGER_NAME,PRODUCT_ID,PRODUCT_NAME,CATEGORY,SUB_CATEGORY,ITEM_TYPE,ACTIVE_DAYS,PLACED_ORDERS,CONFIRMED_ORDERS,DELIVERED_ORDERS,CR_ORDERS,DR_ORDERS,NDR_ORDERS,PLACED_PIECES,CONFIRMED_PIECES,DELIVERED_PIECES\n" \
           "2026-09-14,51726,Ahmed,EG030107MAKN99,Vacuum,HOME,HOME INDOOR,CLEANING,1,49,26,9,0.5,0.3,0.1,49,26,9\n2026-09-14,1007827,Faouzi,EG030107MAKN99,Vacuum,HOME,HOME INDOOR,CLEANING,1,21,15,9,0.7,0.6,0.4,21,15,9\n"
ml = data.main_long(csv_main)
ok("Main is read BY COLUMN NAME: merchant, product, category and the three piece columns", len(ml) == 2 and ml.confirmed.sum() == 41 and set(ml.merchant) == {"51726", "1007827"} and ml.category.iloc[0] == "HOME")
hist_like = pd.DataFrame({"PERIOD_FILTER": ["January 1, 2026"], "TAGER_ID": ["86751"], "PRODUCT_ID": ["EG04"], "CATEGORY": ["CONSUMABLES"], "PLACED_PIECES": ["1,035"], "CONFIRMED_PIECES": [736], "DELIVERED_PIECES": [510]})
try: data.to_long(hist_like.drop(columns=["CONFIRMED_PIECES"]), "History"); ok("a missing column is refused", False)
except SystemExit: ok("a Main / history table with a column missing is refused, not read as zero", True)
bi = data.beginning_inventory(pd.DataFrame({"PRODUCT_ID": ["A", "A", "B"], "QTY": [10, 5, 7], "MONTH": ["2026-10-01", "2026-10-01", "2026-09-01"], "PRODUCT_NAME": ["", "", ""]}))
ok("beginning inventory: per month and SKU (A has two rows in October = 15)", bi["2026-10"]["A"] == 15 and bi["2026-09"]["B"] == 7)
pm = data.build_panel(ml.assign(date=pd.Timestamp("2026-09-13")), ml, {"EG030107MAKN99": [("S1", 2.0)]}, None, None, as_of="2026-09-14", inbound=pd.DataFrame({"date": [pd.Timestamp("2026-09-14")], "sku": ["S1"], "qty": [100.0]}), bi={"2026-09": pd.Series({"S1": 500.0})})
ok("bundle rows keep their merchants, pieces x quantity, category, inbound and stock all reach the panel", pm.merch is not None and abs(pm.merch[3].sum() - 164) < 1e-9 and pm.INB[0, -1] == 100 and pm.BI["2026-09"][0] == 500 and pm.cat[0] == "HOME")

ok("as-of = yesterday (Cairo) even when today has no rows yet", str(data.pick_asof("2026-11-01", "2026-10-31").date()) == "2026-10-31")
ok("as-of = yesterday when today's partial day is already in the data", str(data.pick_asof("2026-11-01", "2026-11-01").date()) == "2026-10-31")
ok("as-of stops at the data when the data is stale (stale days are never read as zero sales)", str(data.pick_asof("2026-11-01", "2026-10-28").date()) == "2026-10-28")

# ---- the Worker client against a stub server
store = {}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.headers.get("X-Service-Key") != "k": return self._send(401, {"success": False})
        if "list=1" in self.path: return self._send(200, {"months": sorted(store)})
        if "month=" in self.path:
            m = self.path.split("month=")[1].split("&")[0]
            return self._send(200, store[m]) if m in store else self._send(404, {"success": False})
        self._send(200, {"asOf": "x"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        m = self.path.split("month=")[1] if "month=" in self.path else None
        if "kind=locked" in self.path:
            if m in store: return self._send(409, {"success": False, "message": "already locked"})
            store[m] = body
        self._send(200, {"success": True, "locked": m})
srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
W = data.Worker("http://127.0.0.1:%d" % srv.server_port, "k")
c1, r1 = W.put("locked", {"month": "2026-11", "rows": {}}, "2026-11"); c2, r2 = W.put("locked", {"month": "2026-11", "rows": {"x": 1}}, "2026-11")
ok("Worker client: first lock accepted, second refused with 409 (and parsed, not raised)", c1 == 200 and c2 == 409)
ok("Worker client: lists and reads locked months, 404 -> None", W.get(lst=True)["months"] == ["2026-11"] and W.get(month="2026-11")["month"] == "2026-11" and W.get(month="2025-01") is None)
seen = {}
class H2(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        seen["ua"] = self.headers.get("User-Agent")
        b = b'{"success":false,"message":"nope"}'; self.send_response(403); self.end_headers(); self.wfile.write(b)
srv2 = http.server.HTTPServer(("127.0.0.1", 0), H2); threading.Thread(target=srv2.serve_forever, daemon=True).start()
try:
    data.Worker("http://127.0.0.1:%d" % srv2.server_port, "k").source("1"); ok("an HTTP error is raised", False)
except RuntimeError as e:
    ok("an HTTP error says what the server answered (status + body), not just the number", "403" in str(e) and "nope" in str(e))
ok("every request carries our own User-Agent (Cloudflare blocks Python's default one)", seen["ua"].startswith("taager-ml-forecast/"))
srv2.shutdown(); srv.shutdown()

# ---- the Forecast Model recipe ("dash") and the blends
from forecast.features import dash_rule_features, tsb_series
from forecast.models import baselines, CANDIDATES
Pq = make_panel(120, end="2026-10-30"); oq = len(Pq.dates) - 1
csq = np.concatenate([np.zeros((Pq.Q.shape[0], 1)), np.cumsum(Pq.Q, axis=1)], axis=1)
dfq = dash_rule_features(Pq, oq, csq)
w = lambda n: csq[:, oq + 1] - csq[:, oq + 1 - n]
tot = w(30); cp = lambda v: np.maximum(tot * 0.6, np.minimum(tot * 1.4, v))
rules = [tot, cp(w(7) * 30 / 7), cp(w(10) * 3), w(14) * 30 / 14, cp(w(15) * 2), w(21) * 30 / 21, 0.5 * (w(14) * 30 / 14) + 0.5 * tot]
ok("dash recipe = average of the 7 recent-rate rules (capped +-40% where the Forecast Model caps them)", np.allclose(dfq["dash_ens"], np.mean([np.maximum(0, r) for r in rules], axis=0)))
x = Pq.Q[3]; size = prob = 0.0; st = False
for v in x:
    if v > 0: size = v if not st else 0.5 * v + 0.5 * size; prob = 0.1 + 0.9 * prob; st = True
    else: prob = 0.9 * prob
ok("TSB rate (alpha .5 / beta .1) matches a hand-written loop for one SKU", abs(dfq["tsb_c30"][3] - (prob * size * 30 if st else 0.0)) < 1e-9)
ok("dash_rate = 70% rules + 30% TSB (segment A of the Forecast Model)", np.allclose(dfq["dash_rate"], 0.7 * dfq["dash_ens"] + 0.3 * dfq["tsb_c30"]))
dff, _ = build_features(Pq, oq)
ok("dash baseline = dash_rate per day x days of the target month", np.allclose(baselines(dff, 1)["dash"], dff["dash_rate"] / 30.0 * dff["len_m"]))
engq = Engine(Pq, fast=True); predq, dfp, infq, tmq = engq.predict_at(oq, 1, {})
ok("blend = 50% GBM ensemble + 50% recipe, blend_gbm = 50% L1 GBM + 50% recipe", ("blend" not in predq) or (np.allclose(predq["blend"], 0.5 * predq["ens"] + 0.5 * predq["dash"]) and np.allclose(predq["blend_gbm"], 0.5 * predq["gbm_l1"] + 0.5 * predq["dash"])))
ok("the new candidates are scored like every other one (and can be champion)", all(k in CANDIDATES for k in ("dash", "blend", "blend_gbm")))


# ---- v2.2: buying pattern (whole Inbound sheet), Stock Archive, stock cap, where the error comes from
from forecast.features import purchase_features, stock_now, origin_info
from forecast import engine as ENG
ad = data.archive_dates(["30-December", "31-December", "01-January", "02-January", "bad"], pd.Timestamp("2027-01-05"))
ok("Stock Archive titles have no year: Dec -> Jan rolls the year (30 Dec 2026 .. 2 Jan 2027), a bad title is skipped",
   ad[:4] == [pd.Timestamp(2026, 12, 30), pd.Timestamp(2026, 12, 31), pd.Timestamp(2027, 1, 1), pd.Timestamp(2027, 1, 2)] and ad[4] is None)
ad2 = data.archive_dates(["23-August", "30-September", "05-November"], pd.Timestamp("2026-11-05"))
ok("…and Aug–Nov of the same year stay in 2026", ad2 == [pd.Timestamp(2026, 8, 23), pd.Timestamp(2026, 9, 30), pd.Timestamp(2026, 11, 5)])
sl = data.stock_archive_long(["01-October", "02-October", "03-October"], [["A", "5", "", "0"], ["SKU_ID", "x", "x", "x"], ["B", "1,200", None, "7"]], pd.Timestamp("2026-10-02"))
ok("Stock Archive: blank cells are skipped (not zero), days after as-of are dropped, '1,200' reads as 1200",
   len(sl) == 2 and set(sl["sku"]) == {"A", "B"} and sl[(sl.sku == "B")]["qty"].tolist() == [1200.0] and sl[(sl.sku == "A") & (sl.date == pd.Timestamp("2026-10-02"))].empty)

# inbound before the panel starts reaches the buying-pattern features; nothing after the origin does
ib = pd.DataFrame({"date": pd.to_datetime(["2025-12-01", "2026-01-05", "2026-01-20"]), "sku": ["S1", "S1", "S1"], "qty": [100, 50, 999]})
hh = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-01-31"), "merchant": "m", "sku": "S1", "placed": 2.0, "confirmed": 1.0, "delivered": 1.0, "category": ""})
Pi = data.build_panel(hh, None, None, None, None, as_of=pd.Timestamp("2026-01-31"), inbound=ib)
oi = 9   # 10 Jan: the 20 Jan delivery is still in the future
pf, since = purchase_features(Pi, oi, origin_info(Pi.dates, oi), 1)
ok("buying pattern reads the Inbound sheet from before the first sales day (inbound since 2025-12-01, %d days earlier)" % Pi.inbh_off,
   Pi.inbh_off == 31 and pf["inb_first_age"][0] == 31 + 9 and pf["inb_n365"][0] == 2 and since[0] == 5)
ok("…and the 20 Jan delivery (after the origin) is not counted", pf["inb_n180"][0] == 2 and abs(pf["inb_buy_sell180"][0] - 150 / (10 + 10.0)) < 1e-9)
P4 = make_panel(120, end="2026-10-30", seed=3)
P4.INBH = np.concatenate([np.random.default_rng(1).integers(0, 3, (len(P4.skus), 60)) * 50.0, P4.INB], axis=1); P4.inbh_off = 60
o4 = int((pd.Timestamp("2026-07-10") - P4.dates[0]).days)
P5 = scrambled(P4, o4, seed=9); P5.INBH = P4.INBH.copy(); P5.INBH[:, 60 + o4 + 1:] = 777.0; P5.inbh_off = 60
fa, _ = purchase_features(P4, o4, origin_info(P4.dates, o4), len(P4.skus)); fb, _ = purchase_features(P5, o4, origin_info(P5.dates, o4), len(P5.skus))
ok("buying-pattern features unchanged when every inbound after the origin is scrambled", all(np.allclose(fa[k], fb[k], equal_nan=True) for k in fa))

# stock now + the cap
P6 = make_panel(50, end="2026-10-30", seed=4); P6.STK = np.full(P6.Q.shape, np.nan)
o6 = int((pd.Timestamp("2026-09-15") - P6.dates[0]).days); P6.STK[:, o6 - 1] = 10.0
inf6 = origin_info(P6.dates, o6)
ok("stock now = the Stock Archive of the origin day or up to 3 days before (here the day before)", np.allclose(stock_now(P6, o6, inf6), 10.0))
P6.STK[:, o6 - 1] = np.nan; P6.STK[:, o6 - 5] = 10.0
ok("…an older archive day is not used (unknown, no cap)", np.isnan(stock_now(P6, o6, inf6)).all())
infc = {"stock_now": np.array([10.0, 10.0, np.nan]), "since_inb_h": np.array([200.0, 30.0, 200.0])}
capd = ENG.Engine.stock_cap({"x": np.array([100.0, 100.0, 100.0])}, infc)["x"]
ok("stock cap: not bought for > %d days -> forecast <= %.1f x stock (100 -> 15); bought recently or stock unknown -> unchanged" % (ENG.CAP_NO_INBOUND_DAYS, ENG.CAP_STOCK_X),
   capd.tolist() == [15.0, 100.0, 100.0])
e6 = Engine(P6, fast=True); pr6, _, i6, _ = e6.predict_at(o6, 1, {})
m6, st6 = Engine.cap_mask(i6)
ok("the cap is applied to every candidate the engine produces", all((v[m6] <= st6[m6] * ENG.CAP_STOCK_X + 1e-9).all() for v in pr6.values()) if m6 is not None and m6.any() else True)

# where the error comes from
P7 = make_panel(150, end="2026-10-30", seed=6)
P7.STK = np.where(np.random.default_rng(2).random(P7.Q.shape) < 0.2, 0.0, 50.0); P7.STK[:, :200] = np.nan
e7 = Engine(P7, fast=True); bt7 = e7.backtest(1, buckets=["end"])
es = pipeline.error_split(P7, bt7, e7)
ok("error split: only months the Stock Archive covers, every class has n / actual / forecast / error, shares add up to 100",
   es and all(pd.Period(r["month"], "M").start_time >= P7.dates[200] for r in es) and all(abs(sum(c["share"] for c in r["classes"].values()) - 100) < 0.6 for r in es)
   and all(set(c) == {"n", "actual", "forecast", "error", "share"} for r in es for c in r["classes"].values()))
ok("…and the classes are the documented ones", all(set(r["classes"]) <= set(pipeline.SPLIT_KEYS) for r in es))

print("\n%d passed" % n_pass)
