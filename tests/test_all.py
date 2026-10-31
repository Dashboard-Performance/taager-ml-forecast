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

# ---- features never read the future
for d in ["2026-03-15", "2026-06-30", "2026-09-10"]:
    o = int((pd.Timestamp(d) - P.dates[0]).days)
    a, _ = build_features(P, o)
    P2 = Panel(P.dates, P.skus, P.Q.copy(), P.P.copy(), P.R.copy(), P.cat, None, {})
    rng = np.random.default_rng(1)
    for arr in (P2.Q, P2.P, P2.R): arr[:, o + 1:] = rng.integers(0, 500, arr[:, o + 1:].shape)
    b, _ = build_features(P2, o)
    ok("features at %s identical when every later day is scrambled" % d, a.equals(b))

# ---- end to end: predictions made at an origin do not depend on anything after it (training uses only finished months)
o = int((pd.Timestamp("2026-08-14") - P.dates[0]).days)
P3 = Panel(P.dates, P.skus, P.Q.copy(), P.P.copy(), P.R.copy(), P.cat, None, {})
P3.Q[:, o + 1:] = np.random.default_rng(5).integers(0, 900, P3.Q[:, o + 1:].shape); P3.P[:, o + 1:] = P3.Q[:, o + 1:]; P3.R[:, o + 1:] = P3.Q[:, o + 1:]
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
ok("this month's frozen forecast is carried in the payload to track the month against", pl3["locked"]["month"] == "2026-10")

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
d = data.product_only_daily(po)
ok("Product Only: the IS_PRIMARY true and false rows of a product-day are added together", d[d.sku == "A"].confirmed.iloc[0] == 11 and len(d) == 2)
try: data.product_only_daily(po.drop(columns=["CNF_PIECES_ACTUAL"])); ok("missing column is refused", False)
except SystemExit: ok("a Product Only answer with a column missing is refused, not read as zero", True)
oc = data.overlap_check(hist, newer)
ok("overlap check compares the two sources on the days both cover", oc["days"] == 2)

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
print("\n%d passed" % n_pass)
