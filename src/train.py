#!/usr/bin/env python3
"""
Taager demand forecast — ML backtest & prediction (SHADOW / offline).

What it does, and just as important, what it does NOT do:
  • It READS the public "Forecast History" tab of the dashboard sheet
    (or a local CSV), builds a leakage-safe walk-forward backtest, and
    trains a global gradient-boosted model on ALL available months.
  • It WRITES, into output/:
        predictions.csv        — next-month ML forecast per product
        accuracy_by_segment.csv — measured 1-WAPE by segment: ML vs naive vs MA
        results.html           — a self-contained VIEW: where ML wins, by segment
  • It does NOT touch the live dashboard or the live forecast in any way.
    It is a shadow experiment. Nothing here changes what the Forecast Model
    shows until a human looks at results.html and decides to wire it in.

Data source: the same public Google Sheet the dashboard reads.
  SHEET_ID / HISTORY_GID below. Columns (0-based) in that tab:
    A PERIOD_FILTER(date)  D PRODUCT_ID  P PLACED_PIECES(15)
    Q CONFIRMED_PIECES(16) R DELIVERED_PIECES(17)
The target we forecast is CONFIRMED PIECES per product per month — the same
quantity the Forecast Model plans on.

Usage:
    python src/train.py                 # fetch live from the sheet
    python src/train.py --local hist.csv  # use a local pivot CSV instead
"""
import sys, os, json, csv, io, datetime
import numpy as np

SHEET_ID = "1Vg8P1EL5y_FqQSR7_uDI1XtB-gDe0Bkj7IqbiOzNgxA"
HISTORY_GID = "1338407774"           # "Forecast History" tab
N_MONTHS = 7                          # months present in history (grows over time; auto-detected on live)
SEG_A, SEG_B = 100, 30                # monthly-confirmed thresholds: A>=100, B 30-99, C<30

# ------------------------------------------------------------------ data load
def _gviz(tq):
    import urllib.request, urllib.parse
    url = ("https://docs.google.com/spreadsheets/d/%s/gviz/tq?gid=%s&tq=%s&tqx=out:json"
           % (SHEET_ID, HISTORY_GID, urllib.parse.quote(tq)))
    raw = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    s = raw.find("("); e = raw.rfind(")")
    return json.loads(raw[s + 1:e])

def load_live():
    """Pivot to {pid: {'c':[..],'p':[..],'r':[..]}} of monthly confirmed/placed/delivered."""
    resp = _gviz("select D, year(A), month(A), sum(P), sum(Q), sum(R) group by D, year(A), month(A) order by year(A), month(A)")
    rows = resp["table"]["rows"]
    # map (year,month) -> column index, oldest first
    ym = sorted({(r["c"][1]["v"], r["c"][2]["v"]) for r in rows if r["c"][1] and r["c"][2]})
    idx = {k: i for i, k in enumerate(ym)}
    n = len(ym)
    M = {}
    for r in rows:
        c = r["c"]
        pid = c[0]["v"] if c[0] else None
        if not pid or c[1] is None or c[2] is None:
            continue
        j = idx[(c[1]["v"], c[2]["v"])]
        e = M.setdefault(pid, {"p": [0]*n, "q": [0]*n, "r": [0]*n})
        e["p"][j] += round(c[3]["v"] or 0) if c[3] else 0
        e["q"][j] += round(c[4]["v"] or 0) if c[4] else 0
        e["r"][j] += round(c[5]["v"] or 0) if c[5] else 0
    labels = ["%04d-%02d" % (y, m + 1) for (y, m) in ym]
    return M, labels

def load_local(path):
    M = {}
    with open(path) as f:
        rd = csv.reader(f); next(rd)
        for r in rd:
            if len(r) < 22:
                continue
            try:
                nums = [float(x) for x in r[-21:]]
            except ValueError:
                continue
            M[r[0]] = {"c": nums[0:7], "p": nums[7:14], "r": nums[14:21]}
    # local file stores c/p/r; unify key name to q for confirmed
    for e in M.values():
        e["q"] = e.pop("c")
    labels = ["m%d" % i for i in range(7)]
    return M, labels

# ------------------------------------------------------------------ modelling
def features(e, t):
    q, p, r = e["q"], e["p"], e["r"]
    prior = q[:t]
    return [q[t-1], q[t-2], q[t-3], p[t-1], p[t-2], r[t-1],
            float(np.mean(prior)), float(np.max(prior)), float(np.std(prior)),
            sum(1 for x in prior if x > 0), t]

def wape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float); s = a.sum()
    return float(np.abs(a - f).sum() / s) if s > 0 else float("nan")

def seg(e, t):
    m = float(np.mean(e["q"][:t])) if t > 0 else 0
    return "A" if m >= SEG_A else ("B" if m >= SEG_B else "C")

def run(M, labels):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(labels)
    prods = list(M.values())
    test_months = [t for t in range(3, n)]           # need 3 lags
    scopes = {s: {"a": [], "naive": [], "ma3": [], "ml": []} for s in ["ALL", "A", "B", "C"]}
    final_model = None
    for t in test_months:
        Xtr, ytr = [], []
        for s in range(3, t):
            for e in prods:
                Xtr.append(features(e, s)); ytr.append(e["q"][s])
        if not Xtr:
            continue
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=4,
                                          learning_rate=0.05, min_samples_leaf=30)
        m.fit(np.array(Xtr), np.log1p(ytr))
        final_model = m
        for e in prods:
            a = e["q"][t]; sg = seg(e, t)
            naive = e["q"][t-1]; ma3 = float(np.mean(e["q"][t-3:t]))
            ml = max(0.0, float(np.expm1(m.predict(np.array(features(e, t)).reshape(1, -1))[0])))
            for sc in ("ALL", sg):
                d = scopes[sc]; d["a"].append(a); d["naive"].append(naive)
                d["ma3"].append(ma3); d["ml"].append(ml)
    acc = {}
    for sc, d in scopes.items():
        if not d["a"]:
            continue
        acc[sc] = {"n": len(d["a"]),
                   "naive": (1 - wape(d["a"], d["naive"])) * 100,
                   "ma3":   (1 - wape(d["a"], d["ma3"])) * 100,
                   "ml":    (1 - wape(d["a"], d["ml"])) * 100}
    # final: train on everything with 3 lags, predict the NEXT (unseen) month
    Xtr, ytr = [], []
    for s in range(3, n):
        for e in prods:
            Xtr.append(features(e, s)); ytr.append(e["q"][s])
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    fm = HGB(max_iter=250, max_depth=4, learning_rate=0.05, min_samples_leaf=30)
    fm.fit(np.array(Xtr), np.log1p(ytr))
    preds = {}
    for pid, e in M.items():
        x = features(e, n) if n >= 3 else None
        # shift window: predict month n using lags at n-1..n-3
        ee = {"q": e["q"] + [0], "p": e["p"] + [0], "r": e["r"] + [0]}
        preds[pid] = max(0.0, float(np.expm1(fm.predict(np.array(features(ee, n)).reshape(1, -1))[0])))
    return acc, preds

# ------------------------------------------------------------------ outputs
def write_outputs(acc, preds, M, labels, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "accuracy_by_segment.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["segment", "n", "naive_acc", "ma3_acc", "ml_acc", "ml_minus_naive"])
        for s in ["ALL", "A", "B", "C"]:
            if s in acc:
                a = acc[s]; w.writerow([s, a["n"], round(a["naive"], 1), round(a["ma3"], 1),
                                        round(a["ml"], 1), round(a["ml"] - a["naive"], 1)])
    with open(os.path.join(outdir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["product_id", "ml_next_month_confirmed", "segment", "last_month_confirmed"])
        for pid, e in M.items():
            w.writerow([pid, round(preds[pid]), seg(e, len(labels)), round(e["q"][-1])])
    _write_html(acc, labels, os.path.join(outdir, "results.html"))

def _write_html(acc, labels, path):
    rows = ""
    for s in ["ALL", "A", "B", "C"]:
        if s not in acc:
            continue
        a = acc[s]; delta = a["ml"] - a["naive"]
        win = "win" if delta > 0.5 else ("lose" if delta < -0.5 else "flat")
        rows += ("<tr><td>%s</td><td class=n>%d</td><td class=n>%.1f%%</td>"
                 "<td class=n>%.1f%%</td><td class='n ml'>%.1f%%</td>"
                 "<td class='n %s'>%+.1f</td></tr>") % (s, a["n"], a["naive"], a["ma3"], a["ml"], win, delta)
    span = (labels[0] + " .. " + labels[-1]) if labels else ""
    gen = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tpl = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>ML Forecast — Shadow</title>
<style>
:root{--bg:#fff;--fg:#0f172a;--dim:#64748b;--line:#e2e8f0;--ml:#7c3aed;--win:#16a34a;--lose:#dc2626;--card:#f8fafc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b1120;--fg:#e2e8f0;--dim:#94a3b8;--line:#1e293b;--ml:#a78bfa;--win:#4ade80;--lose:#f87171;--card:#111a2e}}
:root[data-theme=dark]{--bg:#0b1120;--fg:#e2e8f0;--dim:#94a3b8;--line:#1e293b;--ml:#a78bfa;--win:#4ade80;--lose:#f87171;--card:#111a2e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px 16px}
.wrap{max-width:820px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--dim);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
table{width:100%;border-collapse:collapse}th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
td.n{text-align:right;font-variant-numeric:tabular-nums}td.ml{color:var(--ml);font-weight:700}
.win{color:var(--win);font-weight:700}.lose{color:var(--lose);font-weight:700}.flat{color:var(--dim)}
.note{color:var(--dim);font-size:13px}.tag{display:inline-block;background:var(--ml);color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600}
</style></head><body><div class=wrap>
<h1>ML Forecast <span class=tag>Shadow — not live</span></h1>
<p class=sub>Measured on the real Forecast-History months (__SPAN__). Target: confirmed pieces / product / month. Walk-forward, leakage-safe. Accuracy = 1 &minus; WAPE. <b>Nothing here drives the live Forecast Model.</b></p>
<div class=card><table>
<thead><tr><th>Segment</th><th class=n>SKUs&times;months</th><th class=n>Naive</th><th class=n>Avg-3mo</th><th class=n>ML</th><th class=n>ML &minus; Naive</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<p class=note>Segment by monthly confirmed: A &ge; 100, B 30&ndash;99, C &lt; 30. "ML &minus; Naive" &gt; 0 means the ML model beat simply repeating last month, on that segment. This is vs <b>naive</b>, not yet vs the dashboard's tuned engine — the head-to-head against the live engine is the next step before anything is wired in.</p>
<p class=note>Generated __GEN__</p>
</div></body></html>"""
    html = tpl.replace("__SPAN__", span).replace("__ROWS__", rows).replace("__GEN__", gen)
    with open(path, "w") as f:
        f.write(html)

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if "--local" in sys.argv:
        M, labels = load_local(sys.argv[sys.argv.index("--local") + 1])
    else:
        M, labels = load_live()
    print("products=%d months=%d (%s..%s)" % (len(M), len(labels), labels[0], labels[-1]))
    acc, preds = run(M, labels)
    outdir = os.path.join(os.path.dirname(__file__), "..", "output")
    write_outputs(acc, preds, M, labels, outdir)
    for s in ["ALL", "A", "B", "C"]:
        if s in acc:
            a = acc[s]; print("  %-3s n=%-6d naive=%5.1f ma3=%5.1f ML=%5.1f  (ML-naive %+.1f)"
                              % (s, a["n"], a["naive"], a["ma3"], a["ml"], a["ml"] - a["naive"]))
    print("wrote output/predictions.csv, accuracy_by_segment.csv, results.html")
