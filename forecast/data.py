"""Data: the same sources the dashboard reads, merged the same way, as one dense SKU x day panel.

  history tab (Google Sheet, public gviz)  -> daily pieces per product, older months
  Product Only (via the Worker)            -> daily pieces per product, current + previous month (wins on the days it covers)
  Products Debundle Map (via the Worker)   -> bundles are split into their Single SKUs (pieces x quantity), like the Forecast Model
  Products (via the Worker)                -> category
  price history tab (gviz)                 -> price timeline (optional)

Merge rule (mirrors the dashboard's fcBuildSeries): a day that the newer source covers is taken from it, never added to the history tab's.
The last day in the data is treated as partial (still being filled) and dropped, so the model never learns from half a day.
"""
import csv, io, json, urllib.parse, urllib.request
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SHEET_ID = "1Vg8P1EL5y_FqQSR7_uDI1XtB-gDe0Bkj7IqbiOzNgxA"
HISTORY_GID = "1433599782"
PRICE_GID = "1968659230"
GID_PRODUCT_ONLY = "7100000002"
GID_DEBUNDLE = "1409034448"
GID_PRODUCTS = "1779314157"


@dataclass
class Panel:
    dates: pd.DatetimeIndex            # one entry per calendar day, no gaps
    skus: np.ndarray                   # (S,) Single SKU ids
    Q: np.ndarray                      # (S,T) confirmed pieces
    P: np.ndarray                      # (S,T) placed pieces
    R: np.ndarray                      # (S,T) delivered pieces
    cat: np.ndarray = None             # (S,) category or None
    price: np.ndarray = None           # (S,T) price in force that day (NaN = unknown) or None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- fetching
def _http(url, headers=None, data=None, timeout=240):
    req = urllib.request.Request(url, headers=headers or {}, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


class Worker:
    """The dashboard's Cloudflare Worker. The Metabase links never leave it: the job only knows the Worker URL and a service key."""
    def __init__(self, base, key):
        self.base, self.key = base.rstrip("/"), key
    def _h(self): return {"X-Service-Key": self.key}
    def source(self, gid):
        return _http("%s/?action=mlSource&gid=%s" % (self.base, gid), self._h())
    def put(self, kind, payload, month=None):
        q = "action=mlPut&kind=%s" % kind + ("&month=%s" % month if month else "")
        req = urllib.request.Request("%s/?%s" % (self.base, q), data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                                     headers=dict(self._h(), **{"Content-Type": "application/json"}), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")
    def get(self, month=None, lst=False):
        q = "action=mlGet" + ("&list=1" if lst else ("&month=%s" % month if month else ""))
        try:
            return json.loads(_http("%s/?%s" % (self.base, q), self._h()))
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            raise


def parse_table(text):
    """Metabase answers either CSV or a JSON array of objects."""
    t = text.lstrip()
    if t[:1] in "[{":
        j = json.loads(text)
        if isinstance(j, dict): j = j.get("data") or j.get("rows") or []
        return pd.DataFrame(j)
    return pd.read_csv(io.StringIO(text))


def _gviz(gid, tq, sheet_id=SHEET_ID):
    url = "https://docs.google.com/spreadsheets/d/%s/gviz/tq?gid=%s&tq=%s&tqx=out:json" % (sheet_id, gid, urllib.parse.quote(tq))
    raw = _http(url)
    return json.loads(raw[raw.find("(") + 1:raw.rfind(")")])


def _gv_date(v):
    if isinstance(v, str) and v.startswith("Date("):
        y, m, d = [int(x) for x in v[5:-1].split(",")[:3]]
        return pd.Timestamp(y, m + 1, d)
    return pd.to_datetime(v, errors="coerce")


def fetch_history(start, end):
    """Daily pieces per product from the history tab. One query per month so no single answer is huge."""
    probe = _gviz(HISTORY_GID, "select * limit 1")
    labels = [(c.get("label") or c.get("id") or "").strip().upper() for c in probe["table"]["cols"]]
    need = {"PERIOD_FILTER", "PRODUCT_ID", "PLACED_PIECES", "CONFIRMED_PIECES", "DELIVERED_PIECES"}
    if not need.issubset(labels):   # Google answers with the FIRST sheet when a gid does not exist — never train on the wrong tab
        raise SystemExit("HISTORY_GID %s is not the daily history tab (columns seen: %s)" % (HISTORY_GID, labels[:12]))
    L = {n: chr(65 + labels.index(n)) for n in need}
    out = []
    for m in pd.period_range(start, end, freq="M"):
        a, b = m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d")
        tq = ("select %(d)s, %(p)s, sum(%(pp)s), sum(%(cp)s), sum(%(dp)s) where %(d)s >= date '%(a)s' and %(d)s <= date '%(b)s' group by %(d)s, %(p)s"
              % {"d": L["PERIOD_FILTER"], "p": L["PRODUCT_ID"], "pp": L["PLACED_PIECES"], "cp": L["CONFIRMED_PIECES"], "dp": L["DELIVERED_PIECES"], "a": a, "b": b})
        for r in _gviz(HISTORY_GID, tq)["table"]["rows"]:
            c = r["c"]
            if not c[0] or not c[1]: continue
            out.append((_gv_date(c[0]["v"]), str(c[1]["v"]).strip(), (c[2] or {}).get("v") or 0, (c[3] or {}).get("v") or 0, (c[4] or {}).get("v") or 0))
    return pd.DataFrame(out, columns=["date", "sku", "placed", "confirmed", "delivered"])


def fetch_prices():
    """Price timeline: (sku, valid_from, price). Optional — returns None when the tab does not look like we expect."""
    try:
        probe = _gviz(PRICE_GID, "select * limit 1")
        labels = [(c.get("label") or c.get("id") or "").strip().upper() for c in probe["table"]["cols"]]
        need = ["PRODUCT_ID", "VALID_FROM_DATE", "PRODUCT_BASE_PRICE"]
        if not all(n in labels for n in need): return None
        L = [chr(65 + labels.index(n)) for n in need]
        rows = _gviz(PRICE_GID, "select %s, %s, %s" % tuple(L))["table"]["rows"]
        out = [(str(r["c"][0]["v"]).strip(), _gv_date(r["c"][1]["v"]), r["c"][2]["v"]) for r in rows if r["c"][0] and r["c"][1] and r["c"][2]]
        return pd.DataFrame(out, columns=["sku", "from", "price"]).dropna()
    except Exception:
        return None


# --------------------------------------------------------------------------- shaping
def product_only_daily(df):
    c = {x.upper(): x for x in df.columns}
    need = ["DATE_GROUP", "PRODUCT_ID", "PLC_PIECES_ACTUAL", "CNF_PIECES_ACTUAL", "DLV_PIECES_ACTUAL"]
    miss = [n for n in need if n not in c]
    if miss: raise SystemExit("Product Only is missing columns: %s" % miss)
    d = pd.DataFrame({"date": pd.to_datetime(df[c["DATE_GROUP"]].astype(str).str[:10], errors="coerce"), "sku": df[c["PRODUCT_ID"]].astype(str).str.strip(),
                      "placed": pd.to_numeric(df[c["PLC_PIECES_ACTUAL"]], errors="coerce").fillna(0), "confirmed": pd.to_numeric(df[c["CNF_PIECES_ACTUAL"]], errors="coerce").fillna(0),
                      "delivered": pd.to_numeric(df[c["DLV_PIECES_ACTUAL"]], errors="coerce").fillna(0)}).dropna(subset=["date"])
    # IS_PRIMARY true / false rows of the same product-day are simply added (the false rows carry the extra pieces)
    return d.groupby(["date", "sku"], as_index=False)[["placed", "confirmed", "delivered"]].sum()


def debundle_map(df):
    c = {x.upper(): x for x in df.columns}
    need = ["PRODUCT_ID", "SINGLE_ID", "PRODUCT_QUANTITY"]
    if any(n not in c for n in need): raise SystemExit("Products Debundle Map is missing columns: %s" % [n for n in need if n not in c])
    m = {}
    for p, s, q in zip(df[c["PRODUCT_ID"]].astype(str).str.strip(), df[c["SINGLE_ID"]].astype(str).str.strip(), pd.to_numeric(df[c["PRODUCT_QUANTITY"]], errors="coerce").fillna(1)):
        if p and s and p != "nan" and s != "nan": m.setdefault(p, []).append((s, float(q) or 1.0))
    return m


def category_map(df):
    c = {x.upper(): x for x in df.columns}
    pid = next((c[k] for k in ("PRODUCT_ID", "SKU_ID", "ID") if k in c), None)
    cat = next((c[k] for k in ("CATEGORY_L1", "CATEGORY") if k in c), None)
    if not pid or not cat: return {}
    return {str(a).strip(): str(b).strip().upper() for a, b in zip(df[pid], df[cat]) if str(b) not in ("", "nan", "None")}


def build_panel(hist, newer, dmap=None, cats=None, prices=None, as_of=None):
    """hist / newer: DataFrames [date, sku, placed, confirmed, delivered]. `newer` wins on the days it covers. as_of = last FULL day (default: day before the last day in the data)."""
    frames = []
    newer = newer if newer is not None and len(newer) else None
    cut = newer["date"].min() if newer is not None else None
    if hist is not None and len(hist):
        frames.append(hist[hist["date"] < cut] if cut is not None else hist)
    if newer is not None: frames.append(newer)
    d = pd.concat(frames, ignore_index=True)
    last = d["date"].max()
    if as_of is None: as_of = last - pd.Timedelta(days=1)          # the last day is still being filled
    as_of = pd.Timestamp(as_of).normalize()
    d = d[d["date"] <= as_of]
    dmap = dmap or {}
    exp = []
    for r in d.itertuples(index=False):
        for s, q in dmap.get(r.sku, [(r.sku, 1.0)]):
            exp.append((r.date, s, r.placed * q, r.confirmed * q, r.delivered * q))
    e = pd.DataFrame(exp, columns=["date", "sku", "placed", "confirmed", "delivered"]).groupby(["date", "sku"], as_index=False).sum()
    dates = pd.date_range(e["date"].min(), as_of, freq="D")
    skus = np.array(sorted(e["sku"].unique()))
    si = {s: i for i, s in enumerate(skus)}; di = {d_: i for i, d_ in enumerate(dates)}
    Q = np.zeros((len(skus), len(dates))); P = Q.copy(); R = Q.copy()
    ii = e["sku"].map(si).to_numpy(); jj = e["date"].map(di).to_numpy()
    np.add.at(Q, (ii, jj), e["confirmed"].to_numpy()); np.add.at(P, (ii, jj), e["placed"].to_numpy()); np.add.at(R, (ii, jj), e["delivered"].to_numpy())
    cat = np.array([(cats or {}).get(s, "") for s in skus], dtype=object) if cats else None
    price = None
    if prices is not None and len(prices):
        price = np.full((len(skus), len(dates)), np.nan)
        for s, g in prices.groupby("sku"):
            if s not in si: continue
            g = g.sort_values("from")
            for f, p in zip(g["from"], g["price"]):
                j = dates.searchsorted(max(f, dates[0]))
                if j < len(dates): price[si[s], j:] = p
    meta = {"firstDay": str(dates[0].date()), "asOf": str(as_of.date()), "skus": int(len(skus)), "historyRows": int(len(hist) if hist is not None else 0), "newerRows": int(len(newer) if newer is not None else 0),
            "newerFrom": str(cut.date()) if cut is not None else None}
    return Panel(dates, skus, Q, P, R, cat, price, meta)


def overlap_check(hist, newer):
    """Days that BOTH sources cover: do they agree? Ratio history / newer of the confirmed pieces (1.0 = identical). A value far from 1 means duplicate rows or a different definition."""
    if hist is None or newer is None or not len(hist) or not len(newer): return None
    a = hist.groupby("date")["confirmed"].sum(); b = newer.groupby("date")["confirmed"].sum()
    j = a.index.intersection(b.index)
    if not len(j): return None
    return {"days": int(len(j)), "ratio": float(a.loc[j].sum() / b.loc[j].sum()) if b.loc[j].sum() else None}


def pick_asof(today, data_max):
    """The last FULL day: yesterday (Cairo), unless the data itself stops earlier (a stale cache) - never pretend missing days were zero-sales days."""
    y = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    return min(y, pd.Timestamp(data_max).normalize())
