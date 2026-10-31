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
GID_MAIN = "2099497960"            # Main = the Matches query (merchant x product x day): the SAME columns as the history tab
GID_BEGIN_INV = "22283311"
INBOUND_GID = "565878313"
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
    merch: tuple = None                # (day_idx, sku_idx, merchant_code, pieces) sorted by day — who sold what (confirmed pieces)
    INB: np.ndarray = None             # (S,T) pieces received into the warehouse that day (Inbound sheet) or None
    BI: dict = None                    # {"2026-10": (S,) beginning inventory of that month} or None
    _cache: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- fetching
UA = "taager-ml-forecast/2.0 (+https://github.com/Dashboard-Performance/taager-ml-forecast)"   # Cloudflare answers 403 to Python's default User-Agent


class HttpError(RuntimeError):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


def _http(url, headers=None, data=None, timeout=240):
    h = {"User-Agent": UA}; h.update(headers or {})
    req = urllib.request.Request(url, headers=h, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:     # say WHAT the server answered, not just the number
        body = ""
        try: body = e.read().decode("utf-8", "replace")[:300]
        except Exception: pass
        raise HttpError(e.code, "HTTP %s from %s -> %s" % (e.code, url.split("?")[0] + "?" + (url.split("?")[1].split("&")[0] if "?" in url else ""), body)) from None


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
                                     headers=dict(self._h(), **{"Content-Type": "application/json", "User-Agent": UA}), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")
    def get(self, month=None, lst=False):
        q = "action=mlGet" + ("&list=1" if lst else ("&month=%s" % month if month else ""))
        try:
            return json.loads(_http("%s/?%s" % (self.base, q), self._h()))
        except HttpError as e:
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


STD = ["date", "merchant", "sku", "placed", "confirmed", "delivered", "category"]


def _pick(df, *names):
    up = {str(c).strip().upper(): c for c in df.columns}
    for n in names:
        if n in up: return up[n]
    return None


def to_long(df, what):
    """A Main / history table -> [date, merchant, sku, placed, confirmed, delivered, category], looked up BY COLUMN NAME (both have the same layout)."""
    cols = {"date": _pick(df, "CREATION_DATE", "PERIOD_FILTER", "DATE"), "merchant": _pick(df, "TAGER_ID", "MERCHANT_ID"), "sku": _pick(df, "PRODUCT_ID", "SKU"),
            "placed": _pick(df, "PLACED_PIECES"), "confirmed": _pick(df, "CONFIRMED_PIECES"), "delivered": _pick(df, "DELIVERED_PIECES")}
    miss = [k for k, v in cols.items() if v is None]
    if miss: raise SystemExit("%s is missing columns for: %s (columns seen: %s)" % (what, miss, list(df.columns)[:14]))
    cat = _pick(df, "CATEGORY")
    d = pd.DataFrame({"date": pd.to_datetime(df[cols["date"]].astype(str).str[:10], errors="coerce"), "merchant": df[cols["merchant"]].astype(str).str.strip(),
                      "sku": df[cols["sku"]].astype(str).str.strip(), "placed": pd.to_numeric(df[cols["placed"]], errors="coerce").fillna(0),
                      "confirmed": pd.to_numeric(df[cols["confirmed"]], errors="coerce").fillna(0), "delivered": pd.to_numeric(df[cols["delivered"]], errors="coerce").fillna(0),
                      "category": df[cat].astype(str).str.strip().str.upper() if cat else ""})
    return d.dropna(subset=["date"])


def main_long(text):
    return to_long(parse_table(text), "Main")


def fetch_history_long(start, end):
    """The history tab at merchant level (same grain as Main), one query per month."""
    probe = _gviz(HISTORY_GID, "select * limit 1")
    labels = [(c.get("label") or c.get("id") or "").strip().upper() for c in probe["table"]["cols"]]
    need = ["PERIOD_FILTER", "TAGER_ID", "PRODUCT_ID", "CATEGORY", "PLACED_PIECES", "CONFIRMED_PIECES", "DELIVERED_PIECES"]
    if not set(need).issubset(labels):   # Google answers with the FIRST sheet when a gid does not exist - never train on the wrong tab
        raise SystemExit("HISTORY_GID %s is not the daily history tab (columns seen: %s)" % (HISTORY_GID, labels[:12]))
    L = {n: chr(65 + labels.index(n)) for n in need}
    out = []
    for m in pd.period_range(start, end, freq="M"):
        a, b = m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d")
        tq = ("select %(d)s, %(t)s, %(p)s, %(c)s, sum(%(pp)s), sum(%(cp)s), sum(%(dp)s) where %(d)s >= date '%(a)s' and %(d)s <= date '%(b)s' group by %(d)s, %(t)s, %(p)s, %(c)s"
              % {"d": L["PERIOD_FILTER"], "t": L["TAGER_ID"], "p": L["PRODUCT_ID"], "c": L["CATEGORY"], "pp": L["PLACED_PIECES"], "cp": L["CONFIRMED_PIECES"], "dp": L["DELIVERED_PIECES"], "a": a, "b": b})
        for r in _gviz(HISTORY_GID, tq)["table"]["rows"]:
            c = r["c"]
            if not c[0] or not c[2]: continue
            out.append((_gv_date(c[0]["v"]), str(c[1]["v"]).strip() if c[1] else "", str(c[2]["v"]).strip(), str((c[3] or {}).get("v") or "").strip().upper(),
                        (c[4] or {}).get("v") or 0, (c[5] or {}).get("v") or 0, (c[6] or {}).get("v") or 0))
    d = pd.DataFrame(out, columns=["date", "merchant", "sku", "category", "placed", "confirmed", "delivered"])
    return d[STD]


def fetch_inbound():
    """Pieces received per SKU per day (Inbound sheet). Optional: None if the tab does not look as expected."""
    try:
        probe = _gviz(INBOUND_GID, "select * limit 1")
        labels = [(c.get("label") or c.get("id") or "").strip().upper() for c in probe["table"]["cols"]]
        need = ["DATE", "SKU", "RCV_QTY"]
        if not all(n in labels for n in need): return None
        L = [chr(65 + labels.index(n)) for n in need]
        rows = _gviz(INBOUND_GID, "select %s, %s, %s" % tuple(L))["table"]["rows"]
        out = [(_gv_date(r["c"][0]["v"]), str(r["c"][1]["v"]).strip(), r["c"][2]["v"]) for r in rows if r["c"][0] and r["c"][1] and r["c"][2]]
        return pd.DataFrame(out, columns=["date", "sku", "qty"]).dropna()
    except Exception as e:
        print("inbound unavailable:", e); return None


def beginning_inventory(df):
    """[PRODUCT_ID, QTY, MONTH] -> {'2026-10': Series(sku -> qty)}"""
    c = {x.upper(): x for x in df.columns}
    if not all(k in c for k in ("PRODUCT_ID", "QTY", "MONTH")): return None
    d = pd.DataFrame({"sku": df[c["PRODUCT_ID"]].astype(str).str.strip(), "qty": pd.to_numeric(df[c["QTY"]], errors="coerce"), "month": pd.to_datetime(df[c["MONTH"]].astype(str).str[:10], errors="coerce")}).dropna()
    d["m"] = d["month"].dt.to_period("M").astype(str)
    return {m: g.groupby("sku")["qty"].sum() for m, g in d.groupby("m")}


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
        need = ["SKU_NUMBER", "VALID_FROM_DATE", "PRODUCT_BASE_PRICE"]
        if not all(n in labels for n in need): return None
        L = [chr(65 + labels.index(n)) for n in need]
        rows = _gviz(PRICE_GID, "select %s, %s, %s" % tuple(L))["table"]["rows"]
        out = [(str(r["c"][0]["v"]).strip(), _gv_date(r["c"][1]["v"]), r["c"][2]["v"]) for r in rows if r["c"][0] and r["c"][1] and r["c"][2]]
        return pd.DataFrame(out, columns=["sku", "from", "price"]).dropna()
    except Exception:
        return None


# --------------------------------------------------------------------------- shaping
def product_only_daily(df, primary_only=True):
    """primary_only=True keeps only the IS_PRIMARY rows: that is what the history tab counts (measured on September: history 106,146 vs Product Only primary 104,216 = 0.98,
    vs 160,260 with the extra rows = 0.66). Mixing the two definitions would put a +54% step in the series on the day the sources switch."""
    c = {x.upper(): x for x in df.columns}
    if primary_only and "IS_PRIMARY" in c:
        df = df[df[c["IS_PRIMARY"]].astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])]
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


def build_panel(hist, newer, dmap=None, cats=None, prices=None, as_of=None, inbound=None, bi=None):
    """hist / newer: long tables [date, merchant, sku, placed, confirmed, delivered, category]. `newer` (Main) wins on the days it covers.
    as_of = last FULL day (default: the day before the last day in the data)."""
    frames = []
    newer = newer if newer is not None and len(newer) else None
    cut = newer["date"].min() if newer is not None else None
    if hist is not None and len(hist):
        frames.append(hist[hist["date"] < cut] if cut is not None else hist)
    if newer is not None: frames.append(newer)
    d = pd.concat(frames, ignore_index=True)
    for col in ("merchant", "category"):
        if col not in d: d[col] = ""
    last = d["date"].max()
    if as_of is None: as_of = last - pd.Timedelta(days=1)          # the last day is still being filled
    as_of = pd.Timestamp(as_of).normalize()
    d = d[d["date"] <= as_of].copy()
    # bundles -> their Single SKUs (pieces x quantity), like the Forecast Model
    rows = [(p, sg, q) for p, lst in (dmap or {}).items() for sg, q in lst]
    if rows:
        dm = pd.DataFrame(rows, columns=["product", "single", "qty"])
        d = d.merge(dm, left_on="sku", right_on="product", how="left")
        d["single"] = d["single"].fillna(d["sku"]); d["qty"] = d["qty"].fillna(1.0)
    else:
        d["single"] = d["sku"]; d["qty"] = 1.0
    for col in ("placed", "confirmed", "delivered"): d[col] = d[col] * d["qty"]
    if cats is None and d["category"].astype(bool).any():
        cats = d[d["category"] != ""].groupby("single")["category"].agg(lambda x: x.value_counts().index[0]).to_dict()
    daily = d.groupby(["date", "single"], as_index=False)[["placed", "confirmed", "delivered"]].sum().rename(columns={"single": "sku"})
    dates = pd.date_range(daily["date"].min(), as_of, freq="D")
    skus = np.array(sorted(daily["sku"].unique()))
    si = {s_: i for i, s_ in enumerate(skus)}; di = {d_: i for i, d_ in enumerate(dates)}
    Q = np.zeros((len(skus), len(dates))); P = Q.copy(); R = Q.copy()
    ii = daily["sku"].map(si).to_numpy(); jj = daily["date"].map(di).to_numpy()
    np.add.at(Q, (ii, jj), daily["confirmed"].to_numpy()); np.add.at(P, (ii, jj), daily["placed"].to_numpy()); np.add.at(R, (ii, jj), daily["delivered"].to_numpy())
    cat = np.array([(cats or {}).get(s_, "") for s_ in skus], dtype=object) if cats else None
    # who sold it: (day, sku, merchant) confirmed pieces
    mt = None
    if d["merchant"].astype(bool).any():
        g = d.groupby(["date", "single", "merchant"], as_index=False)["confirmed"].sum()
        g = g[g["confirmed"] > 0]
        mcode = pd.Categorical(g["merchant"]).codes.astype(np.int64)
        o = np.argsort(g["date"].map(di).to_numpy(), kind="stable")
        mt = (g["date"].map(di).to_numpy()[o].astype(np.int64), g["single"].map(si).to_numpy()[o].astype(np.int64), mcode[o], g["confirmed"].to_numpy()[o])
    price = None
    if prices is not None and len(prices):
        price = np.full((len(skus), len(dates)), np.nan)
        for s_, g in prices.groupby("sku"):
            if s_ not in si: continue
            g = g.sort_values("from")
            for f, p_ in zip(g["from"], g["price"]):
                j = dates.searchsorted(max(f, dates[0]))
                if j < len(dates): price[si[s_], j:] = p_
    INB = None
    if inbound is not None and len(inbound):
        INB = np.zeros((len(skus), len(dates)))
        ib = inbound[inbound["sku"].isin(si) & (inbound["date"] >= dates[0]) & (inbound["date"] <= as_of)]
        np.add.at(INB, (ib["sku"].map(si).to_numpy(), ib["date"].map(di).to_numpy()), ib["qty"].to_numpy())
    BI = None
    if bi:
        BI = {m: ser.reindex(skus).to_numpy(dtype=float) for m, ser in bi.items()}
    meta = {"firstDay": str(dates[0].date()), "asOf": str(as_of.date()), "skus": int(len(skus)), "historyRows": int(len(hist) if hist is not None else 0), "newerRows": int(len(newer) if newer is not None else 0),
            "newerFrom": str(cut.date()) if cut is not None else None, "merchants": int(mt[2].max() + 1) if mt is not None and len(mt[2]) else 0,
            "pricedSkus": int(np.isfinite(price).any(axis=1).sum()) if price is not None else 0, "inboundSkus": int((INB.sum(axis=1) > 0).sum()) if INB is not None else 0,
            "beginInvMonths": sorted(BI) if BI else []}
    return Panel(dates, skus, Q, P, R, cat, price, meta, mt, INB, BI)


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
