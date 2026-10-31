#!/usr/bin/env python3
"""Daily ML forecast run.
   python run.py --worker https://<worker>.workers.dev --key $SERVICE_KEY            # normal daily run (GitHub Actions)
   python run.py --worker ... --key ... --lock-month 2026-11                        # re-create a missed lock for a month that has just started
   python run.py --demo --out payload.json                                          # synthetic data, nothing is sent anywhere (tests / dashboard demo)
"""
import argparse, json, sys, datetime as dt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker"); ap.add_argument("--key"); ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--lock-month"); ap.add_argument("--out"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    from forecast import data, pipeline
    W = None
    if a.demo:
        sys.path.insert(0, "tests"); from synth import make_panel
        panel = make_panel(600); checks = {"demo": True}
    else:
        if not a.worker or not a.key: sys.exit("--worker and --key are required (or use --demo)")
        W = data.Worker(a.worker, a.key)
        today = pd.Timestamp(dt.date.today())
        hist = data.fetch_history(pd.Timestamp(a.since), today)
        newer = data.product_only_daily(data.parse_table(W.source(data.GID_PRODUCT_ONLY)))
        dmap = data.debundle_map(data.parse_table(W.source(data.GID_DEBUNDLE)))
        try: cats = data.category_map(data.parse_table(W.source(data.GID_PRODUCTS)))
        except Exception as e: print("categories unavailable:", e); cats = None
        prices = data.fetch_prices()
        checks = {"overlap": data.overlap_check(hist, newer), "prices": prices is not None, "categories": bool(cats)}
        cairo_today = pd.Timestamp.now(tz="Africa/Cairo").tz_localize(None).normalize()
        dmax = max(x["date"].max() for x in (hist, newer) if x is not None and len(x))
        asof = data.pick_asof(cairo_today, dmax)
        if asof < cairo_today - pd.Timedelta(days=1): checks["stale"] = "data stops at %s" % dmax.date()
        if a.lock_month:   # replay the day before that month started
            m = pd.Period(a.lock_month, "M"); asof = (m.start_time - pd.Timedelta(days=1)).normalize()
        panel = data.build_panel(hist, newer, dmap, cats, prices, as_of=asof)
    locked = {}
    if W is not None:
        for m in (W.get(lst=True) or {}).get("months", []):
            p = W.get(month=m)
            if p: locked[m] = p
    payload, lock = pipeline.build(panel, fast=a.fast, checks=checks, locked_by_month=locked)
    print("as of %s | %d SKUs | live forecast for %s | champion %s" % (payload["asOf"], len(panel.skus), payload["live"]["month"], payload["live"]["champion"]))
    if a.out: json.dump(payload, open(a.out, "w"), separators=(",", ":"))
    if a.dry_run or W is None: return
    if lock and (not a.lock_month or lock["month"] == a.lock_month):
        code, res = W.put("locked", lock, lock["month"]); print("lock %s ->" % lock["month"], code, res.get("message") or res.get("locked"))
    if a.lock_month: return
    code, res = W.put("latest", payload); print("latest ->", code, res)
    if code != 200: sys.exit(1)


if __name__ == "__main__":
    main()
