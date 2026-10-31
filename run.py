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
        # ONE source for the recent days: Main (the Matches query) - the same columns as the history tab. Worker first, so a wrong key / URL shows up in seconds.
        newer = data.main_long(W.source(data.GID_MAIN))
        dmap = data.debundle_map(data.parse_table(W.source(data.GID_DEBUNDLE)))
        try: bi = data.beginning_inventory(data.parse_table(W.source(data.GID_BEGIN_INV)))
        except Exception as e: print("beginning inventory unavailable:", e); bi = None
        print("Worker sources ok: Main %d rows (%s .. %s), %d bundles mapped" % (len(newer), newer["date"].min().date(), newer["date"].max().date(), len(dmap)), flush=True)
        hist = data.fetch_history_long(pd.Timestamp(a.since), today)
        inbound = data.fetch_inbound(); prices = data.fetch_prices()
        checks = {"overlap": data.overlap_check(hist, newer), "source": "Main + history tab", "prices": prices is not None, "inbound": inbound is not None, "beginInventory": bool(bi)}
        cairo_today = pd.Timestamp.now(tz="Africa/Cairo").tz_localize(None).normalize()
        dmax = max(x["date"].max() for x in (hist, newer) if x is not None and len(x))
        asof = data.pick_asof(cairo_today, dmax)
        if asof < cairo_today - pd.Timedelta(days=1): checks["stale"] = "data stops at %s" % dmax.date()
        if a.lock_month:   # replay the day before that month started
            m = pd.Period(a.lock_month, "M"); asof = (m.start_time - pd.Timedelta(days=1)).normalize()
        panel = data.build_panel(hist, newer, dmap, None, prices, as_of=asof, inbound=inbound, bi=bi)
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
