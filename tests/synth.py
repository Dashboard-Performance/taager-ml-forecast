"""Synthetic daily panel with structure a good model can learn: per-SKU level + random-walk trend + weekday pattern + month shocks + launches / deaths + spikes,
PLUS merchants (a SKU's top merchant sometimes leaves, taking its share of the demand with it), inbound receipts and beginning inventory."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forecast.data import Panel


def make_panel(n=900, start="2026-01-01", end="2026-10-30", seed=3, merchants=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D"); T = len(dates)
    base = np.exp(rng.normal(1.2, 1.5, n))                       # pieces / day, heavy tailed
    drift = rng.normal(0.0, 0.006, n)
    walk = np.cumsum(rng.normal(0, 0.03, (n, T)), axis=1) * 0.35 + drift[:, None] * np.arange(T)
    dow = np.array([1.1, 1.0, 1.0, 0.95, 0.9, 0.85, 1.2])[dates.dayofweek.to_numpy()]
    month_shock = {m: rng.normal(0, 0.12) for m in range(1, 13)}
    shock = np.array([month_shock[d.month] for d in dates])
    mu = base[:, None] * np.exp(walk) * dow[None, :] * np.exp(shock)[None, :]
    launch = rng.integers(-60, 200, n); death = rng.integers(150, 600, n)
    t = np.arange(T)[None, :]
    mu = mu * ((t >= launch[:, None]) & (t <= death[:, None]))
    spike = (rng.random((n, T)) < 0.01) * rng.uniform(2, 6, (n, T))
    mu = mu * (1 + spike)
    r = 2.0
    mt = None
    if merchants:
        K = 6
        w = rng.dirichlet(np.ones(K) * 0.7, n)                    # merchant shares per SKU (some SKUs have one dominant merchant)
        nmer = rng.integers(1, K + 1, n); w = w * (np.arange(K)[None, :] < nmer[:, None]); w = w / w.sum(axis=1, keepdims=True)
        top = w.argmax(axis=1)
        leaves = np.where(rng.random(n) < 0.35, rng.integers(60, T - 20, n), 10 ** 6)   # the biggest merchant leaves on this day
        alive = np.ones((n, K, T))
        alive[np.arange(n), top, :] = (np.arange(T)[None, :] < leaves[:, None]).astype(float)
        share = w[:, :, None] * alive                               # the SKU loses the leaving merchant's share of demand
        Qm = rng.negative_binomial(r, r / (r + np.maximum(mu[:, None, :] * share, 1e-9))).astype(float) * (share > 0)
        Q = Qm.sum(axis=1)
        si, mi, di = np.nonzero(Qm > 0)
        order = np.argsort(di, kind="stable")
        mt = (di[order].astype(np.int64), si[order].astype(np.int64), mi[order].astype(np.int64), Qm[si, mi, di][order])
    else:
        Q = rng.negative_binomial(r, r / (r + mu)).astype(float)
    P = np.round(Q / rng.uniform(0.5, 0.8, (n, 1)) + rng.poisson(0.3, (n, T)))
    R = np.round(Q * rng.uniform(0.55, 0.8, (n, 1)))
    skus = np.array(["SK%04d" % i for i in range(n)])
    cats = np.array(["CAT%d" % (i % 9) for i in range(n)], dtype=object)
    # inbound receipts: a SKU gets stock about every 5-6 weeks, sized around a month of its demand
    INB = np.zeros((n, T))
    for i in range(n):
        d0 = int(rng.integers(0, 40))
        while d0 < T:
            INB[i, d0] = rng.poisson(max(1.0, base[i] * 30 * rng.uniform(0.5, 1.5))); d0 += int(rng.integers(30, 50))
    per = dates.to_period("M")
    BI = {}
    for m in sorted(set(per)):
        idx = int(np.argmax(per == m))
        BI[str(m)] = np.maximum(0.0, (Q[:, max(idx - 30, 0):idx + 1].sum(axis=1) * rng.uniform(0.2, 1.5, n)))
    return Panel(dates, skus, Q, P, R, cats, None, {}, mt, INB, BI)
