"""Synthetic daily panel with structure a good model can learn: per-SKU level + random-walk trend + weekday pattern + month shocks + launches / deaths + spikes."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forecast.data import Panel


def make_panel(n=900, start="2026-01-01", end="2026-10-30", seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D"); T = len(dates)
    base = np.exp(rng.normal(1.2, 1.5, n))                       # pieces / day, heavy tailed
    drift = rng.normal(0.0, 0.006, n)                            # daily log drift
    walk = np.cumsum(rng.normal(0, 0.03, (n, T)), axis=1) * 0.35 + drift[:, None] * np.arange(T)
    dow = np.array([1.1, 1.0, 1.0, 0.95, 0.9, 0.85, 1.2])[dates.dayofweek.to_numpy()]
    month_shock = {m: rng.normal(0, 0.12) for m in range(1, 13)}
    shock = np.array([month_shock[d.month] for d in dates])
    mu = base[:, None] * np.exp(walk) * dow[None, :] * np.exp(shock)[None, :]
    launch = rng.integers(-60, 200, n); death = rng.integers(150, 600, n)       # some SKUs start late, some stop
    t = np.arange(T)[None, :]
    mu = mu * ((t >= launch[:, None]) & (t <= death[:, None]))
    spike = (rng.random((n, T)) < 0.01) * rng.uniform(2, 6, (n, T))
    mu = mu * (1 + spike)
    r = 2.0
    Q = rng.negative_binomial(r, r / (r + mu)).astype(float)
    P = np.round(Q / rng.uniform(0.5, 0.8, (n, 1)) + rng.poisson(0.3, (n, T)))
    R = np.round(Q * rng.uniform(0.55, 0.8, (n, 1)))
    skus = np.array(["SK%04d" % i for i in range(n)])
    cats = np.array(["CAT%d" % (i % 9) for i in range(n)], dtype=object)
    return Panel(dates, skus, Q, P, R, cats, None, {})
