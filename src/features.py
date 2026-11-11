"""Point-in-time feature engine.

Eligibility of an event for prediction timestamp T (all UTC):

    T - lookback < event_timestamp <= T      (lower exclusive, upper inclusive)
    AND available_at <= T                    (no availability leakage)
    AND same customer_id

``build_features`` is the ONE transformation used for training and inference. It never
receives labels. ``build_features_reference`` is a deliberately simple per-request filter
over the whole table used to verify the optimized implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as S
from .config import FeatureConfig, ensure_supported_windows
from .timeutil import NS_PER_DAY, to_ns

FEATURE_DTYPES = {
    "purchase_count_30d": "int64", "purchase_amount_sum_30d": "float64",
    "purchase_amount_avg_30d": "float64", "days_since_last_purchase": "float64",
    "refund_count_30d": "int64", "login_count_30d": "int64", "customer_tenure_days": "int64",
}
_EMPTY_B, _EMPTY_F, _EMPTY_I = np.zeros(0, bool), np.zeros(0, float), np.zeros(0, np.int64)


def _guard(customers, transactions, requests, cfg):
    ensure_supported_windows(cfg)
    for name, df in (("customers", customers), ("transactions", transactions), ("prediction_requests", requests)):
        leaked = S.LABEL_LEAK_COLUMNS & set(df.columns)
        if leaked:
            raise ValueError(f"build_features must not receive label columns; found {sorted(leaked)} in {name}")


def _aggregate(is_p, is_r, is_l, amt, ts_ns, T_ns, signup_ns):
    """Feature values for one request from the (already eligible) event window."""
    tenure = int((T_ns - signup_ns) // NS_PER_DAY)
    if tenure < 0 and (T_ns < signup_ns):
        raise ValueError("prediction_timestamp is before customer signup")
    pc = int(is_p.sum())
    if pc:
        amounts = amt[is_p]
        total = round(float(amounts.sum()), 2)
        avg = round(float(amounts.mean()), 4)
        days = round(float((T_ns - ts_ns[is_p].max()) / NS_PER_DAY), 4)
    else:
        total, avg, days = 0.0, np.nan, np.nan
    return (pc, total, avg, days, int(is_r.sum()), int(is_l.sum()), tenure)


def _assemble(requests: pd.DataFrame, rows: list, cfg: FeatureConfig) -> pd.DataFrame:
    cols = list(FEATURE_DTYPES)
    out = pd.DataFrame(rows, columns=cols, index=requests.index)
    out.insert(0, "feature_version", cfg.feature_version)
    out.insert(0, "prediction_timestamp", requests["prediction_timestamp"].to_numpy())
    out.insert(0, "customer_id", requests["customer_id"].to_numpy())
    out.insert(0, "prediction_id", requests["prediction_id"].to_numpy())
    out = out.astype(FEATURE_DTYPES)
    return out[S.FEATURE_COLUMNS].sort_values("prediction_id", kind="mergesort").reset_index(drop=True)


def _signup_map(customers):
    return dict(zip(customers["customer_id"], to_ns(customers["signup_timestamp"])))


def _flags(tx, cfg):
    et, es = tx["event_type"].to_numpy(), tx["event_status"].to_numpy()
    is_p = (et == cfg.qualifying_purchase_event_type) & (es == cfg.qualifying_purchase_status)
    is_r = (et == "refund") & (es == "completed")
    is_l = (et == "login") & (es == "completed")
    return is_p, is_r, is_l


def _sorted(tx):
    return tx.sort_values(["customer_id", "event_timestamp", "event_id"], kind="mergesort").reset_index(drop=True)


def build_features(customers: pd.DataFrame, transactions: pd.DataFrame,
                   prediction_requests: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return one leakage-safe feature row per prediction request.

    This function must not receive or inspect target labels.

    Optimized: events are sorted once per customer; each request locates its lookback
    window with binary search and only the (small) window is filtered on availability.
    """
    _guard(customers, transactions, prediction_requests, cfg)
    signup = _signup_map(customers)
    tx = _sorted(transactions)
    ts, av = to_ns(tx["event_timestamp"]), to_ns(tx["available_at"])
    amt = tx["amount"].to_numpy(dtype="float64")
    is_p, is_r, is_l = _flags(tx, cfg)
    cust = tx["customer_id"].to_numpy()
    bounds = {}
    if len(cust):
        uniq, first, counts = np.unique(cust, return_index=True, return_counts=True)
        bounds = {c: (int(f), int(f + n)) for c, f, n in zip(uniq, first, counts)}
    lookback_ns = cfg.lookback_days * NS_PER_DAY

    rows = []
    for cid, T in zip(prediction_requests["customer_id"], to_ns(prediction_requests["prediction_timestamp"])):
        if cid not in signup:
            raise ValueError(f"prediction request references unknown customer {cid!r}")
        s, e = bounds.get(cid, (0, 0))
        if e > s:
            seg = ts[s:e]
            lo = s + int(np.searchsorted(seg, T - lookback_ns, side="right"))  # ts >  T - lookback
            hi = s + int(np.searchsorted(seg, T, side="right"))                # ts <= T
            m = av[lo:hi] <= T                                                 # available_at <= T
            rows.append(_aggregate(is_p[lo:hi][m], is_r[lo:hi][m], is_l[lo:hi][m],
                                   amt[lo:hi][m], ts[lo:hi][m], T, signup[cid]))
        else:
            rows.append(_aggregate(_EMPTY_B, _EMPTY_B, _EMPTY_B, _EMPTY_F, _EMPTY_I, T, signup[cid]))
    return _assemble(prediction_requests, rows, cfg)


def build_features_reference(customers, transactions, prediction_requests, cfg) -> pd.DataFrame:
    """Naive reference: filter the whole transaction table once per request."""
    _guard(customers, transactions, prediction_requests, cfg)
    signup = _signup_map(customers)
    tx = _sorted(transactions)
    lookback = pd.Timedelta(days=cfg.lookback_days)
    rows = []
    for cid, T in zip(prediction_requests["customer_id"], prediction_requests["prediction_timestamp"]):
        if cid not in signup:
            raise ValueError(f"prediction request references unknown customer {cid!r}")
        sub = tx[(tx["customer_id"] == cid) & (tx["event_timestamp"] > T - lookback)
                 & (tx["event_timestamp"] <= T) & (tx["available_at"] <= T)]
        is_p, is_r, is_l = _flags(sub, cfg)
        T_ns = int(T.tz_convert("UTC").tz_localize(None).as_unit("ns").value)
        rows.append(_aggregate(is_p, is_r, is_l, sub["amount"].to_numpy(dtype="float64"),
                               to_ns(sub["event_timestamp"]) if len(sub) else _EMPTY_I, T_ns, signup[cid]))
    return _assemble(prediction_requests, rows, cfg)
