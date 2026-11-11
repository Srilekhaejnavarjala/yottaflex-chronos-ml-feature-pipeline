"""Deterministic synthetic data generator.

    python -m src.generate_data --customers 5000 --events 100000 --seed 42

Same seed + parameters => byte-identical files. Business IDs are deterministic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import schemas as S

GENERATOR_VERSION = "1.0"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
DAY = 86_400


def _sec(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


SIGNUP_START, SIGNUP_END = _sec("2026-01-01"), _sec("2026-07-01")
DATA_END = _sec("2026-10-01")  # source completeness watermark (data_complete_through)
PRED_START, TRAIN_END = _sec("2026-07-15"), _sec("2026-10-05")
INFER_START = _sec("2026-09-25")

EVENT_TYPES = np.array(["purchase", "login", "cart_add", "refund", "cancellation"])
EVENT_P = [0.25, 0.45, 0.20, 0.06, 0.04]


def _fmt(arr) -> pd.Series:
    return pd.Series(pd.to_datetime(np.asarray(arr), unit="s", utc=True)).dt.strftime(TS_FMT)


def _customers(rng, n):
    signup = rng.integers(SIGNUP_START, SIGNUP_END, n)
    region = rng.choice(["north", "south", "east", "west"], n).astype(object)
    seg = rng.choice(["consumer", "smb", "enterprise"], n).astype(object)
    region[rng.random(n) < 0.05] = ""
    seg[rng.random(n) < 0.05] = ""
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "signup_timestamp": _fmt(signup),
        "region": region,
        "customer_segment": seg,
    })
    return df, signup


def _events(rng, n_cust, signup, n_events):
    # every 20th customer never has events (covers the "no history" case)
    active = np.array([i for i in range(n_cust) if i % 20 != 0])
    idx = rng.choice(active, n_events)
    s = signup[idx]
    ts = s + (rng.random(n_events) * (DATA_END - s)).astype(np.int64)
    etype = rng.choice(EVENT_TYPES, n_events, p=EVENT_P).astype(object)
    u = rng.random(n_events)
    status = np.full(n_events, "completed", dtype=object)
    p = etype == "purchase"
    status[p & (u < 0.08)] = "pending"
    status[p & (u >= 0.08) & (u < 0.15)] = "failed"
    status[p & (u >= 0.15) & (u < 0.20)] = "cancelled"
    status[(etype == "refund") & (u < 0.10)] = "pending"
    status[(etype == "login") & (u < 0.05)] = "failed"
    status[(etype == "cancellation") & (u < 0.10)] = "pending"
    monetary = np.isin(etype, list(S.MONETARY_EVENT_TYPES))
    amt = np.maximum(np.round(np.exp(rng.normal(3.5, 0.8, n_events)), 2), 1.0)
    late = rng.random(n_events) < 0.03
    delay = np.where(late, rng.integers(2 * DAY, 10 * DAY, n_events), rng.integers(1, 301, n_events))
    return pd.DataFrame({
        "event_id": [f"evt_{i:07d}" for i in range(1, n_events + 1)],
        "customer_id": [f"C{i + 1:05d}" for i in idx],
        "event_timestamp": _fmt(ts),
        "available_at": _fmt(ts + delay),
        "event_type": etype,
        "event_status": status,
        "amount": np.where(monetary, [f"{a:.2f}" for a in amt], ""),
        "currency": np.where(monetary, "USD", ""),
    })


def _predictions(rng, n_cust, n_pred):
    cidx = rng.integers(0, n_cust, n_pred)
    is_inf = rng.random(n_pred) < 0.2
    ts = np.where(is_inf, rng.integers(INFER_START, DATA_END + 1, n_pred),
                  rng.integers(PRED_START, TRAIN_END, n_pred)).astype(np.int64)
    df = pd.DataFrame({"cidx": cidx, "ts": ts})
    dup = df.duplicated(["cidx", "ts"], keep="first")
    while dup.any():  # natural key (customer, timestamp) must be unique
        df.loc[dup, "ts"] += 1
        dup = df.duplicated(["cidx", "ts"], keep="first")
    return pd.DataFrame({
        "prediction_id": [f"pred_{i:07d}" for i in range(1, n_pred + 1)],
        "customer_id": [f"C{i + 1:05d}" for i in df["cidx"]],
        "prediction_timestamp": _fmt(df["ts"].to_numpy()),
        "request_type": np.where(is_inf, "inference", "training"),
    }), df["ts"].to_numpy(), is_inf


def _boundary_events(preds, ts, is_inf, cases):
    """Exact-boundary completed purchases around T for the first N training predictions."""
    offsets = [  # tag, event offset, availability offset (seconds relative to T)
        ("lower_excl", -30 * DAY, -30 * DAY + 1),    # exactly T-30d      -> excluded
        ("lower_incl", -30 * DAY + 1, -30 * DAY + 2),  # T-30d+1s         -> included
        ("at_cutoff", 0, 0),                          # exactly T          -> included
        ("before_cutoff", -1, -1),                    # T-1s               -> included
        ("after_cutoff", 1, 1),                       # T+1s               -> excluded (label-side)
        ("late_arrival", -2 * DAY, 1 * DAY),          # known only at T+1d -> excluded
    ]
    rows, taken = [], 0
    for i in np.flatnonzero(~is_inf):
        if taken >= cases:
            break
        taken += 1
        for tag, eo, ao in offsets:
            rows.append({
                "event_id": f"evt_b{i + 1:05d}_{tag}", "customer_id": preds.loc[i, "customer_id"],
                "event_timestamp": int(ts[i] + eo), "available_at": int(ts[i] + ao),
                "event_type": "purchase", "event_status": "completed", "amount": "10.00", "currency": "USD",
            })
    if not rows:
        return pd.DataFrame(columns=S.TRANSACTION_COLUMNS)
    df = pd.DataFrame(rows)
    df["event_timestamp"] = _fmt(df["event_timestamp"].to_numpy())
    df["available_at"] = _fmt(df["available_at"].to_numpy())
    return df


def _invalid_transactions(base_events: pd.DataFrame) -> pd.DataFrame:
    tpl = dict(event_id="evt_x000000", customer_id="C00002", event_timestamp="2026-08-10T10:00:00Z",
               available_at="2026-08-10T10:00:30Z", event_type="purchase", event_status="completed",
               amount="25.00", currency="USD")
    cases = [
        ("evt_x000001", {"event_id": ""}),
        ("evt_x000002", {"customer_id": ""}),
        ("evt_x000003", {"customer_id": "C99999"}),
        ("evt_x000004", {"event_type": "gift"}),
        ("evt_x000005", {"event_status": "unknown"}),
        ("evt_x000006", {"event_timestamp": "2026-13-45T99:99:99Z"}),
        ("evt_x000007", {"event_timestamp": "2026-08-10T10:00:00"}),  # naive timestamp
        ("evt_x000008", {"amount": ""}),
        ("evt_x000009", {"amount": "-5.00"}),
        ("evt_x000010", {"currency": "EUR"}),
        ("evt_x000011", {"available_at": "2026-08-10T09:00:00Z"}),   # available before event
        ("evt_x000012", {"amount": "0.00"}),                          # zero-value purchase
        ("evt_x000013", {"amount": "abc"}),
    ]
    rows = [{**tpl, "event_id": eid, **ov} if "event_id" not in ov else {**tpl, **ov} for eid, ov in cases]
    first, second = base_events.iloc[0].to_dict(), base_events.iloc[1].to_dict()
    rows.append(dict(first))  # exact replay of an event already in transactions.csv
    conflict = dict(second)   # same event_id, different content -> conflict
    conflict["event_status"] = "failed" if second["event_status"] == "completed" else "completed"
    rows.append(conflict)
    return pd.DataFrame(rows, columns=S.TRANSACTION_COLUMNS)


def generate(out_dir, customers=100, events=3000, predictions=None, seed=42,
             include_invalid=False, boundary_cases=20) -> dict:
    if customers < 2:
        raise ValueError("customers must be >= 2")
    predictions = predictions if predictions is not None else max(events // 5, 10)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cust_df, signup = _customers(rng, customers)
    ev_df = _events(rng, customers, signup, events)
    pred_df, pred_ts, is_inf = _predictions(rng, customers, predictions)
    boundary = _boundary_events(pred_df, pred_ts, is_inf, boundary_cases)
    all_ev = pd.concat([ev_df, boundary], ignore_index=True)
    all_ev = all_ev.iloc[rng.permutation(len(all_ev))].reset_index(drop=True)  # out-of-order rows

    kw = dict(index=False, lineterminator="\n")
    cust_df.to_csv(out / "customers.csv", **kw)
    all_ev[S.TRANSACTION_COLUMNS].to_csv(out / "transactions.csv", **kw)
    pred_df.to_csv(out / "prediction_requests.csv", **kw)
    if include_invalid:
        _invalid_transactions(ev_df).to_csv(out / "invalid_transactions.csv", **kw)

    meta = {
        "generator_version": GENERATOR_VERSION, "seed": seed, "schema_version": "1",
        "customers": customers, "events": events, "boundary_events": int(len(boundary)),
        "predictions": predictions, "include_invalid": include_invalid,
        "data_complete_through": pd.Timestamp(DATA_END, unit="s", tz="UTC").strftime(TS_FMT),
    }
    (out / "source_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customers", type=int, default=100)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--predictions", type=int, default=None, help="default: events // 5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boundary-cases", type=int, default=20)
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--out-dir", default="data/generated")
    a = ap.parse_args(argv)
    meta = generate(a.out_dir, a.customers, a.events, a.predictions, a.seed, a.include_invalid, a.boundary_cases)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
