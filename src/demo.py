"""Leakage walkthrough demo:  python -m src.demo

Shows the design-doc worked example, a late-arriving event, and that appending future
events leaves an already-computed historical feature vector unchanged.
"""
from __future__ import annotations

import pandas as pd

from .config import load_config
from .features import build_features

T = "2026-10-20T12:00:00Z"
EXAMPLE = [  # id, event time, available at, expected eligible, reason
    ("A", "2026-10-01T10:00:00Z", "2026-10-01T10:05:00Z", True, "in window and known"),
    ("B", "2026-10-15T09:00:00Z", "2026-10-15T09:01:00Z", True, "in window and known"),
    ("C", "2026-10-20T11:59:00Z", "2026-10-20T11:59:30Z", True, "before cutoff and known"),
    ("D", "2026-10-20T12:01:00Z", "2026-10-20T12:01:00Z", False, "happened after cutoff"),
    ("E", "2026-10-18T15:00:00Z", "2026-10-21T09:00:00Z", False, "late-arriving: unknown at cutoff"),
    ("F", "2026-09-20T12:00:00Z", "2026-09-20T12:01:00Z", False, "exactly at excluded lower boundary"),
]


def _tx(rows):
    df = pd.DataFrame([{"event_id": i, "customer_id": "C1", "event_timestamp": e, "available_at": a,
                        "event_type": "purchase", "event_status": "completed", "amount": 10.0, "currency": "USD"}
                       for i, e, a in rows])
    for c in ("event_timestamp", "available_at"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    return df


def main():
    cfg = load_config()
    cust = pd.DataFrame({"customer_id": ["C1"], "signup_timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)})
    req = pd.DataFrame({"prediction_id": ["p1"], "customer_id": ["C1"],
                        "prediction_timestamp": pd.to_datetime([T], utc=True)})
    print(f"Prediction timestamp T = {T}  (window: T-30d < event_time <= T  AND  available_at <= T)\n")
    for i, e, a, ok, why in EXAMPLE:
        got = build_features(cust, _tx([(i, e, a)]), req, cfg).iloc[0]["purchase_count_30d"] == 1
        print(f"  event {i}: eligible={got!s:5} expected={ok!s:5} {'OK ' if got == ok else 'BUG'}  {why}")
    base = _tx([(i, e, a) for i, e, a, _, _ in EXAMPLE[:3]])
    before = build_features(cust, base, req, cfg)
    future = _tx([(f"future{k}", f"2026-10-2{k}T12:00:00Z", f"2026-10-2{k}T12:00:00Z") for k in range(1, 6)])
    after = build_features(cust, pd.concat([base, future], ignore_index=True), req, cfg)
    print("\nFeature row before appending 5 future purchases:\n", before.drop(columns=["prediction_timestamp"]).to_string(index=False))
    print("\nAfter appending them:\n", after.drop(columns=["prediction_timestamp"]).to_string(index=False))
    print("\nHistorical features unchanged:", before.equals(after))


if __name__ == "__main__":
    main()
