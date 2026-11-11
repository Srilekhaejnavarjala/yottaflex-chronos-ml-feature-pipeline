import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.generate_data import generate  # noqa: E402


def ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def customers_df(rows=None):
    rows = rows or [("C1", "2026-01-01T00:00:00Z")]
    return pd.DataFrame({"customer_id": [r[0] for r in rows],
                         "signup_timestamp": pd.to_datetime([r[1] for r in rows], utc=True).astype("datetime64[ns, UTC]"),
                         "region": None, "customer_segment": None})


def tx_df(rows):
    """rows: dicts with event_id, customer_id, event_timestamp, available_at; optional type/status/amount."""
    recs = []
    for r in rows:
        t = r.get("event_type", "purchase")
        recs.append({
            "event_id": r["event_id"], "customer_id": r.get("customer_id", "C1"),
            "event_timestamp": r["event_timestamp"], "available_at": r.get("available_at", r["event_timestamp"]),
            "event_type": t, "event_status": r.get("event_status", "completed"),
            "amount": r.get("amount", 10.0 if t in ("purchase", "refund") else float("nan")),
            "currency": "USD" if t in ("purchase", "refund") else None,
            "source_file": "t.csv", "source_row_number": 2, "run_id": "r"})
    df = pd.DataFrame(recs, columns=["event_id", "customer_id", "event_timestamp", "available_at", "event_type",
                                     "event_status", "amount", "currency", "source_file", "source_row_number", "run_id"])
    for c in ("event_timestamp", "available_at"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    df["amount"] = df["amount"].astype("float64")
    return df


def preds_df(rows):
    """rows: (prediction_id, customer_id, prediction_timestamp, request_type)"""
    return pd.DataFrame({
        "prediction_id": [r[0] for r in rows], "customer_id": [r[1] for r in rows],
        "prediction_timestamp": pd.to_datetime([r[2] for r in rows], utc=True).astype("datetime64[ns, UTC]"),
        "request_type": [r[3] if len(r) > 3 else "training" for r in rows]})


def raw_frame(rows, columns, table_file="t.csv"):
    df = pd.DataFrame(rows, columns=columns).astype(str)
    df["source_file"] = table_file
    df["source_row_number"] = range(2, len(df) + 2)
    return df


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    generate(d, customers=40, events=800, predictions=150, seed=7, include_invalid=True, boundary_cases=10)
    return d
