"""SQLite persistence: schema, run ledger, and atomic, idempotent publishing.

All data tables are written inside ONE transaction (``publish_all``): either every table
is updated or none is. Writes are upserts on the logical key, and each table reports
inserted / updated / unchanged rows, so replaying identical input changes nothing.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3

import numpy as np
import pandas as pd

from . import schemas as S
from .config import FeatureConfig
from .timeutil import iso

log = logging.getLogger(__name__)

FEATURE_VALUE_COLS = [c for c in S.FEATURE_COLUMNS if c not in ("prediction_id", "feature_version")]
LABEL_VALUE_COLS = [c for c in S.LABEL_COLUMNS if c not in ("prediction_id", "label_version")]

DDL = f"""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY, signup_timestamp TEXT NOT NULL, region TEXT, customer_segment TEXT);
CREATE TABLE IF NOT EXISTS canonical_events (
    event_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    event_timestamp TEXT NOT NULL, available_at TEXT NOT NULL,
    event_type TEXT NOT NULL, event_status TEXT NOT NULL, amount REAL, currency TEXT,
    source_file TEXT, source_row_number INTEGER, run_id TEXT);
CREATE TABLE IF NOT EXISTS prediction_requests (
    prediction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    prediction_timestamp TEXT NOT NULL, request_type TEXT NOT NULL,
    UNIQUE (customer_id, prediction_timestamp));
CREATE TABLE IF NOT EXISTS customer_features (
    prediction_id TEXT NOT NULL REFERENCES prediction_requests(prediction_id),
    feature_version TEXT NOT NULL, customer_id TEXT NOT NULL, prediction_timestamp TEXT NOT NULL,
    purchase_count_30d INTEGER NOT NULL, purchase_amount_sum_30d REAL NOT NULL,
    purchase_amount_avg_30d REAL, days_since_last_purchase REAL,
    refund_count_30d INTEGER NOT NULL, login_count_30d INTEGER NOT NULL, customer_tenure_days INTEGER NOT NULL,
    run_id TEXT, PRIMARY KEY (prediction_id, feature_version));
CREATE TABLE IF NOT EXISTS training_labels (
    prediction_id TEXT NOT NULL REFERENCES prediction_requests(prediction_id),
    label_version TEXT NOT NULL, customer_id TEXT NOT NULL, prediction_timestamp TEXT NOT NULL,
    label_window_end TEXT NOT NULL, target_purchase_next_7d INTEGER,
    label_status TEXT NOT NULL CHECK (label_status IN ('complete','incomplete')), run_id TEXT,
    PRIMARY KEY (prediction_id, label_version),
    CHECK ((label_status = 'complete' AND target_purchase_next_7d IN (0,1))
        OR (label_status = 'incomplete' AND target_purchase_next_7d IS NULL)));
CREATE TABLE IF NOT EXISTS rejected_records (
    run_id TEXT NOT NULL, source_table TEXT NOT NULL, source_file TEXT NOT NULL,
    source_row_number INTEGER NOT NULL, record_id TEXT, reject_reason TEXT NOT NULL,
    category TEXT NOT NULL, raw_record TEXT,
    PRIMARY KEY (run_id, source_table, source_file, source_row_number, category));
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT, status TEXT, failed_stage TEXT,
    input_rows INTEGER, valid_rows INTEGER, rejected_rows INTEGER, duplicate_rows INTEGER,
    feature_rows INTEGER, label_rows INTEGER, duration_seconds REAL, error_message TEXT,
    config_hash TEXT, details_json TEXT);
"""


def connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # explicit transactions
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn) -> None:
    conn.executescript(DDL)


# ------------------------------------------------------------------ run ledger
def start_run(conn, rec, cfg: FeatureConfig) -> None:
    conn.execute("INSERT INTO pipeline_runs (run_id, started_at, status, config_hash) VALUES (?,?,?,?)",
                 (rec.run_id, rec.started_at, "running", cfg.config_hash))


def finish_run(conn, rec, summary: dict) -> None:
    t = summary["totals"]
    conn.execute(
        """UPDATE pipeline_runs SET ended_at=?, status=?, failed_stage=?, input_rows=?, valid_rows=?,
           rejected_rows=?, duplicate_rows=?, feature_rows=?, label_rows=?, duration_seconds=?,
           error_message=?, details_json=? WHERE run_id=?""",
        (rec.ended_at, rec.status, rec.failed_stage, t["input_rows"], t["valid_rows"], t["rejected_rows"],
         t["duplicate_rows"], summary["feature_rows"], summary["label_rows"], rec.duration_seconds,
         rec.error_message, json.dumps(summary, default=str), rec.run_id))


# ------------------------------------------------------------------ upsert machinery
def _norm(v):
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 6)
    return v


def _rows(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    obj = df[cols].astype(object)
    obj = obj.where(df[cols].notna(), None)
    return [tuple(_norm(v) for v in r) for r in obj.itertuples(index=False, name=None)]


def _upsert(conn, table, key_cols, value_cols, extra_cols, df, where=None, params=()) -> dict:
    cols = key_cols + value_cols + extra_cols
    rows = _rows(df, cols)
    nk, nv = len(key_cols), len(value_cols)
    sel = f"SELECT {', '.join(key_cols + value_cols)} FROM {table}" + (f" WHERE {where}" if where else "")
    existing = {tuple(r[:nk]): tuple(_norm(x) for x in r[nk:]) for r in conn.execute(sel, params)}
    to_write, ins, upd, unch = [], 0, 0, 0
    for r in rows:
        old = existing.get(r[:nk])
        if old is None:
            ins += 1
            to_write.append(r)
        elif old != r[nk:nk + nv]:
            upd += 1
            to_write.append(r)
        else:
            unch += 1
    if to_write:
        sets = ", ".join(f"{c}=excluded.{c}" for c in value_cols + extra_cols)
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {sets}", to_write)
    return {"inserted": ins, "updated": upd, "unchanged": unch}


def _with_iso(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = iso(out[c])
    return out


# ------------------------------------------------------------------ table publishers
def _publish_customers(conn, customers):
    df = _with_iso(customers, ["signup_timestamp"])
    return _upsert(conn, "customers", ["customer_id"], ["signup_timestamp", "region", "customer_segment"], [], df)


def _publish_events(conn, events, run_id):
    df = _with_iso(events, ["event_timestamp", "available_at"]).assign(run_id=run_id)
    return _upsert(conn, "canonical_events", ["event_id"],
                   ["customer_id", "event_timestamp", "available_at", "event_type", "event_status", "amount", "currency"],
                   ["source_file", "source_row_number", "run_id"], df)


def _publish_predictions(conn, preds):
    df = _with_iso(preds, ["prediction_timestamp"])
    return _upsert(conn, "prediction_requests", ["prediction_id"],
                   ["customer_id", "prediction_timestamp", "request_type"], [], df)


def _publish_features(conn, features, run_id, cfg):
    df = _with_iso(features, ["prediction_timestamp"]).assign(run_id=run_id)
    return _upsert(conn, "customer_features", ["prediction_id", "feature_version"], FEATURE_VALUE_COLS, ["run_id"],
                   df, where="feature_version = ?", params=(cfg.feature_version,))


def _publish_labels(conn, labels, run_id, cfg):
    df = _with_iso(labels, ["prediction_timestamp", "label_window_end"]).assign(run_id=run_id)
    return _upsert(conn, "training_labels", ["prediction_id", "label_version"], LABEL_VALUE_COLS, ["run_id"],
                   df, where="label_version = ?", params=(cfg.label_version,))


def _publish_rejects(conn, rejects):
    if rejects.empty:
        return {"inserted": 0, "updated": 0, "unchanged": 0}
    cols = ["run_id", "source_table", "source_file", "source_row_number", "record_id",
            "reject_reason", "category", "raw_record"]
    conn.executemany(f"INSERT OR REPLACE INTO rejected_records ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     _rows(rejects, cols))
    return {"inserted": len(rejects), "updated": 0, "unchanged": 0}


def publish_all(conn, run_id, cfg, customers, events, predictions, features, labels, rejects) -> dict:
    """Atomically publish everything; roll back all tables on any failure."""
    conn.execute("BEGIN")
    try:
        result = {
            "customers": _publish_customers(conn, customers),
            "canonical_events": _publish_events(conn, events, run_id),
            "prediction_requests": _publish_predictions(conn, predictions),
            "customer_features": _publish_features(conn, features, run_id, cfg),
            "training_labels": _publish_labels(conn, labels, run_id, cfg),
            "rejected_records": _publish_rejects(conn, rejects),
        }
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        log.error("publish failed; transaction rolled back")
        raise
    if result["canonical_events"]["updated"]:
        log.warning("%d stored events changed content (batch-snapshot contract: input is authoritative)",
                    result["canonical_events"]["updated"])
    return result
