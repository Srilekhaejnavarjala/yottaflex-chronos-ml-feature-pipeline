"""Business validation, duplicate policy and quarantine.

Every input row ends in exactly one bucket:
  valid      -> canonical table
  reject     -> quarantined with all applicable reasons (``category = reject``)
  duplicate  -> exact replay of an already-kept row (``category = duplicate``)

So per table:  input_rows = valid_rows + rejected_rows + duplicate_rows.

Duplicate policy (same key):
  * identical normalized content   -> first row kept, later ones are replays
  * different content              -> *all* rows with that key are quarantined
    (conflict; order-independent, nothing trusted without a correction contract)
  * different keys, similar content -> never deduplicated
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd

from . import schemas as S
from .config import FeatureConfig
from .timeutil import parse_utc, to_ns

REJECT_COLUMNS = ["run_id", "source_table", "source_file", "source_row_number",
                  "record_id", "reject_reason", "category", "raw_record"]


@dataclass
class TableResult:
    table: str
    input_rows: int
    valid: pd.DataFrame
    rejects: pd.DataFrame

    @property
    def valid_rows(self) -> int:
        return len(self.valid)

    @property
    def rejected_rows(self) -> int:
        return int((self.rejects["category"] == "reject").sum())

    @property
    def duplicate_rows(self) -> int:
        return int((self.rejects["category"] == "duplicate").sum())


def _reasons(index, masks: dict) -> pd.Series:
    out = pd.Series("", index=index, dtype=object)
    for name, mask in masks.items():
        out = out.where(~mask, out + name + ";")
    return out.str.rstrip(";")


def _reject_rows(rows: pd.DataFrame, reasons, table, id_col, raw_cols, category, run_id) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=REJECT_COLUMNS)
    raw = [json.dumps(d, ensure_ascii=False) for d in rows[raw_cols].to_dict("records")]
    return pd.DataFrame({
        "run_id": run_id, "source_table": table,
        "source_file": rows["source_file"].to_numpy(),
        "source_row_number": rows["source_row_number"].to_numpy(),
        "record_id": rows[id_col].map(lambda x: str(x).strip() or None).to_numpy(),
        "reject_reason": reasons.to_numpy() if hasattr(reasons, "to_numpy") else reasons,
        "category": category, "raw_record": raw,
    })


def _content(parts: list[pd.Series]) -> pd.Series:
    return reduce(lambda a, b: a + "|" + b, [p.astype(str) for p in parts])


def _dedupe(canon: pd.DataFrame, key: str, content: pd.Series):
    nunique = content.groupby(canon[key]).transform("nunique")
    conflict = nunique > 1
    replay = (~conflict) & canon.duplicated(subset=key, keep="first")
    return conflict, replay


def _split(raw, row_reasons, canon, key, content, table, id_col, raw_cols, run_id, conflict_reason):
    """Common tail: row-level rejects + dedupe -> (valid, rejects)."""
    row_ok = row_reasons == ""
    parts = [_reject_rows(raw[~row_ok], row_reasons[~row_ok], table, id_col, raw_cols, "reject", run_id)]
    if len(canon):
        conflict, replay = _dedupe(canon, key, content)
        rows = raw.loc[canon.index]
        parts.append(_reject_rows(rows[conflict], pd.Series(conflict_reason, index=rows[conflict].index),
                                  table, id_col, raw_cols, "reject", run_id))
        parts.append(_reject_rows(rows[replay], pd.Series("duplicate_replay", index=rows[replay].index),
                                  table, id_col, raw_cols, "duplicate", run_id))
        valid = canon[~conflict & ~replay]
    else:
        valid = canon
    return valid, parts


def _finish(parts) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=REJECT_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["source_file", "source_row_number", "category"], kind="mergesort").reset_index(drop=True)


def _raw_cols(raw):
    return [c for c in raw.columns if c not in ("source_file", "source_row_number")]


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _nullable(s: pd.Series) -> pd.Series:
    return pd.Series(np.where(s == "", None, s), index=s.index, dtype=object)


# --------------------------------------------------------------------------- customers
def validate_customers(raw: pd.DataFrame, cfg: FeatureConfig, run_id: str) -> TableResult:
    cid, signup = _clean(raw["customer_id"]), parse_utc(raw["signup_timestamp"])
    reasons = _reasons(raw.index, {
        "null_customer_id": cid == "",
        "invalid_signup_timestamp": signup.isna(),
    })
    ok = reasons == ""
    canon = pd.DataFrame({
        "customer_id": cid[ok], "signup_timestamp": signup[ok],
        "region": _nullable(_clean(raw["region"]))[ok], "customer_segment": _nullable(_clean(raw["customer_segment"]))[ok],
        "source_file": raw["source_file"][ok], "source_row_number": raw["source_row_number"][ok],
    })
    content = None
    if len(canon):
        content = _content([canon.customer_id, pd.Series(to_ns(canon.signup_timestamp), index=canon.index),
                            canon.region.fillna(""), canon.customer_segment.fillna("")])
    valid, parts = _split(raw, reasons, canon, "customer_id", content, "customers", "customer_id",
                          _raw_cols(raw), run_id, "conflicting_duplicate_customer_id")
    return TableResult("customers", len(raw), valid.reset_index(drop=True), _finish(parts))


# ------------------------------------------------------------------------- transactions
def validate_transactions(raw: pd.DataFrame, customers_valid: pd.DataFrame, cfg: FeatureConfig,
                          run_id: str) -> TableResult:
    eid, cid = _clean(raw["event_id"]), _clean(raw["customer_id"])
    ts, av = parse_utc(raw["event_timestamp"]), parse_utc(raw["available_at"])
    etype, estatus, cur = _clean(raw["event_type"]), _clean(raw["event_status"]), _clean(raw["currency"])
    amt_s = _clean(raw["amount"])
    has_amt = amt_s != ""
    amount = pd.to_numeric(amt_s.where(has_amt), errors="coerce")
    bad_num = has_amt & (amount.isna() | np.isinf(amount))
    type_ok = etype.isin(S.EVENT_TYPES)
    monetary = etype.isin(S.MONETARY_EVENT_TYPES)
    reasons = _reasons(raw.index, {
        "null_event_id": eid == "",
        "null_customer_id": cid == "",
        "invalid_event_timestamp": ts.isna(),
        "invalid_available_at": av.isna(),
        "invalid_event_type": ~type_ok,
        "invalid_event_status": ~estatus.isin(S.EVENT_STATUSES),
        "unknown_customer": (cid != "") & ~cid.isin(set(customers_valid["customer_id"])),
        "available_before_event": ts.notna() & av.notna() & (av < ts),
        "invalid_amount_format": bad_num,
        "missing_amount": monetary & ~has_amt,
        "non_positive_amount": monetary & has_amt & ~bad_num & (amount <= 0),
        "unexpected_amount": type_ok & ~monetary & has_amt,
        "invalid_currency": monetary & (cur != cfg.currency),
    })
    ok = reasons == ""
    mon_ok = monetary[ok]
    canon = pd.DataFrame({
        "event_id": eid[ok], "customer_id": cid[ok],
        "event_timestamp": ts[ok], "available_at": av[ok],
        "event_type": etype[ok], "event_status": estatus[ok],
        "amount": amount[ok].where(mon_ok).astype("float64"),
        "currency": pd.Series(np.where(mon_ok, cur[ok], None), index=mon_ok.index, dtype=object),
        "source_file": raw["source_file"][ok], "source_row_number": raw["source_row_number"][ok],
    })
    canon["run_id"] = run_id
    content = None
    if len(canon):
        content = _content([
            canon.customer_id,
            pd.Series(to_ns(canon.event_timestamp), index=canon.index),
            pd.Series(to_ns(canon.available_at), index=canon.index),
            canon.event_type, canon.event_status,
            canon.amount.fillna(-1.0).map(lambda x: f"{x:.6f}"),
            canon.currency.fillna(""),
        ])
    valid, parts = _split(raw, reasons, canon, "event_id", content, "transactions", "event_id",
                          _raw_cols(raw), run_id, "conflicting_duplicate_event_id")
    return TableResult("transactions", len(raw), valid.reset_index(drop=True), _finish(parts))


# ------------------------------------------------------------------ prediction requests
def validate_predictions(raw: pd.DataFrame, customers_valid: pd.DataFrame, cfg: FeatureConfig,
                         run_id: str) -> TableResult:
    pid, cid = _clean(raw["prediction_id"]), _clean(raw["customer_id"])
    ts, rtype = parse_utc(raw["prediction_timestamp"]), _clean(raw["request_type"])
    signup = cid.map(customers_valid.set_index("customer_id")["signup_timestamp"])
    reasons = _reasons(raw.index, {
        "null_prediction_id": pid == "",
        "null_customer_id": cid == "",
        "invalid_prediction_timestamp": ts.isna(),
        "invalid_request_type": ~rtype.isin(S.REQUEST_TYPES),
        "unknown_customer": (cid != "") & ~cid.isin(set(customers_valid["customer_id"])),
        "prediction_before_signup": ts.notna() & signup.notna() & (ts < signup),
    })
    ok = reasons == ""
    canon = pd.DataFrame({
        "prediction_id": pid[ok], "customer_id": cid[ok], "prediction_timestamp": ts[ok],
        "request_type": rtype[ok],
        "source_file": raw["source_file"][ok], "source_row_number": raw["source_row_number"][ok],
    })
    content = None
    if len(canon):
        content = _content([canon.customer_id, pd.Series(to_ns(canon.prediction_timestamp), index=canon.index),
                            canon.request_type])
    valid, parts = _split(raw, reasons, canon, "prediction_id", content, "prediction_requests", "prediction_id",
                          _raw_cols(raw), run_id, "conflicting_duplicate_prediction_id")
    # natural key (customer_id, prediction_timestamp) must also be unique; first row (file order) wins
    nk_dup = valid.duplicated(["customer_id", "prediction_timestamp"], keep="first")
    if nk_dup.any():
        rows = raw.loc[valid.index[nk_dup]]
        parts.append(_reject_rows(rows, pd.Series("duplicate_natural_key", index=rows.index),
                                  "prediction_requests", "prediction_id", _raw_cols(raw), "reject", run_id))
        valid = valid[~nk_dup]
    return TableResult("prediction_requests", len(raw), valid.reset_index(drop=True), _finish(parts))
