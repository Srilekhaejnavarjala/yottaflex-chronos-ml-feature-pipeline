import json

import pandas as pd
import pytest

from src.validate import validate_customers, validate_predictions, validate_transactions
from .conftest import customers_df, raw_frame

TX_COLS = ["event_id", "customer_id", "event_timestamp", "available_at", "event_type", "event_status", "amount", "currency"]
OK = ["e1", "C1", "2026-08-01T10:00:00Z", "2026-08-01T10:00:05Z", "purchase", "completed", "25.00", "USD"]


def tx(*rows):
    return raw_frame(rows, TX_COLS)


def run(rows, cfg):
    return validate_transactions(tx(*rows), customers_df(), cfg, "run1")


def variant(**kw):
    row = dict(zip(TX_COLS, OK))
    row.update(kw)
    return [row[c] for c in TX_COLS]


def reasons(res):
    return sorted(res.rejects["reject_reason"])


def test_clean_row_is_valid(cfg):
    r = run([OK], cfg)
    assert (r.valid_rows, r.rejected_rows, r.duplicate_rows) == (1, 0, 0)
    assert str(r.valid["event_timestamp"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("override,reason", [
    ({"event_id": ""}, "null_event_id"),
    ({"customer_id": ""}, "null_customer_id"),
    ({"customer_id": "C404"}, "unknown_customer"),
    ({"event_type": "gift"}, "invalid_event_type"),
    ({"event_status": "weird"}, "invalid_event_status"),
    ({"event_timestamp": "2026-13-45T00:00:00Z"}, "invalid_event_timestamp"),
    ({"event_timestamp": "2026-08-01T10:00:00"}, "invalid_event_timestamp"),   # naive timestamp
    ({"available_at": "garbage"}, "invalid_available_at"),
    ({"available_at": "2026-08-01T09:59:59Z"}, "available_before_event"),
    ({"amount": ""}, "missing_amount"),
    ({"amount": "-1"}, "non_positive_amount"),
    ({"amount": "0"}, "non_positive_amount"),
    ({"amount": "abc"}, "invalid_amount_format"),
    ({"amount": "inf"}, "invalid_amount_format"),
    ({"currency": "EUR"}, "invalid_currency"),
    ({"event_type": "login", "amount": "5.00"}, "unexpected_amount"),
])
def test_row_level_rejects(cfg, override, reason):
    r = run([variant(**override)], cfg)
    assert r.valid_rows == 0 and reason in r.rejects["reject_reason"].iloc[0].split(";")
    assert r.rejects["category"].iloc[0] == "reject"


def test_all_applicable_reasons_recorded(cfg):
    r = run([variant(customer_id="C404", event_type="gift", amount="")], cfg)
    got = set(r.rejects["reject_reason"].iloc[0].split(";"))
    assert {"unknown_customer", "invalid_event_type"} <= got


def test_login_without_amount_is_valid(cfg):
    assert run([variant(event_type="login", amount="", currency="")], cfg).valid_rows == 1


def test_reject_traceability_has_source_and_raw_record(cfg):
    r = run([OK, variant(event_id="e2", event_type="gift")], cfg)
    rej = r.rejects.iloc[0]
    assert rej["source_file"] == "t.csv" and rej["source_row_number"] == 3 and rej["record_id"] == "e2"
    assert json.loads(rej["raw_record"])["event_type"] == "gift"
    assert rej["run_id"] == "run1"


def test_duplicate_same_content_is_replay(cfg):
    r = run([OK, OK], cfg)
    assert (r.valid_rows, r.rejected_rows, r.duplicate_rows) == (1, 0, 1)
    assert r.rejects["reject_reason"].iloc[0] == "duplicate_replay"
    assert r.valid["source_row_number"].iloc[0] == 2  # first occurrence kept


def test_duplicate_conflicting_content_quarantines_all_copies(cfg):
    r = run([OK, variant(amount="99.00")], cfg)
    assert (r.valid_rows, r.rejected_rows) == (0, 2)
    assert set(r.rejects["reject_reason"]) == {"conflicting_duplicate_event_id"}


def test_similar_events_with_different_ids_not_deduplicated(cfg):
    r = run([OK, variant(event_id="e2")], cfg)
    assert r.valid_rows == 2 and r.duplicate_rows == 0


def test_accounting_equation(cfg):
    rows = [OK, OK, variant(event_id="e2", event_type="gift"), variant(event_id="e3", amount="1"), variant(event_id="e3", amount="2")]
    r = run(rows, cfg)
    assert r.input_rows == r.valid_rows + r.rejected_rows + r.duplicate_rows == 5


def test_empty_transactions(cfg):
    r = run([], cfg)
    assert r.input_rows == 0 and r.valid_rows == 0 and r.rejects.empty


# --- customers / predictions
def test_customer_rules(cfg):
    cols = ["customer_id", "signup_timestamp", "region", "customer_segment"]
    raw = raw_frame([["C1", "2026-01-01T00:00:00Z", "n", "s"], ["", "2026-01-01T00:00:00Z", "", ""],
                     ["C2", "nope", "", ""], ["C3", "2026-01-01T00:00:00Z", "", ""],
                     ["C3", "2026-02-01T00:00:00Z", "", ""]], cols)
    r = validate_customers(raw, cfg, "r")
    assert r.valid_rows == 1 and r.rejected_rows == 4 and r.duplicate_rows == 0
    assert r.input_rows == r.valid_rows + r.rejected_rows + r.duplicate_rows


def test_prediction_rules(cfg):
    cols = ["prediction_id", "customer_id", "prediction_timestamp", "request_type"]
    raw = raw_frame([
        ["p1", "C1", "2026-08-01T00:00:00Z", "training"],
        ["p2", "C1", "2025-12-31T23:59:59Z", "training"],   # before signup
        ["p3", "C1", "2026-08-02T00:00:00Z", "other"],      # bad request type
        ["p4", "C9", "2026-08-02T00:00:00Z", "training"],   # unknown customer
        ["p5", "C1", "2026-08-01T00:00:00Z", "inference"],  # natural-key duplicate of p1
        ["p1", "C1", "2026-08-01T00:00:00Z", "training"],   # replay of p1
        ["", "C1", "2026-08-03T00:00:00Z", "training"],     # null id
    ], cols)
    r = validate_predictions(raw, customers_df(), cfg, "r")
    assert r.valid["prediction_id"].tolist() == ["p1"]
    assert r.input_rows == 7 == r.valid_rows + r.rejected_rows + r.duplicate_rows
    got = set(r.rejects["reject_reason"])
    assert {"prediction_before_signup", "invalid_request_type", "unknown_customer", "duplicate_natural_key",
            "duplicate_replay", "null_prediction_id"} <= got
