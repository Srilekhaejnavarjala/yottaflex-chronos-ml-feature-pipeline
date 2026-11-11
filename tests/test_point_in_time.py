import pandas as pd

from src.features import build_features, build_features_reference
from .conftest import customers_df, preds_df, tx_df

T = "2026-10-20T12:00:00Z"


def one(events, cfg, ptype="training", customers=None):
    return build_features(customers if customers is not None else customers_df(), tx_df(events), preds_df([("p1", "C1", T, ptype)]), cfg).iloc[0]


# The worked example from the design doc (events A-F)
WORKED = [
    dict(event_id="A", event_timestamp="2026-10-01T10:00:00Z", available_at="2026-10-01T10:05:00Z"),
    dict(event_id="B", event_timestamp="2026-10-15T09:00:00Z", available_at="2026-10-15T09:01:00Z"),
    dict(event_id="C", event_timestamp="2026-10-20T11:59:00Z", available_at="2026-10-20T11:59:30Z"),
    dict(event_id="D", event_timestamp="2026-10-20T12:01:00Z", available_at="2026-10-20T12:01:00Z"),  # after cutoff
    dict(event_id="E", event_timestamp="2026-10-18T15:00:00Z", available_at="2026-10-21T09:00:00Z"),  # late arrival
    dict(event_id="F", event_timestamp="2026-09-20T12:00:00Z", available_at="2026-09-20T12:01:00Z"),  # lower boundary
]


def test_worked_example_only_A_B_C_count(cfg):
    assert one(WORKED, cfg)["purchase_count_30d"] == 3


def test_event_one_second_after_T_excluded(cfg):
    assert one([dict(event_id="x", event_timestamp="2026-10-20T12:00:01Z")], cfg)["purchase_count_30d"] == 0


def test_event_exactly_at_T_included_upper_bound_inclusive(cfg):
    r = one([dict(event_id="x", event_timestamp=T, available_at=T)], cfg)
    assert r["purchase_count_30d"] == 1 and r["days_since_last_purchase"] == 0.0


def test_event_exactly_at_lower_boundary_excluded(cfg):
    assert one([dict(event_id="x", event_timestamp="2026-09-20T12:00:00Z")], cfg)["purchase_count_30d"] == 0


def test_event_one_second_inside_lower_boundary_included(cfg):
    assert one([dict(event_id="x", event_timestamp="2026-09-20T12:00:01Z")], cfg)["purchase_count_30d"] == 1


def test_late_arriving_event_excluded_by_availability_time(cfg):
    ev = dict(event_id="x", event_timestamp="2026-10-18T15:00:00Z", available_at="2026-10-21T09:00:00Z")
    assert one([ev], cfg)["purchase_count_30d"] == 0
    # ...but the same event IS visible to a later prediction made after it arrived
    later = build_features(customers_df(), tx_df([ev]), preds_df([("p2", "C1", "2026-10-21T09:00:00Z")]), cfg).iloc[0]
    assert later["purchase_count_30d"] == 1


def test_availability_exactly_at_T_is_known(cfg):
    ev = dict(event_id="x", event_timestamp="2026-10-19T00:00:00Z", available_at=T)
    assert one([ev], cfg)["purchase_count_30d"] == 1


def test_other_customer_excluded(cfg):
    cust = customers_df([("C1", "2026-01-01T00:00:00Z"), ("C2", "2026-01-01T00:00:00Z")])
    ev = dict(event_id="x", customer_id="C2", event_timestamp="2026-10-10T00:00:00Z")
    assert one([ev], cfg, customers=cust)["purchase_count_30d"] == 0


def test_appending_future_events_does_not_change_historical_features(cfg):
    base = tx_df(WORKED[:3])
    future = tx_df([dict(event_id=f"f{i}", event_timestamp=f"2026-10-2{i}T12:00:00Z", amount=500.0) for i in range(1, 6)])
    req = preds_df([("p1", "C1", T)])
    before = build_features(customers_df(), base, req, cfg)
    after = build_features(customers_df(), pd.concat([base, future], ignore_index=True), req, cfg)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_optimized_equals_reference_on_generated_data(small_data, cfg):
    from src.ingest import read_customers, read_predictions, read_transactions
    from src.validate import validate_customers, validate_predictions, validate_transactions
    c = validate_customers(read_customers(small_data / "customers.csv"), cfg, "r")
    t = validate_transactions(read_transactions([small_data / "transactions.csv"]), c.valid, cfg, "r")
    p = validate_predictions(read_predictions(small_data / "prediction_requests.csv"), c.valid, cfg, "r")
    fast = build_features(c.valid, t.valid, p.valid, cfg)
    slow = build_features_reference(c.valid, t.valid, p.valid, cfg)
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)
