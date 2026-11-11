import numpy as np
import pandas as pd
import pytest

from src import schemas as S
from src.features import FEATURE_DTYPES, build_features
from .conftest import customers_df, preds_df, tx_df

T = "2026-10-20T12:00:00Z"


def feats(events, cfg, signup="2026-01-01T00:00:00Z"):
    return build_features(customers_df([("C1", signup)]), tx_df(events), preds_df([("p1", "C1", T)]), cfg).iloc[0]


def test_happy_path_values(cfg):
    ev = [dict(event_id="1", event_timestamp="2026-10-10T12:00:00Z", amount=10.0),
          dict(event_id="2", event_timestamp="2026-10-18T12:00:00Z", amount=30.0),
          dict(event_id="3", event_timestamp="2026-10-12T00:00:00Z", event_type="refund", amount=5.0),
          dict(event_id="4", event_timestamp="2026-10-13T00:00:00Z", event_type="login"),
          dict(event_id="5", event_timestamp="2026-10-14T00:00:00Z", event_type="login")]
    r = feats(ev, cfg)
    assert r["purchase_count_30d"] == 2 and r["purchase_amount_sum_30d"] == 40.0
    assert r["purchase_amount_avg_30d"] == 20.0 and r["days_since_last_purchase"] == 2.0
    assert r["refund_count_30d"] == 1 and r["login_count_30d"] == 2
    assert r["customer_tenure_days"] == (pd.Timestamp(T) - pd.Timestamp("2026-01-01", tz="UTC")).days


def test_only_completed_purchases_count(cfg):
    ev = [dict(event_id=str(i), event_timestamp="2026-10-10T00:00:00Z", event_status=s)
          for i, s in enumerate(["pending", "failed", "cancelled"])]
    r = feats(ev, cfg)
    assert r["purchase_count_30d"] == 0 and r["purchase_amount_sum_30d"] == 0.0


def test_refunds_are_not_purchases_and_not_netted(cfg):
    r = feats([dict(event_id="1", event_timestamp="2026-10-10T00:00:00Z", event_type="refund", amount=50.0)], cfg)
    assert r["purchase_count_30d"] == 0 and r["purchase_amount_sum_30d"] == 0.0 and r["refund_count_30d"] == 1


def test_no_purchase_history_zero_vs_null_policy(cfg):
    r = feats([], cfg)
    assert r["purchase_count_30d"] == 0 and r["purchase_amount_sum_30d"] == 0.0
    assert np.isnan(r["purchase_amount_avg_30d"]) and np.isnan(r["days_since_last_purchase"])
    assert r["refund_count_30d"] == 0 and r["login_count_30d"] == 0


def test_empty_transactions_for_many_requests(cfg):
    out = build_features(customers_df(), tx_df([]), preds_df([("p1", "C1", T), ("p2", "C1", "2026-10-21T00:00:00Z")]), cfg)
    assert len(out) == 2 and (out["purchase_count_30d"] == 0).all()


def test_prediction_before_signup_raises(cfg):
    with pytest.raises(ValueError, match="before customer signup"):
        feats([], cfg, signup="2026-11-01T00:00:00Z")


def test_unknown_customer_raises(cfg):
    with pytest.raises(ValueError, match="unknown customer"):
        build_features(customers_df(), tx_df([]), preds_df([("p1", "C9", T)]), cfg)


def test_label_columns_rejected_by_build_features(cfg):
    req = preds_df([("p1", "C1", T)])
    req["target_purchase_next_7d"] = 1
    with pytest.raises(ValueError, match="label columns"):
        build_features(customers_df(), tx_df([]), req, cfg)


def test_column_order_and_dtypes_are_stable(cfg):
    out = build_features(customers_df(), tx_df([dict(event_id="1", event_timestamp="2026-10-10T00:00:00Z")]),
                         preds_df([("p1", "C1", T)]), cfg)
    assert list(out.columns) == S.FEATURE_COLUMNS
    for col, dt in FEATURE_DTYPES.items():
        assert str(out[col].dtype) == dt
    empty = build_features(customers_df(), tx_df([]), preds_df([("p1", "C1", T)]), cfg)
    assert list(empty.dtypes.astype(str)) == list(out.dtypes.astype(str))


def test_training_inference_parity(cfg):
    ev = [dict(event_id=str(i), event_timestamp=f"2026-10-{d:02d}T00:00:00Z") for i, d in enumerate(range(2, 19))]
    rows = [("p1", "C1", T, "training"), ("p2", "C1", "2026-10-25T00:00:00Z", "inference")]
    both = build_features(customers_df(), tx_df(ev), preds_df(rows), cfg)
    train = build_features(customers_df(), tx_df(ev), preds_df(rows[:1]), cfg)
    infer = build_features(customers_df(), tx_df(ev), preds_df([("p1", "C1", T, "inference")]), cfg)
    pd.testing.assert_frame_equal(both.iloc[[0]].reset_index(drop=True), train, check_exact=True)
    pd.testing.assert_frame_equal(train, infer, check_exact=True)   # request type does not change values
    assert not (S.LABEL_LEAK_COLUMNS & set(infer.columns))
