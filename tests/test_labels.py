import pandas as pd

from src.features import build_features
from src.labels import assemble_training_dataset, build_labels
from .conftest import customers_df, preds_df, ts, tx_df

T = "2026-10-01T00:00:00Z"
WM_FULL = ts("2026-10-10T00:00:00Z")   # >= T + 7d  -> complete
WM_SHORT = ts("2026-10-05T00:00:00Z")  # <  T + 7d  -> incomplete


def test_future_qualifying_purchase_gives_1(cfg):
    r = build_labels(tx_df([dict(event_id="x", event_timestamp="2026-10-03T00:00:00Z")]), preds_df([("p1", "C1", T)]), cfg, WM_FULL).iloc[0]
    assert r["target_purchase_next_7d"] == 1 and r["label_status"] == "complete"


def test_no_future_purchase_complete_horizon_gives_0(cfg):
    r = build_labels(tx_df([]), preds_df([("p1", "C1", T)]), cfg, WM_FULL).iloc[0]
    assert r["target_purchase_next_7d"] == 0 and r["label_status"] == "complete"


def test_incomplete_horizon_is_null_not_zero(cfg):
    r = build_labels(tx_df([]), preds_df([("p1", "C1", T)]), cfg, WM_SHORT).iloc[0]
    assert pd.isna(r["target_purchase_next_7d"]) and r["label_status"] == "incomplete"


def test_incomplete_even_if_purchase_already_seen(cfg):
    ev = [dict(event_id="x", event_timestamp="2026-10-02T00:00:00Z")]
    r = build_labels(tx_df(ev), preds_df([("p1", "C1", T)]), cfg, WM_SHORT).iloc[0]
    assert pd.isna(r["target_purchase_next_7d"])


def test_unknown_watermark_means_incomplete(cfg):
    r = build_labels(tx_df([]), preds_df([("p1", "C1", T)]), cfg, None).iloc[0]
    assert r["label_status"] == "incomplete"


def test_watermark_exactly_at_window_end_is_complete(cfg):
    r = build_labels(tx_df([]), preds_df([("p1", "C1", T)]), cfg, ts("2026-10-08T00:00:00Z")).iloc[0]
    assert r["label_status"] == "complete" and r["target_purchase_next_7d"] == 0


def test_label_window_boundaries(cfg):
    # exactly T is NOT in the future window; exactly T+7d IS
    at_t = [dict(event_id="a", event_timestamp=T)]
    at_end = [dict(event_id="b", event_timestamp="2026-10-08T00:00:00Z")]
    after_end = [dict(event_id="c", event_timestamp="2026-10-08T00:00:01Z")]
    req = preds_df([("p1", "C1", T)])
    assert build_labels(tx_df(at_t), req, cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 0
    assert build_labels(tx_df(at_end), req, cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 1
    assert build_labels(tx_df(after_end), req, cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 0


def test_only_completed_purchases_count_for_label(cfg):
    ev = [dict(event_id="a", event_timestamp="2026-10-02T00:00:00Z", event_status="pending"),
          dict(event_id="b", event_timestamp="2026-10-02T00:00:00Z", event_type="refund"),
          dict(event_id="c", event_timestamp="2026-10-02T00:00:00Z", event_type="login")]
    assert build_labels(tx_df(ev), preds_df([("p1", "C1", T)]), cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 0


def test_future_purchase_changes_label_but_not_features(cfg):
    hist = [dict(event_id="h", event_timestamp="2026-09-20T00:00:00Z")]
    fut = hist + [dict(event_id="f", event_timestamp="2026-10-03T00:00:00Z")]
    req = preds_df([("p1", "C1", T)])
    f0 = build_features(customers_df(), tx_df(hist), req, cfg)
    f1 = build_features(customers_df(), tx_df(fut), req, cfg)
    pd.testing.assert_frame_equal(f0, f1, check_exact=True)
    assert build_labels(tx_df(hist), req, cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 0
    assert build_labels(tx_df(fut), req, cfg, WM_FULL).iloc[0]["target_purchase_next_7d"] == 1


def test_training_dataset_joins_only_complete_labels(cfg):
    req = preds_df([("p1", "C1", T), ("p2", "C1", "2026-10-04T00:00:00Z")])
    feats = build_features(customers_df(), tx_df([]), req, cfg)
    labels = build_labels(tx_df([]), req, cfg, ts("2026-10-09T00:00:00Z"))  # p1 complete, p2 incomplete
    ds = assemble_training_dataset(feats, labels)
    assert ds["prediction_id"].tolist() == ["p1"] and "target_purchase_next_7d" in ds.columns
