"""Target labels, generated separately from features.

Qualifying future purchase:  T < event_timestamp <= T + horizon, completed purchase.
A label is final only when  data_complete_through >= T + horizon.  Otherwise
label_status = 'incomplete' and the target is NULL: missing future data is not
proof that no purchase happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as S
from .config import FeatureConfig, ensure_supported_windows
from .timeutil import NS_PER_DAY, to_ns


def build_labels(transactions: pd.DataFrame, prediction_requests: pd.DataFrame,
                 cfg: FeatureConfig, data_complete_through) -> pd.DataFrame:
    ensure_supported_windows(cfg)
    horizon_ns = cfg.target_horizon_days * NS_PER_DAY
    p = transactions[(transactions["event_type"] == cfg.qualifying_purchase_event_type)
                     & (transactions["event_status"] == cfg.qualifying_purchase_status)]
    by_cust = {c: np.sort(to_ns(g["event_timestamp"])) for c, g in p.groupby("customer_id")}
    watermark_ns = None if data_complete_through is None else int(
        data_complete_through.tz_convert("UTC").tz_localize(None).as_unit("ns").value)

    T_all = to_ns(prediction_requests["prediction_timestamp"])
    target, status = [], []
    for cid, T in zip(prediction_requests["customer_id"], T_all):
        end = T + horizon_ns
        if watermark_ns is not None and watermark_ns >= end:
            arr = by_cust.get(cid, np.zeros(0, np.int64))
            n = np.searchsorted(arr, end, side="right") - np.searchsorted(arr, T, side="right")  # T < ts <= end
            target.append(1 if n > 0 else 0)
            status.append("complete")
        else:
            target.append(None)
            status.append("incomplete")
    out = pd.DataFrame({
        "prediction_id": prediction_requests["prediction_id"].to_numpy(),
        "customer_id": prediction_requests["customer_id"].to_numpy(),
        "prediction_timestamp": prediction_requests["prediction_timestamp"].to_numpy(),
        "label_window_end": (prediction_requests["prediction_timestamp"]
                             + pd.Timedelta(days=cfg.target_horizon_days)).to_numpy(),
        "target_purchase_next_7d": pd.array(target, dtype="Int64"),
        "label_status": status,
        "label_version": cfg.label_version,
    })
    out["prediction_timestamp"] = pd.to_datetime(out["prediction_timestamp"], utc=True)
    out["label_window_end"] = pd.to_datetime(out["label_window_end"], utc=True)
    return out[S.LABEL_COLUMNS].sort_values("prediction_id", kind="mergesort").reset_index(drop=True)


def assemble_training_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join features to FINAL labels by prediction_id. Only place features meet labels."""
    final = labels[labels["label_status"] == "complete"][["prediction_id", "target_purchase_next_7d"]]
    return features.merge(final, on="prediction_id", how="inner").sort_values("prediction_id").reset_index(drop=True)
