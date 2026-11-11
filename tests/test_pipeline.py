import json
import sqlite3

import pandas as pd
import pytest

from src import publish
from src.ingest import SchemaError
from src.pipeline import run_pipeline


def go(data, tmp_path, include_invalid=True, db="chronos.db", out="out", **kw):
    tx = [data / "transactions.csv"] + ([data / "invalid_transactions.csv"] if include_invalid else [])
    return run_pipeline(data / "customers.csv", tx, data / "prediction_requests.csv",
                        tmp_path / db, tmp_path / out, **kw)


def table(db, name, drop=("run_id",)):
    conn = sqlite3.connect(db)
    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df.drop(columns=[c for c in drop if c in df.columns]).sort_values(list(df.columns[:2])).reset_index(drop=True)


def test_end_to_end_success_and_reconciliation(small_data, tmp_path):
    s = go(small_data, tmp_path)
    assert s["status"] == "success" and all(c["passed"] for c in s["checks"])
    t = s["totals"]
    assert t["input_rows"] == t["valid_rows"] + t["rejected_rows"] + t["duplicate_rows"]
    assert s["feature_rows"] == s["sources"]["prediction_requests"]["valid_rows"]
    assert s["sources"]["transactions"]["rejected_rows"] > 0 and s["sources"]["transactions"]["duplicate_rows"] == 1
    for f in ("features/features_v1.csv", "features/inference_features_v1.csv", "labels/labels_v1.csv",
              "features/training_dataset_v1.csv"):
        assert (tmp_path / "out" / f).exists()
    assert list((tmp_path / "out" / "reports").glob("run_*.txt"))


def test_inference_output_has_no_label_columns_and_only_inference_rows(small_data, tmp_path):
    go(small_data, tmp_path)
    inf = pd.read_csv(tmp_path / "out/features/inference_features_v1.csv")
    assert not ({"target_purchase_next_7d", "label_status", "label_window_end"} & set(inf.columns))
    preds = pd.read_csv(small_data / "prediction_requests.csv")
    assert set(inf["prediction_id"]) == set(preds[preds["request_type"] == "inference"]["prediction_id"])


def test_incomplete_labels_are_null_in_published_labels(small_data, tmp_path):
    go(small_data, tmp_path)
    lab = pd.read_csv(tmp_path / "out/labels/labels_v1.csv")
    assert (lab[lab["label_status"] == "incomplete"]["target_purchase_next_7d"].isna()).all()
    assert (lab[lab["label_status"] == "complete"]["target_purchase_next_7d"].isin([0, 1])).all()
    assert (lab["label_status"] == "incomplete").any()


def test_boundary_fixtures_end_to_end(small_data, tmp_path):
    """Recompute purchase_count_30d for boundary predictions straight from the raw CSV with plain
    pandas and the documented rule; the pipeline output must agree, and the three excluded
    boundary events (lower_excl, after_cutoff, late_arrival) must not be counted."""
    go(small_data, tmp_path)
    feats = pd.read_csv(tmp_path / "out/features/features_v1.csv").set_index("prediction_id")
    tx = pd.read_csv(small_data / "transactions.csv")
    preds = pd.read_csv(small_data / "prediction_requests.csv").set_index("prediction_id")
    for c in ("event_timestamp", "available_at"):
        tx[c] = pd.to_datetime(tx[c], utc=True)
    nums = sorted({int(e.split("_")[1][1:]) for e in tx["event_id"] if e.startswith("evt_b")})
    boundary_ids = [f"pred_{n:07d}" for n in nums]
    assert len(boundary_ids) == 10
    for pid, n in zip(boundary_ids, nums):
        T = pd.Timestamp(preds.loc[pid, "prediction_timestamp"])
        mine = tx[(tx["customer_id"] == preds.loc[pid, "customer_id"]) & (tx["event_type"] == "purchase")
                  & (tx["event_status"] == "completed")]
        elig = mine[(mine["event_timestamp"] > T - pd.Timedelta(days=30)) & (mine["event_timestamp"] <= T)
                    & (mine["available_at"] <= T)]
        assert feats.loc[pid, "purchase_count_30d"] == len(elig)
        eligible_ids = set(elig["event_id"])
        for tag in ("lower_excl", "after_cutoff", "late_arrival"):
            assert f"evt_b{n:05d}_{tag}" not in eligible_ids
        for tag in ("lower_incl", "at_cutoff", "before_cutoff"):
            assert f"evt_b{n:05d}_{tag}" in eligible_ids


def test_rerun_is_idempotent(small_data, tmp_path):
    first = go(small_data, tmp_path)
    snap = {t: table(tmp_path / "chronos.db", t, drop=("run_id", "source_file", "source_row_number"))
            for t in ("customers", "canonical_events", "prediction_requests", "customer_features", "training_labels")}
    second = go(small_data, tmp_path)
    assert second["run_id"] != first["run_id"]
    for t, before in snap.items():
        after = table(tmp_path / "chronos.db", t, drop=("run_id", "source_file", "source_row_number"))
        pd.testing.assert_frame_equal(before, after)                       # identical business values
    for t in ("customer_features", "training_labels", "canonical_events", "customers", "prediction_requests"):
        p = second["publish"][t]
        assert p["inserted"] == 0 and p["updated"] == 0 and p["unchanged"] > 0, (t, p)
    conn = sqlite3.connect(tmp_path / "chronos.db")
    assert conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 2   # new run record each time
    n = conn.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    assert n == first["feature_rows"]                                               # no duplicate logical rows
    assert conn.execute("SELECT COUNT(*) FROM (SELECT 1 FROM customer_features GROUP BY prediction_id, feature_version HAVING COUNT(*)>1)").fetchone()[0] == 0
    conn.close()


def test_failure_during_publish_leaves_no_partial_data_and_retry_works(small_data, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk exploded")
    monkeypatch.setattr(publish, "_publish_labels", boom)
    with pytest.raises(RuntimeError):
        go(small_data, tmp_path)
    db = tmp_path / "chronos.db"
    conn = sqlite3.connect(db)
    for t in ("customers", "canonical_events", "prediction_requests", "customer_features", "training_labels"):
        assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0, t      # rolled back, nothing partial
    run = conn.execute("SELECT status, failed_stage, error_message FROM pipeline_runs").fetchone()
    conn.close()
    assert run[0] == "failed" and run[1] == "publish" and "disk exploded" in run[2]
    assert not list((tmp_path / "out" / "features").glob("*.csv"))                  # no outputs from a failed run

    monkeypatch.undo()                                                              # retry is safe
    ok = go(small_data, tmp_path)
    assert ok["status"] == "success"
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0] == ok["feature_rows"]
    assert [r[0] for r in conn.execute("SELECT status FROM pipeline_runs ORDER BY started_at")] == ["failed", "success"]
    conn.close()


def test_missing_column_fails_at_ingest_with_ledger_entry(small_data, tmp_path):
    bad = tmp_path / "bad_customers.csv"
    bad.write_text("customer_id,region\nC1,north\n")
    with pytest.raises(SchemaError):
        run_pipeline(bad, [small_data / "transactions.csv"], small_data / "prediction_requests.csv",
                     tmp_path / "db.sqlite", tmp_path / "out")
    conn = sqlite3.connect(tmp_path / "db.sqlite")
    status, stage, msg = conn.execute("SELECT status, failed_stage, error_message FROM pipeline_runs").fetchone()
    conn.close()
    assert (status, stage) == ("failed", "ingest") and "signup_timestamp" in msg


def test_unsupported_schema_version_fails(small_data, tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    for f in ("customers.csv", "transactions.csv", "prediction_requests.csv"):
        (d / f).write_text((small_data / f).read_text())
    meta = json.loads((small_data / "source_metadata.json").read_text())
    meta["schema_version"] = "99"
    (d / "source_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaError, match="unsupported schema_version"):
        go(d, tmp_path, include_invalid=False)


def test_reconciliation_failure_marks_run_failed(small_data, tmp_path, monkeypatch):
    import src.pipeline as pl
    from src.reconcile import Check
    monkeypatch.setattr(pl, "run_checks", lambda *a, **k: [Check("forced", False, "x")])
    with pytest.raises(Exception, match="forced"):
        go(small_data, tmp_path)
    conn = sqlite3.connect(tmp_path / "chronos.db")
    assert conn.execute("SELECT status, failed_stage FROM pipeline_runs").fetchone() == ("failed", "reconcile")
    conn.close()
    assert not list((tmp_path / "out" / "features").glob("*.csv"))


def test_rejects_persisted_with_traceability(small_data, tmp_path):
    s = go(small_data, tmp_path)
    conn = sqlite3.connect(tmp_path / "chronos.db")
    rows = conn.execute("SELECT source_file, source_row_number, reject_reason, raw_record FROM rejected_records WHERE run_id=?",
                        (s["run_id"],)).fetchall()
    conn.close()
    assert len(rows) == s["totals"]["rejected_rows"] + s["totals"]["duplicate_rows"]
    assert all(r[0] and r[1] and r[2] and json.loads(r[3]) for r in rows)
