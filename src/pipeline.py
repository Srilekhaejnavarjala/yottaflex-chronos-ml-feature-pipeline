"""Chronos pipeline orchestrator and CLI.

    python -m src.pipeline                      # runs on data/sample
    python -m src.pipeline --customers ... --transactions a.csv b.csv --predictions ...

Stages: ingest -> validate -> features -> labels -> publish -> reconcile -> export.
Each stage is timed; a failure records the failed stage in the run ledger, marks the run
'failed', and re-raises. Publishing is one atomic SQLite transaction, so a failed run never
leaves partial published data, and retrying is safe (upserts on logical keys).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from . import ingest, publish
from .config import DEFAULT_CONFIG_PATH, load_config
from .features import build_features
from .labels import assemble_training_dataset, build_labels
from .metrics import RunRecorder
from .reconcile import ReconciliationError, run_checks
from .timeutil import ISO_FMT, iso
from .validate import validate_customers, validate_predictions, validate_transactions

log = logging.getLogger("chronos")


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _ts_cols(df):
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime64"):
            out[c] = iso(out[c])
    return out


def _source_counts(results) -> dict:
    return {n: {"input_rows": r.input_rows, "valid_rows": r.valid_rows,
                "rejected_rows": r.rejected_rows, "duplicate_rows": r.duplicate_rows}
            for n, r in results.items()}


def _summary(rec, cfg, watermark, results, features, labels, pub, checks) -> dict:
    src = _source_counts(results)
    totals = {k: sum(v[k] for v in src.values()) for k in ("input_rows", "valid_rows", "rejected_rows", "duplicate_rows")}
    complete = 0 if labels is None else int((labels["label_status"] == "complete").sum())
    return {
        "run_id": rec.run_id, "status": rec.status, "failed_stage": rec.failed_stage,
        "error_message": rec.error_message, "started_at": rec.started_at, "ended_at": rec.ended_at,
        "duration_seconds": rec.duration_seconds, "stage_seconds": rec.stage_seconds,
        "feature_version": cfg.feature_version, "label_version": cfg.label_version, "config_hash": cfg.config_hash,
        "data_complete_through": None if watermark is None else watermark.strftime(ISO_FMT),
        "sources": src, "totals": totals,
        "feature_rows": 0 if features is None else len(features),
        "label_rows": 0 if labels is None else len(labels),
        "complete_label_rows": complete,
        "incomplete_label_rows": 0 if labels is None else len(labels) - complete,
        "publish": pub, "checks": [c.as_dict() for c in (checks or [])],
    }


def format_report(s: dict) -> str:
    t = s["sources"].get("transactions", {})
    p = s["sources"].get("prediction_requests", {})
    lines = [
        f"run_id: {s['run_id']}", f"status: {s['status']}",
        f"failed_stage: {s['failed_stage']}" if s["failed_stage"] else None,
        f"error: {s['error_message']}" if s["error_message"] else None,
        f"input_events: {t.get('input_rows')}", f"valid_events: {t.get('valid_rows')}",
        f"rejected_events: {t.get('rejected_rows')}", f"duplicate_events: {t.get('duplicate_rows')}",
        f"prediction_requests: {p.get('input_rows')}  (valid: {p.get('valid_rows')})",
        f"feature_rows: {s['feature_rows']}",
        f"complete_label_rows: {s['complete_label_rows']}", f"incomplete_label_rows: {s['incomplete_label_rows']}",
        f"feature_version: {s['feature_version']}", f"data_complete_through: {s['data_complete_through']}",
        f"config_hash: {s['config_hash']}", f"duration_seconds: {s['duration_seconds']}",
        "", "all sources (input = valid + rejected + duplicate):",
    ]
    for n, c in s["sources"].items():
        lines.append(f"  {n}: {c}")
    lines.append(f"  TOTAL: {s['totals']}")
    lines += ["", "publish (inserted / updated / unchanged):"]
    for n, c in (s["publish"] or {}).items():
        lines.append(f"  {n}: {c['inserted']} / {c['updated']} / {c['unchanged']}")
    lines += ["", "stage_seconds: " + json.dumps(s["stage_seconds"]), "", "reconciliation checks:"]
    for c in s["checks"]:
        lines.append(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  {c['detail']}")
    return "\n".join(l for l in lines if l is not None) + "\n"


def run_pipeline(customers_path, transactions_paths, predictions_path, db_path, out_dir,
                 config_path=DEFAULT_CONFIG_PATH, data_complete_through=None, metadata_path=None,
                 run_id=None) -> dict:
    cfg = load_config(config_path)
    rec = RunRecorder(run_id)
    out = Path(out_dir)
    for sub in ("features", "labels", "rejects", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = publish.connect(db_path)
    publish.init_db(conn)
    publish.start_run(conn, rec, cfg)

    watermark = results = features = labels = pub = checks = None
    try:
        with rec.stage("ingest"):
            meta = ingest.load_source_metadata(metadata_path or Path(customers_path).parent / "source_metadata.json")
            ingest.check_schema_version(meta, cfg)
            watermark = ingest.resolve_watermark(data_complete_through, meta)
            raw_c = ingest.read_customers(customers_path)
            raw_t = ingest.read_transactions(transactions_paths)
            raw_p = ingest.read_predictions(predictions_path)
        with rec.stage("validate"):
            rc = validate_customers(raw_c, cfg, rec.run_id)
            rt = validate_transactions(raw_t, rc.valid, cfg, rec.run_id)
            rp = validate_predictions(raw_p, rc.valid, cfg, rec.run_id)
            results = {"customers": rc, "transactions": rt, "prediction_requests": rp}
            rejects = pd.concat([r.rejects for r in results.values() if len(r.rejects)], ignore_index=True) \
                if any(len(r.rejects) for r in results.values()) else pd.DataFrame(columns=rc.rejects.columns)
        with rec.stage("features"):
            features = build_features(rc.valid, rt.valid, rp.valid, cfg)
        with rec.stage("labels"):
            training = rp.valid[rp.valid["request_type"] == "training"]
            labels = build_labels(rt.valid, training, cfg, watermark)
        with rec.stage("publish"):
            pub = publish.publish_all(conn, rec.run_id, cfg, rc.valid, rt.valid, rp.valid, features, labels, rejects)
        with rec.stage("reconcile"):
            checks = run_checks(conn, run_id=rec.run_id, cfg=cfg, results=results, customers=rc.valid,
                                transactions=rt.valid, predictions=rp.valid, features=features, labels=labels,
                                watermark=watermark)
            failed = [c for c in checks if not c.passed]
            if failed:
                raise ReconciliationError("; ".join(f"{c.name}: {c.detail}" for c in failed))
        with rec.stage("export"):  # files replace previous outputs only after reconciliation passed
            v = cfg.feature_version
            feats = _ts_cols(features)
            inference_ids = set(rp.valid.loc[rp.valid["request_type"] == "inference", "prediction_id"])
            _atomic_csv(feats, out / "features" / f"features_{v}.csv")
            _atomic_csv(feats[feats["prediction_id"].isin(inference_ids)], out / "features" / f"inference_features_{v}.csv")
            _atomic_csv(_ts_cols(labels), out / "labels" / f"labels_{cfg.label_version}.csv")
            _atomic_csv(_ts_cols(assemble_training_dataset(features, labels)), out / "features" / f"training_dataset_{v}.csv")
            _atomic_csv(rejects, out / "rejects" / f"rejects_{rec.run_id}.csv")
        rec.finish("success")
    except Exception as exc:
        if rec.failed_stage is None:
            rec.failed_stage, rec.error_message = "unknown", f"{type(exc).__name__}: {exc}"
        rec.finish("failed")
        summary = _summary(rec, cfg, watermark, results or {}, features, labels, pub, checks)
        _finalize(conn, rec, summary, out)
        raise
    summary = _summary(rec, cfg, watermark, results, features, labels, pub, checks)
    _finalize(conn, rec, summary, out)
    return summary


def _finalize(conn, rec, summary, out: Path) -> None:
    publish.finish_run(conn, rec, summary)
    conn.close()
    (out / "reports" / f"{rec.run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "reports" / f"{rec.run_id}.txt").write_text(format_report(summary))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chronos point-in-time feature pipeline")
    ap.add_argument("--customers", default="data/sample/customers.csv")
    ap.add_argument("--transactions", nargs="+", default=["data/sample/transactions.csv"])
    ap.add_argument("--predictions", default="data/sample/prediction_requests.csv")
    ap.add_argument("--db", default="outputs/chronos.db")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--data-complete-through", default=None,
                    help="ISO-8601 UTC watermark; default: source_metadata.json next to --customers")
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        s = run_pipeline(a.customers, a.transactions, a.predictions, a.db, a.out_dir, a.config,
                         a.data_complete_through, a.metadata, a.run_id)
    except Exception as exc:  # ledger + report already written by run_pipeline
        print(f"\nPIPELINE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\n" + format_report(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
