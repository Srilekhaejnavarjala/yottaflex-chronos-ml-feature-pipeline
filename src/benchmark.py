"""Reproducible benchmark.

    python -m src.benchmark                       # 1k, 10k, 100k events, seed 42
    python -m src.benchmark --sizes 1000 10000

For each size: generate data (customers = events/20, predictions = events/5), run the full
pipeline and record per-stage timings, then time the naive per-request reference filter
against the optimized builder on the SAME requests and verify identical output.
The naive implementation is O(requests x events); above --naive-max-requests it is timed on
the first N requests only (stated in the output) so the benchmark finishes.
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import ingest
from .config import load_config
from .features import build_features, build_features_reference
from .generate_data import generate
from .pipeline import run_pipeline
from .validate import validate_customers, validate_predictions, validate_transactions


def bench_one(n_events: int, seed: int, naive_max: int) -> dict:
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta = generate(tmp / "data", customers=max(50, n_events // 20), events=n_events,
                        predictions=max(10, n_events // 5), seed=seed, include_invalid=False)
        s = run_pipeline(tmp / "data/customers.csv", [tmp / "data/transactions.csv"],
                         tmp / "data/prediction_requests.csv", tmp / "bench.db", tmp / "out")
        st = s["stage_seconds"]

        c = validate_customers(ingest.read_customers(tmp / "data/customers.csv"), cfg, "b")
        t = validate_transactions(ingest.read_transactions([tmp / "data/transactions.csv"]), c.valid, cfg, "b")
        p = validate_predictions(ingest.read_predictions(tmp / "data/prediction_requests.csv"), c.valid, cfg, "b")
        reqs = p.valid.sort_values("prediction_id").head(naive_max)

        t0 = time.perf_counter()
        fast = build_features(c.valid, t.valid, reqs, cfg)
        fast_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        slow = build_features_reference(c.valid, t.valid, reqs, cfg)
        slow_s = time.perf_counter() - t0
        try:
            pd.testing.assert_frame_equal(fast, slow, check_exact=True)
            identical = True
        except AssertionError:
            identical = False

        total_events = s["sources"]["transactions"]["input_rows"]
        return {
            "requested_events": n_events, "input_events": total_events, "customers": meta["customers"],
            "prediction_requests": s["sources"]["prediction_requests"]["input_rows"],
            "feature_rows": s["feature_rows"], "status": s["status"],
            "ingest_s": st["ingest"], "validate_s": st["validate"], "features_s": st["features"],
            "labels_s": st["labels"], "publish_s": st["publish"], "reconcile_s": st["reconcile"],
            "export_s": st["export"], "total_s": s["duration_seconds"],
            "throughput_events_per_s": round(total_events / s["duration_seconds"], 1),
            "optimization": {
                "requests_compared": len(reqs), "naive_s": round(slow_s, 4), "optimized_s": round(fast_s, 4),
                "speedup_x": round(slow_s / fast_s, 1) if fast_s else None, "outputs_identical": identical,
            },
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--naive-max-requests", type=int, default=5000)
    ap.add_argument("--out", default="outputs/reports/benchmark.json")
    a = ap.parse_args(argv)
    results = [bench_one(n, a.seed, a.naive_max_requests) for n in a.sizes]
    payload = {"python": platform.python_version(), "pandas": pd.__version__, "platform": platform.platform(),
               "cpu_count": __import__("os").cpu_count(), "seed": a.seed, "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2))
    hdr = f"{'events':>8} {'preds':>7} {'ingest':>7} {'valid':>7} {'feat':>7} {'publ':>7} {'recon':>7} {'total':>7} {'ev/s':>9}  naive(N)  opt(N)  speedup"
    print(hdr)
    for r in results:
        o = r["optimization"]
        print(f"{r['input_events']:>8} {r['prediction_requests']:>7} {r['ingest_s']:>7} {r['validate_s']:>7} "
              f"{r['features_s']:>7} {r['publish_s']:>7} {r['reconcile_s']:>7} {r['total_s']:>7} "
              f"{r['throughput_events_per_s']:>9}  {o['naive_s']:>7} {o['optimized_s']:>7}  {o['speedup_x']}x "
              f"(N={o['requests_compared']}, identical={o['outputs_identical']})")


if __name__ == "__main__":
    main()
