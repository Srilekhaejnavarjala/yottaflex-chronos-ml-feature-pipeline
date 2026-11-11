"""Post-publish reconciliation: accounting, key integrity, leakage audit, parity, label horizons."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from . import schemas as S
from .features import build_features, build_features_reference
from .timeutil import ISO_FMT

SQL_PATH = Path(__file__).resolve().parent.parent / "sql" / "reconciliation.sql"
AUDIT_SAMPLE = 300


class ReconciliationError(Exception):
    """One or more reconciliation checks failed."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self):
        return asdict(self)


def load_sql_checks(path=SQL_PATH) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        out[name.strip()] = "\n".join(l for l in body.splitlines() if not l.startswith("--")).strip()
    return out


def _sample(requests: pd.DataFrame) -> pd.DataFrame:
    ordered = requests.sort_values("prediction_id")
    step = max(1, -(-len(ordered) // AUDIT_SAMPLE))
    return ordered.iloc[::step]


def _same(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True, ""
    except AssertionError as exc:
        return False, str(exc).splitlines()[0][:200]


def run_checks(conn, *, run_id, cfg, results: dict, customers, transactions, predictions,
               features, labels, watermark) -> list[Check]:
    checks: list[Check] = []

    # 1. row accounting: input = valid + rejected + duplicate (per source and in total)
    for name, r in results.items():
        ok = r.input_rows == r.valid_rows + r.rejected_rows + r.duplicate_rows
        checks.append(Check(f"accounting_{name}", ok,
                            f"input={r.input_rows} valid={r.valid_rows} rejected={r.rejected_rows} duplicate={r.duplicate_rows}"))
    # 2. output counts explainable from valid prediction requests, not from transaction rows
    checks.append(Check("feature_rows_equal_valid_requests", len(features) == len(predictions),
                        f"features={len(features)} valid_requests={len(predictions)}"))
    n_train = int((predictions["request_type"] == "training").sum())
    checks.append(Check("label_rows_equal_training_requests", len(labels) == n_train,
                        f"labels={len(labels)} training_requests={n_train}"))
    # 3. contract: feature columns exact and label-free
    checks.append(Check("feature_columns_match_contract_and_label_free",
                        list(features.columns) == S.FEATURE_COLUMNS and not (S.LABEL_LEAK_COLUMNS & set(features.columns)),
                        ",".join(features.columns)))
    # 4. independent leakage audit: naive per-request filter must reproduce published features exactly
    sample = _sample(predictions)
    ref = build_features_reference(customers, transactions, sample, cfg)
    pub = features[features["prediction_id"].isin(sample["prediction_id"])].reset_index(drop=True)
    ok, why = _same(pub, ref)
    checks.append(Check("leakage_audit_vs_reference", ok, f"sample={len(sample)} {why}"))
    # 5. training/inference parity: results must not depend on request type or batch composition
    parity_ok, parity_msg = True, f"sample={len(sample)}"
    for rtype in ("training", "inference"):
        part = sample[sample["request_type"] == rtype]
        if part.empty:
            continue
        got = build_features(customers, transactions, part, cfg)
        exp = features[features["prediction_id"].isin(part["prediction_id"])].reset_index(drop=True)
        ok, why = _same(got, exp)
        if not ok:
            parity_ok, parity_msg = False, f"{rtype}: {why}"
    checks.append(Check("training_inference_parity", parity_ok, parity_msg))

    # 6. SQL integrity + label-horizon checks (0 violations expected)
    params = {"feature_version": cfg.feature_version, "label_version": cfg.label_version, "run_id": run_id,
              "watermark": None if watermark is None else watermark.strftime(ISO_FMT)}
    for name, sql in load_sql_checks().items():
        n = conn.execute(sql, params).fetchone()[0]
        checks.append(Check(f"sql_{name}", n == 0, f"violations={n}"))
    return checks
