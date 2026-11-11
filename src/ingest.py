"""Ingestion: read CSVs as raw strings and attach source traceability.

Schema-level problems (missing file, missing required column, unsupported schema
version) raise ``SchemaError`` and fail the run at the ``ingest`` stage.
Row-level problems are handled later in ``validate`` and quarantined.

Policy: extra/unknown columns are ignored for processing but preserved in the
``raw_record`` of any rejected row. ``source_row_number`` is the 1-based file line
number (the header is line 1, so the first data row is 2).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import schemas as S
from .config import FeatureConfig
from .timeutil import utc_timestamp

log = logging.getLogger(__name__)


class SchemaError(Exception):
    """Source file or metadata violates the schema contract."""


def read_csv_source(path, columns: list[str], required: list[str], kind: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"{kind}: file not found: {path}")
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"{kind}: file is empty (no header row): {path}") from exc
    except pd.errors.ParserError as exc:
        raise SchemaError(f"{kind}: cannot parse CSV {path}: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{kind}: missing required column(s) {missing} in {path}")
    for c in columns:  # optional columns absent from the file are treated as blank
        if c not in df.columns:
            df[c] = ""
    df["source_file"] = p.as_posix()
    df["source_row_number"] = range(2, len(df) + 2)
    log.info("ingested %s: %d rows from %s", kind, len(df), p.as_posix())
    return df


def read_customers(path) -> pd.DataFrame:
    return read_csv_source(path, S.CUSTOMER_COLUMNS, S.CUSTOMER_REQUIRED, "customers")


def read_predictions(path) -> pd.DataFrame:
    return read_csv_source(path, S.PREDICTION_COLUMNS, S.PREDICTION_REQUIRED, "prediction_requests")


def read_transactions(paths: Iterable) -> pd.DataFrame:
    """Read one or more transaction files; rows keep file order (then row order)."""
    frames = [read_csv_source(p, S.TRANSACTION_COLUMNS, S.TRANSACTION_REQUIRED, "transactions") for p in paths]
    if not frames:
        raise SchemaError("transactions: no input files given")
    return pd.concat(frames, ignore_index=True)


def load_source_metadata(path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"source metadata is not valid JSON: {path}: {exc}") from exc


def check_schema_version(meta: dict, cfg: FeatureConfig) -> None:
    version = meta.get("schema_version")
    if version is not None and str(version) != cfg.schema_version:
        raise SchemaError(f"unsupported schema_version {version!r}; this pipeline supports {cfg.schema_version!r}")


def resolve_watermark(cli_value, meta: dict):
    """``data_complete_through``: CLI value wins, then source metadata, else None.

    None is conservative: no label can be finalised, so all labels are 'incomplete'.
    """
    value = cli_value or meta.get("data_complete_through")
    return utc_timestamp(value) if value else None
