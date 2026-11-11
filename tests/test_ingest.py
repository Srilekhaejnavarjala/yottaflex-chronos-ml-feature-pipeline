import json

import pytest

from src.ingest import (SchemaError, check_schema_version, load_source_metadata, read_customers,
                        read_transactions, resolve_watermark)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_missing_required_column_fails_clearly(tmp_path):
    p = write(tmp_path, "c.csv", "customer_id,region\nC1,north\n")
    with pytest.raises(SchemaError, match="signup_timestamp"):
        read_customers(p)


def test_empty_file_and_missing_file(tmp_path):
    with pytest.raises(SchemaError, match="empty"):
        read_customers(write(tmp_path, "e.csv", ""))
    with pytest.raises(SchemaError, match="not found"):
        read_customers(tmp_path / "nope.csv")


def test_source_traceability_and_extra_columns_kept_in_frame(tmp_path):
    p = write(tmp_path, "c.csv", "customer_id,signup_timestamp,extra\nC1,2026-01-01T00:00:00Z,x\nC2,2026-01-02T00:00:00Z,y\n")
    df = read_customers(p)
    assert list(df["source_row_number"]) == [2, 3]
    assert df["source_file"].iloc[0].endswith("c.csv")
    assert "extra" in df.columns  # preserved so it appears in raw_record of rejects
    assert df["region"].tolist() == ["", ""]  # absent optional column -> blank


def test_multiple_transaction_files_keep_file_order(tmp_path):
    hdr = "event_id,customer_id,event_timestamp,available_at,event_type,event_status,amount,currency\n"
    a = write(tmp_path, "a.csv", hdr + "e1,C1,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,login,completed,,\n")
    b = write(tmp_path, "b.csv", hdr + "e2,C1,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,login,completed,,\n")
    df = read_transactions([a, b])
    assert list(df["event_id"]) == ["e1", "e2"]


def test_unsupported_schema_version(cfg):
    with pytest.raises(SchemaError, match="unsupported schema_version"):
        check_schema_version({"schema_version": "2"}, cfg)
    check_schema_version({"schema_version": "1"}, cfg)
    check_schema_version({}, cfg)  # absent metadata is tolerated


def test_watermark_resolution(tmp_path):
    assert resolve_watermark(None, {}) is None
    assert str(resolve_watermark(None, {"data_complete_through": "2026-10-01T00:00:00Z"})) == "2026-10-01 00:00:00+00:00"
    assert resolve_watermark("2026-09-01T00:00:00Z", {"data_complete_through": "2026-10-01T00:00:00Z"}).month == 9
    bad = write(tmp_path, "m.json", "{nope")
    with pytest.raises(SchemaError):
        load_source_metadata(bad)
