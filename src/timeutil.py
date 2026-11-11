"""UTC timestamp helpers. All pipeline timestamps are timezone-aware UTC."""
from __future__ import annotations

import numpy as np
import pandas as pd

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
NS_PER_DAY = 86_400 * 1_000_000_000
# A timestamp must carry an explicit UTC designator or offset; naive values are rejected.
_TZ_RE = r"(?:Z|[+-]\d{2}:?\d{2})$"


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse ISO-8601 strings with an explicit timezone to datetime64[ns, UTC].

    Malformed values and values without a timezone designator become NaT.
    """
    s = series.astype(str).str.strip()
    has_tz = s.str.contains(_TZ_RE, regex=True)
    parsed = pd.to_datetime(s.where(has_tz), utc=True, errors="coerce", format="ISO8601")
    return parsed.astype("datetime64[ns, UTC]")


def to_ns(series: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 nanoseconds since epoch (UTC)."""
    naive = series.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    return naive.astype("int64").to_numpy()


def iso(series: pd.Series) -> pd.Series:
    """tz-aware datetime Series -> ISO-8601 'Z' strings (NaT stays null)."""
    return series.dt.tz_convert("UTC").dt.strftime(ISO_FMT)


def utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
