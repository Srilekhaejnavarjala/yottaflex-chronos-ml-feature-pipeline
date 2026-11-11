"""Single source of truth for feature/label semantics.

Every module reads these values from here; nothing else hardcodes
window sizes, boundaries or vocabularies.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "feature_config.json"


class ConfigError(ValueError):
    """Raised when the feature configuration is invalid."""


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str
    label_version: str
    schema_version: str
    lookback_days: int
    target_horizon_days: int
    timezone: str
    lower_bound: str
    upper_bound: str
    availability_cutoff_required: bool
    qualifying_purchase_event_type: str
    qualifying_purchase_status: str
    currency: str
    config_hash: str = ""


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FeatureConfig:
    raw_text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config is not valid JSON: {exc}") from exc

    fields = FeatureConfig.__dataclass_fields__
    expected = {k for k in fields if k != "config_hash"}
    missing = expected - raw.keys()
    unknown = raw.keys() - expected
    if missing:
        raise ConfigError(f"Missing config keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")

    _validate(raw)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return FeatureConfig(**raw, config_hash=config_hash)


def _validate(raw: dict) -> None:
    for key in ("lookback_days", "target_horizon_days"):
        if not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] <= 0:
            raise ConfigError(f"{key} must be a positive integer")
    if raw["timezone"] != "UTC":
        raise ConfigError("timezone must be 'UTC'")
    if raw["lower_bound"] != "exclusive" or raw["upper_bound"] != "inclusive":
        raise ConfigError("window must be lower=exclusive, upper=inclusive")
    if raw["availability_cutoff_required"] is not True:
        raise ConfigError("availability_cutoff_required must be true (no availability leakage)")
    for key in ("feature_version", "label_version", "schema_version", "currency",
                "qualifying_purchase_event_type", "qualifying_purchase_status"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")


def ensure_supported_windows(cfg: FeatureConfig) -> None:
    """Feature/label column names embed the window sizes (``*_30d``, ``*_next_7d``).

    Changing a window therefore changes the output schema, which requires a new
    feature_version and column contract rather than a silent config edit.
    """
    if cfg.lookback_days != 30 or cfg.target_horizon_days != 7:
        raise ConfigError(
            "Feature columns for this schema are named for a 30d lookback and 7d horizon; "
            "define a new feature_version and column set before changing windows."
        )
