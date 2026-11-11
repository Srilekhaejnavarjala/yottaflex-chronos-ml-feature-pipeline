import json
import pytest
from src.config import ConfigError, load_config


def _write(tmp_path, **overrides):
    base = json.loads(open("config/feature_config.json").read())
    base.update(overrides)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(base))
    return p


def test_default_config_loads():
    cfg = load_config()
    assert cfg.lookback_days == 30 and cfg.target_horizon_days == 7
    assert cfg.lower_bound == "exclusive" and cfg.upper_bound == "inclusive"


def test_config_hash_is_stable_and_sensitive(tmp_path):
    assert load_config().config_hash == load_config().config_hash
    changed = load_config(_write(tmp_path, lookback_days=14))
    assert changed.config_hash != load_config().config_hash


@pytest.mark.parametrize("override", [
    {"lookback_days": 0}, {"lookback_days": -3}, {"lookback_days": True},
    {"timezone": "IST"}, {"lower_bound": "inclusive"},
    {"availability_cutoff_required": False}, {"feature_version": ""},
])
def test_invalid_config_rejected(tmp_path, override):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, **override))


def test_missing_and_unknown_keys(tmp_path):
    base = json.loads(open("config/feature_config.json").read())
    del base["lookback_days"]
    p = tmp_path / "m.json"; p.write_text(json.dumps(base))
    with pytest.raises(ConfigError):
        load_config(p)
    base["lookback_days"] = 30; base["bogus"] = 1
    p.write_text(json.dumps(base))
    with pytest.raises(ConfigError):
        load_config(p)
