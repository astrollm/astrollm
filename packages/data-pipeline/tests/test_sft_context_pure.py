"""Pure provenance helpers in context.py (retrieval itself needs the DB and is not tested here)."""

from __future__ import annotations

import re

from context import ARM, POOL, K, config_fingerprint, make_run_id, utc_stamp


def test_frozen_retrieval_constants():
    # The harness's single-variable discipline: these are the beta stage-1 defaults, frozen.
    assert (ARM, POOL, K) == ("hybrid", 100, 100)


def test_config_fingerprint_is_deterministic_and_key_order_insensitive():
    a = {"arm": "hybrid", "pool": 100, "k": 100}
    b = {"k": 100, "arm": "hybrid", "pool": 100}
    assert config_fingerprint(a) == config_fingerprint(b)
    assert len(config_fingerprint(a)) == 8


def test_config_fingerprint_changes_with_config():
    assert config_fingerprint({"pool": 100}) != config_fingerprint({"pool": 50})


def test_make_run_id_format():
    config = {"arm": "hybrid", "pool": 100, "k": 100}
    run_id = make_run_id(config, "sha256:abcdef0123456789", "20260721T120000Z")
    assert run_id == f"rrf-pool100-abcdef01-{config_fingerprint(config)}-20260721T120000Z"


def test_utc_stamp_shape():
    assert re.fullmatch(r"\d{8}T\d{6}Z", utc_stamp())
