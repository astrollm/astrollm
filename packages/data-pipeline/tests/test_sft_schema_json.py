"""data/sft/schema.json (the SFT training-record format) carries the D1 delivered_k contract.

No jsonschema dependency is available, so the schema's constraints are pinned structurally, and a
minimal checker of exactly those constraints is applied to a fixture record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from context import POOL

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "sft" / "schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _delivered_k_errors(schema: dict, record: dict) -> list[str]:
    """Check only the record-level/metadata/delivered_k constraints the schema declares."""
    errors = [f"missing {f}" for f in schema["required"] if f not in record]
    meta_schema = schema["properties"]["metadata"]
    meta = record.get("metadata", {})
    errors += [f"missing metadata.{f}" for f in meta_schema["required"] if f not in meta]
    dk_schema = meta_schema["properties"]["delivered_k"]
    if "delivered_k" in meta:
        dk = meta["delivered_k"]
        if not isinstance(dk, int) or isinstance(dk, bool):
            errors.append("delivered_k not an integer")
        elif not dk_schema["minimum"] <= dk <= dk_schema["maximum"]:
            errors.append("delivered_k out of range")
    return errors


def test_metadata_and_delivered_k_are_required(schema: dict) -> None:
    assert "metadata" in schema["required"]
    assert "delivered_k" in schema["properties"]["metadata"]["required"]


def test_delivered_k_bounds_match_the_pool(schema: dict) -> None:
    dk = schema["properties"]["metadata"]["properties"]["delivered_k"]
    assert dk["type"] == "integer"
    assert dk["minimum"] == 1
    assert dk["maximum"] == POOL == 100
    assert "POOL" in dk["description"]


def _record(**meta) -> dict:
    return {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "metadata": {"source": "curated", **meta},
    }


@pytest.mark.parametrize(
    ("record", "ok"),
    [
        (_record(delivered_k=30), True),
        (_record(delivered_k=1), True),
        (_record(delivered_k=100), True),
        (_record(delivered_k=0), False),
        (_record(delivered_k=101), False),
        (_record(), False),
        ({"messages": _record()["messages"]}, False),
    ],
)
def test_fixture_records_against_the_delivered_k_contract(schema, record, ok) -> None:
    assert (_delivered_k_errors(schema, record) == []) is ok
