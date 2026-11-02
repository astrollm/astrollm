"""Offline tests for tools.py: rendering, the live-mode contract guard, and SIMBAD fallbacks.

No test here touches the network. The astroquery clients are replaced with fakes; where the
contract says "no network", the fake fails loudly if it is touched at all.
"""

from __future__ import annotations

from typing import Any

import pytest
import tools
from astropy.table import Table
from tools import (
    LiveDoc,
    LiveModeRequired,
    ObjectCard,
    ads_live_search,
    render_card,
    render_live_docs,
    simbad_card,
)

BIBCODE_A = "2023Natur.614..664R"
BIBCODE_B = "2023Natur.614..649A"


def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("astroquery / network was touched when the contract forbids it")


class Tripwire:
    """A stand-in that records being called (the code under test may swallow its error)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        raise ConnectionError("tripwire: this must not be reached")


# ── render_card ──────────────────────────────────────────────────────────────


def test_render_card_full() -> None:
    card = ObjectCard(
        queried_name="vega",
        main_id="* alf Lyr",
        ra_deg=279.23473,
        dec_deg=38.78369,
        object_type="PulsV*delSct",
        aliases=("Vega", "HD 172167", "HR 7001"),
    )
    text = render_card(card)
    assert text.splitlines()[0] == "SIMBAD: * alf Lyr"
    assert "queried as : vega" in text
    assert "PulsV*delSct" in text
    assert "279.23473, 38.78369 deg" in text
    assert "also known : Vega; HD 172167; HR 7001" in text
    assert "not a citable source" in text


def test_render_card_missing_fields_and_no_aliases() -> None:
    card = ObjectCard("x", "X 1", None, None, None, ())
    text = render_card(card)
    assert "type       : n/a" in text
    assert "RA, Dec    : n/a, n/a" in text
    assert "also known" not in text


def test_card_is_frozen_and_has_no_bibcode_field() -> None:
    card = ObjectCard("x", "X 1", None, None, None, ())
    with pytest.raises(AttributeError):
        card.main_id = "Y"  # type: ignore[misc]
    assert not hasattr(card, "bibcode")


# ── render_live_docs ─────────────────────────────────────────────────────────


def test_render_live_docs_tags_every_block() -> None:
    docs = [
        LiveDoc(BIBCODE_A, "Title A", "Abstract A.", 2023),
        LiveDoc(BIBCODE_B, "Title B", "Abstract B.", None),
    ]
    text = render_live_docs(docs)
    assert text == (
        f"[{BIBCODE_A}] (live ADS)\nTitle A\nAbstract A.\n"
        "\n"
        f"[{BIBCODE_B}] (live ADS)\nTitle B\nAbstract B.\n"
    )
    assert text.count("(live ADS)") == 2


def test_render_live_docs_marks_missing_abstract() -> None:
    text = render_live_docs([LiveDoc(BIBCODE_A, "Title A", "", 2023)])
    assert text == f"[{BIBCODE_A}] (live ADS)\nTitle A\n(no abstract in ADS)\n"


def test_render_live_docs_empty() -> None:
    assert render_live_docs([]) == ""


def test_live_doc_source_default_is_tagged() -> None:
    assert LiveDoc(BIBCODE_A, "t", "a", 2023).source == "ads_live"


# ── the contract guard: live results need an explicit live=True ──────────────


@pytest.mark.parametrize("live", [False, None, 0, 1, "yes"])
def test_ads_live_search_requires_literal_true_and_never_touches_network(
    monkeypatch: pytest.MonkeyPatch, live: Any
) -> None:
    monkeypatch.setattr(tools, "_make_ads_client", _boom)
    monkeypatch.setattr(tools, "_ads_token", _boom)
    with pytest.raises(LiveModeRequired):
        ads_live_search("WASP-39b JWST", live=live)


def test_ads_live_search_live_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        ads_live_search("q", True)  # type: ignore[misc]


def test_live_mode_required_is_a_distinct_exception() -> None:
    assert issubclass(LiveModeRequired, RuntimeError)
    assert LiveModeRequired is not RuntimeError


def test_ads_live_search_without_token_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tripwire()
    monkeypatch.setattr(tools, "_ads_token", lambda: None)
    monkeypatch.setattr(tools, "_make_ads_client", client)
    assert ads_live_search("WASP-39b", live=True) == []
    assert client.calls == 0


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeADSClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query_simple(self, query: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((query, kwargs))
        return _FakeResponse(self.payload)


def test_ads_live_search_parses_docs_with_a_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "response": {
            "docs": [
                {
                    "bibcode": BIBCODE_A,
                    "title": ["Identification of carbon dioxide in an exoplanet atmosphere"],
                    "abstract": "We  report\nCO2 in WASP-39b.",
                    "year": "2023",
                },
                {"bibcode": BIBCODE_B, "title": ["No abstract here"]},
                {"title": ["dropped: no bibcode"]},
            ]
        }
    }
    client = _FakeADSClient(payload)
    monkeypatch.setattr(tools, "_ads_token", lambda: "fake-token")
    monkeypatch.setattr(tools, "_make_ads_client", lambda token, rows, timeout_s: client)

    docs = ads_live_search("WASP-39b JWST", live=True, rows=3)

    assert [d.bibcode for d in docs] == [BIBCODE_A, BIBCODE_B]
    assert docs[0].title.startswith("Identification of carbon dioxide")
    assert docs[0].abstract == "We report CO2 in WASP-39b."
    assert docs[0].year == 2023
    assert docs[1].abstract == "" and docs[1].year is None
    assert all(d.source == "ads_live" for d in docs)
    assert client.calls[0][0] == "WASP-39b JWST"


def test_ads_live_search_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing:
        def query_simple(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError("down")

    monkeypatch.setattr(tools, "_ads_token", lambda: "fake-token")
    monkeypatch.setattr(tools, "_make_ads_client", lambda *_a: Failing())
    assert ads_live_search("q", live=True) == []


# ── simbad_card ──────────────────────────────────────────────────────────────


def _fake_simbad(
    monkeypatch: pytest.MonkeyPatch,
    table: Table | None = None,
    reachable: tuple[str, ...] = tools.SIMBAD_MIRRORS,
) -> dict[str, list[str]]:
    """Fake mirror probing and querying; returns the mirrors probed / queried so far."""
    seen: dict[str, list[str]] = {"probed": [], "queried": []}

    def probe(server: str, _timeout_s: float) -> bool:
        seen["probed"].append(server)
        return server in reachable

    def query(server: str, _adql: str) -> Table | None:
        seen["queried"].append(server)
        return table

    monkeypatch.setattr(tools, "_probe_simbad_mirror", probe)
    monkeypatch.setattr(tools, "_query_simbad_mirror", query)
    monkeypatch.setattr(tools, "_simbad_good_mirror", None)
    return seen


def _vega_table() -> Table:
    ids = [f"HD 17216{i}" for i in range(10)] + [
        "NAME Vega",
        "NAME Alpha Lyrae",
        "* alf Lyr",
        "vega",
    ]
    n = len(ids)
    return Table(
        {
            "main_id": ["* alf Lyr"] * n,
            "ra": [279.23473] * n,
            "dec": [38.78369] * n,
            "otype": ["PulsV*delSct"] * n,
            "alias": ids,
        }
    )


def test_simbad_card_unresolved_name_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = Table(
        {"main_id": [], "ra": [], "dec": [], "otype": [], "alias": []},
        dtype=[str, float, float, str, str],
    )
    seen = _fake_simbad(monkeypatch, empty)
    assert simbad_card("definitely not an object") is None
    assert len(seen["queried"]) == 1  # an empty answer is definitive: no second mirror


def test_simbad_card_builds_a_card_with_capped_sorted_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_simbad(monkeypatch, _vega_table())
    card = simbad_card("vega")
    assert card is not None
    assert card.main_id == "* alf Lyr"
    assert (card.ra_deg, card.dec_deg) == (279.23473, 38.78369)
    assert card.object_type == "PulsV*delSct"
    assert len(card.aliases) <= 8
    assert card.aliases[0] == "Alpha Lyrae"  # NAME-prefixed common name first, prefix stripped
    assert "* alf Lyr" not in card.aliases  # main_id is not repeated
    assert all(a.casefold() != "vega" for a in card.aliases)  # queried name not repeated


def test_simbad_card_masked_coordinates_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Table(
        {"main_id": ["X 1"], "ra": [1.0], "dec": [2.0], "otype": ["Star"], "alias": ["X 1"]},
        masked=True,
    )
    table["ra"].mask = [True]
    table["dec"].mask = [True]
    _fake_simbad(monkeypatch, table)
    card = simbad_card("x 1")
    assert card is not None and card.ra_deg is None and card.dec_deg is None
    assert card.aliases == ()


def test_simbad_card_skips_unreachable_mirror_and_remembers_the_good_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = tools.SIMBAD_MIRRORS
    seen = _fake_simbad(monkeypatch, _vega_table(), reachable=(second,))
    card = simbad_card("vega")
    assert card is not None and card.main_id == "* alf Lyr"
    assert seen["probed"] == [first, second]
    assert seen["queried"] == [second]
    # The mirror that answered is remembered: no re-probe on the next call.
    simbad_card("vega")
    assert seen["probed"] == [first, second]
    assert seen["queried"] == [second, second]


def test_simbad_card_all_mirrors_down_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_simbad(monkeypatch, _vega_table(), reachable=())
    assert simbad_card("vega") is None
    assert seen["probed"] == list(tools.SIMBAD_MIRRORS)
    assert seen["queried"] == []


def test_simbad_card_query_error_returns_none_and_forces_a_reprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _fake_simbad(monkeypatch, _vega_table())

    def failing(_server: str, _adql: str) -> Table:
        raise ConnectionError("down")

    monkeypatch.setattr(tools, "_query_simbad_mirror", failing)
    assert simbad_card("vega") is None
    assert tools._simbad_good_mirror is None
    simbad_card("vega")
    assert len(seen["probed"]) == 2  # probed again after the failure


def test_simbad_card_blank_name_makes_no_query(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_simbad(monkeypatch, _vega_table())
    assert simbad_card("   ") is None
    assert seen == {"probed": [], "queried": []}


def test_simbad_query_escapes_quotes() -> None:
    assert "q.id = 'Barnard''s Galaxy'" in tools._simbad_query("Barnard's Galaxy")
