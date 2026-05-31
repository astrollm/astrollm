"""Generate the offline smoke-test fixture for the pilot retrieval pipeline.

The corpus is SYNTHETIC test data — short, made-up abstracts about real exoplanet
targets, with synthetic 19-char bibcodes (journal code ``TEST``). It exists only to
exercise the embed -> pgvector -> FTS5 -> RRF -> metrics machinery end-to-end without
needing an ADS key. It is NOT a scientific corpus and its numbers are a machinery
smoke test, not the real deliverable.

Run:  uv run python tests/fixtures/pilot/make_fixture.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# (suffix, title, abstract) — suffix padded to make a 19-char "2024TEST<suffix>" bibcode.
DOCS = [
    ("WASP39B-CO2", "JWST CO2 detection in the atmosphere of WASP-39b",
     "We report a robust detection of carbon dioxide in the transmission spectrum of the "
     "hot Saturn WASP-39b using JWST NIRSpec. The 4.3 micron CO2 feature constrains the "
     "atmospheric metallicity and carbon-to-oxygen ratio."),
    ("WASP39B-SO2", "Photochemically produced sulfur dioxide on WASP-39b",
     "Follow-up analysis of WASP-39b reveals sulfur dioxide, a photochemical product, "
     "alongside water and carbon monoxide. The detection implies active photochemistry in "
     "this hot giant exoplanet atmosphere."),
    ("K218B--DMS", "Carbon-bearing molecules and a possible biosignature on K2-18b",
     "Transmission spectroscopy of the habitable-zone sub-Neptune K2-18b shows methane and "
     "carbon dioxide, with a tentative dimethyl sulfide signal. We discuss implications for "
     "a possible Hycean world."),
    ("TRAP1E-ATM", "Search for a terrestrial atmosphere on TRAPPIST-1e",
     "We present transmission observations of the temperate rocky planet TRAPPIST-1e. The "
     "flat spectrum disfavors a cloud-free hydrogen-dominated atmosphere but cannot rule out "
     "a high-mean-molecular-weight secondary atmosphere."),
    ("TRAP1B-BARE", "Thermal emission from TRAPPIST-1b consistent with bare rock",
     "Secondary-eclipse photometry of TRAPPIST-1b yields a high dayside brightness "
     "temperature consistent with a dark bare rock and little to no atmosphere."),
    ("HD189733-NA", "Sodium and high-altitude haze in HD 189733b",
     "High-resolution spectroscopy of the hot Jupiter HD 189733b detects neutral sodium "
     "absorption and strong Rayleigh scattering from a high-altitude haze layer."),
    ("HD209458H2O", "Water vapor and atmospheric escape from HD 209458b",
     "We measure water vapor in the atmosphere of HD 209458b and characterize an extended, "
     "escaping hydrogen envelope driven by stellar irradiation."),
    ("55CNCE-LAVA", "A tenuous outgassed atmosphere on the lava world 55 Cancri e",
     "JWST thermal measurements of the ultra-short-period super-Earth 55 Cancri e are "
     "consistent with a secondary atmosphere outgassed from a magma ocean, rich in CO and CO2."),
    ("GJ1214BFLAT", "A high-metallicity, hazy atmosphere on GJ 1214b",
     "The mini-Neptune GJ 1214b shows a flat, featureless transmission spectrum indicating "
     "a high-metallicity atmosphere muted by high-altitude clouds or haze."),
    ("WASP121BINV", "Thermal inversion and metal emission in WASP-121b",
     "Emission spectroscopy of the ultra-hot Jupiter WASP-121b reveals a temperature "
     "inversion and emission features from gaseous metals on the dayside."),
    ("LTT9779BREF", "Reflective clouds on the ultra-hot Neptune LTT 9779b",
     "Phase-curve photometry of LTT 9779b indicates a high geometric albedo produced by "
     "reflective silicate clouds on the dayside of this ultra-hot Neptune."),
    ("WASP96B-H2O", "A cloud-free water detection in WASP-96b with JWST NIRISS",
     "JWST NIRISS transmission spectroscopy of the hot Saturn WASP-96b shows a clear water "
     "absorption signature with little evidence for clouds."),
    ("PHOTOCHEM-1", "Photochemical modeling of sulfur species in irradiated exoplanets",
     "We develop a photochemical model predicting sulfur dioxide production in hot giant "
     "exoplanet atmospheres under intense stellar UV irradiation."),
    ("RETRIEVAL-1", "Bayesian atmospheric retrieval methods for transmission spectra",
     "We describe a free-chemistry Bayesian retrieval framework for inferring molecular "
     "abundances and temperature structure from exoplanet transmission spectra."),
]

# Smoke-test query set: each maps to the synthetic bibcode(s) it should retrieve.
QUERIES = [
    ("q01", "JWST CO2 detection in WASP-39b", ["WASP39B-CO2"]),
    ("q02", "sulfur dioxide photochemistry on WASP-39b", ["WASP39B-SO2", "PHOTOCHEM-1"]),
    ("q03", "dimethyl sulfide biosignature K2-18b", ["K218B--DMS"]),
    ("q04", "does TRAPPIST-1e have an atmosphere", ["TRAP1E-ATM"]),
    ("q05", "TRAPPIST-1b bare rock thermal emission", ["TRAP1B-BARE"]),
    ("q06", "sodium absorption haze HD 189733b", ["HD189733-NA"]),
    ("q07", "atmospheric escape water HD 209458b", ["HD209458H2O"]),
    ("q08", "55 Cancri e lava world outgassed atmosphere", ["55CNCE-LAVA"]),
    ("q09", "GJ 1214b flat featureless spectrum high metallicity", ["GJ1214BFLAT"]),
    ("q10", "thermal inversion ultra-hot Jupiter WASP-121b", ["WASP121BINV"]),
    ("q11", "reflective clouds high albedo LTT 9779b", ["LTT9779BREF"]),
    ("q12", "cloud-free water WASP-96b NIRISS", ["WASP96B-H2O"]),
    ("q13", "Bayesian retrieval transmission spectra method", ["RETRIEVAL-1"]),
]


def bibcode(suffix: str) -> str:
    """Return a synthetic 19-char bibcode: '2024TEST' + 11-char right-padded suffix."""
    code = "2024TEST" + suffix.ljust(11, ".")
    assert len(code) == 19, (code, len(code))
    return code


def main() -> None:
    corpus = HERE / "corpus.jsonl"
    with corpus.open("w") as fh:
        for suffix, title, abstract in DOCS:
            rec = {
                "bibcode": bibcode(suffix),
                "title": title,
                "year": 2024,
                "authors": ["Synthetic, A.", "Fixture, B."],
                "abstract": abstract,
            }
            fh.write(json.dumps(rec) + "\n")

    queries = HERE / "queries.yaml"
    header = "# SYNTHETIC smoke-test query set (see make_fixture.py). Not real science."
    lines = [header, "queries:"]
    for qid, text, suffixes in QUERIES:
        bibs = ", ".join(bibcode(s) for s in suffixes)
        lines += [f"  - id: {qid}", f"    query: {text!r}", f"    expected_bibcodes: [{bibs}]"]
    queries.write_text("\n".join(lines) + "\n")
    print(f"wrote {corpus} ({len(DOCS)} docs) and {queries} ({len(QUERIES)} queries)")


if __name__ == "__main__":
    main()
