# Tools Integration Package

The unique differentiator of AstroLLM: deep integration with the scientific tools astronomers use daily.

## Why This Matters

No existing astronomy LLM (including AstroSage) can:
- Look up an object in SIMBAD and reason about its properties
- Search NASA ADS for relevant papers and synthesize findings
- Run Astropy calculations (coordinate transforms, cosmological distances)
- Query observation archives (JWST, HST, Kepler) for available data
- Cross-match catalogs via VizieR

AstroLLM treats these as first-class tool calls, trained into the model during SFT.

## Tool Implementations

### NASA ADS (`ads_search`, `ads_paper`)
- Uses the ADS API (api.adsabs.harvard.edu)
- Requires API key (free, register at adsabs.harvard.edu)
- Capabilities: keyword search, author search, citation network traversal
- Returns: titles, abstracts, bibcodes, citation counts, full metadata

### SIMBAD (`simbad_query`)
- Uses SIMBAD TAP service (simbad.u-strasbg.fr/simbad/sim-tap)
- No API key required
- Capabilities: object lookup by name, coordinate cone search, type filtering
- Returns: coordinates, object type, redshift, magnitudes, cross-identifications

### VizieR (`vizier_catalog`)
- Uses VizieR TAP service
- No API key required
- Capabilities: query any of 23,000+ astronomical catalogs
- Common use: photometry lookups, cross-matching, survey data

### Astropy (`astropy_calc`)
- Local Python execution (sandboxed)
- Capabilities: coordinate transforms (ICRS, Galactic, ecliptic), unit conversions,
  cosmological calculations (luminosity distance, lookback time), FITS header parsing
- Executed in a restricted Python environment with only astropy + numpy

### MAST (`mast_search`)
- Uses MAST API (mast.stsci.edu/api)
- No API key required for most queries
- Capabilities: search HST/JWST/Kepler/TESS observations by coordinates or target name
- Returns: observation metadata, proposal info, data product availability

## SFT Data Generation

For each tool, generate training examples:

1. **Direct tool use**: "What is the redshift of NGC 4151?" → SIMBAD call → interpreted answer
2. **Multi-tool chains**: "Find recent JWST papers about NGC 4151 and check what observations exist" → ADS search → MAST search → synthesis
3. **Tool + reasoning**: "Is NGC 4151 a Seyfert galaxy? How does its luminosity compare to other AGN?" → SIMBAD → calculation → contextual knowledge
4. **Tool selection**: Model must learn WHEN to call a tool vs answer from knowledge

Generate 500-1000 examples per tool, covering edge cases and error handling.

## Directory Structure
```
tools-integration/
├── src/
│   ├── ads.py          # NASA ADS client
│   ├── simbad.py       # SIMBAD TAP client
│   ├── vizier.py       # VizieR TAP client
│   ├── astropy_calc.py # Sandboxed Astropy execution
│   ├── mast.py         # MAST archive client
│   ├── ned.py          # NED client
│   ├── registry.py     # Tool registry and dispatch
│   └── sandbox.py      # Secure execution environment
├── tests/
│   ├── test_ads.py
│   ├── test_simbad.py
│   └── ...
├── data/
│   └── tool_use_examples.jsonl   # SFT training data for tool use
└── README.md           # This file
```
