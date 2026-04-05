# AstroLLM Data Sources — Unified Astronomical Knowledge Layer

> **Scope note (post peer review):** This document covers ALL data sources across all phases.
> For v1, only **ADS + SIMBAD + thin Exoplanet Archive** are in scope. NED, PDS, Gaia, MAST,
> and VizieR integrations are deferred to Phase 2+. See **V1_FINAL_PLAN.md** for execution scope.
> All API patterns and code examples in this document remain valid reference material.

Five databases, one unified knowledge graph. Each covers a different slice of astronomy, and together they give AstroLLM complete coverage from exoplanets to the cosmic web.

---

## The Big Picture

```
                          AstroLLM Knowledge Layer
                                   │
        ┌──────────┬───────────┬───┴───┬───────────┬──────────┐
        │          │           │       │           │          │
     ┌──▼──┐  ┌───▼───┐  ┌───▼──┐ ┌──▼───┐  ┌───▼───┐  ┌──▼───┐
     │ ADS │  │SIMBAD │  │ NED  │ │ NEA  │  │ PDS  │  │VizieR│
     │     │  │       │  │      │ │      │  │      │  │      │
     │Pubs │  │Stars  │  │Galax │ │Exo-  │  │Solar │  │Cata- │
     │Cita │  │Nebulae│  │Quasar│ │planet│  │System│  │logues│
     │Bib  │  │Clust. │  │AGN   │ │Data  │  │Missi │  │Surv. │
     └──┬──┘  └───┬───┘  └───┬──┘ └──┬───┘  └───┬──┘  └──┬───┘
        │         │           │       │          │         │
        └─────────┴───────┬───┴───────┴──────────┴─────────┘
                          │
                   ┌──────▼──────┐
                   │ Training    │
                   │ Data + RAG  │
                   │ + Live Tool │
                   └─────────────┘
```

Each database serves AstroLLM in THREE ways:
1. **Training data** — bulk harvest for SFT dataset generation
2. **RAG source** — indexed for retrieval-augmented generation
3. **Live tool** — called at inference time for up-to-date queries

---

## 1. NASA ADS — The Publication Graph

**What**: 15M+ bibliographic records, complete citation networks, co-readership data
**Coverage**: All published astronomy, astrophysics, planetary science, physics
**API**: REST + Python `ads` package — free, 5,000 req/day
**Already in plan**: Yes (Architecture V2)

### Role in AstroLLM
- Primary source for paper abstracts, metadata, and citation graphs
- Powers literature search and "what should I read" recommendations
- Links every other database back to the literature

---

## 2. SIMBAD — The Object Identity Layer

**What**: 20.5M+ astronomical objects outside the solar system with cross-identifications, coordinates, types, magnitudes, redshifts, proper motions, spectral types, and full bibliography per object.
**Coverage**: Stars, nebulae, galaxies, clusters, radio/X-ray/IR sources — everything extrasolar
**API**: TAP service (ADQL/SQL) + astroquery Python — free, no key required
**Access**: `simbad.cds.unistra.fr/simbad/sim-tap`

### What SIMBAD uniquely provides
- **Cross-identification**: An object might have 50+ names across different catalogs. SIMBAD resolves them all to one canonical entry. M31 = NGC 224 = Andromeda Galaxy = UGC 454 = PGC 2557 = ... SIMBAD knows.
- **Object types**: Hierarchical classification (star → variable star → Cepheid → classical Cepheid)
- **Basic physical data**: Coordinates, magnitudes (UBVRIJHK), spectral types, radial velocities, parallaxes, proper motions
- **Bibliography per object**: Every paper that mentions a given object

### Training data generation

```python
from astroquery.simbad import Simbad

# Customize returned fields
custom = Simbad()
custom.add_votable_fields('otype', 'sp', 'flux(V)', 'flux(B)',
                          'rv_value', 'plx', 'pmra', 'pmdec',
                          'morphtype', 'dim_majaxis')

# Example: Get all Seyfert galaxies
result = custom.query_criteria(
    "otype='Sy1' | otype='Sy2'",
    otype='Seyfert'
)
```

**SFT data from SIMBAD** (target: 15,000 pairs):

| Type | Example | Volume |
|------|---------|--------|
| Object lookup | "What type of object is NGC 1275?" | 5,000 |
| Cross-ID resolution | "Are M45 and the Pleiades the same?" | 2,000 |
| Property comparison | "Compare the spectral types of Sirius A and B" | 3,000 |
| Neighborhood queries | "What objects are within 1° of Sgr A*?" | 2,000 |
| Classification | "List all known Wolf-Rayet stars in the LMC" | 3,000 |

### Live tool integration

```json
{
  "tool_call": {
    "name": "simbad_query",
    "arguments": {
      "object_name": "Betelgeuse",
      "fields": ["coordinates", "sp_type", "flux_V", "plx", "rv_value", "otype"]
    }
  }
}
```

### Phase integration
- **Phase 1**: Bulk harvest top 100K most-cited objects for training data
- **Phase 2**: SFT examples with SIMBAD tool calls
- **Phase 3**: Live tool in RAG pipeline — object name → properties + papers
- **Phase 4**: Cross-link with NED for extragalactic objects

---

## 3. NED — The Extragalactic Universe

**What**: Comprehensive multiwavelength database of extragalactic objects — galaxies, quasars, AGN, galaxy clusters, radio sources, IR sources, gravitational lenses.
**Coverage**: Everything outside the Milky Way, including images, spectra, photometry, redshifts, distances
**API**: TAP service + astroquery — free, no key required
**Object count**: Hundreds of millions of objects, integrated from hundreds of sky surveys and tens of thousands of papers

### What NED uniquely provides (beyond SIMBAD)
- **Extragalactic specialization**: While SIMBAD covers all objects, NED goes deeper on galaxies and AGN with detailed photometric, spectroscopic, and morphological data
- **Redshift-independent distances**: Critical for cosmological research
- **Spectral energy distributions (SEDs)**: Multi-wavelength photometric data assembled from many surveys
- **Gravitational wave counterpart tracking**: Electromagnetic follow-up for GW events
- **LEVEL5 knowledgebase**: Review-level articles and educational content about extragalactic astronomy — excellent training data
- **Cosmological calculators**: Distance, age, volume calculations at different redshifts

### Training data generation

```python
from astroquery.ipac.ned import Ned

# Lookup a galaxy
result = Ned.query_object("NGC 4151")

# Get all objects near a position
near = Ned.query_region("12h10m32.6s +39d24m21s", radius=5 * u.arcmin)

# Get photometric data
phot = Ned.get_table("NGC 4151", table='photometry')

# Get spectral data
spec = Ned.get_spectra("NGC 4151")

# Get redshift data
z = Ned.get_table("NGC 4151", table='redshifts')
```

**SFT data from NED** (target: 10,000 pairs):

| Type | Example | Volume |
|------|---------|--------|
| Galaxy properties | "What is the Hubble type and redshift of NGC 4151?" | 3,000 |
| Distance/cosmology | "How far away is the Coma Cluster?" | 2,000 |
| Multi-wavelength | "What does NGC 1275 look like at different wavelengths?" | 2,000 |
| AGN characterization | "Is Markarian 421 a blazar? What makes it special?" | 1,500 |
| Galaxy groups/clusters | "What galaxies are in the Virgo Cluster?" | 1,500 |

### Live tool integration

```json
{
  "tool_call": {
    "name": "ned_query",
    "arguments": {
      "object_name": "M87",
      "data": ["basic", "photometry", "redshifts", "references"]
    }
  }
}
```

### Phase integration
- **Phase 2**: Harvest top 50K galaxies, AGN, clusters for SFT data
- **Phase 3**: Live tool for extragalactic queries
- **Phase 4**: SED data for multimodal training (spectral understanding)
- **Phase 5**: Combine with SIMBAD for complete object coverage

---

## 4. NASA Exoplanet Archive (NEA) — Planetary Systems

**What**: The definitive database of confirmed exoplanets, candidate planets (KOIs, TOIs), host star properties, orbital parameters, and associated time-series data.
**Coverage**: 5,700+ confirmed planets, 10,000+ candidates, light curves from Kepler/K2/TESS/CoRoT
**API**: TAP service (IVOA standard) + astroquery — free, no key required
**URL**: `exoplanetarchive.ipac.caltech.edu/TAP`

### What NEA uniquely provides
- **Planetary parameters**: Mass, radius, orbital period, eccentricity, semi-major axis, equilibrium temperature, insolation flux — for every confirmed exoplanet
- **Host star parameters**: Temperature, luminosity, metallicity, mass, age — linked to their planets
- **Discovery metadata**: Method (transit, RV, imaging, microlensing), facility, year, reference
- **Composite parameters table**: Best-available values from multiple references in one row per planet
- **Time-series data**: Kepler/K2/TESS light curves, radial velocity curves
- **Transit prediction**: Ephemeris service for planning observations
- **Periodogram tools**: Built-in period analysis

### Training data generation

```python
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Get all confirmed planets with key parameters
planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name,hostname,pl_masse,pl_rade,pl_orbper,"
           "pl_eqt,pl_orbsmax,st_teff,st_mass,st_rad,"
           "discoverymethod,disc_year,disc_facility",
    where="default_flag=1"
)

# Get all TESS candidates
tess = NasaExoplanetArchive.query_criteria(
    table="ps",
    where="disc_facility like '%TESS%'"
)

# Query specific planet
trappist = NasaExoplanetArchive.query_object("TRAPPIST-1 e")
```

**SFT data from NEA** (target: 10,000 pairs):

| Type | Example | Volume |
|------|---------|--------|
| Planet properties | "What is the mass and radius of TRAPPIST-1 e?" | 2,500 |
| Habitability | "Which exoplanets are in the habitable zone?" | 1,500 |
| System comparison | "Compare the TRAPPIST-1 system to our solar system" | 1,500 |
| Discovery methods | "How was 51 Pegasi b discovered? What method?" | 1,000 |
| Statistics | "How many Earth-sized planets has TESS found?" | 1,500 |
| Transit analysis | "Predict the next transit of HD 209458 b" | 1,000 |
| Trends | "How has the exoplanet discovery rate changed?" | 1,000 |

### Live tool integration

```json
{
  "tool_call": {
    "name": "exoplanet_query",
    "arguments": {
      "planet_name": "TRAPPIST-1 e",
      "fields": ["mass", "radius", "period", "equilibrium_temp",
                 "insolation", "host_star", "discovery_method"]
    }
  }
}
```

### Phase integration
- **Phase 1**: Download full confirmed planets table (~6K rows) — small, do immediately
- **Phase 2**: Generate SFT pairs from planet properties + host star data
- **Phase 3**: Live tool with transit prediction and planet comparison
- **Phase 4**: Light curve data for multimodal training (time-series understanding)
- **Phase 5**: Habitable zone analysis assistant

---

## 5. NASA Planetary Data System (PDS) — Solar System Missions

**What**: Long-term archive of digital data from NASA's planetary missions — rovers, orbiters, flyby missions, ground-based observations, laboratory experiments.
**Coverage**: Mars (Curiosity, Perseverance, MRO), Jupiter (Juno, Galileo), Saturn (Cassini), Moon, asteroids, comets, and every NASA planetary mission since the 1960s
**API**: REST API (PDS Search API) + Python `pds.peppi` client — free, no key required
**Nodes**: Atmospheres, Geosciences, Imaging, Plasma/Fields, Rings, Small Bodies, plus Engineering Node

### What PDS uniquely provides
- **Mission data**: Raw and calibrated data from every NASA planetary mission
- **Planetary surfaces**: High-resolution imaging, topography, mineralogy maps
- **Atmospheric data**: Composition, temperature profiles, weather data for planets and moons
- **Small bodies**: Asteroid and comet observations, sample return data
- **Laboratory data**: Spectral libraries, meteorite analyses
- **PDS4 metadata standard**: Richly described data products with full provenance

### Training data generation

```python
import pds.peppi as pep

# Connect to PDS
client = pep.PDSRegistryClient()

# Search for Mars Curiosity data
context = pep.Context()
curiosity = context.INSTRUMENT_HOSTS.search("curiosity")[0]

products = pep.Products(client) \
    .has_target("Mars") \
    .has_instrument_host(curiosity.lid) \
    .observationals()

# Browse available targets
targets = context.TARGETS.search("Jupiter")
```

**SFT data from PDS** (target: 8,000 pairs):

| Type | Example | Volume |
|------|---------|--------|
| Mission knowledge | "What instruments does Perseverance carry?" | 2,000 |
| Planetary properties | "What is the atmospheric composition of Titan?" | 1,500 |
| Surface analysis | "What minerals has Curiosity found at Gale Crater?" | 1,500 |
| Data discovery | "What Cassini data is available for Enceladus?" | 1,000 |
| Comparative planetology | "Compare the atmospheres of Venus and Mars" | 1,000 |
| Mission planning | "What upcoming observations of Europa are planned?" | 1,000 |

### Live tool integration

```json
{
  "tool_call": {
    "name": "pds_search",
    "arguments": {
      "target": "Mars",
      "instrument_host": "Perseverance",
      "instrument": "SHERLOC",
      "data_type": "observational"
    }
  }
}
```

### Phase integration
- **Phase 2**: Harvest mission metadata and instrument descriptions for SFT
- **Phase 3**: Live tool for "what data exists for [body/mission]?" queries
- **Phase 4**: Planetary surface images for multimodal training
- **Phase 5**: Spectral library data for spectroscopy understanding

---

## 6. VizieR — The Catalog Universe (Bonus: Already Planned)

Briefly noting VizieR since it cross-links with all of the above:
- 23,000+ astronomical catalogs from published papers
- TAP service, free access
- Covers photometry, spectroscopy, astrometry from every major survey
- Used for cross-matching objects across catalogs

---

## Unified Data Harvesting Plan

### Phase 1 — Quick Wins (Week 5-6)

| Source | What to harvest | Volume | Time |
|--------|----------------|--------|------|
| NEA | Full confirmed planets table | ~6K planets | 1 hour |
| ADS | Metadata for top 100K astro-ph papers | 100K records | 3 days |
| SIMBAD | Top 50K most-studied objects | 50K objects | 2 days |
| NED | Top 10K galaxies/AGN by citation count | 10K objects | 1 day |

### Phase 2 — Full Harvest (Week 7-8)

| Source | What to harvest | Volume | Time |
|--------|----------------|--------|------|
| ADS | Full astro-ph metadata + citation graph | 500K+ papers | 1 week |
| SIMBAD | All objects with >5 references | 500K objects | 3 days |
| NED | All galaxies with redshifts | 200K+ objects | 2 days |
| PDS | Mission metadata + instrument descriptions | All missions | 1 day |

### Phase 3 — SFT Generation (Week 8-10)

Use the harvested data to generate training pairs via Claude API:

| Source | Q&A Type | Target Count |
|--------|----------|-------------|
| ADS | Literature, citations, trends | 15,000 |
| SIMBAD | Object properties, cross-IDs, types | 15,000 |
| NED | Galaxies, AGN, cosmology | 10,000 |
| NEA | Exoplanets, habitability, systems | 10,000 |
| PDS | Missions, planetary science | 8,000 |
| Cross-database | Multi-source synthesis | 5,000 |
| **Total** | | **63,000** |

### Phase 4+ — Multimodal Data

| Source | Data Type | Use |
|--------|-----------|-----|
| NED | SEDs, spectra | Spectral understanding |
| NEA | Kepler/TESS light curves | Time-series understanding |
| PDS | Surface images, spectral libraries | Vision + spectroscopy |
| SIMBAD→VizieR | Multi-band photometry | SED analysis |

---

## Cross-Database Knowledge Graph

The real power is in the connections between databases:

```
"Tell me about M87"
    │
    ├── SIMBAD → Object type: cD galaxy, coordinates, redshift z=0.00428
    ├── NED → Detailed photometry, SED, distance 16.4 Mpc, jet properties
    ├── ADS → 12,000+ papers, top-cited: Event Horizon Telescope 2019
    ├── NEA → No exoplanets (not a star — explains why)
    └── PDS → No mission data (extragalactic — explains why)

"Tell me about Mars"
    │
    ├── SIMBAD → Planet in Solar System (basic coordinates, magnitude)
    ├── NED → Not applicable (not extragalactic)
    ├── ADS → 50,000+ papers across all aspects
    ├── NEA → No exoplanets (it IS a planet)
    └── PDS → MASSIVE: Curiosity, Perseverance, MRO, InSight, Phoenix, ...

"Tell me about TRAPPIST-1 e"
    │
    ├── SIMBAD → Host star TRAPPIST-1: M8V dwarf, V=18.8, d=12.4 pc
    ├── NED → Not applicable (star, not extragalactic)
    ├── ADS → 500+ papers on the TRAPPIST-1 system
    ├── NEA → M=0.69 M⊕, R=0.92 R⊕, P=6.1 days, T_eq=251K, habitable zone
    └── PDS → Not applicable (no spacecraft mission)
```

AstroLLM learns WHICH database to query for which type of question — this is part of the tool-use SFT training.

---

## Python Access Summary

All five databases are accessible through `astroquery`, which we already have in requirements.txt:

```python
# SIMBAD
from astroquery.simbad import Simbad

# NED
from astroquery.ipac.ned import Ned

# NASA Exoplanet Archive
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# VizieR
from astroquery.vizier import Vizier

# ADS
from astroquery import nasa_ads

# PDS (separate package)
import pds.peppi as pep
```

All free. All no-key-required except ADS (free key). All support TAP (IVOA standard).

---

## Cost Impact

Zero additional cost for data access. All databases are publicly funded and freely accessible. The only costs are:

- **Claude API** for generating SFT pairs from harvested data: ~$50-100 for 63K pairs
- **Storage**: ~5-10 GB for full metadata harvest (negligible)
- **Compute**: Same GPU budget — the data just makes training more effective

This is one of the beautiful things about astronomy: the community has spent decades building open, interoperable data infrastructure. AstroLLM gets to stand on those shoulders for free.
