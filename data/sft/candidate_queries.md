# Gold-Seed Candidate Queries — for human triage

Drafted 2026-07-21 against corpus snapshot `sha256:08736961…` (2,500 abstracts). Every object/topic
below was grep-verified against the corpus; `(N)` = abstract hit count. Absence claims for
abstention-`absent` candidates are verified zero-hit. **Nothing here is an example — these are
query candidates.** Triage by deleting lines or unchecking; then feed survivors to
`author.py prepare`.

Conventions:
- Queries must stay **unique file-wide** (calibration/eval disjointness is enforced at the query
  level by the harness; same *object* in both partitions is allowed but flagged `⚠same-obj` where
  it happens, in case you prefer object-level separation for eval hygiene).
- Targets (from `partition.py`): eval 40 lit_qa / 30 summarization / 25 abstention;
  calibration 40+ positives / 40–60 negatives covering all 4 failure modes; total 170–230.
- Counts drafted: 50 / 36 / 38 / 48 / 48 = 220 candidates — ~15–20% triage headroom per slot.
- The smoke-test query is L01 (dry-run only; never committed, safe to use).

---

## EVAL — lit_qa (50 candidates, target ≥40)

- [ ] L01 — What did JWST measure for the C/O ratio of WASP-39b? (WASP-39: 50)
- [ ] L02 — Which molecules has JWST detected in the atmosphere of WASP-39b? (50)
- [ ] L03 — What is the evidence for photochemistry in WASP-39b's atmosphere? (SO2/photochem)
- [ ] L04 — Which atoms and molecules have been detected in the ultra-hot Jupiter WASP-121b? (45)
- [ ] L05 — What does the asymmetric iron signal in WASP-76b imply about its atmosphere? (42)
- [ ] L06 — What did the JWST phase curve of WASP-43b reveal about its nightside? (25)
- [ ] L07 — Which species have been detected in the atmosphere of KELT-9b, the hottest known giant planet? (29)
- [ ] L08 — What is known about helium escape from WASP-107b? (24)
- [ ] L09 — What did JWST reveal about the composition and clouds of WASP-107b? (⚠L08 same-obj)
- [ ] L10 — What has high-resolution spectroscopy found in the atmosphere of KELT-20b? (19)
- [ ] L11 — What water abundance has been measured for HD 209458b? (64)
- [ ] L12 — What is the observational evidence for atmospheric escape from HD 189733b? (79)
- [ ] L13 — What did Hubble observations of K2-18b in 2019 claim to detect? (K2-18: 51)
- [ ] L14 — Which carbon-bearing molecules did JWST detect on K2-18b? (⚠L13 same-obj)
- [ ] L15 — Why is the transmission spectrum of GJ 1214b flat? (36)
- [ ] L16 — What did the JWST MIRI phase curve of GJ 1214b reveal? (⚠L15 same-obj)
- [ ] L17 — Does TRAPPIST-1b have a thick atmosphere according to JWST thermal emission measurements? (TRAPPIST-1: 114)
- [ ] L18 — How does stellar contamination affect transmission spectra of the TRAPPIST-1 planets? (55 stellar-contam)
- [ ] L19 — What did JWST secondary-eclipse observations show for TRAPPIST-1c? (⚠same-obj)
- [ ] L20 — What does the thermal emission of LHS 3844b imply about an atmosphere? (7)
- [ ] L21 — What sodium feature was observed in WASP-96b and what does it imply about clouds? (12)
- [ ] L22 — What did JWST NIRISS transmission spectroscopy show for WASP-96b? (⚠L21 same-obj)
- [ ] L23 — What evidence exists for a helium tail escaping WASP-69b? (28)
- [ ] L24 — What was detected escaping from HAT-P-11b? (14)
- [ ] L25 — What does the phase curve of LTT 9779b indicate about clouds on ultra-hot Neptunes? (6)
- [ ] L26 — What atmospheric characterization exists for the sub-Neptune TOI-270d? (15)
- [ ] L27 — What have transmission spectra revealed about aerosols on WASP-6b? (38)
- [ ] L28 — What is known about the atmospheric composition of WASP-127b? (14)
- [ ] L29 — What cloud species were identified in WASP-17b with JWST? (27)
- [ ] L30 — What evidence supports a thermal inversion in WASP-18b? (18)
- [ ] L31 — What molecules have been measured in the directly imaged HR 8799 planets? (20)
- [ ] L32 — What is known about the atmosphere (or lack of one) of 55 Cancri e? (25)
- [ ] L33 — What wind speeds have been inferred on HD 189733b from high-resolution spectroscopy? (⚠L12 same-obj)
- [ ] L34 — What did ESPRESSO observations reveal about iron condensation on WASP-76b? (⚠L05 same-obj)
- [ ] L35 — What is the detection status of TiO in WASP-33b? (20)
- [ ] L36 — What role does H⁻ opacity play in ultra-hot Jupiter daysides? (ultra-hot: 107)
- [ ] L37 — What did Spitzer phase curves establish about heat redistribution in hot Jupiters? (Spitzer: 138)
- [ ] L38 — What methane detection has JWST made for WASP-80b? (14)
- [ ] L39 — What atmospheric metallicity was inferred for GJ 436b and why is it puzzling? (32)
- [ ] L40 — What is known about the atmosphere of GJ 3470b? (14)
- [ ] L41 — What has been observed of atmospheric escape from young planets around AU Mic? (12)
- [ ] L42 — What brightness temperature did JWST measure for TRAPPIST-1b at 15 microns? (⚠L17 same-obj)
- [ ] L43 — What alkali metals have been detected in WASP-52b? (10)
- [ ] L44 — What did JWST observe in the atmosphere of the warm Neptune WASP-69b beyond helium? (⚠L23 same-obj)
- [ ] L45 — How was CO first detected in an exoplanet atmosphere at high spectral resolution? (high-res: 111)
- [ ] L46 — What quartz or silicate cloud evidence has JWST found in giant-planet atmospheres? (condensat: 106)
- [ ] L47 — What constraints exist on the albedo of hot Jupiters from optical phase curves? (albedo: 115)
- [ ] L48 — What did helium 10830 Å observations show for WASP-69b vs HD 189733b? (helium: 120; ⚠multi-obj)
- [ ] L49 — What morning–evening terminator asymmetry has been observed in transmission spectra? (terminator: 62)
- [ ] L50 — What has JWST measured for the mini-Neptune GJ 9827d? (8)

## EVAL — summarization (36 candidates, target ≥30)

- [ ] S01 — Summarize what the literature says about water vapor detections in hot Jupiter atmospheres. (water vapor: 77)
- [ ] S02 — Summarize the use of the helium 10830 Å line as a probe of atmospheric escape. (helium: 120)
- [ ] S03 — Summarize current understanding of clouds and hazes in sub-Neptune atmospheres. (clouds: 334)
- [ ] S04 — Summarize the proposed mechanisms of atmospheric escape and how observations distinguish them. (escape: 131)
- [ ] S05 — Summarize the evidence for and causes of thermal inversions in ultra-hot Jupiters. (47)
- [ ] S06 — Summarize how high-resolution cross-correlation spectroscopy characterizes exoplanet atmospheres. (111)
- [ ] S07 — Summarize JWST's early exoplanet-atmosphere results across planet classes. (JWST: 533)
- [ ] S08 — Summarize the transit light source effect and mitigation strategies for M-dwarf planet spectra. (55)
- [ ] S09 — Summarize atmospheric retrieval methods and their known biases. (retrieval: 438)
- [ ] S10 — Summarize GCM predictions for hot Jupiter atmospheric circulation. (GCM/circulation: ~124)
- [ ] S11 — Summarize what phase curves reveal about day–night heat transport. (122)
- [ ] S12 — Summarize the role of photochemistry in irradiated exoplanet atmospheres. (215)
- [ ] S13 — Summarize how the C/O ratio is used as a planet-formation tracer. (138)
- [ ] S14 — Summarize the observed mass–metallicity trend in exoplanet atmospheres. (275)
- [ ] S15 — Summarize the prospects for detecting secondary atmospheres on rocky M-dwarf planets. (36)
- [ ] S16 — Summarize JWST evidence on whether rocky planets around M dwarfs retain atmospheres. (M dwarf: 176)
- [ ] S17 — Summarize proposed biosignature gases and their false-positive scenarios. (171)
- [ ] S18 — Summarize the case for ozone as a detectable biosignature. (34)
- [ ] S19 — Summarize the state of habitability assessment for the TRAPPIST-1 system. (114)
- [ ] S20 — Summarize the hycean-world hypothesis and its candidate planets. (17)
- [ ] S21 — Summarize the water-world versus gas-dwarf interpretations of sub-Neptunes. (water world: 20)
- [ ] S22 — Summarize what is known about magma-ocean and lava-planet atmospheres. (magma: 58)
- [ ] S23 — Summarize the chemistry unique to ultra-hot Jupiters (dissociation, H⁻). (107)
- [ ] S24 — Summarize how brown dwarf studies inform exoplanet atmosphere science. (137)
- [ ] S25 — Summarize spectroscopy results for directly imaged giant planets. (72)
- [ ] S26 — Summarize approaches to modeling aerosol microphysics in exoplanet atmospheres. (135)
- [ ] S27 — Summarize the history of sodium detections in hot Jupiters. (48)
- [ ] S28 — Summarize search results for TiO and VO in hot Jupiter atmospheres. (62/17)
- [ ] S29 — Summarize the use of isotopologue ratios as formation tracers. (isotop: 55)
- [ ] S30 — Summarize Lyman-α versus helium observations of atmospheric mass loss. (17/120)
- [ ] S31 — Summarize machine-learning applications to atmospheric retrieval. (42)
- [ ] S32 — Summarize the science goals of the ARIEL mission for atmosphere surveys. (116)
- [ ] S33 — Summarize ELT-era prospects for exoplanet atmosphere characterization. (179)
- [ ] S34 — Summarize how stellar flares affect the atmospheres of M-dwarf planets. (86)
- [ ] S35 — Summarize evidence for nightside clouds on hot Jupiters. (nightside: 112)
- [ ] S36 — Summarize what limb asymmetries reveal about 3D atmosphere structure. (62)

## EVAL — abstention (38 candidates, target ≥25)

`absent` — object/topic verified 0 hits in corpus (the q15-style trap: plausible, famous, absent):
- [ ] A01 — What has JWST measured for the atmosphere of LHS 475b? (0 — the canonical phrase-excluded probe)
- [ ] A02 — What does the emission spectrum of the lava planet K2-141b show? (0)
- [ ] A03 — What is known about the atmosphere of Ross 128 b? (0)
- [ ] A04 — Could Kepler-186f retain a habitable atmosphere? (0)
- [ ] A05 — What atmospheric observations exist for HD 40307g? (0)
- [ ] A06 — What is known about the habitable-zone planet Wolf 1069 b? (0)
- [ ] A07 — What are the atmosphere prospects for YZ Ceti b? (0)
- [ ] A08 — Has the atmosphere of Luyten's Star b (GJ 273b) been characterized? (0)
- [ ] A09 — What do we know about the atmospheres of the HD 110067 sub-Neptunes? (0)
- [ ] A10 — What has been measured for TOI-4860 b? (0)
- [ ] A11 — What atmospheric studies exist for the SPECULOOS-discovered planets? (0)
- [ ] A12 — Have atmospheres of exo-Trojan companions been detected? (0)
- [ ] A13 — What is the atmosphere of NGTS-10b like? (0)
- [ ] A14 — Has methyl chloride been detected in any exoplanet atmosphere? (CH3Cl: 0, methyl chloride: 2 — near-absent)

`thin` — 1–5 hits; retrieved context cannot support the asked detail:
- [ ] A15 — What is the atmospheric composition of the iron planet GJ 367b? (1)
- [ ] A16 — What remains of the atmosphere of the stripped giant core TOI-849b? (1)
- [ ] A17 — What is known about the atmosphere of GJ 357b? (1)
- [ ] A18 — What has been measured for the ultra-short-period planet TOI-1685b? (1)
- [ ] A19 — What does volcanism imply for the atmosphere of LP 791-18d? (2)
- [ ] A20 — What atmospheric observations exist for Teegarden's Star b? (5, no observations)
- [ ] A21 — Has Barnard's Star b's atmosphere been studied? (2)
- [ ] A22 — What is the detectability of exomoon atmospheres? (8)
- [ ] A23 — Has lightning been detected in an exoplanet atmosphere? (9)
- [ ] A24 — Can ocean glint be used to detect exo-oceans with current telescopes? (3)
- [ ] A25 — What is known about Kepler-1649c? (2)
- [ ] A26 — What atmospheric constraints exist for LTT 1445Ab? (5)
- [ ] A27 — What is known about the atmosphere of Gliese 12 b? (2)
- [ ] A28 — What seasonal variability has been observed in exoplanet atmospheres? (11)

`off_topic` — retrieval returns plausible-looking but irrelevant abstracts:
- [ ] A29 — What is the current status of the Hubble-constant tension?
- [ ] A30 — How does AGN feedback quench star formation in massive galaxies?
- [ ] A31 — What resolves the solar coronal heating problem?
- [ ] A32 — What does Gaia reveal about the Milky Way's merger history?

`contradictory` — corpus genuinely disagrees; correct behavior flags the dispute:
- [ ] A33 — Has dimethyl sulfide been detected on K2-18b? (DMS: 15, disputed claims)
- [ ] A34 — Was the 2019 K2-18b detection water vapor or methane? (contested reinterpretation)
- [ ] A35 — Does GJ 1132b have a secondary atmosphere? (12; claim + non-confirmation)
- [ ] A36 — Is TiO definitively present in WASP-121b's atmosphere? (disputed detections)
- [ ] A37 — Is phosphine a reliable biosignature gas? (10, contested)
- [ ] A38 — Do the TRAPPIST-1 planets have atmospheres? (partial/contested per-planet evidence)

## CALIBRATION — positives (48 candidates, target ≥40)

lit_qa-style (30):
- [ ] CP01 — What causes the near-UV absorption observed during WASP-12b transits? (WASP-12: 73)
- [ ] CP02 — What has been inferred about WASP-12b's orbital decay and atmospheric loss? (⚠CP01 same-obj)
- [ ] CP03 — What emission features indicate a temperature inversion in WASP-33b? (20)
- [ ] CP04 — What did CHEOPS observe for the ultra-hot Jupiter WASP-189b? (8)
- [ ] CP05 — Which metals were detected in MASCARA-2b/KELT-20b's transmission spectrum? (17)
- [ ] CP06 — What is the helium escape signature of GJ 436b? (32)
- [ ] CP07 — Why is GJ 436b's low methane abundance surprising? (⚠CP06 same-obj)
- [ ] CP08 — What water detection was reported for GJ 9827d? (8)
- [ ] CP09 — What is known about the atmosphere of HAT-P-7b from its phase curve? (8)
- [ ] CP10 — What helium tail structure was observed around HAT-P-32b? (6)
- [ ] CP11 — What has transmission spectroscopy shown for WASP-31b? (6)
- [ ] CP12 — What aerosol properties were inferred for WASP-19b? (8)
- [ ] CP13 — What was measured for the low-density planet KELT-11b? (6)
- [ ] CP14 — What metallicity constraint exists for HD 149026b? (6)
- [ ] CP15 — What escape observations exist for GJ 3470b's helium? (14)
- [ ] CP16 — What did JWST reveal about ammonia in cool giant atmospheres? (35)
- [ ] CP17 — What HCN detections have been claimed in hot Jupiter atmospheres? (67)
- [ ] CP18 — What role does CO2 play as a metallicity indicator in giant planets? (49)
- [ ] CP19 — What have Kepler-51's super-puff planets revealed about hazes? (5)
- [ ] CP20 — What potassium detections exist in exoplanet atmospheres? (17)
- [ ] CP21 — What has been learned from the phase curve of the ultra-hot Neptune LTT 9779b at optical wavelengths? (⚠L25 same-obj, different aspect)
- [ ] CP22 — What ionized species appear in KELT-9b's transmission spectrum? (⚠L07 same-obj)
- [ ] CP23 — What X-ray irradiation effects on atmospheres have been modeled or observed? (125)
- [ ] CP24 — What deuterium-based measurements exist for exoplanet atmospheres? (11)
- [ ] CP25 — What eastward hotspot offsets have been measured in hot Jupiter phase curves? (27)
- [ ] CP26 — What did polarimetry contribute to exoplanet atmosphere characterization? (31)
- [ ] CP27 — What atmospheric variability has been observed in brown dwarf analogs? (137)
- [ ] CP28 — What rain-out / condensation chemistry shapes hot Jupiter terminators? (106)
- [ ] CP29 — What has ESPRESSO measured in ultra-hot Jupiter atmospheres besides WASP-76b? (40)
- [ ] CP30 — What super-puff planets challenge standard atmosphere models? (44 mini-Nept)

summarization-style (10):
- [ ] CP31 — Summarize atmospheric characterization results for warm Neptunes. (—)
- [ ] CP32 — Summarize wind-speed measurements across hot Jupiter atmospheres. (261)
- [ ] CP33 — Summarize how tidally locked rotation shapes atmospheric dynamics. (92)
- [ ] CP34 — Summarize outgassing scenarios for rocky-planet secondary atmospheres. (56)
- [ ] CP35 — Summarize the runaway-greenhouse limit and its exoplanet applications. (20)
- [ ] CP36 — Summarize chemical-equilibrium versus disequilibrium modeling of atmospheres. (98/74)
- [ ] CP37 — Summarize Bayesian model-comparison practice in atmosphere retrievals. (125)
- [ ] CP38 — Summarize stellar-wind interactions with close-in planet atmospheres. (—)
- [ ] CP39 — Summarize the atmospheric signatures of planet migration and accretion history. (56/90)
- [ ] CP40 — Summarize ionization processes in the upper atmospheres of close-in planets. (60)

abstention-style (8; calibrates the verifier on citation-free examples):
- [ ] CP41 — What is known about the atmosphere of K2-72e? (0 — absent)
- [ ] CP42 — What atmospheric data exist for HD 219134b? (2 — thin)
- [ ] CP43 — What is the atmosphere of CoRoT-7b like? (1 — thin)
- [ ] CP44 — Has a ring system's atmosphere signature been observed transiting a star? (4 — thin)
- [ ] CP45 — What CFC-based technosignature searches target exoplanet atmospheres? (2 — thin)
- [ ] CP46 — How do gravitational waves constrain exoplanet atmospheres? (off_topic)
- [ ] CP47 — Is there water ice on the poles of Mercury? (off_topic — solar system)
- [ ] CP48 — Has vegetation red-edge been detected on an exoplanet? (1 — thin)

## CALIBRATION — negatives (48 candidates, target 40–60, all 4 modes)

Authoring note: the *query* is ordinary; the negative lives in the deliberately mislabeled claims
you write at fill time (`prepare --negative --negative-type <mode>`). Hints suggest the intended
mislabel; adapt freely to what the retrieved abstracts actually say.

`claim_not_supported` (12) — cite a real, on-topic paper for a detail it doesn't state:
- [ ] CN01 — What is the water abundance of WASP-127b? (hint: cite its detection paper for a precise value it doesn't give)
- [ ] CN02 — What is the dayside temperature of WASP-18b? (hint: attribute a specific K value absent from the cited abstract)
- [ ] CN03 — What clouds exist on WASP-107b? (hint: name a condensate species the abstract doesn't identify)
- [ ] CN04 — How fast is HD 209458b losing mass? (hint: cite escape paper for an unstated rate)
- [ ] CN05 — What is the metallicity of WASP-43b? (hint: unstated numeric enrichment)
- [ ] CN06 — What aerosols does WASP-6b host? (hint: unstated particle size)
- [ ] CN07 — What did NIRISS measure for WASP-96b's sodium? (hint: unstated abundance)
- [ ] CN08 — What wind speed does KELT-9b exhibit? (hint: unstated km/s figure)
- [ ] CN09 — What is TOI-270d's methane abundance? (hint: unstated mixing ratio)
- [ ] CN10 — What haze production rate fits GJ 1214b? (hint: unstated rate)
- [ ] CN11 — What is HR 8799e's C/O ratio? (hint: unstated precise value)
- [ ] CN12 — What helium absorption depth does WASP-69b show? (hint: unstated percentage)
- 
`contradicts` (12) — claim the opposite of what the cited abstract reports:
- [ ] CN13 — Was methane detected in WASP-39b by JWST NIRCam? (hint: claim detection where the abstract reports an upper limit)
- [ ] CN14 — Is WASP-96b's atmosphere cloudy? (hint: invert the clear-atmosphere finding)
- [ ] CN15 — Does TRAPPIST-1b's 15μm emission indicate a thick CO2 atmosphere? (hint: invert the bare-rock conclusion)
- [ ] CN16 — Does GJ 1214b have a cloud-free atmosphere? (hint: invert)
- [ ] CN17 — Did JWST find WASP-80b methane-free? (hint: invert the detection)
- [ ] CN18 — Is WASP-76b's iron signal symmetric across its terminators? (hint: invert asymmetry)
- [ ] CN19 — Does WASP-121b lack a thermal inversion? (hint: invert)
- [ ] CN20 — Is the atmosphere of 55 Cancri e permanent and stable? (hint: invert variability finding)
- [ ] CN21 — Did the WASP-107b JWST spectrum show high methane? (hint: invert depletion)
- [ ] CN22 — Was helium absent from HAT-P-11b's transmission spectrum? (hint: invert)
- [ ] CN23 — Does stellar contamination negligibly affect TRAPPIST-1 spectra? (hint: invert)
- [ ] CN24 — Did WASP-17b's JWST spectrum rule out silicate clouds? (hint: invert quartz detection)

`wrong_paper` (12) — support a correct-ish claim with a real corpus paper about the wrong object/topic
(the mode where an out-of-pool citation is a warning, not an error — you may cite corpus papers
outside the query's retrieved pool on purpose):
- [ ] CN25 — What did JWST detect on WASP-39b? (hint: cite a WASP-96b paper)
- [ ] CN26 — What helium escape was measured for WASP-69b? (hint: cite the GJ 436b/HAT-P-11b escape paper)
- [ ] CN27 — What thermal inversion was found in WASP-33b? (hint: cite the KELT-9b paper)
- [ ] CN28 — What water was measured in HD 209458b? (hint: cite HD 189733b)
- [ ] CN29 — What clouds were inferred for GJ 1214b? (hint: cite a GJ 436b paper)
- [ ] CN30 — What did the phase curve of WASP-43b show? (hint: cite the WASP-18b phase curve)
- [ ] CN31 — What iron chemistry occurs in WASP-76b? (hint: cite the WASP-121b iron paper)
- [ ] CN32 — What was found in K2-18b's atmosphere? (hint: cite a TOI-270d paper)
- [ ] CN33 — What sodium detection exists for WASP-52b? (hint: cite the WASP-96b sodium paper)
- [ ] CN34 — What did MIRI measure for TRAPPIST-1b? (hint: cite the LHS 3844b emission paper)
- [ ] CN35 — What escape tail was seen at WASP-107b? (hint: cite the WASP-69b tail paper)
- [ ] CN36 — What C/O was inferred for HR 8799 planets? (hint: cite a WASP-39b C/O paper)

`overgeneralization` (12) — cited abstract shows one case; claim asserts a population-level rule:
- [ ] CN37 — Do hot Jupiters have thermal inversions? (hint: one detection → "hot Jupiters generally do")
- [ ] CN38 — Are sub-Neptune atmospheres always hazy? (hint: GJ 1214b → all)
- [ ] CN39 — Do M-dwarf rocky planets lack atmospheres? (hint: TRAPPIST-1b → all)
- [ ] CN40 — Is helium escape ubiquitous in warm Neptunes? (hint: one tail → all)
- [ ] CN41 — Are ultra-hot Jupiter daysides always cloud-free? (hint: one case → all)
- [ ] CN42 — Do all hot Jupiters show eastward hotspot offsets? (hint: some → all)
- [ ] CN43 — Is methane depleted in every warm giant atmosphere? (hint: WASP-107b/GJ 436b → all)
- [ ] CN44 — Are quartz clouds standard in hot Jupiter atmospheres? (hint: WASP-17b → all)
- [ ] CN45 — Does photochemical SO2 appear in all irradiated giants? (hint: WASP-39b → all)
- [ ] CN46 — Are super-puff planets always explained by hazes? (hint: Kepler-51 → all)
- [ ] CN47 — Do young planets universally show inflated escaping atmospheres? (hint: AU Mic → all)
- [ ] CN48 — Is stellar contamination the dominant systematic for all transmission spectra? (hint: M-dwarf case → all)

---

Triage guidance (delete freely): the ⚠same-obj flags mark where eval and eval, or calibration and
eval, share an *object* though queries differ — the harness only enforces query-level disjointness,
but trimming same-obj pairs buys extra contamination insurance for the H1–H3 evals. Every hit
count is from the frozen snapshot; if you add your own queries, verify with a corpus grep first
and prefer objects/topics with ≥6 hits for positives and verified 0–5 hits for abstention.
