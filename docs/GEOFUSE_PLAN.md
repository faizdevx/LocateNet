# GeoFuse: location assist for sighting photos (plan)

One codebase for both outputs: the **LocateNet feature** and the **research paper**. The paper's
experiments run through the existing eval harness (`python -m eval run`), so the numbers in the
paper and the behaviour of the app come from the same code.

Claim we defend: an Indian multi-city benchmark of degraded pedestrian / CCTV-like photos, and a
tested, calibrated way of combining existing evidence. Not "finds any photo on Earth", not the first
text + map + image fusion (VPRTextImage, Hettiarachchi 2022, did that).

## What we reuse instead of writing

| Need | Already here |
|---|---|
| Geocoding sign text and the reporter's location | `geopy` Nominatim (`app.py`, `geolocator`) |
| Map | `folium` (used for the regional map) |
| Calibration of confidences | `services/calibration.py` (isotonic, `predict_many`) |
| Officer verdicts to calibration data | the review / feedback loop pattern (`docs/FEEDBACK_LOOP.md`) |
| Audit, roles, CSRF | `log_audit`, `login_required`, Flask-WTF |
| Face blur before anything leaves the server | `services/face_service.py` (SCRFD boxes) |
| Benchmark runs, results JSON, report, regression gate | `eval/` |
| Retrieval index | `faiss-cpu` |

New dependencies only in an optional `requirements-geo.txt`: `easyocr` (Indic OCR) and, from
phase 3, a place-recognition model plus `lightglue`. Core install and CI stay unchanged.

## Phase 1: the feature, rules only (done)

Officer-only, off by default (`LOCATENET_GEO_ASSIST=0`).

1. `services/geo.py`, one module:
   - `exif_gps(path)`: GPS from metadata (Pillow, already installed).
   - `sign_texts(img)`: EasyOCR (`hi`, `en`, plus the state's script), drop short / low-confidence text.
   - `candidates(texts, prior_latlon)`: Nominatim search per text, bounded to about 10 km
     around the prior (case city or reporter location).
     `shortcut: public Nominatim is limited to 1 request/s; self-host Nominatim or Photon before any pilot.`
   - `fuse(...)`: merge candidates within 25 m, score = number and quality of agreeing evidence,
     **abstain** ("Not sure") when no candidate has evidence beyond the prior.
2. Two routes in `app.py`, next to `sighting_detail`:
   - `POST /sighting/<id>/geo`: case officer or Admin only, CSRF, per-officer rate limit, `log_audit`.
   - `POST /sighting/<id>/geo/<cand>/verdict`: Confirm / Wrong / Unsure. Confirm stores the verified
     location on the sighting page; it never overwrites the reporter's text (the hotspot rule still uses the typed city).
3. Tables `geo_requests`, `geo_candidates` (lat, lon, radius_m, confidence, evidence JSON, verdict,
   reviewer) in `create_db()`.
4. `templates/officer_sighting_detail.html`: a "Locate this photo" button and a results block:
   - the folium map with candidate circles and the reporter's location,
   - evidence chips,
   - a Street View embed (Google Maps Embed API, needs `GOOGLE_MAPS_EMBED_KEY`), else a link to Mapillary.
   - a "location mismatch" badge when the reporter's location and the best candidate are more than 15 km apart.
5. Safety, enforced in code: works only on photos already attached to a sighting (no free upload
   box), never runs automatically, never shown on public pages, no image leaves the server (only sign text goes to the geocoder).
6. Tests: an EXIF fixture, a stub OCR and a stub geocoder, fusion (missing, conflicting, abstain),
   route permissions (other officer 403, public redirect, missing CSRF rejected), the flag off hides the button.

## Phase 2: benchmark and baselines (paper weeks 1-7)

1. Week 1-2 pilot: about 50 phone-GPS photos per candidate city; count Mapillary images in each city's
   bounding box (Mapillary API v4, free token); go / no-go per city. Ethics committee application.
2. `scripts/geo_collect.py`: download the Mapillary database for the chosen cities (CC BY-SA 4.0,
   attribution). Never download Google Street View images.
3. Queries: independently captured phone photos (log GPS accuracy, drop fixes worse than 10 m), real
   changes (night, viewpoint, occlusion, months apart).
4. Degraded variants in the same split as their original: "CCTV-like" downscale, JPEG, blur, and a
   real WhatsApp send-and-receive round trip (procedure documented).
5. Blur faces (SCRFD) and number plates before storing or releasing anything.
6. `eval/datasets/geo_india.py` (manifest, splits) and `eval/protocols/geo.py` (metrics: Recall@1/5
   within 25/50/100 m, % within 1 / 25 / 200 km, median error, ECE, Brier, abstention rate).
7. Baselines through the same protocol: GeoCLIP; retrieval with CosPlace, EigenPlaces, AnyLoc, MegaLoc;
   plus LightGlue reranking; a VPRTextImage reimplementation; a vision-language model on blurred images.

## Phase 3: GeoFuse (paper weeks 8-12, then into the app)

1. Add retrieval evidence to `services/geo.py`: FAISS index of the city's database descriptors, top-100
   inside the prior region, LightGlue on the top-20, inliers as evidence.
2. Replace the rule score in `fuse` with logistic regression over the evidence features, then the
   existing isotonic calibrator. Event being calibrated: "top-1 merged place within 25 m of the camera".
3. Ablations (each is one YAML config): no OCR, oracle transcripts, no prior / correct / coarse / wrong
   prior, no reranking, database density 10-100 %, OSM POI completeness audit per city.
4. Officer verdicts from the app feed `services/recalibration.py`; a geolocation card on
   `/admin/model-health` ("insufficient data" below 30 verdicts).

## Phase 4: user study (weeks 13-14)

12-20 participants, within-subject, counterbalanced, 10 photos each, with and without the tool.
Measures: error distance, time, false acceptance, NASA-TLX. Ethics approval and consent first.

## Phase 5: paper (weeks 15-16)

`python -m eval report` writes the result tables and plots; the paper cites those files.

1. Introduction: sighting photos without location; dual-use risk.
2. Related work: global geolocation (PIGEON, GeoCLIP, OSV-5M), VPR (CosPlace, EigenPlaces, AnyLoc,
   MegaLoc, IDD-VPR), text-based localization (VPRTextImage), VPR uncertainty (SUE, KappaPlace),
   LLM geolocation and privacy (DoxBench).
3. Benchmark: collection, degradations, blurring, statistics, map-completeness audit.
4. Method: evidence, merging, missing and conflicting evidence, abstention, calibration.
5. Experiments: RQ1-RQ4 with bootstrap CIs, per city and per degradation.
6. User study: RQ5.
7. Ethics and governance: restricted access, audit logging, no free upload, blurred release.
8. Limitations: three cities, GPS noise, Mapillary bias toward main roads.

Venues: ICVGIP, WACV, CVPR / ICCV workshops, ACM COMPASS. Check every citation against the paper
itself before submission.

## Not doing (until a need shows up)

- A separate research repository: the eval harness already does the job.
- An OSM POI database: Nominatim bounded search is enough until the pilot.
- A coarse global geolocation model inside the app: the case city is a stronger prior in practice.
- A job queue: phase 1 takes about 1-3 s; revisit when retrieval lands.
