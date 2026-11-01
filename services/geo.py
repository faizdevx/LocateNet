"""Location assist for sighting photos, phase 1 (rules only, no training). See docs/GEOFUSE_PLAN.md.

Evidence: GPS in the photo's EXIF data, and shop-sign text that the geocoder finds near the
reporter's area. Points within MERGE_M of each other are merged into one candidate place. Without
evidence beyond the reporter's area the result abstains ("Not sure", no candidates).

Levels are evidence counts, not probabilities: there is no calibration data until officers have
given verdicts (phase 3 replaces the rule score with a calibrated one).
"""
from geopy.distance import distance
from PIL import Image

MERGE_M = 25          # candidates closer than this are one place
SEARCH_KM = 10        # sign text is only looked up within this distance of the reporter's area
MISMATCH_KM = 15      # reporter areas are often a whole city: smaller values flag normal sightings
MAX_TEXTS = 4         # sign texts looked up per photo (the public geocoder allows 1 request/s)
MAX_CANDIDATES = 5

_GPS_IFD = 0x8825
_reader = None


def _dms(value, ref):
    d, m, s = (float(x) for x in value)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def exif_gps(path):
    """(lat, lon) from the photo's EXIF GPS tags, or None. Messaging apps usually strip these."""
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(_GPS_IFD)
        lat, lon = _dms(gps[2], gps[1]), _dms(gps[4], gps[3])
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat, lon) == (0.0, 0.0):
        return None
    return lat, lon


def clean_texts(items, min_conf=0.4, max_texts=MAX_TEXTS):
    """Best distinct sign texts from (text, confidence) pairs. Drops short text and anything with 6+
    digits (phone numbers, number plates), which is neither stored nor sent to the geocoder."""
    seen, out = set(), []
    for text, conf in sorted(items, key=lambda t: -t[1]):
        text = " ".join(str(text).split())
        if conf < min_conf or len(text) < 4 or sum(ch.isdigit() for ch in text) >= 6 or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out[:max_texts]


def sign_texts(path, langs=("hi", "en")):
    """Sign text in the photo, or None when the optional easyocr package is not installed."""
    global _reader
    try:
        import easyocr
    except ImportError:
        return None
    if _reader is None:  # first use downloads the OCR models (about 100 MB)
        _reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
    return clean_texts([(text, conf) for _box, text, conf in _reader.readtext(path)])


def search_box(prior, km=SEARCH_KM):
    """((south, west), (north, east)) around prior = (lat, lon)."""
    north = distance(kilometers=km).destination(prior, 0)
    east = distance(kilometers=km).destination(prior, 90)
    dlat, dlon = north.latitude - prior[0], east.longitude - prior[1]
    return (prior[0] - dlat, prior[1] - dlon), (prior[0] + dlat, prior[1] + dlon)


def sign_hits(texts, prior, geocode):
    """geocode(text, box) -> [(lat, lon, place_name)], searched only inside the box around prior."""
    box = search_box(prior)
    (s, w), (n, e) = box
    hits = []
    for text in texts:
        for lat, lon, name in geocode(text, box):
            if s <= lat <= n and w <= lon <= e:  # geocoders may return results outside a bounded box
                hits.append({"source": "sign", "text": text, "name": name, "lat": lat, "lon": lon})
    return hits


def fuse(gps, hits, prior=None):
    """Merge evidence into ranked candidate places.

    Returns {"candidates": [...], "abstain": bool, "mismatch_km": float | None}.
    """
    points = ([{"source": "exif", "lat": gps[0], "lon": gps[1]}] if gps else []) + list(hits)
    places = []
    for p in points:
        for place in places:
            if distance((place["lat"], place["lon"]), (p["lat"], p["lon"])).m <= MERGE_M:
                if p.get("text") is None or p["text"] not in {e.get("text") for e in place["evidence"]}:
                    place["evidence"].append(p)  # one sign often maps to several OSM objects of one place
                break
        else:
            places.append({"lat": p["lat"], "lon": p["lon"], "evidence": [p]})
    for place in places:
        ev = place["evidence"]
        place["lat"] = sum(e["lat"] for e in ev) / len(ev)
        place["lon"] = sum(e["lon"] for e in ev) / len(ev)
        has_gps = any(e["source"] == "exif" for e in ev)
        n_texts = len({e["text"] for e in ev if e["source"] == "sign"})
        place["score"] = 3 * has_gps + n_texts  # GPS outranks agreeing signs
        place["level"] = "strong" if has_gps or n_texts >= 2 else "weak"
        place["radius_m"] = MERGE_M if has_gps else 75  # a sign is seen from across the street
        place["km_from_prior"] = distance(prior, (place["lat"], place["lon"])).km if prior else None
    places.sort(key=lambda c: (-c["score"], c["km_from_prior"] or 0))
    places = places[:MAX_CANDIDATES]
    best = places[0] if places else None
    mismatch = best["km_from_prior"] if best and best["km_from_prior"] and best["km_from_prior"] > MISMATCH_KM else None
    return {"candidates": places, "abstain": not places, "mismatch_km": mismatch}


def locate(path, prior, geocode, ocr=None):
    """Run every evidence source on one photo. Returns fuse()'s result plus "texts" and "notes"."""
    notes = []
    gps = exif_gps(path)
    if gps is None:
        notes.append("No GPS data in the photo (messaging apps remove it).")
    texts = (ocr or sign_texts)(path)
    hits = []
    if texts is None:
        notes.append("Sign reading is not installed (pip install -r requirements-geo.txt).")
        texts = []
    elif not texts:
        notes.append("No readable sign text in the photo.")
    elif prior is None:
        notes.append("The reporter's area could not be located, so sign text was not looked up.")
    else:
        hits = sign_hits(texts, prior, geocode)
        if not hits:
            notes.append(f"None of the sign text was found on the map within {SEARCH_KM} km of the reporter's area.")
    result = fuse(gps, hits, prior)
    result.update(texts=texts, notes=notes)
    return result
