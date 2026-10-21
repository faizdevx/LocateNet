# LocateNet

<p align="center">


```

  _     ___   ____    _  _____ _____   _   _ _____ _____ 
 | |   / _ \ / ___|  / \|_   _| ____| | \ | | ____|_   _|
 | |  | | | | |     / _ \ | | |  _|   |  \| |  _|   | |  
 | |__| |_| | |___ / ___ \| | | |___  | |\  | |___  | |  
 |_____\___/ \____/_/   \_\_| |_____| |_| \_|_____| |_|  
                                                         

```

</p>



<p align="center">

[![CI](https://github.com/faizdevx/LocateNet/actions/workflows/ci.yml/badge.svg)](https://github.com/faizdevx/LocateNet/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-WebFramework-000000?logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-FaceRecognition-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ScientificComputing-013243?logo=numpy&logoColor=white)
![Geopy](https://img.shields.io/badge/Geopy-Geolocation-green)
![Folium](https://img.shields.io/badge/Folium-InteractiveMaps-darkgreen)

</p>



LocateNet is a face-matching assistant for missing-person cases.
Officers enrol a reference photo of a missing person. The public can upload sighting photos
(or videos). The system detects every face, compares each one with the enrolled faces, and shows
likely matches to the responsible officer, who must confirm or reject each one.

**A match is a lead for a human to review, never an identification.** See
[Limitations & Responsible Use](#limitations--responsible-use).

---

## Features

- Detects and extracts multiple faces from group photos (and, optionally, video)
- Generates 512-dimension face embeddings with InsightFace (ArcFace)
- Similarity search using FAISS vector indexing, inspecting the top 5 neighbours per face
- Similarity tiers ("High Similarity", "Possible Match – Needs Review") with a configurable threshold
- Officer review screen with a face gallery and Confirm / False Positive / Ignore decisions
- Hotspot alert when the same case is matched repeatedly in one city (see below)

---

## System Architecture

| Layer | Component | Description |
|------|------|------|
| Vision | InsightFace (buffalo_l) | Face detection, alignment, and embedding extraction |
| Vector Search | FAISS | Cosine similarity search over active (not yet found) cases |
| Backend | Flask + SQLite | Routes, sessions, CSRF protection and relational storage (`sqlite3`) |
| Video (optional) | YOLOv8 + DeepSORT | Person detection and tracking before face matching |
| Mapping | Folium / Geopy | Map of cases per city, using coordinates stored at registration |

![workflow](workflow.png) 

Body re-identification is **not** part of the working system. A placeholder exists behind
`ENABLE_REID` (off by default); it uses an untrained projection head and is not a valid ReID model.

---

## Project Structure

```
LocateNet
│
├── app.py                  Flask app and routes
├── config.py               Thresholds and settings (environment variables)
├── models.py               SQLite schema and queries
├── init_system.py          Creates the database, admin user and empty vector index
├── clean_system.py         DESTRUCTIVE: wipes local data
│
├── pipeline
│   ├── detection_pipeline.py   Image pipeline
│   ├── video_pipeline.py       Video pipeline (optional extras)
│   └── scoring.py              Shared threshold and category logic
│
├── services
│   ├── face_service.py         InsightFace wrapper
│   ├── alert_service.py        Hotspot rule
│   └── reid_service.py         Placeholder, disabled by default
│
├── vector_db
│   ├── search_service.py       FAISS index
│   ├── rebuild.py              Rebuild / sync the index from SQL
│   └── storage                 Index files (created at runtime)
│
├── scripts                 rebuild_index.py, download_models.py, dev.ps1
├── templates               HTML templates
├── tests                   pytest suite
├── docs                    HISTORY_CLEANUP.md
├── samples                 Synthetic test images (see samples/README.md)
│
├── resources               Face crops and sighting media (created at runtime, git-ignored)
├── uploads                 Case photos (created at runtime, git-ignored)
└── logs                    system.log (created at runtime, git-ignored)
```

---

## WORKING 
## Application Interface

### 1. Common Interface
<img src="app_snaps/01.png" width="900">

### 2. Admin / Officer Login
<img src="app_snaps/04.png" width="900">

### 3. Role Based Task Dashboard
<img src="app_snaps/05.png" width="900">

### 4. Officer Registers New Case
<img src="app_snaps/07.png" width="900">

### 5. Reporting by Public / Other Officers
<img src="app_snaps/03.png" width="900">

### 6. Face Match Detection Notification
<img src="app_snaps/11.png" width="900">

### 7. Officer Verification (Found / Not Found)
<img src="app_snaps/12.png" width="900">

## Week 1 Upgrade (October 2026)

Correctness, security and reproducibility pass. No new features and no UI redesign.

### 1. Login (CSRF-protected form, secret key from `.env`)
<img src="app_snaps/week1_01_login.png" width="900">

### 2. Hotspot alert: same case matched 3 times in one city within 48 hours
<img src="app_snaps/week1_02_hotspot_flash.png" width="900">

### 3. Officer alerts show raw similarity and say a human must review
<img src="app_snaps/week1_03_officer_dashboard.png" width="900">

### 4. Upload validation (type and size are checked, errors are friendly)
<img src="app_snaps/week1_04_upload_validation.png" width="900">

### 5. Officer review screen: possible match (53.8% similarity) with Confirm / False Positive / Ignore
<img src="app_snaps/week1_05_review_faces.png" width="900">

### 6. Public registry: only missing cases, with name, city, date and photo
<img src="app_snaps/week1_06_public_registry.png" width="900">

Screenshots are from a demo run with two test photos.

## Installation

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/faizdevx/LocateNet.git
cd LocateNet
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

| File | Adds |
|------|------|
| `requirements-video.txt` | Video uploads (`ultralytics`, `deep-sort-realtime`) |
| `requirements-reid.txt` | Placeholder body ReID (`torch`, `torchvision`), not recommended |
| `requirements-dev.txt` | Lightweight test and lint tools (pytest, ruff) |

Create your settings file and set a secret key:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into LOCATENET_SECRET_KEY
```

`make setup`, `make init` and `make run` wrap these steps; on Windows use `.\scripts\dev.ps1 setup|init|run`.

### System initialization

```bash
python init_system.py
```

This creates the database tables, the `admin` user and an empty FAISS index. The admin password is
taken from `LOCATENET_ADMIN_PASSWORD`; if that is not set, a random password is generated and
**printed once**. Store it straight away. There is no password-change screen yet.

Optionally pre-download the models (InsightFace `buffalo_l`, and `yolov8n.pt` for video):

```bash
python scripts/download_models.py
```

### Running

```bash
python app.py
```

The app listens on http://127.0.0.1:5000. Without `LOCATENET_SECRET_KEY` it refuses to start,
unless `FLASK_DEBUG=1` is set (then it uses a random key and sessions reset on restart).
`use_reloader=False` is used so the AI model is not loaded twice.

If the FAISS index file is missing or does not match the cases in the database, the app rebuilds
it at start-up (and logs a warning). You can also run `python scripts/rebuild_index.py`.

### Docker

```bash
cp .env.example .env        # set LOCATENET_SECRET_KEY
docker compose up --build
docker compose run --rm locatenet python init_system.py
```

The image runs `gunicorn -w 1 --threads 4` (one worker because the models are memory-heavy) and
stores the database, media and vector index in named volumes. Set the build argument
`INSTALL_VIDEO` to `"1"` in `docker-compose.yml` for video support.

### Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests mock the face models, so they need no GPU, model download, insightface or torch.

---

## Processing Workflow

### Enrollment

1. Officer uploads a reference image
2. Face embedding (512D vector) is generated
3. Case row is saved in SQLite, then the vector is added to FAISS under the case id
   (if FAISS fails, the case row is rolled back)

### Sighting

1. Citizen uploads a group image (or a video if the extras are installed)
2. System detects all faces and stores a crop of each
3. Each face is embedded and searched against the index (top 5)

### Matching

1. The first neighbour that is at or above `MATCH_THRESHOLD` **and** belongs to a case that is still
   missing (not resolved) is the match. A resolved case at rank 1 does not hide an active case at rank 2.
2. A match creates one alert for the officer who owns the case. Faces below the threshold create nothing.
3. When an officer marks a case as found, its vector is removed from the index.

---

## Match Scoring

Scores are **raw ArcFace cosine similarity, not probabilities**. Thresholds are provisional pending
evaluation (Week 2).

| Cosine similarity | Label | What happens |
|------|------|------|
| ≥ `HIGH_CONF_THRESHOLD` (0.60) | High Similarity | Alert, officer reviews |
| ≥ `MATCH_THRESHOLD` (0.45) | Possible Match – Needs Review | Alert, officer reviews |
| below `MATCH_THRESHOLD` | No Match | No alert |

The percentage shown in the interface is the similarity multiplied by 100. It is a display value,
not a likelihood that the person is the same.

---

## Hotspot alert

A hotspot is raised when **the same case is matched at least 3 times in the same city within 48
hours**. "City" is the location text entered with the sighting, compared after trimming and
lower-casing. Three sightings of different people do not make a hotspot.

---

## Configuration

Settings are read from environment variables (or a `.env` file). See `.env.example`.

| Variable | Default | Meaning |
|------|------|------|
| `LOCATENET_SECRET_KEY` | none | Session signing key. Required unless `FLASK_DEBUG=1` |
| `LOCATENET_ADMIN_PASSWORD` | random | Initial admin password used by `init_system.py` |
| `FLASK_DEBUG` | `0` | Debug mode; enables a random secret key when none is set |
| `LOCATENET_DB` | `database.db` | SQLite database path |
| `MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity for a match |
| `HIGH_CONF_THRESHOLD` | `0.60` | Similarity for the "High Similarity" label |
| `DET_SCORE_MIN` | `0.6` | Minimum face-detector score |
| `TOP_K` | `5` | Nearest neighbours inspected per face |
| `ENABLE_REID` | `0` | Placeholder body ReID (not a valid ReID model) |

Uploads: images up to 10 MB (jpg, jpeg, png, webp), videos up to 100 MB (mp4, mov, avi).

---

## Security

- Sessions are signed with your own secret key, cookies are `HttpOnly` and `SameSite=Lax`, and
  the session is reset on login.
- All forms carry CSRF tokens; resolving a case and dismissing an alert are POST-only.
- Officers can only dismiss alerts, review faces and resolve cases that belong to them (Admins can act on all).
- Case photos, sighting media and face crops require a login. The public registry shows only the
  name, city, date and photo of cases that are still missing.
- Resolved cases are removed from the vector index and ignored for future alerts.

---

## Limitations & Responsible Use

- **Every match needs human confirmation.** A high similarity score is not proof of identity.
  Face recognition makes mistakes, especially with poor lighting, low resolution, occlusion, age
  change and similar-looking people.
- **Not evaluated yet.** Accuracy and the thresholds have not been measured on a labelled data set,
  and the system has **not been evaluated for demographic bias** (skin tone, gender, age). Do not
  treat the thresholds as validated.
- **Not for surveillance.** LocateNet is for locating missing people with the aim of helping them.
  Do not use it to track, profile or monitor people.
- **Personal data and retention.** The system stores face images, face embeddings, and the names,
  mobile numbers and emails of people who report sightings. There is no automatic deletion: decide
  a retention period, and delete resolved cases and unmatched sighting photos you no longer need.
  Under India's Digital Personal Data Protection Act, 2023, personal data should be collected for a
  specific lawful purpose with notice, kept only as long as needed, and protected with reasonable
  safeguards. Get legal advice before using this with real data.
- **No body re-identification.** Only faces are matched.
- Earlier versions of this repository stored real data in git history. See
  [docs/HISTORY_CLEANUP.md](docs/HISTORY_CLEANUP.md).

---

## Tech Stack

- Python 3.11 / 3.12
- Flask, Flask-WTF
- SQLite (`sqlite3`)
- InsightFace, ONNX Runtime
- FAISS
- OpenCV, NumPy
- Geopy, Folium
- Optional: Ultralytics YOLOv8, DeepSORT

---

## License

See `LICENSE` file.

---

## Author

faizdevx
