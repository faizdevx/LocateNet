# ZIPNET sync

LocateNet can keep a local, searchable copy of the public records that the
[ZIPNET portal](https://zipnet.delhipolice.gov.in/) publishes (missing persons, unidentified persons
found, unidentified dead bodies, stolen vehicles, unclaimed/seized vehicles, missing mobiles).
It is a background worker next to the Flask app: it never runs inside a web request.

> **Starting LocateNet with `ZIPNET_SYNC_ENABLED=1` makes outbound HTTPS requests to
> `ZIPNET_BASE_URL`** (about one request every 2 seconds while it works). Set it to `0` to make
> none. Field-level details of the source: [ZIPNET_SOURCE_INVENTORY.md](ZIPNET_SOURCE_INVENTORY.md).

## What it does and does not do

- Imports **external public records into their own dataset** (`zipnet_external.db`). They are not
  LocateNet cases, are never matched against faces, never create alerts and never mark anything found.
- Face matching is untouched. There is **no code path** from an imported record to InsightFace, FAISS or
  `case_embeddings`. The only bridge is `zipnet_case_links`: an administrator can explicitly link one
  external record to one native case (`POST /admin/zipnet-sync/records/<id>/link`). A link changes
  nothing else and is audit-logged.
- Deny-by-default fields: only allow-listed fields are read from the source. Phone numbers, addresses,
  guardian/family names, officers' mobile numbers, e-mail, religion, engine/chassis numbers and similar
  are never read, so they cannot reach the database, logs, API responses or exports. Phone numbers and
  e-mail addresses that appear inside free-text locations are masked as `[redacted]`.
- Photographs are not copied. Only the validated source-relative image path is stored.
- Map coordinates are not invented. ZIPNET publishes only state / district / police-station names.
- Scrapling runs with browser impersonation and stealth headers **off**, with an identifying
  `User-Agent`. A 401/403/429 or an HTML login/block page stops that section for a long cool-down;
  nothing tries to evade it.

## Architecture

```
python app.py ──(dev only)──► zipnet_worker.py ──► services/zipnet_sync_service.py   lease, scheduling, control flags
Docker: separate service  ──►                        │
                                                      ├─► pipeline/zipnet_pipeline.py   one bounded turn of one section
                                                      │      └─► services/zipnet_scraper.py   Scrapling FetcherSession, retries
                                                      │      └─► services/zipnet_adapters.py  allow-listed normalization
                                                      └─► services/zipnet_store.py          SQLite schema (v1), upserts, checkpoints
Flask: blueprints/zipnet_sync.py  (/admin/zipnet-sync/*, admin only, CSRF)  reads the same SQLite file
```

| File | Role |
|---|---|
| `zipnet_worker.py` | Process entry point (`--once`, `--status`, `--healthcheck`). |
| `services/zipnet_sync_service.py` | Single-worker lease and heartbeat, pause/exit flags, round-robin scheduling. |
| `pipeline/zipnet_pipeline.py` | discover → fetch → validate → normalize → dedupe → persist → diff → checkpoint. |
| `services/zipnet_store.py` | Versioned schema (`PRAGMA user_version`), idempotent upserts, jobs, failures. |
| `services/zipnet_supervisor.py` | Starts/stops the worker as a child of `python app.py` (development). |
| `blueprints/zipnet_sync.py` + `templates/zipnet_sync.html` | Admin status page and API. |

### Database (`ZIPNET_DB`, default `zipnet_external.db` beside the main database)

A separate SQLite file in WAL mode, so a multi-million-row import never blocks case registration.
Existing LocateNet tables are not touched. Tables: `zipnet_records` (unique on `section` + `source_id`),
`zipnet_record_changes` (field-level diffs), `zipnet_checkpoints` (one row per section), `zipnet_jobs`,
`zipnet_failures` (error text only, no payload), `zipnet_lease`, `zipnet_control`, `zipnet_case_links`.
`python init_system.py` creates or upgrades it; it is also created on first use, and repeating it is safe.

## Lifecycle

**Startup** (`python app.py`, `ZIPNET_SYNC_ENABLED=1`, `ZIPNET_AUTOSTART_WORKER=1`)

1. The Flask app starts and serves requests.
2. `app.py`'s `__main__` block calls `autostart_if_enabled()`. Importing `app` (Gunicorn, tests) never does.
3. If no live worker holds the database lease, a child `python zipnet_worker.py --parent-pid <app> --exit-token <t>` is spawned. Log: `logs/zipnet_worker.log`.
4. The worker migrates its database, takes the lease, marks jobs left `running` by a crash as `interrupted` (checkpoints are never reset) and starts fetching immediately.

**Work loop.** Due incremental turns first, otherwise one bounded backfill turn (`ZIPNET_PAGES_PER_TURN`
pages) of the next unfinished section, rotating through the six sections. Each page is committed in
**one transaction together with its checkpoint**. Fresh install → backfill of every section from the
newest record down; interrupted → resumes at the committed offset; backfill finished → incremental
sync every `ZIPNET_INCREMENTAL_INTERVAL_MINUTES`.

- A section is marked `backfill_complete` only when the source reports no further rows (`offset >= recordsTotal`). An empty page before the reported total, a pagination loop or a page-limit stop are reported as **incomplete** and retried after a cool-down.
- `ZIPNET_BACKFILL_MAX_PAGES` > 0 caps total backfill pages per section; a capped section is shown as "backfill incomplete (page limit)" until the cap is raised.
- Incremental: stops when a page is entirely at or below the section's high-water source ID and older than the overlap window. If it cannot catch up within `ZIPNET_INCREMENTAL_MAX_PAGES` it keeps its offset, shows a warning, and continues on the next turn; the high-water mark moves only once it has caught up. The source's date filters are **not** used (they proved unreliable).
- Failures: transient errors are retried with exponential backoff and jitter (`ZIPNET_BACKOFF_*`); a section that still fails cools down (doubling) while the others continue.

**Shutdown.** `python app.py` registers an exit handler that writes an `exit_request` token and waits
`ZIPNET_SHUTDOWN_GRACE_SECONDS` for the worker to finish its current page, escalating to terminate only
if it does not. Ctrl+C/SIGTERM on a standalone worker does the same. If the app is killed without
running exit handlers, the worker notices its parent is gone and exits by itself. On exit the worker
releases the lease and records `stopped`. The next start resumes automatically.

**Duplicates.** One worker per database: a lease row with heartbeat (`ZIPNET_LOCK_TTL_SECONDS`).
A second worker exits with code 3. A crashed worker's lease expires and can be taken over. Gunicorn
workers never start a worker, and Docker runs exactly one `zipnet-worker` service.

## Commands

```bash
python init_system.py                 # creates/upgrades the ZIPNET tables (safe to repeat)
python app.py                         # app + automatic worker
python zipnet_worker.py               # worker only (production without Docker, systemd, ...)
python zipnet_worker.py --status      # per-section progress from the checkpoints
python zipnet_worker.py --once        # run turns until nothing is due, then exit
docker compose up --build             # app + zipnet-worker service
```

## Monitoring and control (Admin role)

| Request | Purpose |
|---|---|
| `GET /admin/zipnet-sync` | Status page (also linked from the admin dashboard). |
| `GET /admin/zipnet-sync/status` | Worker state, per-section phase/progress/warnings, last success. |
| `POST /admin/zipnet-sync/start` | Clear a stop request and sync now. Starts a worker in dev if none is alive. Idempotent. `409` when sync is disabled. |
| `POST /admin/zipnet-sync/stop` | Ask for a stop at the next page boundary. Idempotent. `stop_acknowledged` becomes true only after the worker confirms (`202` until then). |
| `GET /admin/zipnet-sync/jobs?limit=` | Recent turns and recent failures (no payloads). |
| `GET /admin/zipnet-sync/records?section=&q=&state=&limit=&offset=` | Allow-listed fields of imported records. |
| `GET /admin/zipnet-sync/export.csv` | CSV of the same fields (max 5000 rows, audit-logged, formula-safe). |
| `POST /admin/zipnet-sync/records/<id>/link` | Explicit link `{"case_id": n}` of a record to a native case. |

Anonymous callers get `401`, non-admins `403`, state-changing requests need the CSRF token
(`X-CSRFToken` header; the status page sends it). Errors are short JSON messages: no paths, secrets or
stack traces.

## Settings

All in `.env` (see `.env.example`): `ZIPNET_SYNC_ENABLED`, `ZIPNET_AUTOSTART_WORKER`, `ZIPNET_BASE_URL`,
`ZIPNET_SECTIONS`, `ZIPNET_DB`, `ZIPNET_PAGE_SIZE`, `ZIPNET_REQUEST_DELAY_SECONDS`,
`ZIPNET_REQUEST_TIMEOUT_SECONDS`, `ZIPNET_MAX_RETRIES`, `ZIPNET_BACKOFF_BASE_SECONDS`,
`ZIPNET_BACKOFF_MAX_SECONDS`, `ZIPNET_PAGES_PER_TURN`, `ZIPNET_BACKFILL_MAX_PAGES`,
`ZIPNET_INCREMENTAL_INTERVAL_MINUTES`, `ZIPNET_INCREMENTAL_MAX_PAGES`, `ZIPNET_INCREMENTAL_OVERLAP_DAYS`,
`ZIPNET_LOCK_TTL_SECONDS`, `ZIPNET_FAILURE_COOLDOWN_SECONDS`, `ZIPNET_BLOCKED_COOLDOWN_SECONDS`,
`ZIPNET_SHUTDOWN_GRACE_SECONDS`.

At the default 2 s request delay and 50 rows per request, the historical import of about 6.2 million
records takes roughly three days of continuous running. Progress is shown as "offset N of T"; nothing
is presented as complete before it is.

## Tests

`pytest tests/test_zipnet_*.py` uses synthetic records and mocked transports (no network, models or
GPU). One test spawns the real worker process against a local stub server and needs Scrapling
(`pip install -r requirements.txt`); it is skipped otherwise. `ZIPNET_LIVE_SMOKE=1 pytest
tests/test_zipnet_live_smoke.py` makes two real requests to ZIPNET and is never part of the normal run.

## Known limitations

- Offsets are relative to ZIPNET's newest-first order. During a multi-day backfill, new source records
  shift later pages down (harmless overlap), and a record deleted at the source could shift one page
  boundary; a later re-run of the backfill (`UPDATE zipnet_checkpoints SET backfill_complete=0, backfill_cursor=0`) reconciles it.
- ZIPNET offers no per-record page and no "changed since" filter. Status changes to records older than
  the overlap window are noticed only when a backfill revisits them.
- Section order is by `CreatedOn` (stolen vehicles: `FIRDate`), not by ID; a record entered late with an old date can be missed by the incremental sweep until the next backfill.
- The sync is not an official police system; statuses are as last published by ZIPNET.
