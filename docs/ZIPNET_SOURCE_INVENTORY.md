# ZIPNET source inventory (as verified on 2026-10-26)

Inspected live on **2026-10-26** (IST) against `https://zipnet.delhipolice.gov.in/`.
Everything below was observed directly (page HTML, the page's own JavaScript, and
small sample requests of 2–6 rows). Nothing here is assumed from documentation.

## Portal-wide facts

| Item | Finding |
|---|---|
| Framework | ASP.NET MVC, jQuery 3.7.1, DataTables (server-side processing mode) |
| `robots.txt` | Not present (redirects to `/Error/NotFound`). No crawl rules are published. We still apply conservative limits. |
| Authentication | Listing grids are public. `/Login/Login` exists for police users; add/edit/delete endpoints (`AddMissingPerson`, `UpdateMissingPersonRecord`, `Set*DeletedStatus`, `/Home/DeleteImages`) are for logged-in staff. **We never call them.** |
| Record loading | Each listing page renders an empty `<table>` and its script (`/Scripts/app/<Section>.js`) POSTs to a section endpoint. The response is JSON in DataTables format: `{draw, recordsTotal, recordsFiltered, data[]}`. |
| Pagination | DataTables `start` / `length`. Page sizes the UI offers: 10–50. A 100-row request returned 100 rows in ~2.8 s. We default to 50. |
| Ordering | Requesting `order[0][column]=1, dir=desc` returns roughly newest-first by internal ID. The first few rows can be out of order (recently edited rows appear first). |
| Filter string | DataTables `search[value]` carries a pipe-delimited filter: `State|District|PoliceStation|FromDDDate|ToDDDate|FromCreatedDate|ToCreatedDate` (Missing Persons prepends `Status|`). Dates are `yyyy-mm-dd`. |
| Date formats in responses | **Inconsistent.** Seen: `dd/mm/yyyy`, `dd/mm/yyyy HH:MM:SS`, and `M/D/YYYY h:mm:ss AM` (US order, only when a filter is applied on some sections). The normalizer treats values with AM/PM as month-first and all others as day-first; unparseable values are kept raw and flagged. |
| Images | `ImageUrls[]` holds same-host relative paths such as `/ImageUploads/2026/MissingUploads/<uuid>.jpg`. A sample GET returned `200 image/jpeg` (~44 KB) with no auth. Placeholder used by the portal: `/Public/images/image_not_available.jpg`. |
| Update indicators | `CreatedOn` (date the record was entered on ZIPNET), `UpdatedOn` (often null). No ETag/Last-Modified on JSON responses. No "changed since" filter exists, so changes are detected by re-checking a rolling window and comparing fingerprints. |
| Record detail URLs | **None.** The portal has no per-record public page; details expand inline in the grid. Every stored record links to its section listing URL plus its source ID, and the UI says so. |

## Sensitive fields in the JSON (dropped at ingestion)

The public JSON returns more than the grid displays, including complainant phone
numbers, home addresses, guardian/parent names, police officer mobile numbers,
email addresses and religion. Adapters in `services/zipnet_adapters.py` use a **deny-by-default allow-list**: only fields listed
in the per-section tables below are read. Everything else is discarded before
normalization and is never written to the database, logs or test fixtures.

## Sections

Endpoint base: `https://zipnet.delhipolice.gov.in`. Totals are `recordsTotal` at inspection time.

### 1. Missing Persons
- Listing: `/Victims/MissingPersons` · grid `#missingPersonGrid`
- Data: `POST /Victims/GetMissingPersonsData/` · ID field `MissingPersonId` · total **669,266**
- Search fields (UI): status (Untraced/Traced), state, district, police station, FIR/DD/GD date range, "Report Date on ZIPNET" range; advanced modal: missing date range, name, guardian name, age range, height, religion, gender, has-image.
- Displayed columns: photo, state, district, police station, FIR no., DD date, serial no., missing-from place, tracing status, report date.
- Kept: ID, state, district, police station, FIR no., DD no., DD date, serial no., missing-from place, reporting date, missing-since date, name, sex, birth year, height, build, complexion, hair, eyes, upper/lower dress and colours, tracing status, tracing date, created/updated dates, image URLs.
- Dropped: address, guardian name/occupation, complainant phone, "report inform" (contains officer mobile), remarks (officer names), police post, religion, community, mother/spouse/sibling names, house/road/village/PIN, email.
- Status filter verified: `Untraced` → 367,630 rows; created-date filter verified (25–26 Oct → 88 rows).

### 2. Un-Identified Dead Bodies
- Listing: `/Victims/UnIdentifiedDeadBodies` · grid `#unidentifiedDeadBodiesGrid`
- Data: `POST /Victims/GetUnIdentifiedDeadBodiesData/` · ID `UnIdentifiedDeadBodyId` · total **122,007**
- Displayed: photo, state, district, police station, DD date, UIDB serial no., found date/time, report date.
- Kept: ID, state, district, PS, FIR no., DD no., DD date, UIDB serial, found date/time, found place, sex, age from/to, height, build, complexion, hair, dress, matched flag/date, created/updated, image URLs.
- Dropped: name, parentage, address, post-mortem result, kept place, remarks, religion.
- Created-date filter verified (20–26 Oct → 75 rows).

### 3. Un-Identified Persons Found
- Listing: `/Victims/UnIdentifiedPersons` · grid `#unidentifiedPersonsGrid`
- Data: `POST /Victims/GetUnIdentifiedPersonsData/` · ID `UnIdentifiedPersonId` · total **14,520**
- Displayed: photo, state, district, PS, DD no., DD date, UIPF serial, found date/time, report date.
- Kept: ID, state, district, PS, DD no., DD date, UIPF serial, found status, found date/time, found place, name (as published), sex, age from/to, height, build, complexion, hair, dress, matched flag/date, created/updated, image URLs.
- Dropped: address, parentage, kept-place address (shelter location of a vulnerable person), remarks, religion.
- **Uncertainty:** with a date filter applied, this endpoint returns US-format dates and only 3 rows for a 26-day window although unfiltered paging shows more. Incremental sync therefore does not depend on the date filter.

### 4. Stolen Vehicles (portal label: "Missing Vehicles")
- Listing: `/VehiclesMobiles/MissingVehicles` · grid `#missingVehiclesGrid`
- Data: `POST /VehiclesMobiles/GetMissingVehiclesData/` · ID `MissingVehiclesId` · total **1,572,773**
- Displayed: state, district, PS, FIR no., FIR date, registration no., vehicle type, status. Free-text search covers registration, engine and chassis numbers.
- Kept: ID, state, district, PS, FIR no., FIR date, report date-time, registration no. (raw + normalized), type, make, colour, model, status, stolen date, stolen-from place, year, created/updated.
- Dropped: complainant, complainant phone, address, control-room no., engine no., chassis no.
- No images.

### 5. Un-claimed / Seized Vehicles
- Listing: `/VehiclesMobiles/UnclaimedSeizedVehicles` · grid `#unclaimedSeizedVehiclesGrid`
- Data: `POST /VehiclesMobiles/GetUnclaimedSeizedVehiclesData/` · ID `UnclaimedSeizedVehiclesId` · total **1,216,247**
- Displayed: state, district, PS, DD no., DD date, registration no., model, status, report date.
- Kept: ID, state, district, PS, DD no., DD date, reporting date, registration no., make, type, colour, model, status, year, created/updated.
- Dropped: engine no., chassis no.
- **Uncertainty:** the created-date filter returned `recordsFiltered` 1,550,176, larger than the total — the filter is not reliable. Not used.

### 6. Missing Mobiles
- Listing: `/VehiclesMobiles/MissingMobiles` · grid `#missingMobilesGrid`
- Data: `POST /VehiclesMobiles/GetMissingMobilesData/` · ID `MissingMobileId` · total **2,626,873**
- Displayed: state, district, PS, FIR no., FIR date, IMEI, other IMEI, make, status, report date.
- Status values seen: `Theft/Stolen`, `Stolen/Theft`, `RECOVERED`, `Found/Recovered/Seized`.
- Kept: ID, state, district, PS, FIR no., FIR date, reporting date, stolen date, IMEI, other IMEI, make, type, status, created/updated.
- Dropped: complainant, complainant contact no., **the phone's own mobile numbers**, SHO mobile no., recovered-from.
- **Uncertainty:** every date filter tried returned 0 rows. Not used.

### Other portal pages (no records)
- `/Preamble/ZIPNETProject` — project description.
- `/Preamble/MemberStates` — project coordinator details (officer contact info; **not ingested**).
- Member-state portals linked from the home page (Chandigarh, HP, Haryana, Punjab, Rajasthan, UP, Uttarakhand) are separate sites and out of scope.

## Extraction strategy (all sections)

1. **Discovery** — GET the listing page with Scrapling `FetcherSession`; assert the expected grid `<table id>` and section script exist. If not, the adapter reports `selector_changed` and stops.
2. **Fetch** — POST the section endpoint with the same DataTables parameters the page sends. No browser is needed: the data is plain JSON, so only Scrapling's `FetcherSession` is used.
3. **Honest client** — `impersonate=None`, `stealthy_headers=False`, an identifying `User-Agent`. No proxies, no fingerprint spoofing, no CAPTCHA handling. A 403/429 or block page stops the adapter and surfaces `blocked`.
4. **Incremental sweep** — page newest-first; stop when a page lies entirely at or below the stored high-water source ID **and** older than the `CreatedOn` overlap window (default 1 day). A sweep that hits its page limit first keeps its offset and continues next turn, and never advances the high-water mark until it has caught up.
5. **Backfill** — the same pages walked to the end in bounded turns with a per-section offset checkpoint. A full backfill of ~6.2 M rows needs ~124,000 requests at 50 rows each; at the default 2 s delay that is roughly three days of continuous running. Offsets are relative to a newest-first ordering, so records added during a long backfill shift later pages down: the resulting overlap is absorbed by idempotent upserts, and the newest records are picked up by the incremental sweep that follows.

## Known limitations

- No per-record detail URL; links point to the section listing.
- Status changes on records older than the re-check window are only seen if `UpdatedOn` rows surface at the top of the newest-first ordering (observed for some edited rows) or during a backfill.
- Geography is published as names only (state / district / police station). No coordinates are published. The map uses generalized centroids from a curated gazetteer and labels the precision.
- Hindi localisation (`/Home/ChangeCurrentCulture/1`) was not ingested; English values only.
