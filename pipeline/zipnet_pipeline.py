"""One synchronization turn for one ZIPNET section.

discover -> fetch -> validate -> normalize (allow-list) -> deduplicate -> persist -> detect changes -> checkpoint

Every page is committed in ONE transaction together with its checkpoint, so an interruption at any
instant leaves either the whole page and its checkpoint, or neither. Nothing here is ever marked
complete while known pages remain.

Statuses of a turn:
  completed         backfill reached the end of the source / incremental caught up with known records
  incomplete        progress saved, more work remains (turn limit, page limit, empty page before the end)
  stopped           a stop/pause/shutdown request was honoured at a page boundary
  blocked           the source refused access (never retried or evaded)
  selector_changed  the page or JSON no longer has the expected structure
  failed            network or database failure after retries
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from services import zipnet_store as store
from services.zipnet_adapters import NormalizedRecord, RowRejected, SourceAdapter
from services.zipnet_scraper import (
    SchemaChanged,
    SourceBlocked,
    SourceClient,
    SourceError,
    SyncHalted,
)
from services.zipnet_settings import ZipnetSettings
from services.zipnet_time import IST, utcnow

log = logging.getLogger("locatenet.zipnet.pipeline")


@dataclass
class SectionResult:
    section: str
    mode: str
    job_id: int | None = None
    status: str = "running"
    stop_reason: str | None = None
    message: str | None = None
    pages: int = 0
    scanned: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    source_total: int | None = None
    warnings: list[str] = field(default_factory=list)


def validate_payload(adapter: SourceAdapter, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise SchemaChanged(f"{adapter.category}: response has no 'data' list")
    total = payload.get("recordsTotal")
    total_i = int(total) if isinstance(total, (int, str)) and str(total).isdigit() else None
    rows = [r for r in data if isinstance(r, dict)]
    if rows:
        missing = [adapter.missing_required_keys(r) for r in rows]
        broken = sum(1 for m in missing if m)
        if broken > len(rows) / 2:
            keys = sorted({k for m in missing for k in m})
            raise SchemaChanged(f"{adapter.category}: expected fields missing from source rows: {', '.join(keys)}")
    return rows, total_i


def run_section(
    conn: sqlite3.Connection,
    adapter: SourceAdapter,
    client: SourceClient,
    st: ZipnetSettings,
    mode: str,
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.time,
) -> SectionResult:
    """Run one bounded turn. Never raises for source problems; the outcome is in the result."""
    section = adapter.category
    backfill = mode == "backfill"
    res = SectionResult(section=section, mode=mode)
    with store.transaction(conn):
        cp = store.get_checkpoint(conn, section)
        res.job_id = store.start_job(conn, section, mode)
    req0, retry0 = client.stats.requests, client.stats.retries

    start = cp["backfill_cursor"] if backfill else cp["incr_cursor"]
    known_hw = cp["high_water_id"]
    running_max = cp["high_water_id"] if backfill else cp["incr_pending_hw"]
    cap = st.backfill_max_pages if backfill else 0
    pages_before = cp["backfill_pages"]
    max_pages = st.pages_per_turn if backfill else st.incremental_max_pages
    overlap_cutoff = (utcnow().astimezone(IST) - timedelta(days=st.incremental_overlap_days)).date()
    seen_signatures: set[tuple[str, ...]] = set()
    end_of_data = caught_up = False
    stop_reason = "turn_limit" if backfill else "max_pages"
    status: str | None = None
    message: str | None = None

    try:
        if cap and pages_before >= cap:
            stop_reason = "page_limit"
        else:
            # 1. Discovery: the listing page must still contain the grid this adapter expects.
            page = client.get_page(adapter.listing_url)
            if page.has_selector is not None:
                if not page.has_selector(f"table#{adapter.grid_id}"):
                    raise SchemaChanged(f"listing page no longer contains table#{adapter.grid_id}")
                if not page.has_selector(f'script[src*="{adapter.script_path}"]'):
                    res.warnings.append(f"listing page no longer references {adapter.script_path}")

            while res.pages < max_pages:
                if cap and pages_before + res.pages >= cap:
                    stop_reason = "page_limit"
                    break
                if should_stop():
                    raise SyncHalted("stop requested")
                payload = client.post_json(adapter.data_url, adapter.form_data(start, st.page_size),
                                           referer=adapter.listing_url)
                rows, total = validate_payload(adapter, payload)
                if total is not None:
                    res.source_total = total
                if not rows:
                    if total is None or start >= total:
                        end_of_data = True
                        break
                    stop_reason = "empty_page_before_end"
                    message = (f"the source returned an empty page at offset {start} although it reports {total} "
                               "records; section kept incomplete and will be retried")
                    break
                signature = tuple(str(r.get(adapter.id_field)) for r in rows)
                if signature in seen_signatures:
                    stop_reason = "pagination_loop"
                    message = f"pagination loop detected at offset {start}; stopped"
                    break
                seen_signatures.add(signature)

                records: list[NormalizedRecord] = []
                bad: list[tuple[str | None, str]] = []
                for row in rows:
                    try:
                        records.append(adapter.normalize(row))
                    except (RowRejected, ValueError, TypeError) as e:
                        sid = row.get(adapter.id_field)
                        bad.append((str(sid) if sid is not None else None, str(e)[:200]))

                ids = [r.source_id_num for r in records if r.source_id_num is not None]
                page_max = max(ids) if ids else None
                if page_max is not None:
                    running_max = page_max if running_max is None else max(running_max, page_max)
                next_start = start + len(rows)

                # 2. Persist this page and its checkpoint atomically.
                try:
                    with store.transaction(conn):
                        for sid, err in bad:
                            store.add_failure(conn, res.job_id, section, "normalize", err, sid, start)
                        n, u, unch = store.upsert_records(conn, section, adapter.listing_url, records, res.job_id)
                        fields: dict[str, Any] = {"indexed_count": cp["indexed_count"] + res.new + n,
                                                  "source_total": res.source_total if res.source_total is not None
                                                  else cp["source_total"]}
                        if backfill:
                            fields.update(backfill_cursor=next_start, backfill_pages=pages_before + res.pages + 1,
                                          high_water_id=running_max)
                        else:
                            fields.update(incr_cursor=next_start, incr_pending_hw=running_max, incr_in_progress=1)
                        store.update_checkpoint(conn, section, **fields)
                        store.update_job(conn, res.job_id, pages=res.pages + 1, scanned=res.scanned + len(rows),
                                         new_records=res.new + n, updated_records=res.updated + u,
                                         unchanged_records=res.unchanged + unch, failed_records=res.failed + len(bad),
                                         source_total=res.source_total)
                except sqlite3.Error as e:
                    raise _PersistError(f"{type(e).__name__}: {e}") from e
                res.pages += 1
                res.scanned += len(rows)
                res.new += n
                res.updated += u
                res.unchanged += unch
                res.failed += len(bad)
                start = next_start

                if total is not None and start >= total:
                    end_of_data = True
                    break
                if not backfill:
                    created = [r.source_created_date for r in records]
                    old = bool(created) and all(d is not None and d < overlap_cutoff for d in created)
                    if known_hw is not None and ids and max(ids) <= known_hw and old:
                        caught_up = True
                        break
    except SyncHalted:
        status, stop_reason, message = "stopped", "stop_requested", "stopped at a page boundary; progress saved"
    except SourceBlocked as e:
        status, stop_reason, message = "blocked", "blocked", str(e)
    except SchemaChanged as e:
        status, stop_reason, message = "selector_changed", "schema_changed", str(e)
    except SourceError as e:
        status, stop_reason, message = "failed", "source_error", str(e)
    except _PersistError as e:
        status, stop_reason, message = "failed", "persist_error", str(e)[:300]
        try:
            with store.transaction(conn):
                store.add_failure(conn, res.job_id, section, "persist", str(e), None, start)
        except sqlite3.Error:  # pragma: no cover - database unusable
            log.exception("could not record persist failure")
    except Exception as e:
        log.exception("section %s crashed", section)
        status, stop_reason, message = "failed", "internal_error", f"{type(e).__name__}"

    # 3. Final state, decided from facts rather than from how the loop happened to exit.
    now_s = clock()
    final: dict[str, Any] = {}
    if status is None:
        if end_of_data and backfill:
            status, stop_reason = "completed", "end_of_data"
            final.update(backfill_complete=1, backfill_completed_at=store.now_iso(), backfill_limit_reached=0)
        elif (end_of_data or caught_up) and not backfill:
            status, stop_reason = "completed", "end_of_data" if end_of_data else "caught_up"
            hw = max([v for v in (known_hw, running_max) if v is not None], default=None)
            final.update(high_water_id=hw, incr_cursor=0, incr_pending_hw=None, incr_in_progress=0, caught_up=1,
                         last_incremental_at=store.now_iso())
        else:
            status = "incomplete"
            if stop_reason == "page_limit":
                final["backfill_limit_reached"] = 1
                message = (f"backfill page limit ({cap} pages) reached before the end of the source; "
                           "raise ZIPNET_BACKFILL_MAX_PAGES to continue. NOT complete.")
            elif not backfill and stop_reason == "max_pages":
                final["caught_up"] = 0
                message = ("incremental sync has not yet reached previously indexed records; "
                           "it will continue from its checkpoint on the next turn")
                res.warnings.append(message)
    if backfill and status == "incomplete":
        final["backfill_limit_reached"] = 1 if stop_reason == "page_limit" else 0

    no_progress = status == "incomplete" and stop_reason in ("empty_page_before_end", "pagination_loop")
    if status in ("failed", "blocked", "selector_changed") or no_progress:
        fails = cp["consecutive_failures"] + 1
        base = st.blocked_cooldown if status == "blocked" else st.failure_cooldown
        final.update(consecutive_failures=fails, next_attempt_at=now_s + min(base * (2 ** (fails - 1)), 6 * 3600))
        if not backfill and status != "stopped":
            final["caught_up"] = 0
    elif status in ("completed", "incomplete"):
        final.update(consecutive_failures=0, next_attempt_at=None)
        if res.pages:
            final["last_success_at"] = store.now_iso()
    res.status, res.stop_reason, res.message = status, stop_reason, message
    with store.transaction(conn):
        store.update_checkpoint(conn, section, last_status=status, last_message=message, **final)
        store.update_job(conn, res.job_id, status=status, finished_at=store.now_iso(), stop_reason=stop_reason,
                         message=message, requests=client.stats.requests - req0, retries=client.stats.retries - retry0,
                         source_total=res.source_total, pages=res.pages, scanned=res.scanned, new_records=res.new,
                         updated_records=res.updated, unchanged_records=res.unchanged, failed_records=res.failed)
    log.info("zipnet %s/%s: %s (%s) pages=%d scanned=%d new=%d updated=%d unchanged=%d failed=%d",
             section, mode, status, stop_reason, res.pages, res.scanned, res.new, res.updated, res.unchanged,
             res.failed)
    return res


class _PersistError(Exception):
    pass
