"""Runtime settings of the ZIPNET sync, read from config at call time (tests construct their own)."""

from __future__ import annotations

from dataclasses import dataclass, field

import config

from .zipnet_store import SECTIONS


@dataclass
class ZipnetSettings:
    enabled: bool = False
    base_url: str = "https://zipnet.delhipolice.gov.in"
    sections: tuple[str, ...] = SECTIONS
    user_agent: str = "LocateNet-ZIPNET-Sync/1.0"
    page_size: int = 50
    request_delay: float = 2.0
    request_timeout: float = 30.0
    max_retries: int = 4
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    pages_per_turn: int = 20
    backfill_max_pages: int = 0
    incremental_interval_minutes: int = 60
    incremental_max_pages: int = 100
    incremental_overlap_days: int = 1
    lock_ttl: float = 120.0
    failure_cooldown: float = 60.0
    blocked_cooldown: float = 6 * 3600.0
    shutdown_grace: float = 20.0
    db_path: str | None = field(default=None)

    @classmethod
    def from_config(cls) -> ZipnetSettings:
        wanted = [s.strip() for s in config.ZIPNET_SECTIONS.split(",") if s.strip()]
        return cls(
            enabled=config.ZIPNET_SYNC_ENABLED,
            base_url=config.ZIPNET_BASE_URL,
            sections=tuple(s for s in SECTIONS if s in wanted),
            user_agent=config.ZIPNET_USER_AGENT,
            page_size=config.ZIPNET_PAGE_SIZE,
            request_delay=config.ZIPNET_REQUEST_DELAY_SECONDS,
            request_timeout=config.ZIPNET_REQUEST_TIMEOUT_SECONDS,
            max_retries=config.ZIPNET_MAX_RETRIES,
            backoff_base=config.ZIPNET_BACKOFF_BASE_SECONDS,
            backoff_max=config.ZIPNET_BACKOFF_MAX_SECONDS,
            pages_per_turn=config.ZIPNET_PAGES_PER_TURN,
            backfill_max_pages=config.ZIPNET_BACKFILL_MAX_PAGES,
            incremental_interval_minutes=config.ZIPNET_INCREMENTAL_INTERVAL_MINUTES,
            incremental_max_pages=config.ZIPNET_INCREMENTAL_MAX_PAGES,
            incremental_overlap_days=config.ZIPNET_INCREMENTAL_OVERLAP_DAYS,
            lock_ttl=float(config.ZIPNET_LOCK_TTL_SECONDS),
            failure_cooldown=float(config.ZIPNET_FAILURE_COOLDOWN_SECONDS),
            blocked_cooldown=float(config.ZIPNET_BLOCKED_COOLDOWN_SECONDS),
            shutdown_grace=float(config.ZIPNET_SHUTDOWN_GRACE_SECONDS),
            db_path=config.ZIPNET_DB_PATH,
        )
