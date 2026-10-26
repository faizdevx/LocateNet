"""Opt-in live connectivity check. Skipped unless ZIPNET_LIVE_SMOKE=1 (never part of the normal suite).

    ZIPNET_LIVE_SMOKE=1 pytest tests/test_zipnet_live_smoke.py

Makes two requests to the real ZIPNET portal (one listing page, one 10-row data page) in a throwaway database.
"""

import os

import pytest

from services import zipnet_store as store
from services.zipnet_adapters import get_adapter
from services.zipnet_scraper import ScraplingTransport, SourceClient
from pipeline.zipnet_pipeline import run_section
from tests.zipnet_fakes import make_settings

pytestmark = pytest.mark.skipif(os.environ.get("ZIPNET_LIVE_SMOKE") != "1", reason="set ZIPNET_LIVE_SMOKE=1 to run")


def test_one_live_page_of_missing_persons(tmp_path):
    pytest.importorskip("scrapling")
    path = str(tmp_path / "live.db")
    st = make_settings(path, sections=("missing_persons",), pages_per_turn=1, request_delay=1.0)
    st.base_url = "https://zipnet.delhipolice.gov.in"
    conn = store.connect(path)
    store.migrate(conn)
    client = SourceClient(ScraplingTransport(30, "LocateNet-ZIPNET-Sync/1.0 (smoke test)"), delay_seconds=1.0,
                          max_retries=2, backoff_base=1, backoff_max=5)
    try:
        r = run_section(conn, get_adapter("missing_persons", st.base_url), client, st, "backfill")
    finally:
        client.close()
    assert r.status == "incomplete" and r.stop_reason == "turn_limit" and r.new >= 1
    assert r.source_total and r.source_total > 1000
