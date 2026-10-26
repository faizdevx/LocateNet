"""Entry point of the ZIPNET sync worker.

    python zipnet_worker.py              # run until stopped (also started automatically by `python app.py`)
    python zipnet_worker.py --once       # run turns until nothing is due, then exit
    python zipnet_worker.py --healthcheck  # exit 0 when a live worker holds the lease (Docker healthcheck)
    python zipnet_worker.py --status     # print per-section progress

Exit codes: 0 clean, 3 another worker already holds the lease, 1 error.
"""

import argparse
import json
import logging
import os
import signal
import sys

import config
from services import zipnet_store as store
from services.zipnet_settings import ZipnetSettings
from services.zipnet_supervisor import worker_alive
from services.zipnet_sync_service import EXIT_DUPLICATE, EXIT_OK, ZipnetSyncService


def _setup_logging():
    os.makedirs(os.path.join(config.BASE_DIR, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("scrapling").setLevel(logging.WARNING)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--healthcheck", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--parent-pid", type=int, default=None, help="exit when this process disappears")
    ap.add_argument("--exit-token", default=None, help="exit when the exit_request control flag equals this token")
    args = ap.parse_args(argv)
    st = ZipnetSettings.from_config()

    if args.healthcheck:
        return 0 if (worker_alive(st) or not st.enabled) else 1  # a disabled worker idles on purpose
    if args.status:
        conn = store.connect(st.db_path)
        store.migrate(conn)
        for s in st.sections:
            cp = store.get_checkpoint(conn, s)
            print(json.dumps({"section": s, **{k: cp[k] for k in (
                "backfill_cursor", "backfill_complete", "source_total", "indexed_count", "last_status",
                "last_incremental_at")}}))
        return 0
    _setup_logging()
    svc = ZipnetSyncService(st, parent_pid=args.parent_pid, exit_token=args.exit_token)

    def _graceful(signum, _frame):
        logging.getLogger("locatenet.zipnet.worker").info("signal %s received; finishing the current page", signum)
        svc.stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _graceful)
    if not st.enabled:
        msg = "ZIPNET_SYNC_ENABLED is off: this worker makes no requests. Set ZIPNET_SYNC_ENABLED=1 and restart it."
        print(msg)
        if args.once:
            return EXIT_OK
        # Stay up (idle) instead of exiting, so a process manager with restart=always does not crash-loop.
        svc.stop_event.wait()
        return EXIT_OK
    if args.once:
        conn = svc.connect()
        store.migrate(conn)
        if not store.acquire_lease(conn, svc.holder, st.lock_ttl):
            return EXIT_DUPLICATE
        try:
            while not svc.should_exit(conn) and svc.step(conn) is not None:
                pass
        finally:
            store.release_lease(conn, svc.holder)
            conn.close()
        return EXIT_OK
    return svc.run()


if __name__ == "__main__":
    sys.exit(main())
