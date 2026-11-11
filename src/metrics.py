"""Run metadata: run id, stage timing, failure stage."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

log = logging.getLogger("chronos")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RunRecorder:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.started_at = utc_now_iso()
        self.ended_at = None
        self.status = "running"
        self.failed_stage = None
        self.error_message = None
        self.stage_seconds: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self.duration_seconds = 0.0

    @contextmanager
    def stage(self, name: str):
        log.info("[%s] stage %s: start", self.run_id, name)
        t = time.perf_counter()
        try:
            yield
        except Exception as exc:
            if self.failed_stage is None:
                self.failed_stage = name
                self.error_message = f"{type(exc).__name__}: {exc}"
            log.error("[%s] stage %s: FAILED %s", self.run_id, name, self.error_message)
            raise
        finally:
            self.stage_seconds[name] = round(time.perf_counter() - t, 4)
        log.info("[%s] stage %s: done in %.3fs", self.run_id, name, self.stage_seconds[name])

    def finish(self, status: str):
        self.status = status
        self.ended_at = utc_now_iso()
        self.duration_seconds = round(time.perf_counter() - self._t0, 4)
