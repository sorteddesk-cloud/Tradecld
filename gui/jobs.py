"""Background jobs for the Trading tab: a paper-trading session or a backtest.

Both take minutes to hours, so they run on a worker thread and report through a
log the page polls. One job runs at a time, and never alongside an analysis:
on a laptop running a local model, two graphs at once only slow both down.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from .pause import GATE

logger = logging.getLogger(__name__)

JobFn = Callable[[Callable[[str], None], Callable[[], bool]], Any]


class Job:
    def __init__(self, kind: str, title: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.status = "running"
        self.log: list[str] = []
        self.result: Any = None
        self.error: str | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None
        self._stop = threading.Event()

    def write(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"{stamp}  {line}")
        del self.log[:-500]  # a long backtest must not grow the log without bound

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def snapshot(self, since: int = 0) -> dict:
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
            "log": self.log[since:], "log_size": len(self.log), "result": self.result,
            "error": self.error, "started_at": self.started_at, "ended_at": self.ended_at,
            "paused": self.status == "running" and GATE.paused,
        }


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.current: Job | None = None

    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def start(self, kind: str, title: str, fn: JobFn) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError(f"{self.current.title} is still running")
            job = self.current = Job(kind, title)
            GATE.resume()
        threading.Thread(target=self._run, args=(job, fn), name=f"job-{job.id}",
                         daemon=True).start()
        return job

    def stop(self) -> bool:
        job = self.current
        if job is None or job.status != "running":
            return False
        job._stop.set()
        job.write("Stop requested: finishing the current step first.")
        return True

    @staticmethod
    def _run(job: Job, fn: JobFn) -> None:
        try:
            job.result = fn(job.write, job.stop_requested)
            job.status = "stopped" if job.stop_requested() else "completed"
            job.write("Stopped." if job.status == "stopped" else "Done.")
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.write(f"Failed: {job.error}")
            logger.debug(traceback.format_exc())
        finally:
            GATE.resume()
            job.ended_at = time.time()
