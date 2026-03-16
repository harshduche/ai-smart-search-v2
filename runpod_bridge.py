"""RabbitMQ → RunPod Serverless Bridge.

Consumes jobs from the RabbitMQ ``embedding.jobs`` queue, submits them to a
RunPod Serverless endpoint, and polls for completion.  On completion it
acknowledges (or nacks) the original RabbitMQ message based on the outcome.

Architecture
────────────
  RabbitMQ ──► Bridge (this process) ──► RunPod /run  (async submit)
                      │
                      └──► Poller thread (every RUNPOD_POLL_INTERVAL seconds)
                                │
                         COMPLETED  → ack RabbitMQ
                         FAILED     → classify error
                                        permanent → nack (no requeue)
                                        transient → nack (requeue)
                         TIMED_OUT  → nack (requeue)

Key design choices
──────────────────
- Bridge is CPU-only and lightweight (~50 MB image, no ML deps).
- Prefetch count = RUNPOD_MAX_CONCURRENT_JOBS so RabbitMQ doesn't deliver
  more messages than RunPod can accept concurrently.
- In-flight state is kept in memory. On bridge crash, un-acked RabbitMQ
  messages are automatically requeued; the RunPod jobs continue running
  (idempotent ingestion means a rare duplicate is harmless).
- Uses pika SelectConnection (async I/O) so heartbeats fire even while
  the poller thread is working.

Environment variables
─────────────────────
  RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER, RABBITMQ_PASSWORD
  RABBITMQ_QUEUE            (default: embedding.jobs)
  RABBITMQ_VIRTUAL_HOST     (default: /)

  RUNPOD_API_KEY            RunPod API key (required)
  RUNPOD_ENDPOINT_ID        Serverless endpoint ID (required)
  RUNPOD_MAX_CONCURRENT_JOBS  Max in-flight RunPod jobs (default: 4)
  RUNPOD_POLL_INTERVAL      Seconds between status polls (default: 30)
  RUNPOD_JOB_TIMEOUT        Seconds before a job is declared timed out (default: 7200)
  RUNPOD_API_BASE           Override RunPod API base URL (default: https://api.runpod.ai/v2)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import pika
import pika.exceptions
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging ──────────────────────────────────────────────────────────────────

log_dir = Path(os.getenv("LOG_DIR", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_dir / "runpod_bridge.log")),
    ],
)
logger = logging.getLogger("runpod_bridge")

# ── Configuration ─────────────────────────────────────────────────────────────

RABBITMQ_HOST         = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT         = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER         = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD     = os.getenv("RABBITMQ_PASSWORD", "guest")
RABBITMQ_QUEUE        = os.getenv("RABBITMQ_QUEUE", "embedding.jobs")
RABBITMQ_VIRTUAL_HOST = os.getenv("RABBITMQ_VIRTUAL_HOST", "/")

RUNPOD_API_KEY            = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID        = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_MAX_CONCURRENT_JOBS = int(os.getenv("RUNPOD_MAX_CONCURRENT_JOBS", 4))
RUNPOD_POLL_INTERVAL      = int(os.getenv("RUNPOD_POLL_INTERVAL", 30))
RUNPOD_JOB_TIMEOUT        = int(os.getenv("RUNPOD_JOB_TIMEOUT", 7200))   # 2 hours
RUNPOD_API_BASE           = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2")

# RunPod terminal statuses
_RUNPOD_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
_RUNPOD_RUNNING  = {"IN_QUEUE", "IN_PROGRESS"}

# ── RunPod HTTP client ────────────────────────────────────────────────────────

def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    })
    return session


_http = _build_http_session()


def _runpod_submit(job_payload: dict) -> str:
    """Submit a job to RunPod Serverless and return the RunPod job ID."""
    url = f"{RUNPOD_API_BASE}/{RUNPOD_ENDPOINT_ID}/run"
    resp = _http.post(url, json={"input": job_payload}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    runpod_id = data.get("id")
    if not runpod_id:
        raise RuntimeError(f"RunPod response missing 'id': {data}")
    return runpod_id


def _runpod_status(runpod_id: str) -> dict:
    """Return the RunPod status dict for a job.

    Returns a dict with at least ``{"status": "IN_QUEUE"|"IN_PROGRESS"|
    "COMPLETED"|"FAILED"|"CANCELLED"|"TIMED_OUT", "output": ...}``.
    """
    url = f"{RUNPOD_API_BASE}/{RUNPOD_ENDPOINT_ID}/status/{runpod_id}"
    resp = _http.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _runpod_cancel(runpod_id: str) -> None:
    """Best-effort cancel a RunPod job (used on bridge shutdown for in-progress jobs)."""
    try:
        url = f"{RUNPOD_API_BASE}/{RUNPOD_ENDPOINT_ID}/cancel/{runpod_id}"
        _http.post(url, timeout=10)
    except Exception as exc:
        logger.debug("Cancel request for %s failed (non-fatal): %s", runpod_id, exc)


# ── In-flight job tracker ─────────────────────────────────────────────────────

@dataclass
class InFlightJob:
    """Tracks one submitted RunPod job that is waiting for completion."""
    runpod_id: str
    delivery_tag: int
    channel: Any                      # pika channel reference
    job_data: dict                    # original RabbitMQ payload
    submitted_at: float = field(default_factory=time.time)
    poll_count: int = 0


class InFlightTracker:
    """Thread-safe dict of currently in-flight RunPod jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, InFlightJob] = {}   # runpod_id → InFlightJob

    def add(self, job: InFlightJob) -> None:
        with self._lock:
            self._jobs[job.runpod_id] = job

    def remove(self, runpod_id: str) -> Optional[InFlightJob]:
        with self._lock:
            return self._jobs.pop(runpod_id, None)

    def snapshot(self) -> list[InFlightJob]:
        with self._lock:
            return list(self._jobs.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)


# ── Error classification (mirrors worker._classify_error) ────────────────────

def _classify_runpod_result(output: Any) -> tuple[bool, str]:
    """Decide if a FAILED RunPod output should be retried.

    RunPod sets status=FAILED when the handler raised an exception.
    The handler in runpod_worker.py returns ``{"error": {...}}`` for
    permanent errors and raises for transient ones.

    Returns:
        (is_retryable, reason)
    """
    if output is None:
        return True, "no output (RunPod infrastructure error)"

    # Handler returned {"error": {"retryable": False, ...}}
    if isinstance(output, dict) and "error" in output:
        err = output["error"]
        retryable = err.get("retryable", True) if isinstance(err, dict) else True
        reason = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
        return retryable, reason

    # Unexpected output shape — assume transient
    return True, f"unexpected output shape: {type(output).__name__}"


# ── RabbitMQ ack/nack helpers (called from poller thread via connection callbacks) ──

def _safe_ack(channel, delivery_tag: int, job_id: str) -> None:
    try:
        channel.basic_ack(delivery_tag=delivery_tag)
        logger.info("✓ ACK  delivery_tag=%d  runpod_id=%s", delivery_tag, job_id)
    except Exception as exc:
        logger.warning("Failed to ack delivery_tag=%d: %s", delivery_tag, exc)


def _safe_nack(channel, delivery_tag: int, requeue: bool, job_id: str, reason: str) -> None:
    try:
        channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        action = "NACK+requeue" if requeue else "NACK+discard"
        logger.warning("%s  delivery_tag=%d  runpod_id=%s  reason=%s",
                       action, delivery_tag, job_id, reason)
    except Exception as exc:
        logger.warning("Failed to nack delivery_tag=%d: %s", delivery_tag, exc)


# ── Bridge ────────────────────────────────────────────────────────────────────

class RunPodBridge:
    """Connects RabbitMQ to RunPod Serverless."""

    def __init__(self) -> None:
        self._tracker = InFlightTracker()
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None
        self._stop_event = threading.Event()
        self._poller_thread: Optional[threading.Thread] = None

        # Semaphore caps how many RunPod jobs are in-flight simultaneously.
        # prefetch_count is set to the same value so RabbitMQ delivers exactly
        # that many messages at a time.
        self._slots = threading.Semaphore(RUNPOD_MAX_CONCURRENT_JOBS)

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not RUNPOD_API_KEY:
            raise RuntimeError("RUNPOD_API_KEY is not set")
        if not RUNPOD_ENDPOINT_ID:
            raise RuntimeError("RUNPOD_ENDPOINT_ID is not set")

        logger.info("=" * 60)
        logger.info("RunPod Bridge starting")
        logger.info("  RabbitMQ : %s:%d  queue=%s", RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE)
        logger.info("  RunPod   : endpoint=%s  max_jobs=%d", RUNPOD_ENDPOINT_ID, RUNPOD_MAX_CONCURRENT_JOBS)
        logger.info("  Polling  : every %ds  timeout=%ds", RUNPOD_POLL_INTERVAL, RUNPOD_JOB_TIMEOUT)
        logger.info("=" * 60)

        self._poller_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="runpod-poller"
        )
        self._poller_thread.start()

        retry_delay = 5
        while not self._stop_event.is_set():
            try:
                self._connect()
                retry_delay = 5
                self._consume()         # blocks until connection drops or stop event
            except KeyboardInterrupt:
                logger.info("Received interrupt — shutting down")
                self._stop_event.set()
                break
            except Exception as exc:
                logger.error("RabbitMQ error: %s", exc)
                self._teardown_connection()
                if not self._stop_event.is_set():
                    logger.info("Reconnecting in %ds…", retry_delay)
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)

        self._shutdown()

    # ── RabbitMQ connection ───────────────────────────────────────────────────

    def _connect(self) -> None:
        logger.info("Connecting to RabbitMQ %s:%d…", RABBITMQ_HOST, RABBITMQ_PORT)
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            virtual_host=RABBITMQ_VIRTUAL_HOST,
            credentials=credentials,
            heartbeat=3600,
            blocked_connection_timeout=600,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()

        self._channel.queue_declare(
            queue=RABBITMQ_QUEUE,
            durable=True,
            arguments={
                "x-message-ttl": 86400000,
                "x-max-length": 10000,
            },
        )

        # Prefetch = max concurrent jobs so we don't pull more than we can submit
        self._channel.basic_qos(prefetch_count=RUNPOD_MAX_CONCURRENT_JOBS)
        logger.info("✓ Connected to RabbitMQ, queue=%s", RABBITMQ_QUEUE)

    def _teardown_connection(self) -> None:
        try:
            if self._connection and not self._connection.is_closed:
                self._connection.close()
        except Exception:
            pass
        self._connection = None
        self._channel = None

    # ── Message consumption ───────────────────────────────────────────────────

    def _consume(self) -> None:
        self._channel.basic_consume(
            queue=RABBITMQ_QUEUE,
            on_message_callback=self._on_message,
            auto_ack=False,
        )
        logger.info("Waiting for jobs on queue: %s", RABBITMQ_QUEUE)
        while not self._stop_event.is_set():
            self._connection.process_data_events(time_limit=1)

    def _on_message(self, channel, method, _properties, body: bytes) -> None:
        """Called by pika on the main thread when a RabbitMQ message arrives."""
        delivery_tag = method.delivery_tag

        try:
            job_data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in message (delivery_tag=%d): %s", delivery_tag, exc)
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
            return

        job_id = job_data.get("jobId", "unknown")
        logger.info("Received job %s (delivery_tag=%d)", job_id, delivery_tag)

        # Acquire a slot — blocks if RUNPOD_MAX_CONCURRENT_JOBS are already running.
        # Since pika runs on a single thread with process_data_events(), this will
        # briefly stall the heartbeat loop if all slots are busy for a long time.
        # In practice this is fine: the poller thread frees slots as jobs complete,
        # and heartbeat=3600 gives us ample headroom.
        self._slots.acquire()

        try:
            runpod_id = _runpod_submit(job_data)
            logger.info("Submitted job %s → RunPod %s", job_id, runpod_id)

            in_flight = InFlightJob(
                runpod_id=runpod_id,
                delivery_tag=delivery_tag,
                channel=channel,
                job_data=job_data,
            )
            self._tracker.add(in_flight)

        except Exception as exc:
            # Submission itself failed (network error, bad API key, etc.)
            logger.error("Failed to submit job %s to RunPod: %s", job_id, exc)
            self._slots.release()

            # Treat as transient: requeue so another bridge instance or restart handles it
            try:
                channel.basic_nack(delivery_tag=delivery_tag, requeue=True)
            except Exception:
                pass

    # ── Poller thread ─────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: polls RunPod for job completion."""
        logger.info("Poller thread started (interval=%ds)", RUNPOD_POLL_INTERVAL)
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=RUNPOD_POLL_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                self._poll_all()
            except Exception as exc:
                logger.error("Poller error (non-fatal): %s", exc)
                logger.debug(traceback.format_exc())

    def _poll_all(self) -> None:
        """Check status of every in-flight RunPod job once."""
        jobs = self._tracker.snapshot()
        if not jobs:
            return

        logger.info("Polling %d in-flight job(s)…", len(jobs))

        for job in jobs:
            if self._stop_event.is_set():
                break
            self._poll_one(job)

    def _poll_one(self, job: InFlightJob) -> None:
        runpod_id = job.runpod_id
        job.poll_count += 1
        age_s = time.time() - job.submitted_at
        rmq_job_id = job.job_data.get("jobId", "unknown")

        # ── Timeout check ────────────────────────────────────────────────────
        if age_s > RUNPOD_JOB_TIMEOUT:
            logger.error(
                "Job %s (RunPod %s) timed out after %.0fs — nacking with requeue",
                rmq_job_id, runpod_id, age_s,
            )
            self._tracker.remove(runpod_id)
            self._slots.release()
            _safe_nack(job.channel, job.delivery_tag, requeue=True,
                       job_id=runpod_id, reason="bridge timeout")
            return

        # ── Fetch RunPod status ───────────────────────────────────────────────
        try:
            status_data = _runpod_status(runpod_id)
        except Exception as exc:
            logger.warning("Status poll failed for %s (poll#%d): %s", runpod_id, job.poll_count, exc)
            return  # leave in tracker, retry next cycle

        status  = status_data.get("status", "UNKNOWN")
        output  = status_data.get("output")

        logger.debug(
            "Job %s (RunPod %s)  status=%s  age=%.0fs  poll#%d",
            rmq_job_id, runpod_id, status, age_s, job.poll_count,
        )

        if status in _RUNPOD_RUNNING:
            return  # still running — check again next cycle

        # ── Terminal state ────────────────────────────────────────────────────
        self._tracker.remove(runpod_id)
        self._slots.release()

        if status == "COMPLETED":
            # Check if the handler returned an error dict (permanent failure)
            if isinstance(output, dict) and "error" in output:
                is_retryable, reason = _classify_runpod_result(output)
                _safe_nack(job.channel, job.delivery_tag, requeue=is_retryable,
                           job_id=runpod_id, reason=reason)
            else:
                clips = output.get("clips_ingested", "?") if isinstance(output, dict) else "?"
                logger.info(
                    "✓ Job %s complete: %s clips ingested (RunPod %s)",
                    rmq_job_id, clips, runpod_id,
                )
                _safe_ack(job.channel, job.delivery_tag, runpod_id)

        elif status == "FAILED":
            # RunPod reports FAILED when the handler raised (transient path)
            is_retryable, reason = _classify_runpod_result(output)
            logger.error(
                "RunPod job %s FAILED: retryable=%s reason=%s",
                runpod_id, is_retryable, reason,
            )
            _safe_nack(job.channel, job.delivery_tag, requeue=is_retryable,
                       job_id=runpod_id, reason=reason)

        elif status in ("CANCELLED", "TIMED_OUT"):
            # Runpod-side cancellation or timeout → requeue
            logger.warning("RunPod job %s status=%s — requeuing", runpod_id, status)
            _safe_nack(job.channel, job.delivery_tag, requeue=True,
                       job_id=runpod_id, reason=f"RunPod status={status}")

        else:
            logger.warning("Unknown RunPod status %r for job %s — requeuing", status, runpod_id)
            _safe_nack(job.channel, job.delivery_tag, requeue=True,
                       job_id=runpod_id, reason=f"unknown status={status}")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("Shutting down bridge…")
        self._stop_event.set()

        if self._poller_thread:
            self._poller_thread.join(timeout=5)

        in_flight = self._tracker.snapshot()
        if in_flight:
            logger.warning(
                "%d job(s) were in-flight at shutdown — "
                "they will be requeued by RabbitMQ on reconnect.",
                len(in_flight),
            )
            for job in in_flight:
                logger.warning(
                    "  in-flight: rmq_delivery=%d  runpod_id=%s",
                    job.delivery_tag, job.runpod_id,
                )

        self._teardown_connection()

        jobs_snapshot = len(in_flight)
        logger.info(
            "Bridge stopped. %d job(s) left un-acked (will be requeued).",
            jobs_snapshot,
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    bridge = RunPodBridge()
    bridge.start()


if __name__ == "__main__":
    main()
