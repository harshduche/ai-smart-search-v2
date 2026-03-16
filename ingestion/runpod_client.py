"""RunPod Serverless client for submitting and tracking ingestion jobs.

Replaces RabbitMQ as the job queue.  Import this wherever your backend
previously published to the ``embedding.jobs`` queue.

Usage
─────
    from runpod_client import RunPodClient

    client = RunPodClient()

    # Submit a job (non-blocking)
    job_id = client.submit(job_payload)

    # Poll until done (blocking — use in background tasks or workers)
    result = client.wait(job_id, timeout=3600)

    # Or just check once
    status = client.status(job_id)

Environment variables
─────────────────────
    RUNPOD_API_KEY        RunPod API key (required)
    RUNPOD_ENDPOINT_ID    Serverless endpoint ID (required)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# RunPod terminal job statuses
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

# Default poll backoff: start at 15s, max 60s
_POLL_INITIAL   = int(os.getenv("RUNPOD_POLL_INITIAL",  15))
_POLL_MAX       = int(os.getenv("RUNPOD_POLL_MAX",      60))
_POLL_BACKOFF   = float(os.getenv("RUNPOD_POLL_BACKOFF", 1.5))


def _build_session() -> requests.Session:
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
    return session


class RunPodClient:
    """Thin wrapper around the RunPod Serverless REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        self.api_key     = api_key     or os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID", "")

        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY is not set")
        if not self.endpoint_id:
            raise RuntimeError("RUNPOD_ENDPOINT_ID is not set")

        self._base = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self._session = _build_session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    # ── Core API calls ────────────────────────────────────────────────────────

    def submit(self, job_payload: dict, webhook: Optional[str] = None) -> str:
        """Submit a job to the RunPod queue.

        Args:
            job_payload: The ``embedding.jobs`` payload dict (same schema as
                         the previous RabbitMQ message body).
            webhook:     Optional URL RunPod will POST the result to on
                         completion.  When provided, polling is not needed.

        Returns:
            RunPod job ID string (store this to poll status later).
        """
        body: dict[str, Any] = {"input": job_payload}
        if webhook:
            body["webhook"] = webhook

        resp = self._session.post(f"{self._base}/run", json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod response missing 'id': {data}")

        logger.info(
            "Submitted job %s → RunPod %s",
            job_payload.get("jobId", "unknown"),
            job_id,
        )
        return job_id

    def status(self, runpod_job_id: str) -> dict:
        """Return the current status dict for a RunPod job.

        Returns a dict with at least::

            {
                "id":     "...",
                "status": "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" |
                          "FAILED"   | "CANCELLED"   | "TIMED_OUT",
                "output": {...} | None,
            }
        """
        resp = self._session.get(f"{self._base}/status/{runpod_job_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, runpod_job_id: str) -> None:
        """Best-effort cancel a queued or running job."""
        try:
            self._session.post(f"{self._base}/cancel/{runpod_job_id}", timeout=10)
            logger.info("Cancelled RunPod job %s", runpod_job_id)
        except Exception as exc:
            logger.warning("Cancel request for %s failed (non-fatal): %s", runpod_job_id, exc)

    # ── Polling helpers ───────────────────────────────────────────────────────

    def wait(
        self,
        runpod_job_id: str,
        timeout: int = 3600,
        on_progress: Optional[callable] = None,
    ) -> dict:
        """Block until the job reaches a terminal state or timeout expires.

        Args:
            runpod_job_id: RunPod job ID returned by :meth:`submit`.
            timeout:       Maximum seconds to wait (default 1 hour).
            on_progress:   Optional callback called with the status dict on
                           each poll while the job is still running.

        Returns:
            Final status dict (same shape as :meth:`status`).

        Raises:
            TimeoutError:   If the job does not finish within ``timeout``.
            RuntimeError:   If RunPod reports FAILED with a permanent error.
        """
        deadline   = time.time() + timeout
        interval   = _POLL_INITIAL
        poll_count = 0

        while time.time() < deadline:
            data   = self.status(runpod_job_id)
            state  = data.get("status", "UNKNOWN")
            poll_count += 1

            logger.debug("Poll #%d  job=%s  status=%s", poll_count, runpod_job_id, state)

            if state in _TERMINAL:
                return data

            if on_progress:
                try:
                    on_progress(data)
                except Exception:
                    pass

            time.sleep(interval)
            interval = min(interval * _POLL_BACKOFF, _POLL_MAX)

        raise TimeoutError(
            f"RunPod job {runpod_job_id} did not finish within {timeout}s "
            f"(polled {poll_count} times)"
        )

    def submit_and_wait(
        self,
        job_payload: dict,
        timeout: int = 3600,
        on_progress: Optional[callable] = None,
    ) -> dict:
        """Submit a job and block until it completes.

        Convenience wrapper around :meth:`submit` + :meth:`wait`.
        Useful for scripts and tests; use :meth:`submit` + async polling
        in production APIs.
        """
        job_id = self.submit(job_payload)
        return self.wait(job_id, timeout=timeout, on_progress=on_progress)

    # ── Queue health ──────────────────────────────────────────────────────────

    def queue_health(self) -> dict:
        """Return queue depth and worker counts for the endpoint."""
        resp = self._session.get(f"{self._base}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

