"""RunPod Serverless handler for video embedding jobs.

Wraps the existing EmbeddingWorker.process_job() logic in a RunPod-compatible
handler function.  The embedding model and all services are initialised once at
module level (cold start) and reused across every job dispatched to this worker.

Usage — local testing:
    python runpod_worker.py          # reads test_input.json automatically

Usage — RunPod Serverless:
    Set as the Docker CMD in Dockerfile.runpod.
"""

import os
import sys
import logging
import time
from pathlib import Path

import runpod

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = Path(os.getenv("LOG_DIR", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "runpod_worker.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file)),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time service initialisation (survives across jobs on a warm worker)
# ---------------------------------------------------------------------------
import config
from ingestion.ingest_pipeline import IngestPipeline
from ingestion.embedding_factory import get_embedding_service
from ingestion.s3_service import get_s3_service
from ingestion.mongodb_service import get_mongodb_service
from search.vector_store import get_vector_store
from observability.langfuse_integration import flush_langfuse
from worker import EmbeddingWorker

logger.info("=== RunPod worker cold start — initialising services ===")
_init_start = time.time()

logger.info("Loading embedding service...")
_embedding_service = get_embedding_service()
if hasattr(_embedding_service, "initialize") and hasattr(_embedding_service, "_initialized"):
    if not _embedding_service._initialized:
        _embedding_service.initialize()
logger.info("Embedding service ready.")

logger.info("Connecting to vector store...")
_vector_store = get_vector_store()
logger.info("Vector store connected.")

logger.info("Building ingestion pipeline...")
_pipeline = IngestPipeline(
    embedding_service=_embedding_service,
    vector_store=_vector_store,
)
logger.info("Ingestion pipeline ready.")

_mongodb_service = get_mongodb_service()
if _mongodb_service:
    logger.info("MongoDB connected.")
else:
    logger.info("MongoDB disabled (MONGODB_URI not set).")

logger.info(
    "=== Cold start complete in %.1fs ===",
    time.time() - _init_start,
)


def _build_worker_facade() -> EmbeddingWorker:
    """Build a thin EmbeddingWorker instance with services pre-wired.

    We bypass __init__ (which sets up RabbitMQ params we don't need) and
    directly assign the services that process_job() depends on.
    """
    w = object.__new__(EmbeddingWorker)
    w.ingest_pipeline = _pipeline
    w.embedding_service = _embedding_service
    w.vector_store = _vector_store
    w.mongodb_service = _mongodb_service
    w.jobs_processed = 0
    w.jobs_failed = 0
    w.start_time = time.time()
    return w


_worker_facade = _build_worker_facade()


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """Process a single embedding job dispatched by RunPod.

    ``event["input"]`` must contain the same JSON schema as a RabbitMQ
    ``embedding.jobs`` message (see EmbeddingWorker._parse_job for schema).

    Error handling strategy:
    - **Permanent errors** (S3 403/404, bad credentials, invalid input):
      Return ``{"error": ...}`` so RunPod marks the job as failed without retry.
    - **Transient errors** (network timeout, service unavailable):
      Raise an exception so RunPod retries the job automatically.
    """
    job_data = event["input"]

    job_id = job_data.get("jobId", "unknown")
    logger.info("=" * 60)
    logger.info("Received job: %s", job_id)
    logger.info("=" * 60)

    try:
        result = _worker_facade.process_job(job_data)

        flush_langfuse()

        logger.info("Job %s completed successfully", job_id)
        return result

    except Exception as exc:
        is_retryable, reason = EmbeddingWorker._classify_error(exc)

        if is_retryable:
            logger.warning(
                "Transient error on job %s (%s) — raising for RunPod retry",
                job_id,
                reason,
            )
            flush_langfuse()
            raise

        logger.error(
            "Permanent error on job %s (%s) — returning error (no retry)",
            job_id,
            reason,
        )
        flush_langfuse()
        return {
            "error": {
                "message": str(exc),
                "reason": reason,
                "retryable": False,
                "job_id": job_id,
            }
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
