"""RabbitMQ worker for processing video embedding jobs.

This worker:
1. Loads the embedding model on startup
2. Consumes messages from RabbitMQ queue "embedding.jobs"
3. Downloads videos from URLs or processes local paths
4. Generates embeddings and stores them in Qdrant
5. Acknowledges processed jobs
"""

import os
import sys
import json
import time
import logging
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pika
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker, StreamLostError
from botocore.exceptions import ClientError

import config
from ingestion.ingest_pipeline import IngestPipeline
from ingestion.embedding_factory import get_embedding_service
from ingestion.s3_service import get_s3_service
from ingestion.mongodb_service import get_mongodb_service
from ingestion.geo_coverage import (
    build_flight_path,
    build_coverage_polygon,
    build_bounds,
    build_frames,
)
from search.vector_store import get_vector_store
from observability.langfuse_integration import trace_operation, flush_langfuse

# Configure logging
log_dir = Path(os.getenv('LOG_DIR', 'logs'))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'worker.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Worker that processes video embedding jobs from RabbitMQ."""

    def __init__(
        self,
        rabbitmq_host: str = "localhost",
        rabbitmq_port: int = 5672,
        rabbitmq_user: str = "guest",
        rabbitmq_password: str = "guest",
        queue_name: str = "embedding.jobs",
        prefetch_count: int = 1,
    ):
        """
        Initialize the embedding worker.

        Args:
            rabbitmq_host: RabbitMQ server host
            rabbitmq_port: RabbitMQ server port
            rabbitmq_user: RabbitMQ username
            rabbitmq_password: RabbitMQ password
            queue_name: Queue name to consume from
            prefetch_count: Number of messages to prefetch (QoS)
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.queue_name = queue_name
        self.prefetch_count = prefetch_count

        self.connection = None
        self.channel = None
        self.ingest_pipeline = None
        self.embedding_service = None
        self.vector_store = None
        self.mongodb_service = None

        # Statistics
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.start_time = time.time()

    def initialize_services(self):
        """Initialize ML models and services."""
        logger.info("Initializing embedding service...")
        self.embedding_service = get_embedding_service()

        # Initialize if using local embedding service
        if hasattr(self.embedding_service, 'initialize'):
            if hasattr(self.embedding_service, '_initialized') and not self.embedding_service._initialized:
                self.embedding_service.initialize()

        logger.info("✓ Embedding service ready")

        logger.info("Initializing vector store...")
        self.vector_store = get_vector_store()
        logger.info("✓ Vector store connected")

        logger.info("Initializing ingestion pipeline...")
        self.ingest_pipeline = IngestPipeline(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )
        logger.info("✓ Ingestion pipeline ready")

        self.mongodb_service = get_mongodb_service()
        if self.mongodb_service:
            logger.info("✓ MongoDB connected")
        else:
            logger.info("MongoDB disabled (MONGODB_URI not set)")

    def connect_to_rabbitmq(self):
        """Establish connection to RabbitMQ server."""
        logger.info(f"Connecting to RabbitMQ at {self.rabbitmq_host}:{self.rabbitmq_port}...")

        credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
            virtual_host="rlgzouhm",
            credentials=credentials,
            heartbeat=3600,
            blocked_connection_timeout=600,
        )

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare queue (idempotent - safe to call multiple times)
        self.channel.queue_declare(
            queue=self.queue_name,
            durable=True,  # Queue survives broker restart
            arguments={
                'x-message-ttl': 86400000,  # 24 hours message TTL
                'x-max-length': 10000,  # Max queue length
            }
        )

        # Set QoS (prefetch only 1 message at a time for fair dispatch)
        self.channel.basic_qos(prefetch_count=self.prefetch_count)

        logger.info(f"✓ Connected to RabbitMQ queue: {self.queue_name}")

    def _parse_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate an ``embedding.jobs`` RMQ message.

        Expected schema::

            {
              "jobId": "uuid",
              "organizationId": "org_uuid",
              "siteId": "site_uuid",
              "media": {
                "mediaId": "media_uuid",
                "flightId": "flight_uuid",
                "flightType": "mission",
                "missionId": "mission_uuid",
                "missionType": "inspection",
                "fileName": "video_123.mp4",
                "fileType": "video/mp4",
                "fileSizeBytes": 524288000,
                "captureTimestamp": "ISO8601",
                "latitude": 18.5204,
                "longitude": 73.8567,
                "storagePath": "s3://bucket/path/video_123.mp4"
              },
              "pipelineVersion": "v1",
              "chunking": {
                "chunkDurationSec": 4,
                "maxFramesPerChunk": 32
              },
              "priority": "high",
              "createdAt": "ISO8601"
            }

        Returns a normalised dict consumed by ``process_job``.
        """
        # --- Required top-level fields ---
        organization_id = job_data.get("organizationId")
        if not organization_id:
            raise ValueError("Missing 'organizationId' in job data")

        site_id = job_data.get("siteId")
        if not site_id:
            raise ValueError("Missing 'siteId' in job data")

        media = job_data.get("media")
        if not media:
            raise ValueError("Missing 'media' block in job data")

        storage_path = media.get("storagePath")
        if not storage_path:
            raise ValueError("Missing 'media.storagePath' in job data")

        chunking = job_data.get("chunking", {})

        # --- Build normalised output ---
        return {
            "job_id": job_data.get("jobId"),
            "organization_id": organization_id,
            "site_id": site_id,
            "storage_path": storage_path,  # raw S3 key / path (persisted in Qdrant)
            "video_url": storage_path,     # resolved to a presigned URL in process_job
            # Batch processing request this job belongs to (used to update counters)
            "processing_request_id": (
                job_data.get("processingRequestId") or job_data.get("requestId")
            ),
            # Media metadata (stored on every vector point)
            "media_id": media.get("mediaId"),
            "flight_id": media.get("flightId"),
            "flight_type": media.get("flightType"),
            "mission_id": media.get("missionId"),
            "mission_type": media.get("missionType"),
            "file_name": media.get("fileName"),
            "file_type": media.get("fileType"),
            "file_size_bytes": media.get("fileSizeBytes"),
            "capture_timestamp": media.get("captureTimestamp"),
            "latitude": media.get("latitude"),
            "longitude": media.get("longitude"),
            # Chunking config
            "clip_duration": chunking.get("chunkDurationSec", config.SEMANTIC_CLIP_DURATION),
            "max_frames_per_clip": chunking.get("maxFramesPerChunk", config.SEMANTIC_CLIP_MAX_FRAMES),
            # Extra
            "pipeline_version": job_data.get("pipelineVersion", "v1"),
            "priority": job_data.get("priority"),
            "created_at": job_data.get("createdAt"),
        }

    def _set_ingestion_status(
        self,
        media_id: Optional[str],
        pipeline_version: str,
        status: str,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Best-effort update of the ingestion_state collection."""
        if not media_id or self.mongodb_service is None:
            return
        try:
            ok = self.mongodb_service.update_ingestion_status(
                media_id=media_id,
                status=status,
                pipeline_version=pipeline_version,
                error=error,
                request_id=request_id,
            )
            logger.info(
                "%s ingestion_state → %s for media_id=%s",
                "✓" if ok else "⚠ (not found)",
                status,
                media_id,
            )
        except Exception as exc:
            logger.warning("ingestion_state update failed (non-fatal): %s", exc)

    def _update_processing_request(
        self,
        transition: str,
        processing_request_id: Optional[str],
        media_id: Optional[str],
    ) -> None:
        """Best-effort counter increment on the processing_requests collection."""
        if self.mongodb_service is None:
            return
        if not processing_request_id and not media_id:
            return
        try:
            ok = self.mongodb_service.increment_processing_request(
                transition=transition,
                processing_request_id=processing_request_id,
                media_id=media_id,
            )
            logger.info(
                "%s processing_requests counter update (%s) for request_id=%s media_id=%s",
                "✓" if ok else "⚠ (not found)",
                transition,
                processing_request_id,
                media_id,
            )
        except Exception as exc:
            logger.warning("processing_requests update failed (non-fatal): %s", exc)

    def process_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single embedding job.

        Accepts the raw RMQ ``embedding.jobs`` payload, parses it, downloads
        the video from the storage path, generates embeddings, and stores
        them in the organisation-specific Qdrant collection.

        Args:
            job_data: Raw job payload from RabbitMQ.

        Returns:
            Dictionary with processing results and statistics.
        """
        parsed = self._parse_job(job_data)

        organization_id = parsed["organization_id"]
        storage_path = parsed["storage_path"]
        media_id = parsed.get("media_id")
        pipeline_version = parsed.get("pipeline_version", "v1")
        processing_request_id = parsed.get("processing_request_id")

        # Mark ingestion as processing and move the request counter queued→processing
        self._set_ingestion_status(
            media_id, pipeline_version, "processing", request_id=processing_request_id,
        )
        self._update_processing_request(
            "queued_to_processing", processing_request_id, media_id,
        )

        # Normalise bare S3 keys to full s3://bucket/key URIs so the path is
        # self-contained everywhere it is stored (Qdrant metadata, telemetry
        # paths, etc.) and does not depend on the env var at search time.
        s3_service = get_s3_service()
        if s3_service is not None and not storage_path.startswith("s3://"):
            storage_path = f"s3://{s3_service.bucket_name}/{storage_path}"
            logger.info(f"Normalised storage_path to: {storage_path}")

        # Resolve the storage path to a downloadable URL.
        # If S3 is configured, generate a presigned URL so the pipeline can
        # download the object directly via HTTPS.  Otherwise fall back to
        # using the raw storage path (works for plain http:// / https:// URLs).
        if s3_service is not None:
            logger.info(f"Generating presigned download URL for: {storage_path}")
            video_url = s3_service.generate_presigned_download_url(storage_path)
            logger.info("Presigned URL generated successfully")
        else:
            video_url = storage_path

        logger.info(f"Processing job {parsed.get('job_id')} for org {organization_id}")
        logger.info(f"Storage path: {storage_path}")

        # Build the metadata dict that will be stored on every vector point
        additional_metadata = {
            "organization_id": organization_id,
            "site_id": parsed["site_id"],
            "storage_path": storage_path,
            "media_id": parsed.get("media_id"),
            "flight_id": parsed.get("flight_id"),
            "flight_type": parsed.get("flight_type"),
            "mission_id": parsed.get("mission_id"),
            "mission_type": parsed.get("mission_type"),
            "file_type": parsed.get("file_type"),
            "capture_timestamp": parsed.get("capture_timestamp"),
            "latitude": parsed.get("latitude"),
            "longitude": parsed.get("longitude"),
            "pipeline_version": parsed.get("pipeline_version"),
        }
        # Remove None values so Qdrant payload stays clean
        additional_metadata = {k: v for k, v in additional_metadata.items() if v is not None}

        clip_duration = parsed["clip_duration"]
        max_frames_per_clip = parsed["max_frames_per_clip"]

        with trace_operation(
            name="worker-process-job",
            operation_type="span",
            user_id=organization_id,
            session_id=parsed.get("flight_id"),
            metadata={
                "job_id": parsed.get("job_id"),
                "site_id": parsed["site_id"],
                "media_id": parsed.get("media_id"),
                "mission_id": parsed.get("mission_id"),
            },
            tags=["worker", "ingestion"],
            input={"storage_path": storage_path[:100]},
        ) as trace:
            start_time = time.time()

            try:
                logger.info(f"Downloading and ingesting video from: {video_url}")

                ingest_result = self.ingest_pipeline.ingest_video_from_url(
                    video_url=video_url,
                    organization_id=organization_id,
                    zone="default",
                    clip_duration=clip_duration,
                    max_frames_per_clip=max_frames_per_clip,
                    batch_size=100,
                    save_full_frames=False,
                    cleanup_after=True,
                    metadata=additional_metadata,
                    s3_storage_path=storage_path if s3_service is not None else None,
                )

                clips_ingested = ingest_result.get("clips_ingested", 0)
                telemetry_array = ingest_result.get("telemetry")

                # Upsert flight telemetry into MongoDB for geospatial queries
                if self.mongodb_service is not None and telemetry_array:
                    try:
                        offsets = [
                            e["seconds_offset"]
                            for e in telemetry_array
                            if "seconds_offset" in e
                        ]
                        duration = max(offsets) if offsets else None
                        num_entries = len(telemetry_array)

                        with trace_operation(
                            name="geo-coverage-build",
                            operation_type="span",
                            metadata={
                                "media_id": parsed.get("media_id"),
                                "telemetry_entries": num_entries,
                            },
                            tags=["geo", "coverage"],
                        ) as geo_trace:
                            t0 = time.time()

                            flight_path = build_flight_path(telemetry_array)
                            t_path = time.time() - t0

                            t1 = time.time()
                            coverage_area = build_coverage_polygon(telemetry_array)
                            t_coverage = time.time() - t1

                            t2 = time.time()
                            bounds = build_bounds(telemetry_array)
                            t_bounds = time.time() - t2

                            t3 = time.time()
                            frames = build_frames(telemetry_array)
                            t_frames = time.time() - t3

                            total_geo = time.time() - t0
                            logger.debug(
                                "Geo coverage computed in %.2fs "
                                "(path=%.3fs, coverage=%.3fs, bounds=%.3fs, frames=%.3fs)",
                                total_geo, t_path, t_coverage, t_bounds, t_frames,
                            )

                            if geo_trace:
                                geo_trace.update(output={
                                    "total_seconds": round(total_geo, 3),
                                    "flight_path_seconds": round(t_path, 3),
                                    "coverage_polygon_seconds": round(t_coverage, 3),
                                    "bounds_seconds": round(t_bounds, 3),
                                    "frames_seconds": round(t_frames, 3),
                                    "telemetry_entries": num_entries,
                                    "frames_produced": len(frames),
                                    "has_coverage": coverage_area is not None,
                                })

                        flight_doc = {
                            "_id": parsed.get("media_id") or parsed.get("job_id"),
                            "organizationId": organization_id,
                            "siteId": parsed.get("site_id"),
                            "flightId": parsed.get("flight_id"),
                            "sourceFile": parsed.get("file_name"),
                            "storagePath": storage_path,
                            "captureTimestamp": parsed.get("capture_timestamp"),
                            "duration": duration,
                            "flightPath": flight_path,
                            "coverageArea": coverage_area,
                            "bounds": bounds,
                            "frames": frames,
                        }
                        flight_doc = {k: v for k, v in flight_doc.items() if v is not None}
                        ok = self.mongodb_service.upsert_flight_telemetry(flight_doc)
                        logger.info(
                            "%s MongoDB flight_telemetry upsert",
                            "✓" if ok else "⚠",
                        )
                    except Exception as exc:
                        logger.warning("MongoDB upsert failed (non-fatal): %s", exc)

                processing_time = time.time() - start_time

                # Mark ingestion as completed and advance the request counter processing→done
                self._set_ingestion_status(
                    media_id, pipeline_version, "completed", request_id=processing_request_id,
                )
                self._update_processing_request(
                    "processing_to_done", processing_request_id, media_id,
                )

                result = {
                    "status": "success",
                    "job_id": parsed.get("job_id"),
                    "organization_id": organization_id,
                    "storage_path": storage_path,
                    "clips_ingested": clips_ingested,
                    "telemetry_entries": len(telemetry_array) if telemetry_array else 0,
                    "processing_time_seconds": round(processing_time, 2),
                }

                logger.info(f"Successfully processed video: {storage_path}")
                logger.info(f"  - Clips: {clips_ingested}")
                logger.info(f"  - Processing time: {processing_time:.2f}s")

                if trace:
                    trace.update(
                        output=result,
                        level="DEFAULT",
                        status_message="Job completed successfully",
                    )

                return result

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Failed to process video: {storage_path}")
                logger.error(f"  - Error: {str(e)}")
                logger.error(f"  - Processing time: {processing_time:.2f}s")

                # Mark ingestion as failed and advance the request counter processing→failed
                self._set_ingestion_status(
                    media_id, pipeline_version, "failed",
                    error=str(e), request_id=processing_request_id,
                )
                self._update_processing_request(
                    "processing_to_failed", processing_request_id, media_id,
                )

                if trace:
                    trace.update(
                        level="ERROR",
                        status_message=str(e),
                        output={
                            "error": str(e),
                            "processing_time_seconds": round(processing_time, 2),
                        },
                    )

                raise

    @staticmethod
    def _classify_error(error: Exception) -> Tuple[bool, str]:
        """Classify an error as retryable or permanent.

        Returns:
            (is_retryable, reason)

        Permanent (non-retryable) errors:
        - S3 403/404: Access denied or file not found
        - S3 InvalidAccessKeyId/InvalidSecretAccessKey: Bad credentials
        - ValueError with "invalid" or "malformed": Bad input data
        - ValueError with specific S3 messages: Permission/config issues
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # S3 Client Errors - check status code
        if isinstance(error, ClientError):
            code = error.response.get("Error", {}).get("Code", "")
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            # Permanent errors
            if code in ("Forbidden", "NotFound", "NoSuchBucket", "NoSuchKey",
                       "InvalidAccessKeyId", "InvalidSecretAccessKey",
                       "SignatureDoesNotMatch", "AccessDenied"):
                return False, f"S3 {code} (permanent permission/config issue)"
            if status == 403:
                return False, f"S3 403 Forbidden (access denied)"
            if status == 404:
                return False, f"S3 404 Not Found (object missing)"

            # Retryable (transient network errors)
            if code in ("ServiceUnavailable", "RequestTimeout", "SlowDown"):
                return True, f"S3 {code} (transient)"

        # ValueError with specific messages (from ingest_pipeline)
        if isinstance(error, ValueError):
            if "invalid" in error_str or "malformed" in error_str:
                return False, "Invalid input data (permanent)"
            # HTTP 404/403 from requests.raise_for_status() wrapped by ingest_pipeline
            if "404" in error_str and ("not found" in error_str or "client error" in error_str):
                return False, "HTTP 404 Not Found (object missing, permanent)"
            if "403" in error_str and ("forbidden" in error_str or "client error" in error_str):
                return False, "HTTP 403 Forbidden (access denied, permanent)"
            if "s3" in error_str and any(x in error_str for x in
                    ["403", "forbidden", "accessdenied", "credentials", "permission"]):
                return False, "S3 permission/credential issue (permanent)"

        # RuntimeError from boto3
        if isinstance(error, RuntimeError) and "failed to download" in error_str:
            if "403" in error_str or "forbidden" in error_str:
                return False, "S3 download permission denied (permanent)"
            # Other download failures might be transient
            return True, "S3 download transient error"

        # Default: assume transient
        return True, f"{error_type} (assumed transient)"

    def on_message(self, channel, method, properties, body):
        """
        Callback function when a message is received from RabbitMQ.

        Runs the actual job in a background thread so the main thread can
        keep processing RabbitMQ heartbeats.  This prevents connection resets
        during long-running jobs (large file downloads, video processing).

        Args:
            channel: Channel instance
            method: Delivery method
            properties: Message properties
            body: Message body (bytes)
        """
        try:
            job_data = json.loads(body.decode('utf-8'))
            logger.info(f"\n{'='*60}")
            logger.info(f"Received job (Delivery tag: {method.delivery_tag})")
            logger.info(f"{'='*60}")

            job_error = [None]

            def _run_job():
                try:
                    self.process_job(job_data)
                except Exception as e:
                    job_error[0] = e

            worker_thread = threading.Thread(target=_run_job, daemon=True)
            worker_thread.start()

            while worker_thread.is_alive():
                try:
                    self.connection.process_data_events(time_limit=1)
                except Exception:
                    break

            worker_thread.join()

            if job_error[0] is not None:
                raise job_error[0]

            channel.basic_ack(delivery_tag=method.delivery_tag)
            self.jobs_processed += 1

            logger.info(f"✓ Job acknowledged and completed")
            logger.info(f"Total jobs processed: {self.jobs_processed}")

        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON in message: {e}")
            logger.error(f"Message body: {body}")
            try:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception:
                logger.warning("Failed to nack message (connection may be lost)")
            self.jobs_failed += 1

        except Exception as e:
            logger.error(f"✗ Error processing job: {e}")
            logger.error(traceback.format_exc())

            # Classify error to decide if we should retry
            is_retryable, reason = self._classify_error(e)

            try:
                if is_retryable:
                    logger.warning(
                        f"Retryable error ({reason}): "
                        f"requeuing job delivery_tag={method.delivery_tag}"
                    )
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                else:
                    logger.error(
                        f"Permanent error ({reason}): "
                        f"discarding job delivery_tag={method.delivery_tag}"
                    )
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception:
                logger.warning("Failed to nack message (connection may be lost)")
            self.jobs_failed += 1

    def start_consuming(self):
        """Start consuming messages from the queue."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Worker started - waiting for jobs on queue: {self.queue_name}")
        logger.info(f"Prefetch count: {self.prefetch_count}")
        logger.info(f"Press CTRL+C to exit")
        logger.info(f"{'='*60}\n")

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_message,
            auto_ack=False,  # Manual acknowledgment for reliability
        )

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal - shutting down...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
            logger.error(traceback.format_exc())
            raise

    def stop(self):
        """Gracefully stop the worker."""
        logger.info("Stopping worker...")

        if self.channel:
            self.channel.stop_consuming()

        if self.connection:
            self.connection.close()

        # Flush Langfuse traces before shutting down
        logger.info("Flushing Langfuse traces...")
        flush_langfuse()

        # Print statistics
        uptime = time.time() - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Worker Statistics:")
        logger.info(f"  - Uptime: {uptime:.2f}s ({uptime/60:.2f} minutes)")
        logger.info(f"  - Jobs processed: {self.jobs_processed}")
        logger.info(f"  - Jobs failed: {self.jobs_failed}")
        logger.info(f"  - Success rate: {self.jobs_processed/(self.jobs_processed+self.jobs_failed)*100:.1f}%" if (self.jobs_processed + self.jobs_failed) > 0 else "N/A")
        logger.info(f"{'='*60}")
        logger.info("Worker stopped")

    def run(self):
        """Main entry point for the worker.

        Initialises services once, then enters a reconnect loop that
        re-establishes the RabbitMQ connection whenever it drops (e.g.
        heartbeat timeout, network blip, broker restart).
        """
        self.initialize_services()

        retry_delay = 5
        max_retry_delay = 60

        while True:
            try:
                self.connect_to_rabbitmq()
                retry_delay = 5  # reset after successful connection
                self.start_consuming()

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.stop()
                break

            except Exception as e:
                logger.error(f"Connection lost or error: {e}")
                logger.error(traceback.format_exc())

                # Tear down the broken connection so connect_to_rabbitmq
                # can create a fresh one on the next iteration.
                try:
                    if self.connection and not self.connection.is_closed:
                        self.connection.close()
                except Exception:
                    pass
                self.connection = None
                self.channel = None

                logger.info(f"Reconnecting in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)


def main():
    """Main function to start the worker."""
    # Load configuration from environment
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
    rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5672))
    rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
    queue_name = os.getenv("RABBITMQ_QUEUE", "embedding.jobs")
    prefetch_count = int(os.getenv("RABBITMQ_PREFETCH_COUNT", 1))

    logger.info("Starting Video-RAG Embedding Worker")
    logger.info(f"Configuration:")
    logger.info(f"  - RabbitMQ Host: {rabbitmq_host}:{rabbitmq_port}")
    logger.info(f"  - Queue: {queue_name}")
    logger.info(f"  - Prefetch Count: {prefetch_count}")
    logger.info(f"  - Device: {config.DEVICE}")
    logger.info(f"  - Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  - Model: {config.MODEL_NAME}")

    worker = EmbeddingWorker(
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        rabbitmq_user=rabbitmq_user,
        rabbitmq_password=rabbitmq_password,
        queue_name=queue_name,
        prefetch_count=prefetch_count,
    )

    worker.run()


if __name__ == "__main__":
    main()
