"""S3 service for generating presigned download URLs and uploading objects.

Supports both explicit ``s3://bucket/key`` URIs and bare S3 keys
(using the configured S3_BUCKET_NAME as the bucket).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config as BotocoreConfig
    from botocore.exceptions import ClientError, NoCredentialsError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed – S3 presigned URL generation unavailable")


class S3Service:
    """Generates presigned S3 URLs for secure, time-limited object access."""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        presigned_url_expiration: int = 3600,
        signature_version: str = "s3",
        addressing_style: str = "path",
        use_accelerate_endpoint: bool = False,
    ):
        """
        Initialise the S3 service.

        Args:
            bucket_name: Default S3 bucket used when storage_path is a bare key.
            region: AWS region (e.g. ``"us-east-1"``).
            access_key: AWS access key ID (falls back to environment/IAM role).
            secret_key: AWS secret access key (falls back to environment/IAM role).
            endpoint_url: Optional custom endpoint (e.g. MinIO URL). ``None``
                          means the standard AWS S3 endpoint is used.
            presigned_url_expiration: Default URL lifetime in seconds (1 hour).
            signature_version: Boto3 signature version for presigned URLs.
                ``"s3"`` produces SigV2 URLs (``AWSAccessKeyId``/``Signature``/
                ``Expires`` query params). ``"s3v4"`` produces SigV4 URLs
                (``X-Amz-*`` query params). Defaults to ``"s3"`` (SigV2).
            addressing_style: S3 URL addressing style. ``"path"`` produces
                ``https://s3.<region>.amazonaws.com/<bucket>/<key>`` URLs.
                ``"virtual"`` produces
                ``https://<bucket>.s3.<region>.amazonaws.com/<key>`` URLs.
                Defaults to ``"path"``.
            use_accelerate_endpoint: If ``True``, use S3 Transfer Acceleration
                endpoints (e.g. ``bucket.s3-accelerate.amazonaws.com``) for
                faster transfers. Requires the bucket to have acceleration
                enabled in AWS. Ignored when ``endpoint_url`` is set.
        """
        self.bucket_name = bucket_name
        self.presigned_url_expiration = presigned_url_expiration
        self._client = None

        if not _BOTO3_AVAILABLE:
            return

        session = boto3.Session(
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region,
        )
        s3_config = {"addressing_style": addressing_style}
        if use_accelerate_endpoint and endpoint_url is None:
            s3_config["use_accelerate_endpoint"] = True
        boto_config = BotocoreConfig(
            signature_version=signature_version,
            s3=s3_config,
        )
        self._client = session.client(
            "s3",
            endpoint_url=endpoint_url or None,
            config=boto_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_presigned_download_url(
        self,
        storage_path: str,
        expiration: Optional[int] = None,
    ) -> str:
        """
        Generate a presigned GET URL for an S3 object.

        Args:
            storage_path: S3 key or ``s3://bucket/key`` URI.
            expiration: URL lifetime in seconds (defaults to
                        ``presigned_url_expiration`` set at construction time).

        Returns:
            Presigned HTTPS URL string.

        Raises:
            RuntimeError: If boto3 is unavailable or AWS credentials are missing.
            ValueError: If ``storage_path`` is malformed.
        """
        if not _BOTO3_AVAILABLE or self._client is None:
            raise RuntimeError(
                "boto3 is not installed or S3 client could not be created; "
                "cannot generate presigned URLs"
            )

        bucket, key = self._parse_storage_path(storage_path)
        exp = expiration if expiration is not None else self.presigned_url_expiration

        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=exp,
            )
            logger.debug("Generated presigned URL for s3://%s/%s (exp=%ds)", bucket, key, exp)
            return url
        except (ClientError, NoCredentialsError) as exc:
            raise RuntimeError(
                f"Failed to generate presigned URL for {storage_path!r}: {exc}"
            ) from exc

    def try_generate_presigned_download_url(
        self,
        storage_path: str,
        expiration: Optional[int] = None,
    ) -> Optional[str]:
        """
        Like :meth:`generate_presigned_download_url` but returns ``None``
        instead of raising on failure.

        Useful for search results where a missing presigned URL is non-fatal.
        """
        try:
            return self.generate_presigned_download_url(storage_path, expiration)
        except Exception as exc:
            logger.warning(
                "Could not generate presigned URL for %r: %s", storage_path, exc
            )
            return None

    def get_object_size(self, storage_path: str) -> Optional[int]:
        """Return the size in bytes of an S3 object, or ``None`` on failure."""
        if not _BOTO3_AVAILABLE or self._client is None:
            return None
        bucket, key = self._parse_storage_path(storage_path)
        try:
            resp = self._client.head_object(Bucket=bucket, Key=key)
            return resp.get("ContentLength")
        except Exception:
            return None

    def download_file(
        self,
        storage_path: str,
        local_path: str,
        max_concurrency: int = 20,
        multipart_threshold_mb: int = 25,
        multipart_chunksize_mb: int = 32,
        callback: Optional[Any] = None,
    ) -> str:
        """Download an S3 object directly to a local file using multipart parallel transfer.

        Uses boto3's managed transfer which automatically splits large files into
        parts and downloads them concurrently, dramatically faster than streaming
        through a presigned URL for large objects.

        Args:
            storage_path: S3 key or ``s3://bucket/key`` URI.
            local_path: Destination file path on local disk.
            max_concurrency: Number of parallel download threads (default 10).
            multipart_threshold_mb: File-size threshold in MB above which
                multipart download is used (default 25).
            multipart_chunksize_mb: Part size in MB for multipart transfers
                (default 25).
            callback: Optional callable invoked with bytes transferred (for
                progress reporting, e.g. a ``tqdm`` update function).

        Returns:
            The *local_path* for convenience.

        Raises:
            RuntimeError: If boto3 is unavailable or the download fails.
        """
        if not _BOTO3_AVAILABLE or self._client is None:
            raise RuntimeError(
                "boto3 is not installed or S3 client could not be created; "
                "cannot download files"
            )

        bucket, key = self._parse_storage_path(storage_path)

        transfer_config = TransferConfig(
            max_concurrency=max_concurrency,
            multipart_threshold=multipart_threshold_mb * 1024 * 1024,
            multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
        )

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        try:
            logger.info(
                "Downloading s3://%s/%s → %s (concurrency=%d, chunk=%dMB)",
                bucket, key, local_path, max_concurrency, multipart_chunksize_mb,
            )
            self._client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=local_path,
                Config=transfer_config,
                Callback=callback,
            )
            size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            logger.info("Download complete: %.2f MB → %s", size_mb, local_path)
            return local_path
        except (ClientError, NoCredentialsError) as exc:
            raise RuntimeError(
                f"Failed to download s3://{bucket}/{key} to {local_path}: {exc}"
            ) from exc

    def upload_json(self, s3_path: str, data: Any) -> None:
        """Upload *data* serialised as JSON to *s3_path*.

        Args:
            s3_path: Destination ``s3://bucket/key`` URI or bare S3 key.
            data: JSON-serialisable object.

        Raises:
            RuntimeError: If boto3 is unavailable or upload fails.
        """
        if not _BOTO3_AVAILABLE or self._client is None:
            raise RuntimeError(
                "boto3 is not installed or S3 client could not be created; "
                "cannot upload objects"
            )

        bucket, key = self._parse_storage_path(s3_path)
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            logger.debug("Uploaded JSON to s3://%s/%s (%d bytes)", bucket, key, len(body))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload JSON to {s3_path!r}: {exc}"
            ) from exc

    def try_upload_json(self, s3_path: str, data: Any) -> bool:
        """Like :meth:`upload_json` but returns ``False`` instead of raising.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            self.upload_json(s3_path, data)
            return True
        except Exception as exc:
            logger.warning("Could not upload JSON to %r: %s", s3_path, exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_storage_path(self, storage_path: str) -> Tuple[str, str]:
        """Return ``(bucket, key)`` from a storage path.

        Accepts:
          - ``s3://bucket-name/path/to/object.mp4``  → explicit bucket + key
          - ``path/to/object.mp4``                   → uses ``self.bucket_name``
        """
        if storage_path.startswith("s3://"):
            remainder = storage_path[5:]  # strip "s3://"
            parts = remainder.split("/", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid s3:// URI: {storage_path!r}")
            return parts[0], parts[1]

        # Bare key – use the configured default bucket
        if not self.bucket_name:
            raise ValueError(
                "storage_path is a bare S3 key but S3_BUCKET_NAME is not configured"
            )
        return self.bucket_name, storage_path


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_s3_service: Optional[S3Service] = None


def get_s3_service() -> Optional[S3Service]:
    """Return the singleton :class:`S3Service`, or ``None`` if S3 is not configured.

    S3 is considered unconfigured when ``S3_BUCKET_NAME`` is not set in the
    environment / config.
    """
    global _s3_service
    if _s3_service is not None:
        return _s3_service

    import config as _config

    bucket = getattr(_config, "S3_BUCKET_NAME", None)
    if not bucket:
        logger.debug("S3_BUCKET_NAME not configured; S3Service disabled")
        return None

    sig_ver = getattr(_config, "S3_SIGNATURE_VERSION", "s3")
    addr_style = getattr(_config, "S3_ADDRESSING_STYLE", "path")
    use_accelerate = getattr(_config, "S3_USE_ACCELERATE", False)
    endpoint_url = getattr(_config, "S3_ENDPOINT_URL", None) or None
    _s3_service = S3Service(
        bucket_name=bucket,
        region=getattr(_config, "AWS_REGION", "us-east-1"),
        access_key=getattr(_config, "AWS_ACCESS_KEY_ID", None) or None,
        secret_key=getattr(_config, "AWS_SECRET_ACCESS_KEY", None) or None,
        endpoint_url=endpoint_url,
        presigned_url_expiration=getattr(_config, "S3_PRESIGNED_URL_EXPIRATION", 3600),
        signature_version=sig_ver,
        addressing_style=addr_style,
        use_accelerate_endpoint=use_accelerate and endpoint_url is None,
    )
    logger.info(
        "S3Service initialised (bucket=%s, region=%s, sig=%s, addressing=%s, accelerate=%s)",
        bucket,
        getattr(_config, "AWS_REGION", "us-east-1"),
        sig_ver,
        addr_style,
        use_accelerate and endpoint_url is None,
    )
    return _s3_service
