"""End-to-end test for the RunPod Serverless worker.

Usage:
    # Submit and poll until completion (blocks):
    python scripts/test_runpod.py

    # Submit only (non-blocking, prints job ID):
    python scripts/test_runpod.py --submit-only

    # Check status of an existing job:
    python scripts/test_runpod.py --status rp_abc1234xyz

    # Check RunPod queue health:
    python scripts/test_runpod.py --health

Prerequisites:
    RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in .env
    Update STORAGE_PATH below to a real video in your S3 bucket.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from runpod_client import RunPodClient

# ── Change this to a real short video (30–60s) in your S3 bucket ──────────────
STORAGE_PATH  = os.getenv("TEST_STORAGE_PATH", "s3://flytbase-media-stag/658295f8dbab9efb302183ab/media/7077ddaf-3bcf-4a45-bfdb-feb7bf9faf5e/DJI_202512241838_013_7077ddaf-3bcf-4a45-bfdb-feb7bf9faf5e/DJI_20251224183958_0003_V.mp4")
ORG_ID        = os.getenv("TEST_ORG_ID",       "org_test")
SITE_ID       = os.getenv("TEST_SITE_ID",      "site_test")
# ──────────────────────────────────────────────────────────────────────────────

TEST_JOB = {
    "jobId":          f"test-{int(time.time())}",
    "organizationId": ORG_ID,
    "siteId":         SITE_ID,
    "media": {
        "mediaId":          f"media-test-{int(time.time())}",
        "flightId":         "flight-test-001",
        "flightType":       "mission",
        "missionId":        "mission-test-001",
        "missionType":      "inspection",
        "fileName":         Path(STORAGE_PATH).name,
        "fileType":         "video/mp4",
        "storagePath":      STORAGE_PATH,
        "captureTimestamp": "2026-02-25T10:00:00Z",
        "latitude":         18.5204,
        "longitude":        73.8567,
    },
    "pipelineVersion": "v1",
    "chunking": {
        "chunkDurationSec":  4,
        "maxFramesPerChunk": 16,
    },
    "priority":  "normal",
    "createdAt": "2026-02-25T10:00:00Z",
}


def cmd_health(client: RunPodClient) -> None:
    print("\n── Queue Health ──────────────────────────────────")
    health = client.queue_health()
    print(json.dumps(health, indent=2))


def cmd_status(client: RunPodClient, job_id: str) -> None:
    print(f"\n── Status: {job_id} ──────────────────────────────")
    status = client.status(job_id)
    print(json.dumps(status, indent=2))


def cmd_submit_only(client: RunPodClient) -> None:
    print(f"\n── Submitting job (non-blocking) ─────────────────")
    print(f"Storage path : {STORAGE_PATH}")
    print(f"Organization : {ORG_ID}")
    job_id = client.submit(TEST_JOB)
    print(f"\n✓ Submitted. RunPod job ID: {job_id}")
    print(f"\nCheck status with:")
    print(f"  python scripts/test_runpod.py --status {job_id}")


def cmd_submit_and_wait(client: RunPodClient) -> None:
    print(f"\n── Submitting + waiting for completion ───────────")
    print(f"Storage path : {STORAGE_PATH}")
    print(f"Organization : {ORG_ID}")
    print(f"(This will take several minutes on first run — model cold start + video processing)\n")

    start = time.time()

    def on_progress(data):
        elapsed = time.time() - start
        print(f"  [{elapsed:5.0f}s]  status={data.get('status')}")

    job_id = client.submit(TEST_JOB)
    print(f"Submitted → RunPod job ID: {job_id}\n")

    try:
        result = client.wait(job_id, timeout=3600, on_progress=on_progress)
    except TimeoutError as e:
        print(f"\n✗ Timed out: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    status  = result.get("status")
    output  = result.get("output", {})

    print(f"\n── Result ────────────────────────────────────────")
    print(f"Status  : {status}")
    print(f"Elapsed : {elapsed:.1f}s")
    print(json.dumps(output, indent=2))

    if status == "COMPLETED" and "error" not in (output or {}):
        clips = output.get("clips_ingested", "?") if output else "?"
        print(f"\n✓ SUCCESS — {clips} clips ingested into Qdrant (org={ORG_ID})")
        print(f"\nNow run a search query against org '{ORG_ID}' to verify vectors are there.")
    else:
        print(f"\n✗ Job ended with status={status}")
        if output and "error" in output:
            print(f"Error: {output['error']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test RunPod Serverless worker")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--submit-only", action="store_true", help="Submit job without waiting")
    group.add_argument("--status",      metavar="JOB_ID",   help="Check status of an existing job")
    group.add_argument("--health",      action="store_true", help="Check endpoint queue health")
    args = parser.parse_args()

    # Validate config
    api_key     = os.getenv("RUNPOD_API_KEY", "")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")

    if not api_key or not endpoint_id:
        print("✗ RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in .env")
        sys.exit(1)

    print(f"Endpoint : {endpoint_id}")

    client = RunPodClient(api_key=api_key, endpoint_id=endpoint_id)

    if args.health:
        cmd_health(client)
    elif args.status:
        cmd_status(client, args.status)
    elif args.submit_only:
        cmd_submit_only(client)
    else:
        cmd_submit_and_wait(client)


if __name__ == "__main__":
    main()

