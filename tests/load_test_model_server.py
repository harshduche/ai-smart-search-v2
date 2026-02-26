"""
Load-test for the batched model server.

Simulates N concurrent workers sending embedding requests and reports
latency percentiles (p50/p95/p99), throughput (req/s), and batch
utilisation from the /stats endpoint.

Usage:
    # Quick smoke test (4 workers, 20 requests)
    python tests/load_test_model_server.py

    # Full sweep across concurrency levels
    python tests/load_test_model_server.py --full

    # Custom run
    python tests/load_test_model_server.py \
        --url http://localhost:8001 \
        --concurrency 1 2 4 8 \
        --requests 50 \
        --image-size 256 \
        --include-video
"""

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp is required: pip install aiohttp")

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image(width: int = 256, height: int = 256) -> str:
    """Generate a random RGB image and return its base64-encoded PNG."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_test_video_clip(n_frames: int = 8, width: int = 128, height: int = 128) -> List[str]:
    """Generate a list of random base64-encoded frames for a video clip."""
    return [make_test_image(width, height) for _ in range(n_frames)]


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def fmt_rps(n: int, elapsed: float) -> str:
    return f"{n / elapsed:.2f}" if elapsed > 0 else "N/A"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    concurrency: int
    total_requests: int
    successful: int = 0
    failed: int = 0
    latencies: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    server_stats_before: Optional[dict] = None
    server_stats_after: Optional[dict] = None
    errors: List[str] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        idx = int(len(self.latencies) * 0.95)
        return sorted(self.latencies)[min(idx, len(self.latencies) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        idx = int(len(self.latencies) * 0.99)
        return sorted(self.latencies)[min(idx, len(self.latencies) - 1)]

    @property
    def throughput(self) -> float:
        return self.successful / self.wall_time if self.wall_time > 0 else 0

    @property
    def batches_created(self) -> int:
        if self.server_stats_before and self.server_stats_after:
            return (
                self.server_stats_after.get("total_image_batches", 0)
                - self.server_stats_before.get("total_image_batches", 0)
            )
        return 0

    @property
    def avg_batch_fill(self) -> float:
        if self.batches_created > 0:
            return self.successful / self.batches_created
        return 0


# ---------------------------------------------------------------------------
# Core load-test logic
# ---------------------------------------------------------------------------

async def fetch_stats(session: aiohttp.ClientSession, base_url: str) -> dict:
    try:
        async with session.get(f"{base_url}/stats", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return await resp.json()
    except Exception:
        return {}


async def send_image_request(
    session: aiohttp.ClientSession,
    url: str,
    image_b64: str,
    request_id: str,
    timeout: int,
) -> tuple:
    """Send a single /embed/image request. Returns (latency_s, error_or_None)."""
    payload = {"image_base64": image_b64, "request_id": request_id}
    t0 = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.json()
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                return elapsed, f"HTTP {resp.status}: {body}"
            dim = body.get("dimension", 0)
            if dim == 0:
                return elapsed, "empty embedding returned"
            return elapsed, None
    except Exception as e:
        return time.monotonic() - t0, str(e)


async def send_video_clip_request(
    session: aiohttp.ClientSession,
    url: str,
    frames_b64: List[str],
    request_id: str,
    timeout: int,
) -> tuple:
    payload = {"images_base64": frames_b64, "request_id": request_id}
    t0 = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.json()
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                return elapsed, f"HTTP {resp.status}: {body}"
            return elapsed, None
    except Exception as e:
        return time.monotonic() - t0, str(e)


async def run_concurrent_load(
    base_url: str,
    concurrency: int,
    total_requests: int,
    endpoint: str = "image",
    image_size: int = 256,
    video_frames: int = 8,
    timeout: int = 300,
) -> RunResult:
    """Fire *total_requests* at *concurrency* parallelism and collect metrics."""

    result = RunResult(concurrency=concurrency, total_requests=total_requests)

    # Pre-generate payloads so generation time doesn't pollute latency numbers
    print(f"  Generating {total_requests} test payloads ({endpoint})...")
    if endpoint == "image":
        payloads = [make_test_image(image_size, image_size) for _ in range(total_requests)]
    else:
        payloads = [make_test_video_clip(video_frames, image_size, image_size) for _ in range(total_requests)]

    sem = asyncio.Semaphore(concurrency)
    url = f"{base_url}/embed/{endpoint}" if endpoint == "image" else f"{base_url}/embed/video-clip"

    async with aiohttp.ClientSession() as session:
        result.server_stats_before = await fetch_stats(session, base_url)

        async def _worker(idx: int):
            async with sem:
                rid = f"load-{concurrency}w-{idx}"
                if endpoint == "image":
                    lat, err = await send_image_request(session, url, payloads[idx], rid, timeout)
                else:
                    lat, err = await send_video_clip_request(session, url, payloads[idx], rid, timeout)

                if err:
                    result.failed += 1
                    result.errors.append(f"[req {idx}] {err}")
                else:
                    result.successful += 1
                result.latencies.append(lat)

        t_start = time.monotonic()
        await asyncio.gather(*[_worker(i) for i in range(total_requests)])
        result.wall_time = time.monotonic() - t_start

        result.server_stats_after = await fetch_stats(session, base_url)

    return result


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def check_health(base_url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                loaded = data.get("model_loaded", False)
                device = data.get("device", "unknown")
                batch = data.get("batch_processing", False)
                print(f"  Model loaded: {loaded}")
                print(f"  Device: {device}")
                print(f"  Batch processing: {batch}")
                return loaded
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_result_table(results: List[RunResult], endpoint: str):
    hdr = (
        f"{'Workers':>8} | {'Reqs':>5} | {'OK':>4} | {'Fail':>4} | "
        f"{'p50':>9} | {'p95':>9} | {'p99':>9} | "
        f"{'Throughput':>12} | {'Batches':>8} | {'Avg Fill':>9}"
    )
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"  RESULTS  --  endpoint: /embed/{endpoint}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for r in results:
        print(
            f"{r.concurrency:>8} | {r.total_requests:>5} | {r.successful:>4} | {r.failed:>4} | "
            f"{fmt_ms(r.p50):>9} | {fmt_ms(r.p95):>9} | {fmt_ms(r.p99):>9} | "
            f"{fmt_rps(r.successful, r.wall_time) + ' req/s':>12} | "
            f"{r.batches_created:>8} | {r.avg_batch_fill:>8.1f}x"
        )

    print(sep)

    if any(r.errors for r in results):
        print("\nErrors (first 5 per concurrency level):")
        for r in results:
            for err in r.errors[:5]:
                print(f"  [concurrency={r.concurrency}] {err}")


def print_summary(results: List[RunResult]):
    if len(results) < 2:
        return
    base = results[0]
    peak = max(results, key=lambda r: r.throughput)
    print(f"\n  Peak throughput: {peak.throughput:.2f} req/s at concurrency={peak.concurrency}")
    if base.throughput > 0:
        speedup = peak.throughput / base.throughput
        print(f"  Speedup over single-worker: {speedup:.2f}x")
    print(f"  Peak batch fill: {peak.avg_batch_fill:.1f} requests/batch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Load-test the batched model server")
    parser.add_argument("--url", default="http://localhost:8001", help="Model server base URL")
    parser.add_argument("--concurrency", nargs="+", type=int, default=None,
                        help="Concurrency levels to test (e.g. 1 2 4 8)")
    parser.add_argument("--requests", type=int, default=20,
                        help="Total requests per concurrency level")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Test image width/height in pixels")
    parser.add_argument("--video-frames", type=int, default=8,
                        help="Frames per video clip")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout in seconds")
    parser.add_argument("--full", action="store_true",
                        help="Run a full sweep: concurrency 1,2,4,8,16 with 40 requests each")
    parser.add_argument("--include-video", action="store_true",
                        help="Also test /embed/video-clip endpoint")

    args = parser.parse_args()

    if args.full:
        concurrency_levels = [1, 2, 4, 8, 16]
        args.requests = 40
    elif args.concurrency:
        concurrency_levels = args.concurrency
    else:
        concurrency_levels = [1, 2, 4, 8]

    print("=" * 60)
    print("  Model Server Load Test")
    print("=" * 60)
    print(f"  URL:             {args.url}")
    print(f"  Concurrency:     {concurrency_levels}")
    print(f"  Requests/level:  {args.requests}")
    print(f"  Image size:      {args.image_size}x{args.image_size}")
    print(f"  Timeout:         {args.timeout}s")
    print()

    print("[1/3] Health check...")
    healthy = await check_health(args.url)
    if not healthy:
        sys.exit("Model server is not healthy. Start it first.")
    print()

    # --- Image endpoint ---
    print("[2/3] Testing /embed/image ...")
    image_results: List[RunResult] = []
    for c in concurrency_levels:
        print(f"\n  Concurrency = {c} ({args.requests} requests)")
        r = await run_concurrent_load(
            base_url=args.url,
            concurrency=c,
            total_requests=args.requests,
            endpoint="image",
            image_size=args.image_size,
            timeout=args.timeout,
        )
        image_results.append(r)
        print(f"    -> {r.successful}/{r.total_requests} OK, "
              f"p50={fmt_ms(r.p50)}, throughput={r.throughput:.2f} req/s")

    print_result_table(image_results, "image")
    print_summary(image_results)

    # --- Video clip endpoint ---
    if args.include_video:
        print(f"\n[3/3] Testing /embed/video-clip ({args.video_frames} frames/clip) ...")
        video_results: List[RunResult] = []
        for c in concurrency_levels:
            print(f"\n  Concurrency = {c} ({args.requests} requests)")
            r = await run_concurrent_load(
                base_url=args.url,
                concurrency=c,
                total_requests=args.requests,
                endpoint="video-clip",
                image_size=args.image_size,
                video_frames=args.video_frames,
                timeout=args.timeout,
            )
            video_results.append(r)
            print(f"    -> {r.successful}/{r.total_requests} OK, "
                  f"p50={fmt_ms(r.p50)}, throughput={r.throughput:.2f} req/s")

        print_result_table(video_results, "video-clip")
        print_summary(video_results)
    else:
        print("\n[3/3] Skipping video-clip test (use --include-video to enable)")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
