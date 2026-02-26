"""Helper script to publish embedding jobs to RabbitMQ queue.

This script demonstrates how to send video processing jobs to the
embedding worker via RabbitMQ.

Usage:
    python scripts/publish_embedding_job.py --video-url "https://example.com/video.mp4" --zone "perimeter_1"
    python scripts/publish_embedding_job.py --video-path "/path/to/local/video.mp4" --zone "gate_2"
    python scripts/publish_embedding_job.py --batch-file "jobs.json"
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pika
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def publish_job(
    channel,
    queue_name: str,
    job_data: Dict[str, Any],
    priority: int = 5
) -> None:
    """
    Publish a single job to RabbitMQ queue.

    Args:
        channel: RabbitMQ channel
        queue_name: Queue name
        job_data: Job data dictionary
        priority: Job priority (0-10, higher = more priority)
    """
    message = json.dumps(job_data, indent=2)

    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # Make message persistent
            priority=priority,
            content_type='application/json',
        )
    )

    print(f"\n✓ Published job to queue '{queue_name}':")
    print(f"  Video URL: {job_data.get('video_url')}")
    print(f"  Zone: {job_data.get('metadata', {}).get('zone', 'N/A')}")
    print(f"  Priority: {priority}")


def main():
    parser = argparse.ArgumentParser(
        description="Publish video embedding jobs to RabbitMQ queue"
    )

    # Job source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video-url",
        help="Video URL (http://, https://, or s3://)"
    )
    source_group.add_argument(
        "--video-path",
        help="Local video file path"
    )
    source_group.add_argument(
        "--batch-file",
        help="JSON file with multiple jobs (array of job objects)"
    )

    # Job metadata
    parser.add_argument("--zone", default="unknown", help="Zone/location identifier")
    parser.add_argument("--organization-id", help="Organization ID")
    parser.add_argument("--site-id", help="Site ID")
    parser.add_argument("--drone-id", help="Drone ID")
    parser.add_argument("--flight-id", help="Flight ID")

    # Processing options
    parser.add_argument(
        "--use-semantic-clips",
        action="store_true",
        help="Use semantic clips mode (groups frames into temporal clips)"
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=4.0,
        help="Clip duration in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=32,
        help="Max frames per clip (default: 32)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )
    parser.add_argument(
        "--save-full-frames",
        action="store_true",
        default=True,
        help="Save full-resolution frames (default: True)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep downloaded video after processing (default: cleanup)"
    )

    # Queue options
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        choices=range(0, 11),
        help="Job priority 0-10 (higher = more priority, default: 5)"
    )

    # RabbitMQ connection
    parser.add_argument("--rabbitmq-host", default=os.getenv("RABBITMQ_HOST", "localhost"))
    parser.add_argument("--rabbitmq-port", type=int, default=int(os.getenv("RABBITMQ_PORT", 5672)))
    parser.add_argument("--rabbitmq-user", default=os.getenv("RABBITMQ_USER", "guest"))
    parser.add_argument("--rabbitmq-password", default=os.getenv("RABBITMQ_PASSWORD", "guest"))
    parser.add_argument("--queue", default=os.getenv("RABBITMQ_QUEUE", "embedding.jobs"))

    args = parser.parse_args()

    # Connect to RabbitMQ
    print(f"Connecting to RabbitMQ at {args.rabbitmq_host}:{args.rabbitmq_port}...")
    credentials = pika.PlainCredentials(args.rabbitmq_user, args.rabbitmq_password)
    parameters = pika.ConnectionParameters(
        host=args.rabbitmq_host,
        port=args.rabbitmq_port,
        credentials=credentials,
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare queue (idempotent)
        # Note: If queue already exists with different arguments, this will fail
        # In that case, delete the queue first or use passive=True to just check existence
        try:
            channel.queue_declare(
                queue=args.queue,
                durable=True,
                arguments={
                    'x-message-ttl': 86400000,  # 24 hours
                    'x-max-length': 10000,
                    'x-max-priority': 10,  # Enable priority queue
                }
            )
        except pika.exceptions.ChannelClosedByBroker as e:
            if "PRECONDITION_FAILED" in str(e):
                # Queue exists with different arguments, just use it
                print(f"⚠ Queue '{args.queue}' exists with different configuration, using existing queue")
                # Reconnect since channel was closed
                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()
                # Declare passively (just check it exists)
                channel.queue_declare(queue=args.queue, passive=True)
            else:
                raise

        print(f"✓ Connected to queue: {args.queue}")

        # Handle batch file
        if args.batch_file:
            batch_path = Path(args.batch_file)
            if not batch_path.exists():
                print(f"✗ Batch file not found: {batch_path}")
                sys.exit(1)

            with open(batch_path, 'r') as f:
                jobs = json.load(f)

            if not isinstance(jobs, list):
                print("✗ Batch file must contain a JSON array of job objects")
                sys.exit(1)

            print(f"\nPublishing {len(jobs)} jobs from batch file...")
            for idx, job in enumerate(jobs):
                publish_job(channel, args.queue, job, priority=job.get("priority", args.priority))
                print(f"  [{idx+1}/{len(jobs)}] Published")

            print(f"\n✓ Successfully published {len(jobs)} jobs")

        else:
            # Single job
            # Build metadata
            metadata = {
                "zone": args.zone,
                "use_semantic_clips": args.use_semantic_clips,
                "clip_duration": args.clip_duration,
                "max_frames_per_clip": args.max_frames_per_clip,
                "batch_size": args.batch_size,
                "save_full_frames": args.save_full_frames,
                "cleanup_after": not args.no_cleanup,
            }

            # Add optional multi-site metadata
            if args.organization_id:
                metadata["organization_id"] = args.organization_id
            if args.site_id:
                metadata["site_id"] = args.site_id
            if args.drone_id:
                metadata["drone_id"] = args.drone_id
            if args.flight_id:
                metadata["flight_id"] = args.flight_id

            # Build job data
            job_data = {
                "video_url": args.video_url or args.video_path,
                "metadata": metadata,
            }

            # Publish
            publish_job(channel, args.queue, job_data, priority=args.priority)

        connection.close()
        print("\n✓ Done!")

    except pika.exceptions.AMQPConnectionError as e:
        print(f"\n✗ Failed to connect to RabbitMQ: {e}")
        print("  Please ensure RabbitMQ is running and accessible")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
