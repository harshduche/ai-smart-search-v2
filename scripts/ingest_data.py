#!/usr/bin/env python3
"""CLI script for ingesting video/image data into the vector store."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.ingest_pipeline import IngestPipeline, create_ingest_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest videos and images into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single video
  python ingest_data.py /path/to/video.mp4 --zone entrance

  # Ingest all media from a directory
  python ingest_data.py /path/to/media_folder

  # Ingest pre-extracted video frames with sampling
  python ingest_data.py /path/to/frames_folder --video-frames --sample-rate 5

  # Ingest frames with full metadata
  python ingest_data.py /path/to/frames --video-frames --sample-rate 10 \\
      --source-video-name "camera1.mp4" --original-fps 30 --zone parking

  # Download and ingest video from URL (S3, HTTP, etc.)
  python ingest_data.py "https://s3.amazonaws.com/bucket/video.mp4" --from-url --zone camera1

  # Keep downloaded file after ingestion
  python ingest_data.py "https://example.com/video.mp4" --from-url --no-cleanup
        """,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to video file, image file, or directory",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Zone/location identifier for the footage",
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Only process video files (for directory ingestion)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only process image files (for directory ingestion)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Path to save ingestion statistics",
    )

    # Frame quality options
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--save-full-frames",
        action="store_true",
        default=None,
        help="Save full-resolution frames for high-quality popup view (increases storage)",
    )
    quality_group.add_argument(
        "--no-save-full-frames",
        action="store_true",
        default=None,
        help="Only save compressed thumbnails (saves storage)",
    )

    # Video frames arguments
    frames_group = parser.add_argument_group(
        "Video Frames Options",
        "Options for ingesting pre-extracted video frames (images from a video)",
    )
    frames_group.add_argument(
        "--video-frames",
        action="store_true",
        help="Treat input directory as pre-extracted video frames",
    )
    frames_group.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Sample every Nth frame (default: 1 = all frames)",
    )
    frames_group.add_argument(
        "--source-video-name",
        type=str,
        default=None,
        help="Name to identify the source video in metadata",
    )
    frames_group.add_argument(
        "--original-fps",
        type=float,
        default=None,
        help="FPS of the original video for timestamp calculation (default: 30)",
    )
    frames_group.add_argument(
        "--base-timestamp",
        type=str,
        default=None,
        help="Base timestamp for first frame (ISO format: YYYY-MM-DDTHH:MM:SS)",
    )

    # Semantic video ingestion options (Qwen3 video clips)
    semantic_group = parser.add_argument_group(
        "Semantic Video Options",
        "Options for ingesting videos as short semantic clips using Qwen3-VL video embeddings",
    )
    semantic_group.add_argument(
        "--semantic-video",
        action="store_true",
        help="Ingest videos as semantic clips (2–5s) using Qwen3-VL video embeddings",
    )
    semantic_group.add_argument(
        "--clip-duration",
        type=float,
        default=4.0,
        help="Target clip duration in seconds for semantic video ingestion (default: 4.0)",
    )
    semantic_group.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=32,
        help="Maximum number of frames per semantic clip (default: 32)",
    )

    # URL ingestion options
    url_group = parser.add_argument_group(
        "URL Ingestion Options",
        "Options for downloading and ingesting videos from URLs (S3, HTTP, etc.)",
    )
    url_group.add_argument(
        "--from-url",
        action="store_true",
        help="Treat input_path as a video URL to download and ingest (uses semantic clips)",
    )
    url_group.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep downloaded video file after ingestion (default: cleanup after processing)",
    )

    args = parser.parse_args()

    # Determine save_full_frames setting (None means use config default)
    save_full_frames = None
    if args.save_full_frames:
        save_full_frames = True
    elif args.no_save_full_frames:
        save_full_frames = False

    # Handle URL ingestion separately
    if args.from_url:
        video_url = args.input_path  # Treat as URL, not file path

        # Validate URL format
        if not video_url.startswith(('http://', 'https://', 's3://')):
            print(f"Error: Invalid URL. Expected http://, https://, or s3://")
            print(f"Got: {video_url}")
            sys.exit(1)

        # Create pipeline
        pipeline = create_ingest_pipeline()

        try:
            result = pipeline.ingest_video_from_url(
                video_url=video_url,
                zone=args.zone,
                clip_duration=args.clip_duration,
                max_frames_per_clip=args.max_frames_per_clip,
                batch_size=args.batch_size,
                save_full_frames=save_full_frames,
                cleanup_after=not args.no_cleanup,
            )

            # Print results
            print(f"\nIngestion Results:")
            print(f"  - Clips ingested: {result['clips_ingested']}")
            print(f"  - File size: {result['file_size_mb']:.2f} MB")
            print(f"  - Video URL: {result['video_url']}")

            if args.stats_output:
                pipeline.save_stats(Path(args.stats_output))

            stats = pipeline.get_stats()
            print(f"\nFinal collection size: {stats['collection']['points_count']} vectors")

        except Exception as e:
            print(f"\nError during URL ingestion: {e}")
            raise

        return

    # Standard file/directory ingestion
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    # Create pipeline
    pipeline = create_ingest_pipeline()

    try:
        # Handle video frames ingestion mode
        if args.video_frames:
            if not input_path.is_dir():
                print("Error: --video-frames requires a directory path")
                sys.exit(1)

            # Parse base timestamp if provided
            base_timestamp = None
            if args.base_timestamp:
                try:
                    base_timestamp = datetime.fromisoformat(args.base_timestamp)
                except ValueError:
                    print(f"Error: Invalid timestamp format: {args.base_timestamp}")
                    print("Use ISO format: YYYY-MM-DDTHH:MM:SS")
                    sys.exit(1)

            pipeline.ingest_video_frames(
                frames_dir=input_path,
                sample_rate=args.sample_rate,
                source_video_name=args.source_video_name,
                original_fps=args.original_fps,
                zone=args.zone,
                base_timestamp=base_timestamp,
                batch_size=args.batch_size,
            )

        elif input_path.is_file():
            # Single file
            ext = input_path.suffix.lower()
            video_exts = {".mp4", ".avi", ".mov", ".mkv"}
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

            if ext in video_exts:
                if args.semantic_video:
                    pipeline.ingest_video_semantic_clips(
                        input_path,
                        zone=args.zone,
                        clip_duration=args.clip_duration,
                        max_frames_per_clip=args.max_frames_per_clip,
                        batch_size=args.batch_size,
                        save_full_frames=save_full_frames,
                    )
                else:
                    pipeline.ingest_video(
                        input_path,
                        zone=args.zone,
                        batch_size=args.batch_size,
                        save_full_frames=save_full_frames,
                    )
            elif ext in image_exts:
                pipeline.ingest_images([input_path], zone=args.zone, batch_size=args.batch_size)
            else:
                print(f"Error: Unsupported file type: {ext}")
                sys.exit(1)

        elif input_path.is_dir():
            # Directory - standard ingestion
            if args.semantic_video and not args.images_only:
                # Semantic video ingestion for all videos in the directory
                video_exts = {".mp4", ".avi", ".mov", ".mkv"}
                video_files = [
                    p for p in input_path.iterdir()
                    if p.is_file() and p.suffix.lower() in video_exts
                ]
                print(f"Found {len(video_files)} video files for semantic ingestion")
                for video_path in video_files:
                    pipeline.ingest_video_semantic_clips(
                        video_path,
                        zone=args.zone,
                        clip_duration=args.clip_duration,
                        max_frames_per_clip=args.max_frames_per_clip,
                        batch_size=args.batch_size,
                        save_full_frames=save_full_frames,
                    )
            else:
                pipeline.ingest_directory(
                    input_path,
                    process_videos=not args.images_only,
                    process_images=not args.videos_only,
                )

        # Save stats if requested
        if args.stats_output:
            pipeline.save_stats(Path(args.stats_output))

        # Print final stats
        stats = pipeline.get_stats()
        print(f"\nFinal collection size: {stats['collection']['points_count']} vectors")

    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
