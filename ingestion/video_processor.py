"""Video processing module for frame extraction."""

import cv2
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Generator, Tuple
from PIL import Image
from tqdm import tqdm

import config


class VideoProcessor:
    """Process videos and extract frames."""

    def __init__(
        self,
        frame_rate: float = None,
        thumbnail_size: int = None,
        output_dir: Path = None,
    ):
        """
        Initialize the video processor.

        Args:
            frame_rate: Frames per second to extract (default from config)
            thumbnail_size: Size of thumbnails (default from config)
            output_dir: Directory to save extracted frames
        """
        self.frame_rate = frame_rate or config.FRAME_RATE
        self.thumbnail_size = thumbnail_size or config.THUMBNAIL_SIZE
        self.frames_dir = output_dir or config.FRAMES_DIR
        self.thumbnails_dir = config.THUMBNAILS_DIR

        # Ensure output directories exist
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def is_long_video(self, video_path: Path) -> bool:
        """
        Check if a video is considered "long" based on duration threshold.

        Args:
            video_path: Path to video file

        Returns:
            True if video duration exceeds LONG_VIDEO_THRESHOLD_SECONDS
        """
        info = self.get_video_info(video_path)
        return info['duration_seconds'] > config.LONG_VIDEO_THRESHOLD_SECONDS

    def get_optimal_frame_rate(self, video_path: Path) -> float:
        """
        Get the optimal frame rate for a video based on its duration.

        For videos longer than the threshold, returns LONG_VIDEO_FRAME_RATE.
        Otherwise, returns the standard FRAME_RATE.

        Args:
            video_path: Path to video file

        Returns:
            Optimal frame rate in fps
        """
        if self.is_long_video(video_path):
            return config.LONG_VIDEO_FRAME_RATE
        return self.frame_rate

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Dict with video info (duration, fps, frame_count, resolution)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "path": str(video_path),
            "filename": video_path.name,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "duration_formatted": str(timedelta(seconds=int(duration))),
        }

    def extract_frames(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        zone: Optional[str] = None,
        base_timestamp: Optional[datetime] = None,
    ) -> Generator[Tuple[Image.Image, Dict[str, Any]], None, None]:
        """
        Extract frames from a video at the specified frame rate.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            zone: Zone/location identifier for metadata
            base_timestamp: Base timestamp for the video start

        Yields:
            Tuple of (PIL Image, metadata dict)
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval for desired extraction rate
        frame_interval = int(fps / self.frame_rate) if self.frame_rate < fps else 1

        # Calculate start and end frames
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames

        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Base timestamp (use file modification time if not provided)
        if base_timestamp is None:
            try:
                file_mtime = os.path.getmtime(video_path)
                base_timestamp = datetime.fromtimestamp(file_mtime)
            except:
                base_timestamp = datetime.now()

        frame_number = start_frame
        extracted_count = 0

        while frame_number < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # Only process at the specified interval
            if (frame_number - start_frame) % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Calculate timestamp for this frame
                seconds_offset = frame_number / fps
                frame_timestamp = base_timestamp + timedelta(seconds=seconds_offset)

                # Determine if it's night (simple heuristic based on average brightness)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = gray.mean()
                is_night = avg_brightness < 50  # Threshold for darkness

                # Build metadata
                metadata = {
                    "source_file": video_path.name,
                    "video_path": str(video_path),
                    "frame_number": extracted_count,
                    "original_frame_number": frame_number,
                    "timestamp": frame_timestamp.isoformat(),
                    "seconds_offset": seconds_offset,
                    "zone": zone or "unknown",
                    "is_night": bool(is_night),
                    "avg_brightness": float(avg_brightness),
                }

                extracted_count += 1
                yield pil_image, metadata

            frame_number += 1

        cap.release()

    def extract_and_save_frames(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        zone: Optional[str] = None,
        base_timestamp: Optional[datetime] = None,
        save_full_frames: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract frames and save thumbnails to disk.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            zone: Zone/location identifier
            base_timestamp: Base timestamp for video start
            save_full_frames: Whether to save full-resolution frames

        Returns:
            List of metadata dicts for each extracted frame
        """
        video_path = Path(video_path)
        video_info = self.get_video_info(video_path)

        print(f"Processing: {video_path.name}")
        print(f"Duration: {video_info['duration_formatted']}")
        print(f"Expected frames at {self.frame_rate} fps: ~{int(video_info['duration_seconds'] * self.frame_rate)}")

        # Create subdirectory for this video
        video_name = video_path.stem
        video_frames_dir = self.frames_dir / video_name
        video_thumbs_dir = self.thumbnails_dir / video_name
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        video_thumbs_dir.mkdir(parents=True, exist_ok=True)

        frames_metadata = []

        # Extract frames with progress bar
        frame_generator = self.extract_frames(
            video_path,
            start_time=start_time,
            end_time=end_time,
            zone=zone,
            base_timestamp=base_timestamp,
        )

        for pil_image, metadata in tqdm(
            frame_generator,
            desc=f"Extracting {video_name}",
            total=int(video_info["duration_seconds"] * self.frame_rate),
        ):
            frame_num = metadata["frame_number"]

            # Generate thumbnail
            thumbnail = pil_image.copy()
            thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size))

            # Save thumbnail
            thumb_path = video_thumbs_dir / f"frame_{frame_num:06d}.jpg"
            thumbnail.save(thumb_path, "JPEG", quality=85)
            metadata["thumbnail_path"] = str(thumb_path)

            # Optionally save full frame
            if save_full_frames:
                frame_path = video_frames_dir / f"frame_{frame_num:06d}.jpg"
                pil_image.save(frame_path, "JPEG", quality=95)
                metadata["frame_path"] = str(frame_path)

            # Store PIL image reference for embedding (will be used by pipeline)
            metadata["_pil_image"] = pil_image

            frames_metadata.append(metadata)

        print(f"Extracted {len(frames_metadata)} frames from {video_path.name}")
        return frames_metadata

    def process_video_batch(
        self,
        video_paths: List[Path],
        zone_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos.

        Args:
            video_paths: List of video file paths
            zone_mapping: Optional mapping of filename to zone

        Returns:
            Combined list of all frame metadata
        """
        all_metadata = []

        for video_path in video_paths:
            video_path = Path(video_path)
            zone = None

            if zone_mapping:
                zone = zone_mapping.get(video_path.name)

            frames_meta = self.extract_and_save_frames(
                video_path,
                zone=zone,
            )
            all_metadata.extend(frames_meta)

        return all_metadata

    def extract_frames_from_stream(
        self,
        rtsp_url: str,
        duration_seconds: Optional[float] = None,
        max_frames: Optional[int] = None,
        zone: Optional[str] = None,
        reconnect_on_failure: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2.0,
        connection_timeout_seconds: float = 10.0,
    ) -> Generator[Tuple[Image.Image, Dict[str, Any]], None, None]:
        """
        Extract frames from an RTSP live stream.

        Key differences from extract_frames():
        - Handles infinite streams (vs fixed-duration videos)
        - Implements reconnection logic
        - Duration-based or frame-count-based termination
        - Real-time timestamp generation (no file modification time)

        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://192.168.1.100:554/stream)
            duration_seconds: Optional duration to capture (None = until max_frames)
            max_frames: Optional maximum number of frames (None = until duration)
            zone: Zone/location identifier
            reconnect_on_failure: Attempt reconnection on stream failure
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay_seconds: Delay between reconnection attempts
            connection_timeout_seconds: Timeout for initial connection

        Yields:
            Tuple of (PIL Image, metadata dict)

        Raises:
            ValueError: If connection fails after all retry attempts
            RuntimeError: If both duration_seconds and max_frames are None
        """
        import time

        # Validation: at least one termination condition must be set
        if duration_seconds is None and max_frames is None:
            raise RuntimeError(
                "Must specify at least one termination condition: "
                "duration_seconds or max_frames"
            )

        base_timestamp = datetime.now()
        start_time = time.time()
        frame_count = 0
        reconnect_attempts = 0
        cap = None

        def connect_to_stream():
            """Attempt to connect to RTSP stream."""
            nonlocal reconnect_attempts

            # Set OpenCV RTSP options for better streaming
            stream_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            stream_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            # Wait for connection with timeout
            timeout_start = time.time()
            while not stream_cap.isOpened():
                if time.time() - timeout_start > connection_timeout_seconds:
                    stream_cap.release()
                    return None
                time.sleep(0.1)

            # Verify we can read a frame
            ret, test_frame = stream_cap.read()
            if not ret or test_frame is None:
                stream_cap.release()
                return None

            return stream_cap

        # Initial connection
        print(f"Connecting to RTSP stream: {rtsp_url}")
        cap = connect_to_stream()

        if cap is None:
            raise ValueError(f"Cannot connect to RTSP stream: {rtsp_url}")

        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:  # Sanity check
            fps = 30.0  # Default for most IP cameras

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Stream connected: {width}x{height} @ {fps} fps")
        print(f"Capture settings: duration={duration_seconds}s, max_frames={max_frames}")

        # Calculate frame interval for desired extraction rate
        frame_interval = int(fps / self.frame_rate) if self.frame_rate < fps else 1
        frames_since_last_extract = 0
        extracted_count = 0

        try:
            while True:
                # Check termination conditions
                elapsed_time = time.time() - start_time

                if duration_seconds is not None and elapsed_time >= duration_seconds:
                    print(f"\nCapture complete: reached duration limit ({duration_seconds}s)")
                    break

                if max_frames is not None and extracted_count >= max_frames:
                    print(f"\nCapture complete: reached frame limit ({max_frames} frames)")
                    break

                # Read frame
                ret, frame = cap.read()

                # Handle read failure (stream drop)
                if not ret or frame is None:
                    print(f"\nStream read failed (frame {frame_count})")

                    if not reconnect_on_failure:
                        print("Reconnection disabled, stopping capture")
                        break

                    if reconnect_attempts >= max_reconnect_attempts:
                        print(f"Max reconnection attempts reached ({max_reconnect_attempts})")
                        raise ValueError(
                            f"Stream connection lost after {reconnect_attempts} attempts"
                        )

                    # Attempt reconnection
                    reconnect_attempts += 1
                    print(f"Attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")

                    cap.release()
                    time.sleep(reconnect_delay_seconds)

                    cap = connect_to_stream()
                    if cap is None:
                        print(f"Reconnection attempt {reconnect_attempts} failed")
                        continue

                    print("Reconnection successful")
                    reconnect_attempts = 0  # Reset counter on success
                    continue

                frame_count += 1
                frames_since_last_extract += 1

                # Only process at the specified interval
                if frames_since_last_extract >= frame_interval:
                    frames_since_last_extract = 0

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Calculate real-time timestamp
                    seconds_offset = time.time() - start_time
                    frame_timestamp = base_timestamp + timedelta(seconds=seconds_offset)

                    # Night detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = gray.mean()
                    is_night = avg_brightness < 50

                    # Build metadata
                    metadata = {
                        "source_file": f"rtsp_stream_{zone or 'unknown'}",
                        "video_path": rtsp_url,
                        "frame_number": extracted_count,
                        "original_frame_number": frame_count,
                        "timestamp": frame_timestamp.isoformat(),
                        "seconds_offset": seconds_offset,
                        "zone": zone or "unknown",
                        "is_night": bool(is_night),
                        "avg_brightness": float(avg_brightness),
                        "source_type": "rtsp_stream",
                        "stream_url": rtsp_url,
                    }

                    extracted_count += 1
                    yield pil_image, metadata

        finally:
            # Always release the stream
            if cap is not None:
                cap.release()
                print(f"\nStream released. Total frames extracted: {extracted_count}")


def get_video_files(directory: Path, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    """
    Find all video files in a directory.

    Args:
        directory: Directory to search
        extensions: Video file extensions to look for

    Returns:
        List of video file paths
    """
    directory = Path(directory)
    video_files = []

    for ext in extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(video_files)
