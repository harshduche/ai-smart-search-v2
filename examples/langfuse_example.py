"""Example demonstrating Langfuse observability integration.

This script shows how to use Langfuse tracing in custom code.

Usage:
    # Make sure Langfuse is enabled in .env
    python examples/langfuse_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
from observability.langfuse_integration import (
    trace_operation,
    observe,
    trace_embedding_generation,
    flush_langfuse,
    LANGFUSE_ENABLED
)


def example_1_context_manager():
    """Example 1: Using trace_operation context manager."""
    print("\n" + "="*60)
    print("Example 1: Context Manager")
    print("="*60)

    with trace_operation(
        name="process-video-frames",
        operation_type="span",
        metadata={"zone": "perimeter_1", "frame_count": 10},
        tags=["example", "video-processing"]
    ) as trace:
        print("Processing video frames...")

        # Simulate frame extraction
        time.sleep(0.5)
        frames_extracted = 10

        # Nested operation
        with trace_operation(
            name="generate-embeddings",
            operation_type="generation",
            model="Qwen3-VL-Embedding-2B",
            metadata={"batch_size": frames_extracted},
            tags=["embedding"]
        ) as embed_trace:
            print("Generating embeddings...")
            time.sleep(1.0)
            embeddings_generated = frames_extracted

            if embed_trace:
                embed_trace.update(
                    output={"embeddings_count": embeddings_generated},
                    usage={"frames_processed": frames_extracted}
                )

        # Update parent trace
        if trace:
            trace.update(
                output={
                    "frames_extracted": frames_extracted,
                    "embeddings_generated": embeddings_generated,
                    "status": "success"
                }
            )

        print(f"✓ Processed {frames_extracted} frames")

    if not LANGFUSE_ENABLED:
        print("⚠ Langfuse is disabled - enable it in .env to see traces")


def example_2_decorator():
    """Example 2: Using @observe decorator."""
    print("\n" + "="*60)
    print("Example 2: Decorator")
    print("="*60)

    @observe(
        name="embed-batch-images",
        operation_type="generation",
        capture_input=True,
        capture_output=True,
        model="Qwen3-VL-Embedding-2B",
        metadata={"input_type": "image"},
        tags=["embedding", "batch"]
    )
    def embed_images(image_paths):
        """Simulate embedding generation."""
        print(f"Embedding {len(image_paths)} images...")
        time.sleep(1.5)
        return [f"embedding_{i}" for i in range(len(image_paths))]

    # Call the decorated function
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    embeddings = embed_images(images)

    print(f"✓ Generated {len(embeddings)} embeddings")


def example_3_specialized_decorator():
    """Example 3: Using specialized decorators."""
    print("\n" + "="*60)
    print("Example 3: Specialized Decorator")
    print("="*60)

    @trace_embedding_generation(
        model_name="Qwen3-VL-Embedding-2B",
        input_type="video_clip",
        batch_size=5
    )
    def embed_video_clip(frames):
        """Simulate video clip embedding."""
        print(f"Embedding video clip with {len(frames)} frames...")
        time.sleep(1.2)
        return [random.random() for _ in range(2048)]  # Mock 2048-D embedding

    # Call the decorated function
    frames = [f"frame_{i}" for i in range(5)]
    embedding = embed_video_clip(frames)

    print(f"✓ Generated embedding of dimension {len(embedding)}")


def example_4_error_handling():
    """Example 4: Error handling and logging."""
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60)

    with trace_operation(
        name="risky-operation",
        operation_type="span",
        metadata={"attempt": 1},
        tags=["example", "error-handling"]
    ) as trace:
        try:
            print("Attempting risky operation...")
            time.sleep(0.5)

            # Simulate random failure
            if random.random() > 0.7:
                raise ValueError("Random failure occurred!")

            print("✓ Operation succeeded")

            if trace:
                trace.update(
                    output={"status": "success"},
                    level="DEFAULT"
                )

        except Exception as e:
            print(f"✗ Operation failed: {e}")

            if trace:
                trace.update(
                    level="ERROR",
                    status_message=str(e),
                    output={"error": str(e)}
                )

            # Re-raise or handle
            print("  Handled gracefully")


def example_5_session_grouping():
    """Example 5: Grouping operations by session."""
    print("\n" + "="*60)
    print("Example 5: Session Grouping")
    print("="*60)

    flight_id = "flight_20260210_001"
    organization_id = "org_flytbase"

    # All operations in this session will be grouped
    with trace_operation(
        name="process-flight-data",
        operation_type="span",
        user_id=organization_id,
        session_id=flight_id,
        metadata={"drone_id": "drone_001", "zone": "perimeter_north"},
        tags=["flight", "batch-processing"]
    ) as trace:
        print(f"Processing flight: {flight_id}")

        # Operation 1
        with trace_operation(
            name="extract-keyframes",
            operation_type="span",
            session_id=flight_id  # Same session
        ) as op1:
            time.sleep(0.5)
            print("  - Extracted keyframes")
            if op1:
                op1.update(output={"keyframes": 20})

        # Operation 2
        with trace_operation(
            name="generate-embeddings",
            operation_type="generation",
            model="Qwen3-VL-Embedding-2B",
            session_id=flight_id  # Same session
        ) as op2:
            time.sleep(0.8)
            print("  - Generated embeddings")
            if op2:
                op2.update(output={"embeddings": 20})

        # Operation 3
        with trace_operation(
            name="store-results",
            operation_type="span",
            session_id=flight_id  # Same session
        ) as op3:
            time.sleep(0.3)
            print("  - Stored results")
            if op3:
                op3.update(output={"stored": 20})

        if trace:
            trace.update(
                output={"total_operations": 3, "status": "completed"}
            )

    print(f"✓ Completed flight processing")
    print(f"  View all operations for this flight by filtering session_id={flight_id}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Langfuse Observability Examples")
    print("="*60)

    if not LANGFUSE_ENABLED:
        print("\n⚠ WARNING: Langfuse is currently DISABLED")
        print("To enable:")
        print("  1. Set LANGFUSE_ENABLED=true in .env")
        print("  2. Add your LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("\nExamples will run but traces won't be sent to Langfuse.\n")
        time.sleep(2)
    else:
        print("\n✓ Langfuse is ENABLED")
        print("Traces will be sent to Langfuse dashboard\n")
        time.sleep(1)

    # Run all examples
    example_1_context_manager()
    example_2_decorator()
    example_3_specialized_decorator()
    example_4_error_handling()
    example_5_session_grouping()

    # Flush traces before exiting
    print("\n" + "="*60)
    print("Flushing Langfuse traces...")
    print("="*60)
    flush_langfuse()

    print("\n✓ All examples completed!")
    if LANGFUSE_ENABLED:
        print("\n🔗 View traces at: https://cloud.langfuse.com")
        print("   Navigate to: Traces → Your Project\n")


if __name__ == "__main__":
    main()
