"""Langfuse integration for observability and monitoring.

This module provides centralized Langfuse tracing for:
- Embedding generation
- Video ingestion
- Search operations
- Worker job processing
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global flag to check if Langfuse is enabled
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

# Lazy imports to avoid errors if langfuse is not installed
if LANGFUSE_ENABLED:
    try:
        from langfuse import get_client
        langfuse = get_client()
        logger.info("Langfuse observability enabled")
    except ImportError:
        logger.warning("Langfuse enabled but langfuse package not installed. Install with: pip install langfuse")
        LANGFUSE_ENABLED = False
        langfuse = None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        LANGFUSE_ENABLED = False
        langfuse = None
else:
    langfuse = None
    logger.info("Langfuse observability disabled")


def get_langfuse_client():
    """Get the Langfuse client instance."""
    return langfuse if LANGFUSE_ENABLED else None


@contextmanager
def trace_operation(
    name: str,
    operation_type: str = "span",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    **kwargs
):
    """
    Context manager for tracing operations with Langfuse.

    Args:
        name: Name of the operation (e.g., "embed_video_clip", "search_vectors")
        operation_type: Type of observation ("span", "generation", or "event")
        user_id: User identifier
        session_id: Session identifier
        metadata: Additional metadata
        tags: Tags for filtering and grouping (note: tags are added to metadata, not as separate parameter)
        **kwargs: Additional arguments passed to langfuse.start_as_current_observation

    Usage:
        with trace_operation("embed-image", operation_type="generation", model="Qwen3-VL"):
            embedding = embed_image(image)
    """
    if not LANGFUSE_ENABLED or langfuse is None:
        # If Langfuse is disabled, just yield None and continue
        yield None
        return

    _yielded = False
    try:
        # Prepare observation parameters (only params accepted by start_as_current_observation)
        params = {
            "as_type": operation_type,
            "name": name,
        }

        # Combine metadata and tags
        combined_metadata = {}
        if metadata:
            combined_metadata.update(metadata)
        if tags:
            combined_metadata["tags"] = tags

        if combined_metadata:
            params["metadata"] = combined_metadata

        # Add any additional kwargs (like model, input, output, etc.)
        # But exclude user_id and session_id as they're trace-level attributes
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['user_id', 'session_id']}
        params.update(filtered_kwargs)

        # Start the observation
        with langfuse.start_as_current_observation(**params) as observation:
            # Set trace-level attributes after observation creation
            if observation and (user_id or session_id or tags):
                trace_updates = {}
                if user_id:
                    trace_updates["user_id"] = user_id
                if session_id:
                    trace_updates["session_id"] = session_id
                if tags:
                    trace_updates["tags"] = tags

                if trace_updates:
                    observation.update_trace(**trace_updates)

            _yielded = True
            yield observation

    except Exception as e:
        logger.error(f"Error in Langfuse tracing: {e}")
        if not _yielded:
            # Setup failed before yielding — yield None so caller still runs.
            yield None
        else:
            # Exception came from inside the caller's body (thrown into the
            # generator by @contextmanager). Re-raise so it propagates normally.
            raise


def observe(
    name: Optional[str] = None,
    operation_type: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
    **trace_kwargs
):
    """
    Decorator for tracing functions with Langfuse.

    Args:
        name: Name of the operation (defaults to function name)
        operation_type: Type of observation ("span", "generation", or "event")
        capture_input: Whether to capture function arguments as input
        capture_output: Whether to capture function result as output
        **trace_kwargs: Additional kwargs passed to trace_operation

    Usage:
        @observe(name="embed-batch", operation_type="generation", model="Qwen3-VL")
        def embed_images_batch(images: List[PIL.Image]) -> np.ndarray:
            return embeddings
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not LANGFUSE_ENABLED:
                return func(*args, **kwargs)

            # Determine operation name
            op_name = name if name else func.__name__

            # Prepare input capture
            input_data = None
            if capture_input:
                input_data = {
                    "args": str(args)[:500],  # Truncate to avoid huge payloads
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
                }

            try:
                with trace_operation(
                    name=op_name,
                    operation_type=operation_type,
                    input=input_data,
                    **trace_kwargs
                ) as observation:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Capture output
                    if observation and capture_output:
                        output_data = str(result)[:500] if result is not None else None
                        observation.update(output=output_data)

                    return result

            except Exception as e:
                # Log the error to Langfuse
                if LANGFUSE_ENABLED and langfuse:
                    logger.error(f"Error in {op_name}: {e}")
                raise

        return wrapper
    return decorator


def trace_embedding_generation(
    model_name: str,
    input_type: str,  # "text", "image", "video_clip", "multimodal"
    batch_size: Optional[int] = None,
):
    """
    Specialized decorator for tracing embedding generation.

    Args:
        model_name: Name of the embedding model
        input_type: Type of input being embedded
        batch_size: Batch size if processing multiple items

    Usage:
        @trace_embedding_generation(model_name="Qwen3-VL-Embedding-2B", input_type="image")
        def embed_image(image: PIL.Image) -> np.ndarray:
            return embedding
    """
    metadata = {
        "input_type": input_type,
        "embedding_dim": 2048,
    }
    if batch_size:
        metadata["batch_size"] = batch_size

    return observe(
        operation_type="generation",
        model=model_name,
        metadata=metadata,
        tags=["embedding", input_type],
    )


def trace_search(
    search_type: str,  # "text", "image", "multimodal", "ocr"
    top_k: int,
    use_reranker: bool = False,
):
    """
    Specialized decorator for tracing search operations.

    Args:
        search_type: Type of search query
        top_k: Number of results requested
        use_reranker: Whether reranking is used

    Usage:
        @trace_search(search_type="text", top_k=20, use_reranker=True)
        def search_text(query: str) -> List[SearchResult]:
            return results
    """
    metadata = {
        "search_type": search_type,
        "top_k": top_k,
        "use_reranker": use_reranker,
    }

    return observe(
        name=f"search-{search_type}",
        operation_type="span",
        metadata=metadata,
        tags=["search", search_type],
    )


def trace_ingestion(
    source_type: str,  # "video", "image", "url", "rtsp"
    use_semantic_clips: bool = False,
):
    """
    Specialized decorator for tracing ingestion operations.

    Args:
        source_type: Type of source being ingested
        use_semantic_clips: Whether using semantic clips mode

    Usage:
        @trace_ingestion(source_type="video", use_semantic_clips=True)
        def ingest_video(video_path: Path) -> int:
            return frames_processed
    """
    metadata = {
        "source_type": source_type,
        "use_semantic_clips": use_semantic_clips,
    }

    return observe(
        name=f"ingest-{source_type}",
        operation_type="span",
        metadata=metadata,
        tags=["ingestion", source_type],
    )


def flush_langfuse():
    """
    Flush pending Langfuse events.

    Call this at the end of short-lived applications (e.g., CLI scripts, workers)
    to ensure all events are sent before the process exits.
    """
    if LANGFUSE_ENABLED and langfuse:
        try:
            langfuse.flush()
            logger.debug("Langfuse events flushed")
        except Exception as e:
            logger.error(f"Error flushing Langfuse: {e}")


def shutdown_langfuse():
    """
    Shutdown Langfuse client.

    Alias for flush_langfuse() for consistency with the SDK documentation.
    """
    flush_langfuse()
