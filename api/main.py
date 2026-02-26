"""FastAPI application for the Visual Search Engine."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    
    Preloads models at startup to keep them warm on CUDA for low latency.
    Set PRELOAD_MODELS=false to disable (faster restarts during development).
    """
    import torch
    
    if config.PRELOAD_MODELS:
        print("=" * 60)
        print("Preloading models at startup (set PRELOAD_MODELS=false to skip)")
        print("=" * 60)
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Enable cuDNN auto-tuner for faster convolutions
            torch.backends.cudnn.benchmark = True
            # Use TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Preload embedding model
        print("\n[1/3] Loading embedding model...")
        from ingestion.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        embedding_service.initialize()
        
        # Warmup embedding model with a dummy request (first inference is slower)
        if not embedding_service.is_mock():
            print("  Warming up embedding model...")
            _ = embedding_service.embed_text("warmup query")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Preload reranker model (if enabled)
        if config.USE_RERANKER:
            print("\n[2/3] Loading reranker model...")
            from search.reranker_service import get_reranker_service
            reranker_service = get_reranker_service()
            reranker_service.initialize()
        else:
            print("\n[2/3] Reranker disabled (set USE_RERANKER=true to enable)")
        
        # Initialize vector store connection
        print("\n[3/3] Connecting to Qdrant...")
        from search.vector_store import get_vector_store
        vector_store = get_vector_store()
        vector_store._connect()
        
        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nCUDA memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        print("\n" + "=" * 60)
        print("All models loaded and ready! Server is accepting requests.")
        print("=" * 60 + "\n")
    else:
        print("=" * 60)
        print("Skipping model preload (PRELOAD_MODELS=false)")
        print("Models will load on first request.")
        print("=" * 60 + "\n")
    
    yield  # Server runs here
    
    # Cleanup on shutdown
    print("\nShutting down - cleaning up resources...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


from api.routes import search, health, ingest, geo as geo_route

# Create FastAPI app with lifespan
app = FastAPI(
    title="Visual Search Engine",
    description="""
    AI-powered visual search system for security operations.

    ## Features
    - **Text Search**: Search footage using natural language queries
    - **Image Search**: Find similar scenes using reference images
    - **Multimodal Search**: Combine text and image for precise queries
    - **OCR Search**: Find text visible in footage (license plates, signs)

    ## Powered by
    - Qwen3-VL-Embedding for multimodal embeddings
    - Qdrant for vector storage and search
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # Preload models at startup
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(ingest.router)
app.include_router(geo_route.router)

# Mount static files for thumbnails, frames, and raw videos
try:
    app.mount(
        "/thumbnails",
        StaticFiles(directory=str(config.THUMBNAILS_DIR)),
        name="thumbnails",
    )
except Exception:
    print("Warning: Thumbnails directory not yet created")

try:
    app.mount(
        "/frames",
        StaticFiles(directory=str(config.FRAMES_DIR)),
        name="frames",
    )
except Exception:
    print("Warning: Frames directory not yet created")

try:
    app.mount(
        "/raw",
        StaticFiles(directory=str(config.RAW_DATA_DIR)),
        name="raw",
    )
except Exception:
    print("Warning: RAW video directory not yet created")

# Mount frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_dir)),
        name="static",
    )


@app.get("/")
async def root():
    """Redirect to the dashboard or return API info."""
    return {
        "name": "Visual Search Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/static/index.html",
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
