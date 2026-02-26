#!/usr/bin/env python3
"""Main entry point to run the Visual Search Engine server."""

import uvicorn
import config
from pathlib import Path


def main():
    """Run the FastAPI server."""
    print("=" * 60)
    print("Visual Search Engine for Security Operations")
    print("=" * 60)
    print(f"Starting server at http://{config.API_HOST}:{config.API_PORT}")
    print(f"Dashboard: http://{config.API_HOST}:{config.API_PORT}/static/index.html")
    print(f"API Docs: http://{config.API_HOST}:{config.API_PORT}/docs")
    print("=" * 60)

    # When preloading models, disable hot-reload to prevent reload loops
    # (model cache files trigger WatchFiles reloads)
    # For development: set PRELOAD_MODELS=false for hot-reload with lazy loading
    use_reload = not config.PRELOAD_MODELS
    
    if use_reload:
        print("Hot-reload enabled (PRELOAD_MODELS=false)")
        # Only watch code directories, not model cache or data
        base_dir = Path(__file__).parent
        reload_dirs = [
            str(base_dir / "api"),
            str(base_dir / "ingestion"),
            str(base_dir / "search"),
            str(base_dir / "scripts"),
        ]
        uvicorn.run(
            "api.main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=True,
            reload_dirs=reload_dirs,
        )
    else:
        print("Hot-reload disabled (PRELOAD_MODELS=true)")
        uvicorn.run(
            "api.main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=False,
        )


if __name__ == "__main__":
    main()
