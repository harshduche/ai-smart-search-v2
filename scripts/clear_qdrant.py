#!/usr/bin/env python3
"""CLI script to delete/clear the Qdrant collection for this project."""

import sys
from pathlib import Path


def main() -> None:
    # Ensure project root is on sys.path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from search.vector_store import get_vector_store  # type: ignore

    print("Connecting to Qdrant and deleting collection...")
    vs = get_vector_store()
    vs.delete_collection(organization_id='org_security_footage')
    print("Done: Qdrant collection deleted.")


if __name__ == "__main__":
    main()

