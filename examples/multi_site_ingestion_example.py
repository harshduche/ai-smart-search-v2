#!/usr/bin/env python3
"""
Example: Multi-Site Drone Footage Ingestion

This script demonstrates how to ingest drone footage from multiple sites
within an organization using a single Qdrant collection with proper metadata.

Usage:
    python examples/multi_site_ingestion_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.ingest_pipeline import create_ingest_pipeline


def ingest_site_footage(
    pipeline,
    video_url: str,
    organization_id: str,
    site_id: str,
    site_name: str,
    drone_id: str,
    drone_model: str,
    flight_id: str,
    flight_purpose: str,
    zone: str = None,
    **kwargs
):
    """
    Ingest drone footage with comprehensive site metadata.

    The metadata will be embedded in each video clip, allowing for
    fast site-specific or cross-site searches.
    """

    # Build zone parameter (combines site_id and zone for uniqueness)
    zone_param = f"{site_id}_{zone}" if zone else site_id

    print(f"\n{'='*60}")
    print(f"Ingesting: {site_name}")
    print(f"{'='*60}")
    print(f"  Site ID: {site_id}")
    print(f"  Drone: {drone_id} ({drone_model})")
    print(f"  Flight: {flight_id}")
    print(f"  Purpose: {flight_purpose}")
    print(f"  Zone: {zone or 'N/A'}")
    print(f"  URL: {video_url[:60]}...")

    try:
        result = pipeline.ingest_video_from_url(
            video_url=video_url,
            zone=zone_param,
            **kwargs
        )

        print(f"\n✓ Success!")
        print(f"  Clips ingested: {result['clips_ingested']}")
        print(f"  File size: {result['file_size_mb']:.2f} MB")

        return result

    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return None


def main():
    """
    Example: Ingest drone footage from 3 different sites.

    Organization: FlytBase Security
    - Site A: Construction site in Mumbai
    - Site B: Warehouse in Delhi
    - Site C: Factory in Bangalore
    """

    print("\n" + "="*60)
    print("MULTI-SITE DRONE FOOTAGE INGESTION EXAMPLE")
    print("="*60)
    print("\nOrganization: FlytBase Security")
    print("Sites: 3 (Construction, Warehouse, Factory)")
    print("Collection: Single collection with site filtering")
    print("="*60)

    # Create pipeline (reuse for all sites)
    pipeline = create_ingest_pipeline()

    # Define organization
    organization_id = "flytbase_security"
    organization_name = "FlytBase Security"

    # -------------------------------------------------------------------------
    # SITE A: Construction Site - Mumbai
    # -------------------------------------------------------------------------

    site_a_result = ingest_site_footage(
        pipeline=pipeline,
        video_url="https://example.com/videos/construction_site_morning.mp4",
        organization_id=organization_id,
        site_id="construction_site_a",
        site_name="Construction Site A - Mumbai",
        drone_id="dji_mavic_001",
        drone_model="DJI Mavic 3 Enterprise",
        flight_id="flight_20240207_001",
        flight_purpose="perimeter_inspection",
        zone="north_perimeter",
        clip_duration=4.0,
        max_frames_per_clip=32,
        cleanup_after=True,
    )

    # -------------------------------------------------------------------------
    # SITE B: Warehouse - Delhi
    # -------------------------------------------------------------------------

    site_b_result = ingest_site_footage(
        pipeline=pipeline,
        video_url="https://example.com/videos/warehouse_roof.mp4",
        organization_id=organization_id,
        site_id="warehouse_b",
        site_name="Warehouse B - Delhi",
        drone_id="dji_phantom_002",
        drone_model="DJI Phantom 4 RTK",
        flight_id="flight_20240207_002",
        flight_purpose="roof_inspection",
        zone="roof",
        clip_duration=5.0,
        max_frames_per_clip=48,
        cleanup_after=True,
    )

    # -------------------------------------------------------------------------
    # SITE C: Factory - Bangalore
    # -------------------------------------------------------------------------

    site_c_result = ingest_site_footage(
        pipeline=pipeline,
        video_url="https://example.com/videos/factory_perimeter.mp4",
        organization_id=organization_id,
        site_id="factory_c",
        site_name="Factory C - Bangalore",
        drone_id="dji_mavic_003",
        drone_model="DJI Mavic 3 Enterprise",
        flight_id="flight_20240207_003",
        flight_purpose="security_patrol",
        zone="east_gate",
        clip_duration=4.0,
        max_frames_per_clip=32,
        cleanup_after=True,
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)

    results = [
        ("Site A (Construction - Mumbai)", site_a_result),
        ("Site B (Warehouse - Delhi)", site_b_result),
        ("Site C (Factory - Bangalore)", site_c_result),
    ]

    total_clips = 0
    successful_sites = 0

    for site_name, result in results:
        if result:
            clips = result['clips_ingested']
            total_clips += clips
            successful_sites += 1
            print(f"  ✓ {site_name}: {clips} clips")
        else:
            print(f"  ✗ {site_name}: Failed")

    print(f"\nTotal:")
    print(f"  Sites processed: {successful_sites}/{len(results)}")
    print(f"  Total clips: {total_clips}")

    # Get final collection stats
    stats = pipeline.get_stats()
    print(f"  Collection size: {stats['collection']['points_count']} vectors")

    # -------------------------------------------------------------------------
    # SEARCH EXAMPLES
    # -------------------------------------------------------------------------

    print("\n" + "="*60)
    print("SEARCH EXAMPLES")
    print("="*60)

    from search.vector_store import get_vector_store
    from ingestion.embedding_service import get_embedding_service

    vector_store = get_vector_store()
    embedding_service = get_embedding_service()

    query_text = "person near fence"
    query_embedding = embedding_service.embed_text(query_text)

    print(f"\nQuery: \"{query_text}\"\n")

    # Example 1: Search specific site
    print("1. Site-Specific Search (Construction Site A only):")
    results_site_a = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={"site_id": "construction_site_a"}
    )
    print(f"   Found {len(results_site_a)} results in Site A")

    # Example 2: Cross-site search
    print("\n2. Cross-Site Search (All sites in organization):")
    results_all = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={"organization_id": organization_id}
    )
    print(f"   Found {len(results_all)} results across all sites")

    # Example 3: Specific drone
    print("\n3. Drone-Specific Search (DJI Mavic 001 only):")
    results_drone = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={"drone_id": "dji_mavic_001"}
    )
    print(f"   Found {len(results_drone)} results from Mavic 001")

    # Example 4: Specific flight
    print("\n4. Flight-Specific Search (Flight 001 only):")
    results_flight = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={"flight_id": "flight_20240207_001"}
    )
    print(f"   Found {len(results_flight)} results from Flight 001")

    # Example 5: Multi-criteria
    print("\n5. Multi-Criteria Search (Site A + perimeter zones):")
    results_multi = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={
            "site_id": "construction_site_a",
            "zone": "construction_site_a_north_perimeter"
        }
    )
    print(f"   Found {len(results_multi)} results")

    print("\n" + "="*60)
    print("BENEFITS OF SINGLE COLLECTION:")
    print("="*60)
    print("✓ Fast site filtering (indexed fields)")
    print("✓ Cross-site search capability")
    print("✓ Simple management (one collection)")
    print("✓ Easy analytics across sites")
    print("✓ Unified model and infrastructure")
    print("="*60)


if __name__ == "__main__":
    print("""
    NOTE: This is an example script with placeholder URLs.

    To run with real videos:
    1. Replace the video URLs with your S3 pre-signed URLs
    2. Update site IDs, names, and metadata
    3. Run: python examples/multi_site_ingestion_example.py

    For batch ingestion from JSON:
    1. Create a JSON file with your sites (see examples/multi_site_urls.json)
    2. Run: python scripts/ingest_from_urls_batch.py your_sites.json
    """)

    # Uncomment to run the example
    # main()
