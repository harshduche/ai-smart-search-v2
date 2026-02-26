#!/usr/bin/env python3
"""Demo script showcasing 15 required search scenarios."""

import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.search_service import SearchService, get_search_service


# Required search scenarios from the hackathon problem statement
DEMO_SCENARIOS = [
    # Object & People Detection
    {
        "id": 1,
        "category": "Object & People Detection",
        "query": "people near the perimeter fence",
        "description": "Find all footage showing people near the perimeter fence",
    },
    {
        "id": 2,
        "category": "Object & People Detection",
        "query": "vehicles in parking zones or unauthorized areas",
        "description": "Locate vehicles in unauthorized parking zones",
    },
    {
        "id": 3,
        "category": "Object & People Detection",
        "query": "ladders or climbing equipment",
        "description": "Search for images containing ladders or climbing equipment",
    },
    {
        "id": 4,
        "category": "Object & People Detection",
        "query": "unattended bags or suspicious packages",
        "description": "Find scenes with unattended bags or suspicious packages",
    },
    {
        "id": 5,
        "category": "Object & People Detection",
        "query": "person wearing safety vest or high visibility clothing",
        "description": "Show all instances of people wearing safety vests",
    },

    # Situational Awareness
    {
        "id": 6,
        "category": "Situational Awareness",
        "query": "nighttime footage from the main gate area",
        "description": "Show me all nighttime footage from the main gate",
        "filters": {"is_night": True},
    },
    {
        "id": 7,
        "category": "Situational Awareness",
        "query": "empty security post or guard station without personnel",
        "description": "Find instances where security guards are not at their posts",
    },
    {
        "id": 8,
        "category": "Situational Awareness",
        "query": "open gate or unlocked entrance",
        "description": "Locate footage showing gates left open",
    },
    {
        "id": 9,
        "category": "Situational Awareness",
        "query": "crowd or gathering of multiple people",
        "description": "Search for crowding or gatherings of multiple people",
    },
    {
        "id": 10,
        "category": "Situational Awareness",
        "query": "rain or adverse weather conditions outdoor",
        "description": "Find all footage during rain or adverse weather conditions",
    },

    # Incident Investigation
    {
        "id": 11,
        "category": "Incident Investigation",
        "query": "security breach or intrusion at fence",
        "description": "Find all footage similar to a security breach scenario",
    },
    {
        "id": 12,
        "category": "Incident Investigation",
        "query": "damage to fence or broken infrastructure",
        "description": "Show damage to fence sections or infrastructure",
    },
    {
        "id": 13,
        "category": "Incident Investigation",
        "query": "white sedan or light colored car",
        "description": "Locate footage showing a white sedan",
    },
    {
        "id": 14,
        "category": "Incident Investigation",
        "query": "emergency vehicle or ambulance or fire truck",
        "description": "Find all incidents involving emergency vehicles",
    },
    {
        "id": 15,
        "category": "Incident Investigation",
        "query": "unusual activity or anomalous behavior",
        "description": "Search for unusual or anomalous activity",
    },
]


def run_demo():
    """Run all demo scenarios and display results."""
    print("=" * 70)
    print("Visual Search Engine - Demo Scenarios")
    print("=" * 70)
    print()

    search_service = get_search_service()

    # Get collection stats
    stats = search_service.get_stats()
    print(f"Collection: {stats['collection'].get('name', 'unknown')}")
    print(f"Total vectors: {stats['collection'].get('points_count', 0)}")
    print(f"Using mock embeddings: {stats['using_mock_embeddings']}")
    print()
    print("=" * 70)

    results_summary = []

    for scenario in DEMO_SCENARIOS:
        print(f"\n{'='*70}")
        print(f"Scenario #{scenario['id']}: {scenario['category']}")
        print(f"{'='*70}")
        print(f"Description: {scenario['description']}")
        print(f"Query: \"{scenario['query']}\"")

        filters = scenario.get("filters")
        if filters:
            print(f"Filters: {filters}")

        print("-" * 40)

        # Run search
        start_time = time.time()
        try:
            results = search_service.search_text(
                query=scenario["query"],
                top_k=5,
                filters=filters,
            )
            search_time = (time.time() - start_time) * 1000

            print(f"Results: {len(results)} matches (in {search_time:.0f}ms)")

            if results:
                print("\nTop 3 results:")
                for i, r in enumerate(results[:3], 1):
                    print(f"  {i}. Score: {r['score']:.3f} | "
                          f"Source: {r.get('source_file', 'unknown')} | "
                          f"Frame: {r.get('frame_number', '?')}")

            results_summary.append({
                "scenario_id": scenario["id"],
                "query": scenario["query"],
                "num_results": len(results),
                "search_time_ms": round(search_time, 2),
                "success": len(results) > 0,
            })

        except Exception as e:
            print(f"Error: {e}")
            results_summary.append({
                "scenario_id": scenario["id"],
                "query": scenario["query"],
                "error": str(e),
                "success": False,
            })

    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results_summary if r.get("success"))
    total = len(results_summary)
    avg_time = sum(r.get("search_time_ms", 0) for r in results_summary) / total

    print(f"Scenarios run: {total}")
    print(f"Successful searches: {successful}/{total}")
    print(f"Average search time: {avg_time:.0f}ms")

    # Save results
    output_path = Path(__file__).parent.parent / "demo_results.json"
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_demo()
