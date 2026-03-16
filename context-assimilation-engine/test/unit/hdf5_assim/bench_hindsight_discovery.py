#!/usr/bin/env python3
"""Hindsight Dataset Discovery Benchmark.

Equivalent to bench_hdf5_discovery.cc but targeting Hindsight's memory system.
Uses the same 20 dataset descriptions and 20 queries, reports identical metrics.

Requirements:
  - Hindsight server running (default: http://localhost:8888)
  - LLM endpoint for Hindsight's memory processing

Usage:
    export HINDSIGHT_BASE_URL=http://localhost:8888   # optional
    python3 bench_hindsight_discovery.py
"""

import json
import sys
import time
import requests

HINDSIGHT_BASE_URL = __import__("os").environ.get(
    "HINDSIGHT_BASE_URL", "http://localhost:8888"
)
BANK_ID = "hdf5_bench_discovery"
API_PREFIX = f"{HINDSIGHT_BASE_URL}/v1/default/banks/{BANK_ID}"

# ============================================================
# Same 20 datasets from the CTE benchmark
# ============================================================
DATASETS = [
    {"tag_name": "weather_forecast", "description":
     "Global weather forecast data including temperature, humidity, wind speed, "
     "and atmospheric pressure measurements from 500 weather stations across "
     "North America recorded hourly"},
    {"tag_name": "particle_physics", "description":
     "High-energy particle collision events from the Large Hadron Collider "
     "including particle trajectories, energy deposits, and detector hit "
     "patterns for proton-proton collisions at 13 TeV"},
    {"tag_name": "ocean_temperature", "description":
     "Sea surface temperature measurements from NOAA satellite observations "
     "covering the Pacific Ocean basin with 0.25 degree spatial resolution "
     "daily averages"},
    {"tag_name": "genome_sequences", "description":
     "Human genome variant call data from whole genome sequencing including "
     "SNP positions, allele frequencies, and quality scores across "
     "chromosome 1-22"},
    {"tag_name": "stock_market", "description":
     "Historical stock market trading data including open, high, low, close "
     "prices and volume for S&P 500 companies from 2010 to 2024 at minute "
     "resolution"},
    {"tag_name": "brain_imaging", "description":
     "Functional MRI brain scan data from cognitive neuroscience experiments "
     "showing blood oxygen level dependent signals across cortical regions "
     "during memory tasks"},
    {"tag_name": "climate_model", "description":
     "Earth system climate model output with global CO2 concentration, "
     "radiative forcing, and temperature anomaly projections under RCP 8.5 "
     "scenario from 2020 to 2100"},
    {"tag_name": "seismic_waves", "description":
     "Earthquake seismograph recordings from the Pacific Ring of Fire "
     "including P-wave and S-wave arrival times, magnitude estimates, and "
     "focal mechanism solutions"},
    {"tag_name": "protein_structure", "description":
     "3D protein structure coordinates from X-ray crystallography including "
     "atomic positions, B-factors, and secondary structure annotations for "
     "enzyme active sites"},
    {"tag_name": "satellite_imagery", "description":
     "Multispectral satellite imagery from Landsat 8 with red, green, blue, "
     "near-infrared, and shortwave infrared bands at 30 meter ground "
     "resolution"},
    {"tag_name": "wind_turbine", "description":
     "Wind turbine performance metrics including power output, rotor speed, "
     "blade pitch angle, nacelle temperature, and vibration sensor data from "
     "an offshore wind farm"},
    {"tag_name": "drug_trials", "description":
     "Clinical drug trial results with patient demographics, dosage levels, "
     "biomarker measurements, adverse events, and efficacy endpoints for a "
     "phase III cardiovascular study"},
    {"tag_name": "traffic_flow", "description":
     "Urban traffic flow sensor data from highway loop detectors measuring "
     "vehicle count, average speed, and lane occupancy at 5-minute intervals "
     "across 200 intersections"},
    {"tag_name": "soil_moisture", "description":
     "Agricultural soil moisture measurements from wireless sensor networks "
     "at multiple depths including volumetric water content, soil temperature, "
     "and electrical conductivity"},
    {"tag_name": "audio_speech", "description":
     "Speech recognition training data with mel-frequency cepstral "
     "coefficients, phoneme labels, speaker embeddings, and acoustic features "
     "extracted from 1000 hours of English speech"},
    {"tag_name": "galaxy_survey", "description":
     "Astronomical galaxy survey catalog with redshift measurements, "
     "photometric magnitudes, morphological classifications, and stellar mass "
     "estimates for 2 million galaxies"},
    {"tag_name": "battery_cycling", "description":
     "Lithium-ion battery charge-discharge cycling data including voltage, "
     "current, capacity, impedance spectroscopy, and temperature measurements "
     "over 500 cycles"},
    {"tag_name": "air_quality", "description":
     "Urban air quality monitoring station data with particulate matter PM2.5 "
     "and PM10, ozone, nitrogen dioxide, sulfur dioxide concentrations, and "
     "meteorological conditions"},
    {"tag_name": "neural_network", "description":
     "Deep neural network training checkpoint with model weights, gradient "
     "statistics, optimizer state, learning rate schedule, and per-layer "
     "activation distributions"},
    {"tag_name": "dna_methylation", "description":
     "Epigenetic DNA methylation array data from Illumina EPIC BeadChip with "
     "beta values, detection p-values, and probe annotations across 850,000 "
     "CpG sites"},
]

# ============================================================
# Same 20 queries from the CTE benchmark
# ============================================================
QUERIES = [
    {"text": "Find the weather forecast dataset",
     "expected_tag": "weather_forecast", "category": "exact"},
    {"text": "Locate particle collision data from CERN",
     "expected_tag": "particle_physics", "category": "synonym"},
    {"text": "Where is the ocean temperature data?",
     "expected_tag": "ocean_temperature", "category": "exact"},
    {"text": "Show me the genomics sequencing results",
     "expected_tag": "genome_sequences", "category": "synonym"},
    {"text": "Find stock trading historical prices",
     "expected_tag": "stock_market", "category": "synonym"},
    {"text": "Locate the fMRI brain scan data",
     "expected_tag": "brain_imaging", "category": "synonym"},
    {"text": "Find climate change projection data",
     "expected_tag": "climate_model", "category": "synonym"},
    {"text": "Where are the earthquake recordings?",
     "expected_tag": "seismic_waves", "category": "synonym"},
    {"text": "Locate protein crystallography coordinates",
     "expected_tag": "protein_structure", "category": "synonym"},
    {"text": "Find the Landsat satellite images",
     "expected_tag": "satellite_imagery", "category": "synonym"},
    {"text": "Show wind farm power generation data",
     "expected_tag": "wind_turbine", "category": "synonym"},
    {"text": "Find the cardiovascular drug trial results",
     "expected_tag": "drug_trials", "category": "synonym"},
    {"text": "Locate highway traffic sensor measurements",
     "expected_tag": "traffic_flow", "category": "synonym"},
    {"text": "Where is the agricultural soil sensor data?",
     "expected_tag": "soil_moisture", "category": "synonym"},
    {"text": "Find the speech recognition features dataset",
     "expected_tag": "audio_speech", "category": "synonym"},
    {"text": "Locate the galaxy redshift catalog",
     "expected_tag": "galaxy_survey", "category": "synonym"},
    {"text": "Find battery charge-discharge test data",
     "expected_tag": "battery_cycling", "category": "synonym"},
    {"text": "Where is the PM2.5 air pollution data?",
     "expected_tag": "air_quality", "category": "synonym"},
    {"text": "Find the neural network model weights",
     "expected_tag": "neural_network", "category": "synonym"},
    {"text": "Locate the DNA epigenetic methylation data",
     "expected_tag": "dna_methylation", "category": "synonym"},
]

# Build tag->description map for keyword fallback matching
DATASET_KEYWORDS = {
    "weather_forecast": ["weather", "forecast", "humidity", "atmospheric pressure"],
    "particle_physics": ["particle", "collider", "hadron", "collision", "proton"],
    "ocean_temperature": ["ocean", "sea surface", "NOAA", "Pacific"],
    "genome_sequences": ["genome", "variant", "SNP", "allele", "chromosome"],
    "stock_market": ["stock", "S&P 500", "trading", "close prices"],
    "brain_imaging": ["brain", "fMRI", "MRI", "cortical", "blood oxygen"],
    "climate_model": ["climate", "CO2", "radiative", "RCP"],
    "seismic_waves": ["earthquake", "seismograph", "P-wave", "S-wave"],
    "protein_structure": ["protein", "crystallography", "atomic positions"],
    "satellite_imagery": ["satellite", "Landsat", "multispectral", "infrared"],
    "wind_turbine": ["wind turbine", "rotor", "blade pitch", "nacelle"],
    "drug_trials": ["drug trial", "dosage", "biomarker", "cardiovascular"],
    "traffic_flow": ["traffic", "highway", "loop detector", "lane occupancy"],
    "soil_moisture": ["soil moisture", "volumetric water", "conductivity"],
    "audio_speech": ["speech", "mel-frequency", "phoneme", "speaker"],
    "galaxy_survey": ["galaxy", "redshift", "photometric", "stellar mass"],
    "battery_cycling": ["battery", "lithium-ion", "charge-discharge", "impedance"],
    "air_quality": ["air quality", "PM2.5", "PM10", "ozone", "nitrogen dioxide"],
    "neural_network": ["neural network", "model weights", "gradient", "optimizer"],
    "dna_methylation": ["methylation", "CpG", "EPIC BeadChip", "Illumina"],
}


def check_health():
    """Check if Hindsight server is reachable."""
    try:
        r = requests.get(f"{HINDSIGHT_BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"  Hindsight server: OK ({HINDSIGHT_BASE_URL})")
        return True
    except Exception as e:
        print(f"  ERROR: Cannot reach Hindsight at {HINDSIGHT_BASE_URL}: {e}")
        return False


def cleanup():
    """Delete the benchmark bank from Hindsight."""
    try:
        r = requests.delete(API_PREFIX, timeout=30)
        if r.status_code in (200, 204):
            print(f"  Cleaned up bank: {BANK_ID}")
        else:
            print(f"  Cleanup status: {r.status_code} (may not exist yet)")
    except Exception as e:
        print(f"  Cleanup warning: {e}")


def create_bank():
    """Create/ensure the memory bank exists."""
    r = requests.put(API_PREFIX, json={}, timeout=30)
    r.raise_for_status()
    print(f"  Bank created: {BANK_ID}")

    # Disable observations to avoid LLM queue contention
    try:
        requests.patch(
            f"{API_PREFIX}/config",
            json={"updates": {"enable_observations": False}},
            timeout=30,
        )
        print("  Observations disabled (avoids LLM contention)")
    except Exception:
        pass


def poll_operation(operation_id, timeout_s=600):
    """Poll an async operation until complete."""
    backoff = 5.0
    elapsed = 0.0
    while elapsed < timeout_s:
        time.sleep(backoff)
        elapsed += backoff
        try:
            r = requests.get(
                f"{API_PREFIX}/operations/{operation_id}",
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "unknown")
                if status == "completed":
                    return True
                if status == "failed":
                    print(f"    Operation FAILED: {data.get('error', '?')}")
                    return False
        except Exception:
            pass
        backoff = min(backoff * 1.2, 15.0)

    print(f"    Operation timed out after {timeout_s}s")
    return False


def ingest():
    """Ingest all 20 dataset descriptions as memories into Hindsight."""
    t0 = time.time()
    operation_ids = []

    for i, ds in enumerate(DATASETS):
        payload = {
            "items": [{
                "content": ds["description"],
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "context": f"HDF5 dataset: {ds['tag_name']}",
                "document_id": ds["tag_name"],
            }],
            "async": True,
        }
        r = requests.post(f"{API_PREFIX}/memories", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        op_id = data.get("operation_id")
        if op_id:
            operation_ids.append(op_id)
        print(f"  [{ds['tag_name']}] ingested (op: {op_id or 'sync'})")

    t1 = time.time()
    ingest_ms = (t1 - t0) * 1000
    print(f"  All {len(DATASETS)} descriptions submitted ({ingest_ms:.1f} ms)")
    return ingest_ms, operation_ids


def await_indexing(operation_ids):
    """Wait for all async ingest operations to complete."""
    if not operation_ids:
        print("  No async operations to wait for")
        return 0.0

    print(f"  Waiting for {len(operation_ids)} operations...")
    t0 = time.time()
    completed = 0
    for op_id in operation_ids:
        if poll_operation(op_id):
            completed += 1
    t1 = time.time()
    index_ms = (t1 - t0) * 1000
    print(f"  {completed}/{len(operation_ids)} operations completed "
          f"({index_ms:.1f} ms)")
    return index_ms


def match_result_to_dataset(result):
    """Match a Hindsight recall result to a dataset tag_name."""
    # Primary: use document_id if available
    doc_id = result.get("document_id", "")
    if doc_id and any(ds["tag_name"] == doc_id for ds in DATASETS):
        return doc_id

    # Fallback: keyword matching on result text
    text = (result.get("text", "") + " " +
            result.get("context", "")).lower()

    best_tag = None
    best_score = 0
    for tag_name, keywords in DATASET_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_score = score
            best_tag = tag_name

    # Also check for explicit [tag_name] label
    for ds in DATASETS:
        if ds["tag_name"] in (result.get("text", "") + result.get("context", "")):
            return ds["tag_name"]

    return best_tag


def search(query_text, top_k=5):
    """Search Hindsight for memories matching the query."""
    t0 = time.time()
    r = requests.post(
        f"{API_PREFIX}/memories/recall",
        json={
            "query": query_text,
            "budget": "mid",
            "max_tokens": 8192,
            "include": {"entities": {"max_tokens": 1000}},
        },
        timeout=120,
    )
    r.raise_for_status()
    t1 = time.time()
    query_ms = (t1 - t0) * 1000

    data = r.json()
    results = data.get("results") or []
    entities = data.get("entities") or []

    # Map results back to datasets, deduplicate, preserve order
    seen = set()
    ranked_tags = []

    # From results
    for res in results:
        tag = match_result_to_dataset(res)
        if tag and tag not in seen:
            seen.add(tag)
            ranked_tags.append(tag)
        if len(ranked_tags) >= top_k:
            break

    # From entities (secondary signal)
    if len(ranked_tags) < top_k:
        for ent in entities:
            name = ent.get("canonical_name", "").lower().replace(" ", "_")
            if name in DATASET_KEYWORDS and name not in seen:
                seen.add(name)
                ranked_tags.append(name)
            if len(ranked_tags) >= top_k:
                break

    return ranked_tags, query_ms


def run_benchmark():
    print("========================================")
    print("Hindsight Dataset Discovery Benchmark")
    print("========================================")
    print(f"Datasets: {len(DATASETS)}, Queries: {len(QUERIES)}")
    print()

    # Health check
    if not check_health():
        return 1

    # Cleanup previous run
    print()
    print("=== Phase 0: Cleanup ===")
    cleanup()
    time.sleep(2)

    # Create bank
    print()
    print("=== Phase 1: Create Bank ===")
    create_bank()

    # Phase 2: Ingest
    print()
    print("=== Phase 2: Ingest ===")
    ingest_ms, operation_ids = ingest()

    # Phase 3: Wait for indexing
    print()
    print("=== Phase 3: Await Indexing ===")
    index_ms = await_indexing(operation_ids)

    # Phase 4: Query
    print()
    print(f"=== Phase 4: Query ({len(QUERIES)} questions) ===")
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_rr = 0.0
    total_query_ms = 0.0

    for i, q in enumerate(QUERIES):
        print()
        print(f"  Q{i+1:02d} [{q['category']}] \"{q['text']}\"")

        ranked_tags, query_ms = search(q["text"])
        total_query_ms += query_ms

        # Format results
        results_str = ", ".join(ranked_tags) if ranked_tags else "(none)"
        print(f"    Results: [{results_str}]")

        # Check if expected tag is in results
        found_rank = -1
        for r_idx, tag in enumerate(ranked_tags):
            if tag == q["expected_tag"]:
                found_rank = r_idx
                break

        if found_rank >= 0:
            print(f"    Expected: {q['expected_tag']} -> FOUND at rank "
                  f"{found_rank + 1} ({query_ms:.1f} ms)")
            if found_rank == 0:
                top1_correct += 1
            if found_rank < 3:
                top3_correct += 1
            if found_rank < 5:
                top5_correct += 1
            total_rr += 1.0 / (found_rank + 1)
        else:
            print(f"    Expected: {q['expected_tag']} -> NOT FOUND ({query_ms:.1f} ms)")

    # Phase 5: Report
    n = len(QUERIES)
    mrr = total_rr / n

    print()
    print("========================================")
    print("Results (Hindsight)")
    print("========================================")
    print(f"  Top-1 Accuracy: {top1_correct}/{n} ({100.0 * top1_correct / n:.1f}%)")
    print(f"  Top-3 Accuracy: {top3_correct}/{n} ({100.0 * top3_correct / n:.1f}%)")
    print(f"  Top-5 Accuracy: {top5_correct}/{n} ({100.0 * top5_correct / n:.1f}%)")
    print(f"  Mean Reciprocal Rank: {mrr:.3f}")
    print(f"  Ingest latency:      {ingest_ms:.1f} ms")
    print(f"  Index latency:       {index_ms:.1f} ms")
    print(f"  Avg query latency:   {total_query_ms / n:.2f} ms")
    print(f"  Total queries:       {n}")
    print("========================================")

    # Write results JSON
    results = {
        "provider": "hindsight",
        "datasets": len(DATASETS),
        "queries": n,
        "top1_accuracy": top1_correct / n,
        "top3_accuracy": top3_correct / n,
        "top5_accuracy": top5_correct / n,
        "mrr": mrr,
        "ingest_latency_ms": ingest_ms,
        "index_latency_ms": index_ms,
        "avg_query_latency_ms": total_query_ms / n,
    }
    with open("/tmp/hindsight_bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: /tmp/hindsight_bench_results.json")

    # Cleanup
    print()
    print("=== Cleanup ===")
    cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
