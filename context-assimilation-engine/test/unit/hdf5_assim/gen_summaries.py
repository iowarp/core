#!/usr/bin/env python3
"""Generate LLM summaries for dataset descriptions.

Calls an OpenAI-compatible endpoint to produce 4-8 word summaries,
then writes a manifest.json that the C++ benchmark reads.

Usage:
    # Start the Qwen server first:
    python3 qwen_server.py --port 8080 &

    # Generate summaries:
    python3 gen_summaries.py [--endpoint http://localhost:8080/v1] [--model qwen2.5-0.5b] [--output /tmp/cte_bench]

    # Run the benchmark with summaries:
    export HDF5_BENCH_DIR=/tmp/cte_bench
    ./build/bin/cae_hdf5_discovery_bench
"""

import argparse
import json
import os
import sys
import time
import requests

DATASETS = [
    {"tag_name": "weather_forecast",
     "description": "Global weather forecast data including temperature, humidity, wind speed, and atmospheric pressure measurements from 500 weather stations across North America recorded hourly"},
    {"tag_name": "particle_physics",
     "description": "High-energy particle collision events from the Large Hadron Collider including particle trajectories, energy deposits, and detector hit patterns for proton-proton collisions at 13 TeV"},
    {"tag_name": "ocean_temperature",
     "description": "Sea surface temperature measurements from NOAA satellite observations covering the Pacific Ocean basin with 0.25 degree spatial resolution daily averages"},
    {"tag_name": "genome_sequences",
     "description": "Human genome variant call data from whole genome sequencing including SNP positions, allele frequencies, and quality scores across chromosome 1-22"},
    {"tag_name": "stock_market",
     "description": "Historical stock market trading data including open, high, low, close prices and volume for S&P 500 companies from 2010 to 2024 at minute resolution"},
    {"tag_name": "brain_imaging",
     "description": "Functional MRI brain scan data from cognitive neuroscience experiments showing blood oxygen level dependent signals across cortical regions during memory tasks"},
    {"tag_name": "climate_model",
     "description": "Earth system climate model output with global CO2 concentration, radiative forcing, and temperature anomaly projections under RCP 8.5 scenario from 2020 to 2100"},
    {"tag_name": "seismic_waves",
     "description": "Earthquake seismograph recordings from the Pacific Ring of Fire including P-wave and S-wave arrival times, magnitude estimates, and focal mechanism solutions"},
    {"tag_name": "protein_structure",
     "description": "3D protein structure coordinates from X-ray crystallography including atomic positions, B-factors, and secondary structure annotations for enzyme active sites"},
    {"tag_name": "satellite_imagery",
     "description": "Multispectral satellite imagery from Landsat 8 with red, green, blue, near-infrared, and shortwave infrared bands at 30 meter ground resolution"},
    {"tag_name": "wind_turbine",
     "description": "Wind turbine performance metrics including power output, rotor speed, blade pitch angle, nacelle temperature, and vibration sensor data from an offshore wind farm"},
    {"tag_name": "drug_trials",
     "description": "Clinical drug trial results with patient demographics, dosage levels, biomarker measurements, adverse events, and efficacy endpoints for a phase III cardiovascular study"},
    {"tag_name": "traffic_flow",
     "description": "Urban traffic flow sensor data from highway loop detectors measuring vehicle count, average speed, and lane occupancy at 5-minute intervals across 200 intersections"},
    {"tag_name": "soil_moisture",
     "description": "Agricultural soil moisture measurements from wireless sensor networks at multiple depths including volumetric water content, soil temperature, and electrical conductivity"},
    {"tag_name": "audio_speech",
     "description": "Speech recognition training data with mel-frequency cepstral coefficients, phoneme labels, speaker embeddings, and acoustic features extracted from 1000 hours of English speech"},
    {"tag_name": "galaxy_survey",
     "description": "Astronomical galaxy survey catalog with redshift measurements, photometric magnitudes, morphological classifications, and stellar mass estimates for 2 million galaxies"},
    {"tag_name": "battery_cycling",
     "description": "Lithium-ion battery charge-discharge cycling data including voltage, current, capacity, impedance spectroscopy, and temperature measurements over 500 cycles"},
    {"tag_name": "air_quality",
     "description": "Urban air quality monitoring station data with particulate matter PM2.5 and PM10, ozone, nitrogen dioxide, sulfur dioxide concentrations, and meteorological conditions"},
    {"tag_name": "neural_network",
     "description": "Deep neural network training checkpoint with model weights, gradient statistics, optimizer state, learning rate schedule, and per-layer activation distributions"},
    {"tag_name": "dna_methylation",
     "description": "Epigenetic DNA methylation array data from Illumina EPIC BeadChip with beta values, detection p-values, and probe annotations across 850,000 CpG sites"},
]

QUERIES = [
    {"text": "Find the weather forecast dataset", "expected_tag": "weather_forecast", "category": "exact"},
    {"text": "Locate particle collision data from CERN", "expected_tag": "particle_physics", "category": "synonym"},
    {"text": "Where is the ocean temperature data?", "expected_tag": "ocean_temperature", "category": "exact"},
    {"text": "Show me the genomics sequencing results", "expected_tag": "genome_sequences", "category": "synonym"},
    {"text": "Find stock trading historical prices", "expected_tag": "stock_market", "category": "synonym"},
    {"text": "Locate the fMRI brain scan data", "expected_tag": "brain_imaging", "category": "synonym"},
    {"text": "Find climate change projection data", "expected_tag": "climate_model", "category": "synonym"},
    {"text": "Where are the earthquake recordings?", "expected_tag": "seismic_waves", "category": "synonym"},
    {"text": "Locate protein crystallography coordinates", "expected_tag": "protein_structure", "category": "synonym"},
    {"text": "Find the Landsat satellite images", "expected_tag": "satellite_imagery", "category": "synonym"},
    {"text": "Show wind farm power generation data", "expected_tag": "wind_turbine", "category": "synonym"},
    {"text": "Find the cardiovascular drug trial results", "expected_tag": "drug_trials", "category": "synonym"},
    {"text": "Locate highway traffic sensor measurements", "expected_tag": "traffic_flow", "category": "synonym"},
    {"text": "Where is the agricultural soil sensor data?", "expected_tag": "soil_moisture", "category": "synonym"},
    {"text": "Find the speech recognition features dataset", "expected_tag": "audio_speech", "category": "synonym"},
    {"text": "Locate the galaxy redshift catalog", "expected_tag": "galaxy_survey", "category": "synonym"},
    {"text": "Find battery charge-discharge test data", "expected_tag": "battery_cycling", "category": "synonym"},
    {"text": "Where is the PM2.5 air pollution data?", "expected_tag": "air_quality", "category": "synonym"},
    {"text": "Find the neural network model weights", "expected_tag": "neural_network", "category": "synonym"},
    {"text": "Locate the DNA epigenetic methylation data", "expected_tag": "dna_methylation", "category": "synonym"},
]


def summarize(endpoint, model, description):
    """Call LLM to generate a 4-8 word summary."""
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content":
                 "Extract exactly 4 to 8 of the most important and distinctive keywords "
                 "from the following dataset description. Return only the keywords "
                 "separated by spaces, nothing else. Keep the original words, do not "
                 "paraphrase or translate."},
                {"role": "user", "content": description},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate LLM summaries for benchmark")
    parser.add_argument("--endpoint", default=os.environ.get("CAE_SUMMARY_ENDPOINT", "http://localhost:8080/v1"))
    parser.add_argument("--model", default=os.environ.get("CAE_SUMMARY_MODEL", "qwen2.5-0.5b"))
    parser.add_argument("--output", default="/tmp/cte_bench")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use descriptions as-is")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    manifest = {"datasets": [], "queries": QUERIES}

    for ds in DATASETS:
        if args.no_llm:
            summary = ds["description"]
            print(f"  {ds['tag_name']}: [raw] {summary[:60]}...")
        else:
            t0 = time.time()
            summary = summarize(args.endpoint, args.model, ds["description"])
            elapsed = (time.time() - t0) * 1000
            print(f"  {ds['tag_name']}: [{elapsed:.0f}ms] \"{summary}\"")

        manifest["datasets"].append({
            "tag_name": ds["tag_name"],
            "description": summary,
        })

    out_path = os.path.join(args.output, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {out_path}")
    print(f"Run benchmark with: HDF5_BENCH_DIR={args.output} ./build/bin/cae_hdf5_discovery_bench")


if __name__ == "__main__":
    main()
