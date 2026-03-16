#!/usr/bin/env python3
"""Generate HDF5 datasets with LLM-generated descriptions + search queries.

Usage:
    export LLM_ENDPOINT=http://localhost:8090/v1/chat/completions
    python3 gen_hdf5_datasets.py [output_dir]

Outputs:
    - 20 HDF5 files (64 MB each) with description in dataset attrs
    - manifest.json with descriptions + search queries for the C++ benchmark
"""

import h5py
import numpy as np
import os
import sys
import json
import requests

DOMAINS = [
    "weather_forecast",
    "particle_physics",
    "ocean_temperature",
    "genome_sequences",
    "stock_market",
    "brain_imaging",
    "climate_model",
    "seismic_waves",
    "protein_structure",
    "satellite_imagery",
    "wind_turbine",
    "drug_trials",
    "traffic_flow",
    "soil_moisture",
    "audio_speech",
    "galaxy_survey",
    "battery_cycling",
    "air_quality",
    "neural_network",
    "dna_methylation",
]

LLM_ENDPOINT = os.environ.get(
    "LLM_ENDPOINT", "http://localhost:8090/v1/chat/completions"
)


def call_llm(prompt, max_tokens=256):
    """Call OpenAI-compatible LLM endpoint."""
    resp = requests.post(
        LLM_ENDPOINT,
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_description(domain):
    """Ask LLM to generate a rich scientific dataset description."""
    human_name = domain.replace("_", " ")
    prompt = (
        f"Generate a concise scientific description (2-3 sentences, under 200 words) "
        f"for an HDF5 dataset named '{human_name}'. "
        f"Include specific data types, measurement instruments, spatial/temporal "
        f"resolution, and source institutions. Return ONLY the description text, "
        f"no quotes or prefixes."
    )
    return call_llm(prompt)


def generate_query(domain, description):
    """Ask LLM to generate a natural-language search query (using synonyms)."""
    human_name = domain.replace("_", " ")
    prompt = (
        f"Given a dataset about '{human_name}' with this description:\n"
        f'"{description}"\n\n'
        f"Write a short natural language search query (under 15 words) that someone "
        f"would use to find this dataset. Use synonyms and rephrase — do NOT copy "
        f"exact phrases from the description. Return ONLY the query text, "
        f"no quotes or prefixes."
    )
    return call_llm(prompt, max_tokens=64)


def generate(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"datasets": [], "queries": []}
    total_bytes = 0

    for domain in DOMAINS:
        filename = f"{domain}.h5"
        filepath = os.path.join(output_dir, filename)

        # Generate description via LLM
        print(f"  [{domain}] Generating description...")
        description = generate_description(domain)
        print(f"    desc: {description[:80]}...")

        # Generate query via LLM
        print(f"  [{domain}] Generating query...")
        query = generate_query(domain, description)
        print(f"    query: {query}")

        # Write HDF5 file (64 MB random data + description attr)
        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                data = np.random.randn(8192, 1024).astype(np.float64)
                ds = f.create_dataset("data", data=data)
                ds.attrs["description"] = description
            sz = os.path.getsize(filepath)
            total_bytes += sz
            print(f"    file: {sz / 1e6:.1f} MB (created)")
        else:
            # Update description attr in existing file
            with h5py.File(filepath, "a") as f:
                f["data"].attrs["description"] = description
            sz = os.path.getsize(filepath)
            total_bytes += sz
            print(f"    file: {sz / 1e6:.1f} MB (updated desc)")

        manifest["datasets"].append(
            {
                "tag_name": domain,
                "filename": filename,
                "description": description,
            }
        )
        manifest["queries"].append(
            {
                "text": query,
                "expected_tag": domain,
                "category": "synonym",
            }
        )

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written: {manifest_path}")
    print(f"  Total: {total_bytes / 1e9:.2f} GB ({len(DOMAINS)} files)")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hdf5_bench_datasets"
    print(f"=== Generating {len(DOMAINS)} HDF5 datasets in {output_dir} ===")
    print(f"=== LLM endpoint: {LLM_ENDPOINT} ===")
    generate(output_dir)
