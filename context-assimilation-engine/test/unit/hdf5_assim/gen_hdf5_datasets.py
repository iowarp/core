#!/usr/bin/env python3
"""Generate 20 HDF5 datasets with domain-specific structure and descriptions.

Each file has:
  - Realistic dataset names and group structure
  - A "description" attribute on the root dataset
  - 64MB of random data shaped to match the domain
  - A manifest.json for the C++ benchmark

Usage:
    python3 gen_hdf5_datasets.py [output_dir]
    # Default: /tmp/hdf5_bench_datasets

Then run benchmark:
    HDF5_BENCH_DIR=/tmp/hdf5_bench_datasets ./build/bin/cae_hdf5_discovery_bench
"""

import h5py
import numpy as np
import os
import sys
import json

# Each dataset has a description (stored as HDF5 attr) and domain-specific shape
DATASETS = [
    {"tag_name": "weather_forecast",
     "description": "Global weather forecast data including temperature, humidity, wind speed, "
                    "and atmospheric pressure measurements from 500 weather stations across "
                    "North America recorded hourly",
     "shape": (500, 8760, 4), "dtype": "float32",
     "datasets": {"temperature": (500, 8760), "humidity": (500, 8760),
                  "wind_speed": (500, 8760), "pressure": (500, 8760)}},
    {"tag_name": "particle_physics",
     "description": "High-energy particle collision events from the Large Hadron Collider "
                    "including particle trajectories, energy deposits, and detector hit "
                    "patterns for proton-proton collisions at 13 TeV",
     "shape": (100000, 128), "dtype": "float64",
     "datasets": {"trajectories": (100000, 3), "energy_deposits": (100000, 64),
                  "hit_patterns": (100000, 128)}},
    {"tag_name": "ocean_temperature",
     "description": "Sea surface temperature measurements from NOAA satellite observations "
                    "covering the Pacific Ocean basin with 0.25 degree spatial resolution "
                    "daily averages",
     "shape": (365, 720, 1440), "dtype": "float32",
     "datasets": {"sst": (365, 720, 1440)}},
    {"tag_name": "genome_sequences",
     "description": "Human genome variant call data from whole genome sequencing including "
                    "SNP positions, allele frequencies, and quality scores across "
                    "chromosome 1-22",
     "shape": (5000000, 4), "dtype": "float32",
     "datasets": {"positions": (5000000,), "allele_freq": (5000000,),
                  "quality": (5000000,)}},
    {"tag_name": "stock_market",
     "description": "Historical stock market trading data including open, high, low, close "
                    "prices and volume for S&P 500 companies from 2010 to 2024 at minute "
                    "resolution",
     "shape": (500, 2048, 5), "dtype": "float64",
     "datasets": {"ohlcv": (500, 2048, 5)}},
    {"tag_name": "brain_imaging",
     "description": "Functional MRI brain scan data from cognitive neuroscience experiments "
                    "showing blood oxygen level dependent signals across cortical regions "
                    "during memory tasks",
     "shape": (200, 64, 64, 40), "dtype": "float32",
     "datasets": {"bold_signal": (200, 64, 64, 40)}},
    {"tag_name": "climate_model",
     "description": "Earth system climate model output with global CO2 concentration, "
                    "radiative forcing, and temperature anomaly projections under RCP 8.5 "
                    "scenario from 2020 to 2100",
     "shape": (960, 180, 360), "dtype": "float64",
     "datasets": {"co2": (960, 180, 360), "temperature_anomaly": (960, 180, 360)}},
    {"tag_name": "seismic_waves",
     "description": "Earthquake seismograph recordings from the Pacific Ring of Fire "
                    "including P-wave and S-wave arrival times, magnitude estimates, and "
                    "focal mechanism solutions",
     "shape": (10000, 3, 2048), "dtype": "float32",
     "datasets": {"waveforms": (10000, 3, 2048), "magnitudes": (10000,)}},
    {"tag_name": "protein_structure",
     "description": "3D protein structure coordinates from X-ray crystallography including "
                    "atomic positions, B-factors, and secondary structure annotations for "
                    "enzyme active sites",
     "shape": (50000, 3), "dtype": "float64",
     "datasets": {"coordinates": (50000, 3), "b_factors": (50000,)}},
    {"tag_name": "satellite_imagery",
     "description": "Multispectral satellite imagery from Landsat 8 with red, green, blue, "
                    "near-infrared, and shortwave infrared bands at 30 meter ground "
                    "resolution",
     "shape": (7, 2048, 2048), "dtype": "uint16",
     "datasets": {"bands": (7, 2048, 2048)}},
    {"tag_name": "wind_turbine",
     "description": "Wind turbine performance metrics including power output, rotor speed, "
                    "blade pitch angle, nacelle temperature, and vibration sensor data from "
                    "an offshore wind farm",
     "shape": (50, 525600, 5), "dtype": "float32",
     "datasets": {"power": (50, 525600), "rotor_speed": (50, 525600),
                  "vibration": (50, 525600)}},
    {"tag_name": "drug_trials",
     "description": "Clinical drug trial results with patient demographics, dosage levels, "
                    "biomarker measurements, adverse events, and efficacy endpoints for a "
                    "phase III cardiovascular study",
     "shape": (5000, 200), "dtype": "float64",
     "datasets": {"demographics": (5000, 10), "biomarkers": (5000, 50),
                  "efficacy": (5000, 20)}},
    {"tag_name": "traffic_flow",
     "description": "Urban traffic flow sensor data from highway loop detectors measuring "
                    "vehicle count, average speed, and lane occupancy at 5-minute intervals "
                    "across 200 intersections",
     "shape": (200, 105120, 3), "dtype": "float32",
     "datasets": {"vehicle_count": (200, 105120), "speed": (200, 105120),
                  "occupancy": (200, 105120)}},
    {"tag_name": "soil_moisture",
     "description": "Agricultural soil moisture measurements from wireless sensor networks "
                    "at multiple depths including volumetric water content, soil temperature, "
                    "and electrical conductivity",
     "shape": (100, 4, 8760), "dtype": "float32",
     "datasets": {"water_content": (100, 4, 8760), "temperature": (100, 4, 8760)}},
    {"tag_name": "audio_speech",
     "description": "Speech recognition training data with mel-frequency cepstral "
                    "coefficients, phoneme labels, speaker embeddings, and acoustic features "
                    "extracted from 1000 hours of English speech",
     "shape": (1000000, 40), "dtype": "float32",
     "datasets": {"mfcc": (1000000, 40), "embeddings": (1000000, 256)}},
    {"tag_name": "galaxy_survey",
     "description": "Astronomical galaxy survey catalog with redshift measurements, "
                    "photometric magnitudes, morphological classifications, and stellar mass "
                    "estimates for 2 million galaxies",
     "shape": (2000000, 8), "dtype": "float64",
     "datasets": {"redshift": (2000000,), "magnitudes": (2000000, 5),
                  "stellar_mass": (2000000,)}},
    {"tag_name": "battery_cycling",
     "description": "Lithium-ion battery charge-discharge cycling data including voltage, "
                    "current, capacity, impedance spectroscopy, and temperature measurements "
                    "over 500 cycles",
     "shape": (500, 10000, 5), "dtype": "float32",
     "datasets": {"voltage": (500, 10000), "current": (500, 10000),
                  "capacity": (500, 10000)}},
    {"tag_name": "air_quality",
     "description": "Urban air quality monitoring station data with particulate matter PM2.5 "
                    "and PM10, ozone, nitrogen dioxide, sulfur dioxide concentrations, and "
                    "meteorological conditions",
     "shape": (50, 8760, 6), "dtype": "float32",
     "datasets": {"pm25": (50, 8760), "ozone": (50, 8760), "no2": (50, 8760)}},
    {"tag_name": "neural_network",
     "description": "Deep neural network training checkpoint with model weights, gradient "
                    "statistics, optimizer state, learning rate schedule, and per-layer "
                    "activation distributions",
     "shape": (100, 1024, 1024), "dtype": "float32",
     "datasets": {"weights": (100, 1024, 1024), "gradients": (100, 1024, 1024)}},
    {"tag_name": "dna_methylation",
     "description": "Epigenetic DNA methylation array data from Illumina EPIC BeadChip with "
                    "beta values, detection p-values, and probe annotations across 850,000 "
                    "CpG sites",
     "shape": (850000, 3), "dtype": "float64",
     "datasets": {"beta_values": (850000,), "p_values": (850000,)}},
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

# Target ~64MB per file. We scale the first dataset in each file to hit this.
TARGET_BYTES = 64 * 1024 * 1024


def create_hdf5_file(filepath, ds_info):
    """Create an HDF5 file with domain-specific datasets and description attribute."""
    with h5py.File(filepath, "w") as f:
        # Store description as root group attribute
        f.attrs["description"] = ds_info["description"]

        # Create each sub-dataset with random data, scaled to ~64MB total
        total_elements = 0
        for name, shape in ds_info["datasets"].items():
            n_elements = 1
            for d in shape:
                n_elements *= d
            total_elements += n_elements

        dtype = np.dtype(ds_info["dtype"])
        bytes_per_element = dtype.itemsize
        target_elements = TARGET_BYTES // bytes_per_element

        # Scale factor to hit ~64MB
        scale = max(1, target_elements // max(total_elements, 1))

        for name, shape in ds_info["datasets"].items():
            # Scale first dimension
            scaled_shape = (min(shape[0] * scale, shape[0] * 10),) + shape[1:] if len(shape) > 1 else (min(shape[0] * scale, shape[0] * 10),)
            # Cap at target size
            n_elem = 1
            for d in scaled_shape:
                n_elem *= d
            if n_elem * bytes_per_element > TARGET_BYTES:
                # Just use a flat array to hit 64MB
                n_elem = TARGET_BYTES // (bytes_per_element * len(ds_info["datasets"]))
                scaled_shape = (n_elem,)

            ds = f.create_dataset(name, shape=scaled_shape, dtype=ds_info["dtype"])
            # Write random data in chunks to avoid memory spike
            chunk_size = min(1000000, n_elem)
            for i in range(0, n_elem, chunk_size):
                end = min(i + chunk_size, n_elem)
                if len(scaled_shape) == 1:
                    ds[i:end] = np.random.randn(end - i).astype(ds_info["dtype"])
                else:
                    # For multi-dim, write first dim slices
                    row_end = min(i + chunk_size, scaled_shape[0])
                    if row_end > i:
                        row_shape = (row_end - i,) + scaled_shape[1:]
                        row_n = 1
                        for d in row_shape:
                            row_n *= d
                        ds[i:row_end] = np.random.randn(*row_shape).astype(ds_info["dtype"])
                    break  # one chunk is enough for random data

            # Store description on each dataset too
            ds.attrs["description"] = ds_info["description"]


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hdf5_bench_datasets"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Generating {len(DATASETS)} HDF5 datasets in {output_dir} ===")

    manifest = {"datasets": [], "queries": QUERIES}

    for ds_info in DATASETS:
        filename = f"{ds_info['tag_name']}.h5"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            sz = os.path.getsize(filepath)
            print(f"  {ds_info['tag_name']}: exists ({sz / 1e6:.1f} MB)")
        else:
            print(f"  {ds_info['tag_name']}: creating...", end=" ", flush=True)
            create_hdf5_file(filepath, ds_info)
            sz = os.path.getsize(filepath)
            print(f"{sz / 1e6:.1f} MB")

        manifest["datasets"].append({
            "tag_name": ds_info["tag_name"],
            "filename": filename,
            "description": ds_info["description"],
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Run: HDF5_BENCH_DIR={output_dir} ./build/bin/cae_hdf5_discovery_bench")


if __name__ == "__main__":
    main()
