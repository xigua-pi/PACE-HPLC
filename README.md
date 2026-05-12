# PACE-HPLC

This repository provides the open indices, matchup mappings, reproducible scripts, and synthetic stress-test data used in our PACE/OCI hyperspectral phytoplankton functional type retrieval study.

The repository is organized around two complementary evidence streams:

1. Real-world PACE-HPLC matchup resources, used to trace which HPLC sample is paired with which processed PACE scene.
2. A semi-theoretical synthetic identifiability dataset, used to test when deep phytoplankton interpretation is physically admissible and when a conservative reject option should be triggered.

## Why this repository exists

The main study does not treat all evidence as equivalent. Real matchup data and synthetic stress-test data support different claims:

- The real PACE-HPLC benchmark supports traceability, external evaluation, and reproducible sample-to-scene linkage.
- The synthetic identifiability dataset supports methodological validation of degeneracy analysis, gate behavior, and selective interpretation under controlled optical conditions.

Keeping both resources in one repository makes it easier to understand what the model is tested on, what each dataset can support scientifically, and where the boundary is between real external evidence and simulation-based stress testing.

## Repository structure

### `data_index/`

Open index tables for the PACE-HPLC benchmark:

- `hplc_202_samples_time_lat_lon.csv`
  - Timestamp, latitude, longitude, and source metadata for the 202 HPLC samples.
- `hplc_202_to_pace_image_mapping.csv`
  - Per-sample mapping table linking each HPLC sample to its corresponding PACE hyperspectral image scene.
- `pace_image_repository_manifest_142.csv`
  - Manifest of the 142 processed PACE scenes used during benchmark construction.
- `pace_matched_scene_list.csv`
  - Subset of processed PACE scenes matched to at least one HPLC sample.
- `pace_valid78_effective_scene_list.csv`
  - Effective PACE scenes used by the current 78-sample valid-spectrum benchmark.

### `scripts/`

- `build_pace_hplc_open_repository_lists.py`
  - Script used to build the repository-facing benchmark index tables.

### `synthetic_identifiability/`

Semi-theoretical identifiability stress-test package:

- `data/synthetic_identifiability_dataset_v2b.csv.gz`
  - Compressed release of the 8000-sample synthetic identifiability dataset.
- `data/synthetic_identifiability_dataset_v2b_summary.json`
  - Summary statistics for label rates, regime-wise admissibility, and split counts.
- `data/synthetic_identifiability_dataset_v2b_preview.csv`
  - Lightweight preview of representative metadata columns.
- `data/SHA256SUMS.txt`
  - Checksums for release integrity.
- `figures/synthetic_identifiability_v2b_label_distributions.png`
  - Distribution figure for the released dataset.
- `scripts/generate_synthetic_identifiability_dataset_v2.py`
  - Script that generates the v2 synthetic dataset.
- `scripts/make_synthetic_identifiability_dataset_v2b.py`
  - Script that relabels v2 into the released v2b formulation.
- `README.md`
  - Detailed explanation of why the synthetic dataset is included and how it should be interpreted.

### `docs/`

- `data_availability_text_template.txt`
  - Draft manuscript wording for data availability statements and repository description.

## Current benchmark summary

### Real PACE-HPLC benchmark

- 202 HPLC samples are indexed.
- 149 HPLC samples are currently matched to at least one processed PACE scene.
- 53 HPLC samples currently have no matched processed PACE scene.
- 23 processed PACE scenes are matched to at least one HPLC sample.
- 14 effective scenes are used by the current 78-sample valid-spectrum benchmark.

### Synthetic identifiability dataset

- 8000 synthetic samples in total.
- Released version: `v2b`.
- Overall allowed rate: 0.411375.
- Main purpose: stress-test the degeneracy structure of the inverse problem and evaluate the gate-controlled reject option.

## Why the synthetic dataset matters

The synthetic identifiability dataset is not included as a substitute for real external validation. Its role is different and narrower:

- It provides a controlled environment for testing whether the model can distinguish interpretable from non-interpretable samples.
- It supports the claim that more spectral bands do not automatically imply universal decomposability.
- It helps verify that the selective interpretation framework is responding to optical degeneracy, nuisance absorption and scattering, dominant-margin collapse, and spectral similarity, rather than simply overfitting one benchmark.

In other words, the synthetic dataset is a methodological stress-test resource, not a replacement for the real PACE-HPLC benchmark.

## How to use

### Read the real benchmark indices

Use the CSV files under `data_index/` to trace:

- where each HPLC sample came from,
- which PACE scene it was matched to,
- which scenes enter the valid 78-sample benchmark.

### Read the synthetic dataset

The released synthetic dataset is compressed as `csv.gz`. In Python:

```python
import pandas as pd
df = pd.read_csv("synthetic_identifiability/data/synthetic_identifiability_dataset_v2b.csv.gz")
```

### Rebuild the synthetic labels

From the repository root:

```bash
python synthetic_identifiability/scripts/generate_synthetic_identifiability_dataset_v2.py
python synthetic_identifiability/scripts/make_synthetic_identifiability_dataset_v2b.py
```

## Data availability notes

This repository releases the open benchmark indices, synthetic stress-test dataset, and reconstruction scripts associated with the study.

Some internally generated AC intermediate products, internal HPLC raw tables, and laboratory optical datasets are not distributed here because they are subject to internal workflow constraints or third-party data-sharing boundaries. However, the released indices, matchup mappings, synthetic benchmark files, and reconstruction scripts are intended to provide full traceability for the reported public benchmark and methodological stress tests.
