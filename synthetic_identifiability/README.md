# Synthetic Identifiability Dataset

This folder contains the released semi-theoretical synthetic identifiability dataset used to support the methodological part of the PACE/OCI hyperspectral phytoplankton retrieval study.

## What this dataset is for

This dataset is designed to answer a specific question:

**When is a hyperspectral ocean-color sample physically interpretable at a deeper decomposition level, and when should the model reject deep interpretation?**

The dataset is therefore not a substitute for real external validation. Instead, it is used to:

- analyze inverse-problem degeneracy under controlled optical scenarios,
- stress-test the recognition and physical gates,
- evaluate whether the reject option activates in difficult or low-identifiability cases,
- support the claim that additional spectral bands do not guarantee universal decomposability.

## What this dataset is not

This dataset should not be interpreted as:

- a replacement for HPLC truth,
- a substitute for real PACE-HPLC matchup validation,
- evidence of absolute biological recovery accuracy in real scenes.

It is a methodological stress-test dataset for selective interpretation and gate behavior.

## Released files

### `data/synthetic_identifiability_dataset_v2b.csv.gz`

Compressed release of the final dataset used in the paper.

- number of samples: 8000
- release variant: v2b
- format: gzip-compressed CSV

### `data/synthetic_identifiability_dataset_v2b_summary.json`

Summary statistics for:

- total sample count,
- allowed-rate statistics,
- regime-wise admissibility,
- sample-type-wise admissibility,
- train/val/test split counts.

### `data/synthetic_identifiability_dataset_v2b_preview.csv`

A lightweight preview file containing selected metadata and label columns for quick inspection on GitHub.

### `figures/synthetic_identifiability_v2b_label_distributions.png`

Visualization of the released label distributions.

### `scripts/generate_synthetic_identifiability_dataset_v2.py`

Script that generates the upstream v2 synthetic dataset.

### `scripts/make_synthetic_identifiability_dataset_v2b.py`

Script that converts the v2 dataset into the released v2b version by applying a non-absolute Level-3 feasibility rule.

## Why v2b was released

The v2b formulation is a safer and more realistic label design for methodological review.

Its main idea is:

- clear water remains the primary admissible domain,
- transition and coastal regimes remain mostly hard negatives,
- but non-clear samples are not rejected as an absolute rule,
- rare low-degeneracy, low-nuisance, high-margin cases are allowed to pass.

This means the released labels treat water regime as an optical-risk proxy rather than an absolute physical law.

## Key column groups

The full dataset contains several column families:

- sample metadata:
  - `sample_uid`, `sample_type`, `regime_code`, `regime_name`, `split`
- composition and dominance:
  - `dominant_pft`, `co_dominant_pft`, `dominant_margin`, `fraction_entropy`
- identifiability diagnostics:
  - `composition_vs_nuisance_ratio`
  - `jacobian_condition_number`
  - `smallest_singular_value`
  - `min_active_pair_sam_deg`
  - `degeneracy_risk`
- gate labels:
  - `decomposition_allowed_soft`
  - `decomposition_allowed_hard`
  - `level3_decomposition_allowed_soft`
  - `level3_decomposition_allowed_hard`
- target composition fields:
  - `target_fraction_*`
  - `target_pft_*`
- spectral fields:
  - `feat_rrs_400` to `feat_rrs_700`
  - `clean_rrs_400` to `clean_rrs_700`
- physics-derived auxiliary descriptors:
  - `phys_*`

## Recommended use in papers or benchmarks

This dataset is best used for:

- gate ablation studies,
- reject-option analysis,
- regime-wise degeneracy analysis,
- coverage-quality evaluation,
- sensitivity studies on conditional interpretation.

It is not the right dataset for claiming real-world external biological accuracy on its own.

## Quick load example

```python
import pandas as pd
df = pd.read_csv("synthetic_identifiability/data/synthetic_identifiability_dataset_v2b.csv.gz")
print(df.shape)
```
