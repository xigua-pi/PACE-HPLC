# PACE-HPLC PFT Benchmark Repository

This repository organizes the open index tables, matchup mappings, and reproducible scripts used to support the PACE-HPLC benchmark in our hyperspectral phytoplankton functional type retrieval study.

## Included contents

- `data_index/hplc_202_samples_time_lat_lon.csv`
  - Time, latitude, longitude, and source metadata for the 202 HPLC samples.
- `data_index/hplc_202_to_pace_image_mapping.csv`
  - Per-sample mapping table linking each HPLC sample to its corresponding PACE hyperspectral image scene.
- `data_index/pace_image_repository_manifest_142.csv`
  - Manifest of the 142 processed PACE scenes used during benchmark construction.
- `data_index/pace_matched_scene_list.csv`
  - Subset of processed PACE scenes matched to at least one HPLC sample.
- `data_index/pace_valid78_effective_scene_list.csv`
  - Effective PACE scenes used by the current 78-sample valid-spectrum benchmark.
- `scripts/build_pace_hplc_open_repository_lists.py`
  - Script used to build the repository-facing index tables.
- `docs/data_availability_text_template.txt`
  - Draft manuscript wording for the data-availability section.

## Current benchmark summary

- 202 HPLC samples are indexed.
- 149 HPLC samples are currently matched to at least one processed PACE scene.
- 53 HPLC samples currently have no matched processed PACE scene.
- 23 processed PACE scenes are matched to at least one HPLC sample.
- 14 effective scenes are used by the current 78-sample valid-spectrum benchmark.

## Notes

This repository is designed to provide sample-level traceability and benchmark reproducibility. Large PACE image files and any internally restricted processing intermediates can be distributed separately, while this repository keeps the lightweight index tables, mapping tables, and build scripts in one place.
