from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MATCHUP_DIR = ROOT / "pace_era_ac_rrs_matchups_Edrive"
OUT_DIR = ROOT / "workspace_docs" / "reports" / "pace_hplc_open_repository_lists"

MATCHED_CSV = MATCHUP_DIR / "pace_era_hplc_ac_rrs_matchups_native_167bands.csv"
VALID_MATCHED_CSV = MATCHUP_DIR / "pace_era_hplc_ac_rrs_matchups_valid_only_native_167bands.csv"
UNMATCHED_CSV = MATCHUP_DIR / "pace_era_hplc_ac_rrs_unmatched_samples.csv"
SCENE_META_CSV = MATCHUP_DIR / "pace_ac_rrs_scene_metadata.csv"

HPLC_REPO_PLACEHOLDER = "<HPLC_OPEN_REPOSITORY_URL>"
PACE_REPO_PLACEHOLDER = "<PACE_IMAGE_REPOSITORY_URL>"


def build_hplc_samples_table(matched: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    matched_cols = ["sample_uid", "timestamp", "sample_time", "latitude", "longitude", "depth_m", "station", "sample", "source_file"]
    matched_part = matched[matched_cols].copy()
    matched_part["match_status"] = "matched"
    matched_part["hplc_repository_url"] = HPLC_REPO_PLACEHOLDER

    unmatched_part = unmatched[["sample_uid", "timestamp", "latitude", "longitude"]].copy()
    unmatched_part["sample_time"] = pd.to_datetime(unmatched_part["timestamp"], format="%Y%m%dT%H:%M:%S", errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    unmatched_part["depth_m"] = pd.NA
    unmatched_part["station"] = pd.NA
    unmatched_part["sample"] = pd.NA
    unmatched_part["source_file"] = pd.NA
    unmatched_part["match_status"] = "unmatched"
    unmatched_part["hplc_repository_url"] = HPLC_REPO_PLACEHOLDER
    unmatched_part = unmatched_part[matched_part.columns]

    full = pd.concat([matched_part, unmatched_part], ignore_index=True)
    full = full.drop_duplicates(subset=["sample_uid"]).sort_values(["timestamp", "sample_uid"]).reset_index(drop=True)
    full["sample_index_202"] = range(1, len(full) + 1)
    ordered = [
        "sample_index_202",
        "sample_uid",
        "timestamp",
        "sample_time",
        "latitude",
        "longitude",
        "depth_m",
        "station",
        "sample",
        "source_file",
        "match_status",
        "hplc_repository_url",
    ]
    return full[ordered]


def build_scene_repository_table(scene_meta: pd.DataFrame) -> pd.DataFrame:
    scene = scene_meta.copy()
    scene["pace_repository_url"] = PACE_REPO_PLACEHOLDER
    scene["repository_relative_path"] = scene["path"].astype(str).str.replace(r"^[A-Za-z]:\\", "", regex=True).str.replace("\\", "/", regex=False)
    ordered = [
        "region",
        "scene_id",
        "scene_file",
        "scene_time",
        "path",
        "repository_relative_path",
        "pace_repository_url",
        "west",
        "east",
        "north",
        "south",
        "width",
        "height",
        "bands",
        "wavelength_min_nm",
        "wavelength_max_nm",
    ]
    return scene[ordered].sort_values(["scene_time", "scene_id"]).reset_index(drop=True)


def build_effective_scene_lists(scene_manifest: pd.DataFrame, matched: pd.DataFrame, valid_matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_paths = matched["image_path"].dropna().astype(str).unique().tolist()
    valid_paths = valid_matched["image_path"].dropna().astype(str).unique().tolist()

    matched_scene_list = scene_manifest[scene_manifest["path"].astype(str).isin(matched_paths)].copy()
    valid_scene_list = scene_manifest[scene_manifest["path"].astype(str).isin(valid_paths)].copy()

    matched_scene_list["scene_usage"] = "matched_by_at_least_one_hplc_sample"
    valid_scene_list["scene_usage"] = "used_by_valid_78_sample_benchmark"
    return (
        matched_scene_list.sort_values(["scene_time", "scene_id"]).reset_index(drop=True),
        valid_scene_list.sort_values(["scene_time", "scene_id"]).reset_index(drop=True),
    )


def build_sample_to_scene_table(hplc_samples: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    map_cols = [
        "sample_uid",
        "scene_region",
        "scene_id",
        "scene_file",
        "scene_time",
        "image_path",
        "time_delta_hours",
        "abs_time_delta_hours",
        "pixel_center_latitude",
        "pixel_center_longitude",
        "point_to_pixel_center_km",
        "valid_spectrum",
    ]
    mapping = matched[map_cols].copy().drop_duplicates(subset=["sample_uid", "scene_id"])
    mapping["pace_repository_url"] = PACE_REPO_PLACEHOLDER
    mapping["repository_relative_path"] = mapping["image_path"].astype(str).str.replace(r"^[A-Za-z]:\\", "", regex=True).str.replace("\\", "/", regex=False)
    merged = hplc_samples.merge(mapping, on="sample_uid", how="left")
    merged["pace_match_status"] = merged["scene_id"].notna().map({True: "matched_to_pace_scene", False: "no_matching_pace_scene"})
    ordered = [
        "sample_index_202",
        "sample_uid",
        "timestamp",
        "sample_time",
        "latitude",
        "longitude",
        "match_status",
        "pace_match_status",
        "scene_region",
        "scene_id",
        "scene_file",
        "scene_time",
        "image_path",
        "repository_relative_path",
        "pace_repository_url",
        "time_delta_hours",
        "abs_time_delta_hours",
        "pixel_center_latitude",
        "pixel_center_longitude",
        "point_to_pixel_center_km",
        "valid_spectrum",
    ]
    return merged[ordered].sort_values(["sample_index_202"]).reset_index(drop=True)


def build_markdown_summary(hplc_samples: pd.DataFrame, scenes: pd.DataFrame, sample_scene: pd.DataFrame) -> str:
    n_samples = len(hplc_samples)
    n_matched = int((hplc_samples["match_status"] == "matched").sum())
    n_unmatched = int((hplc_samples["match_status"] == "unmatched").sum())
    n_scene_total = len(scenes)
    n_effective_scene = int(sample_scene["scene_id"].dropna().nunique())
    return f"""# PACE-HPLC Open Repository Lists

This folder provides repository-facing index tables for the `{n_samples}` HPLC samples used in the paper and their corresponding PACE hyperspectral images.

## Files

- `hplc_202_samples_time_lat_lon.csv`
  - All `{n_samples}` HPLC samples with timestamp, latitude, longitude, and source metadata.
- `pace_image_repository_manifest_142.csv`
  - All `{n_scene_total}` processed PACE image scenes with region, scene id, coverage, and repository path placeholder.
- `pace_matched_scene_list.csv`
  - The subset of processed PACE scenes that are matched to at least one HPLC sample.
- `pace_valid78_effective_scene_list.csv`
  - The subset of PACE scenes used by the current 78-sample valid-spectrum benchmark.
- `hplc_202_to_pace_image_mapping.csv`
  - Sample-to-scene mapping table showing which PACE hyperspectral image corresponds to each HPLC sample.

## Current counts

- Total HPLC samples: `{n_samples}`
- HPLC samples with at least one matched PACE scene: `{n_matched}`
- HPLC samples without a matched PACE scene: `{n_unmatched}`
- Total processed PACE scenes in the repository manifest: `{n_scene_total}`
- Effective matched PACE scenes used by the current benchmark: `{n_effective_scene}`

## Repository placeholders to fill before release

- HPLC repository URL placeholder: `{HPLC_REPO_PLACEHOLDER}`
- PACE image repository URL placeholder: `{PACE_REPO_PLACEHOLDER}`

Replace these placeholders with the actual open repository links before publishing the supplementary materials or README.
"""


def build_manuscript_text() -> str:
    return (
        "补充材料同时提供三个开放索引列表："
        "（1）202 个 HPLC 样本的时间、经纬度及来源信息表；"
        "（2）本文使用的 PACE 高光谱图像仓库清单；"
        "（3）202 个 HPLC 样本与对应 PACE 高光谱图像的逐样本映射表。"
        "HPLC 样本获取方式与 PACE 图像获取方式均通过作者开源仓库公开，"
        f"HPLC 样本仓库链接记为 {HPLC_REPO_PLACEHOLDER}，PACE 图像仓库链接记为 {PACE_REPO_PLACEHOLDER}。"
        "读者可依据补充表中的样本编号、时间、经纬度和场景编号，直接追溯每一个 HPLC 样本及其对应的 PACE 高光谱场景。"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(MATCHED_CSV)
    valid_matched = pd.read_csv(VALID_MATCHED_CSV)
    unmatched = pd.read_csv(UNMATCHED_CSV)
    scene_meta = pd.read_csv(SCENE_META_CSV)

    hplc_samples = build_hplc_samples_table(matched, unmatched)
    scene_manifest = build_scene_repository_table(scene_meta)
    matched_scene_list, valid_scene_list = build_effective_scene_lists(scene_manifest, matched, valid_matched)
    sample_scene_map = build_sample_to_scene_table(hplc_samples, matched)

    hplc_samples.to_csv(OUT_DIR / "hplc_202_samples_time_lat_lon.csv", index=False, encoding="utf-8-sig")
    scene_manifest.to_csv(OUT_DIR / "pace_image_repository_manifest_142.csv", index=False, encoding="utf-8-sig")
    matched_scene_list.to_csv(OUT_DIR / "pace_matched_scene_list.csv", index=False, encoding="utf-8-sig")
    valid_scene_list.to_csv(OUT_DIR / "pace_valid78_effective_scene_list.csv", index=False, encoding="utf-8-sig")
    sample_scene_map.to_csv(OUT_DIR / "hplc_202_to_pace_image_mapping.csv", index=False, encoding="utf-8-sig")

    (OUT_DIR / "README.md").write_text(build_markdown_summary(hplc_samples, scene_manifest, sample_scene_map), encoding="utf-8")
    (OUT_DIR / "manuscript_data_availability_text.txt").write_text(build_manuscript_text(), encoding="utf-8")

    print(f"Output directory: {OUT_DIR}")
    print(f"HPLC samples: {len(hplc_samples)}")
    print(f"Matched scenes used by sample mapping: {sample_scene_map['scene_id'].dropna().nunique()}")
    print(f"Valid-spectrum benchmark scenes: {valid_scene_list['scene_id'].nunique()}")


if __name__ == "__main__":
    main()
