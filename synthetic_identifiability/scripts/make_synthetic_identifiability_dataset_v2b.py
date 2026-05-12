from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    ROOT
    / "external_benchmark_data"
    / "synthetic_identifiability_dataset_v2"
    / "synthetic_identifiability_dataset_v2.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "external_benchmark_data" / "synthetic_identifiability_dataset_v2b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create synthetic_identifiability_dataset_v2b by relabeling v2 with a non-absolute "
            "Level-3 feasibility rule. Transition/coastal samples remain mostly hard negatives, "
            "but rare low-degeneracy, low-nuisance cases may pass."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def relabel_level3_nonabsolute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["decomposition_allowed_soft", "decomposition_allowed_hard", "degeneracy_risk"]:
        if col in out.columns:
            out[f"v2_original_{col}"] = out[col]

    regime = out["regime_name"].astype(str).to_numpy()
    clear_feasibility = pd.to_numeric(out["clear_feasibility_raw"], errors="coerce").fillna(0.0).to_numpy(float)
    ratio = pd.to_numeric(out["composition_vs_nuisance_ratio"], errors="coerce").fillna(0.0).to_numpy(float)
    margin = pd.to_numeric(out["dominant_margin"], errors="coerce").fillna(0.0).to_numpy(float)
    entropy = pd.to_numeric(out["fraction_entropy"], errors="coerce").fillna(1.0).to_numpy(float)
    min_sam = pd.to_numeric(out["min_active_pair_sam_deg"], errors="coerce").fillna(0.0).to_numpy(float)
    nuisance = pd.to_numeric(out["level3_nuisance_strength"], errors="coerce").fillna(9.0).to_numpy(float)
    cond = pd.to_numeric(out["jacobian_condition_number"], errors="coerce").fillna(1.0e6).to_numpy(float)

    # Continuous optical admissibility. These terms are intentionally simple:
    # strong composition signal, clear dominant margin, low entropy, separated active spectra,
    # mild nuisance optical terms, and non-explosive local condition number.
    ratio_ok = sigmoid((ratio - 0.13) / 0.055)
    margin_ok = sigmoid((margin - 0.28) / 0.08)
    entropy_ok = sigmoid((0.84 - entropy) / 0.10)
    sam_ok = sigmoid((min_sam - 2.5) / 1.8)
    nuisance_ok = sigmoid((1.45 - nuisance) / 0.28)
    cond_ok = sigmoid((120.0 - np.minimum(cond, 1000.0)) / 35.0)
    nonclear_mildness = np.power(
        np.clip(ratio_ok * margin_ok * entropy_ok * sam_ok * nuisance_ok * cond_ok, 0.0, 1.0),
        1.0 / 6.0,
    )

    score = np.zeros(len(out), dtype=float)
    clear_mask = regime == "clear"
    transition_mask = regime == "transition"
    coastal_mask = regime == "coastal"

    score[clear_mask] = 0.03 + 0.94 * clear_feasibility[clear_mask]
    score[transition_mask] = (
        0.03
        + 0.72
        * clear_feasibility[transition_mask]
        * np.power(nonclear_mildness[transition_mask], 0.55)
    )
    score[coastal_mask] = (
        0.02
        + 0.62
        * clear_feasibility[coastal_mask]
        * np.power(nonclear_mildness[coastal_mask], 0.75)
    )
    score = np.clip(score, 0.0, 1.0)

    out["level3_nonabsolute_mildness"] = nonclear_mildness
    out["level3_decomposition_allowed_soft"] = score
    out["decomposition_allowed_soft"] = score
    out["level3_decomposition_allowed_hard"] = (score >= 0.55).astype(int)
    out["decomposition_allowed_hard"] = out["level3_decomposition_allowed_hard"]
    out["degeneracy_risk"] = 1.0 - score
    return out


def write_summary(df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    summary = {
        "dataset_name": "synthetic_identifiability_dataset_v2b",
        "source": str(DEFAULT_INPUT_CSV),
        "label_semantics": (
            "Level-3 conditional delta-Rrs decomposition feasibility with non-absolute regime treatment"
        ),
        "decision_threshold": threshold,
        "n_rows": int(len(df)),
        "allowed_rate": float(df["decomposition_allowed_hard"].mean()),
        "allowed_rate_by_regime": df.groupby("regime_name")["decomposition_allowed_hard"].mean().to_dict(),
        "mean_allowed_soft_by_regime": df.groupby("regime_name")["decomposition_allowed_soft"].mean().to_dict(),
        "allowed_rate_by_type": df.groupby("sample_type")["decomposition_allowed_hard"].mean().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (output_dir / "synthetic_identifiability_dataset_v2b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=180)
    for ax, col, title in [
        (axes[0], "decomposition_allowed_soft", "Level-3 allowed score"),
        (axes[1], "level3_nonabsolute_mildness", "Non-clear mildness"),
        (axes[2], "composition_vs_nuisance_ratio", "Composition / nuisance"),
    ]:
        for regime_name, sub in df.groupby("regime_name"):
            ax.hist(sub[col], bins=35, alpha=0.45, label=regime_name)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_identifiability_v2b_label_distributions.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv, low_memory=False)
    out = relabel_level3_nonabsolute(df)
    out_csv = output_dir / "synthetic_identifiability_dataset_v2b.csv"
    out.to_csv(out_csv, index=False)
    write_summary(out, output_dir, args.threshold)
    save_figures(out, output_dir)
    readme = f"""# synthetic_identifiability_dataset_v2b

This dataset is a relabeled version of v2. It keeps the Level-3 conditional delta-Rrs decomposition target, but removes the absolute claim that transition/coastal water can never be decomposed.

Scientific interpretation:

- clear water is still the primary admissible domain;
- transition and coastal water are mostly hard negatives because nuisance absorption/backscattering and local degeneracy dominate;
- rare transition/coastal samples can pass if composition signal is strong, nuisance is mild, dominant margin is high, entropy is low, active spectra are separated, and the local condition number is not extreme.

This is a safer label design for review: regime is treated as an optical-risk proxy, not an absolute physical law.

Output CSV: `{out_csv}`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_csv": str(out_csv), "n_rows": int(len(out))}, indent=2))


if __name__ == "__main__":
    main()
