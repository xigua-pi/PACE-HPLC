from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calibrate_v52_partial_label_physics as cal
import species14_physical_utils as phys
from BDSFormer_v2_owt_qaa_enhanced import AdaptiveQAAEngine, resolve_lab_excel
from benchmark_pft_utils import build_physics_feature_frame
from generate_synthetic_identifiability_dataset import (
    FIT_MASK,
    PFTS,
    REGIME_NAMES,
    WAVELENGTHS,
    add_noise,
    build_pft_templates,
    forward_model,
    local_identifiability_metrics,
    min_active_pair_sam,
    normalize,
    relative_rmse,
    sample_fraction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "external_benchmark_data" / "synthetic_identifiability_dataset_v2"
DEFAULT_CALIBRATION_JSON = (
    ROOT
    / "outputs_BDSFormer_v1"
    / "v52_partial_label_calibration"
    / "v52_partial_label_calibration_bundle.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic_identifiability_dataset_v2 for Level-3 conditional delta-Rrs "
            "decomposition feasibility. Unlike v1, transition/coastal scenes are treated mostly "
            "as hard negatives for decomposition eligibility."
        )
    )
    parser.add_argument("--n-samples", type=int, default=7000)
    parser.add_argument("--n-degenerate-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--lab-excel", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


def sample_scene_level3(scenario: str, rng: np.random.Generator) -> tuple[int, float, float, float, float]:
    if scenario == "clear_feasible":
        regime = 0
        adg_scale = float(np.exp(rng.normal(-0.35, 0.22)))
        bbp_scale = float(np.exp(rng.normal(-0.28, 0.22)))
        noise_sigma = float(rng.uniform(2.0e-5, 5.5e-5))
    elif scenario == "clear_ambiguous":
        regime = 0
        adg_scale = float(np.exp(rng.normal(0.05, 0.38)))
        bbp_scale = float(np.exp(rng.normal(0.05, 0.38)))
        noise_sigma = float(rng.uniform(4.5e-5, 1.2e-4))
    elif scenario == "clear_hard_pair":
        regime = 0
        adg_scale = float(np.exp(rng.normal(-0.05, 0.35)))
        bbp_scale = float(np.exp(rng.normal(-0.05, 0.35)))
        noise_sigma = float(rng.uniform(4.0e-5, 1.1e-4))
    elif scenario == "transition_hard_negative":
        regime = 1
        adg_scale = float(np.exp(rng.normal(0.25, 0.35)))
        bbp_scale = float(np.exp(rng.normal(0.20, 0.35)))
        noise_sigma = float(rng.uniform(5.5e-5, 1.6e-4))
    elif scenario == "coastal_hard_negative":
        regime = 2
        adg_scale = float(np.exp(rng.normal(0.62, 0.42)))
        bbp_scale = float(np.exp(rng.normal(0.62, 0.42)))
        noise_sigma = float(rng.uniform(7.0e-5, 2.1e-4))
    elif scenario == "external_like_shift":
        regime = int(rng.choice([0, 1, 2], p=[0.18, 0.52, 0.30]))
        adg_scale = float(np.exp(rng.normal(0.48 + 0.15 * regime, 0.48)))
        bbp_scale = float(np.exp(rng.normal(0.35 + 0.18 * regime, 0.48)))
        noise_sigma = float(rng.uniform(8.0e-5, 2.5e-4))
    elif scenario == "low_margin":
        regime = int(rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20]))
        adg_scale = float(np.exp(rng.normal(0.05, 0.38)))
        bbp_scale = float(np.exp(rng.normal(0.05, 0.38)))
        noise_sigma = float(rng.uniform(4.0e-5, 1.4e-4))
    elif scenario == "hard_pair":
        regime = int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
        adg_scale = float(np.exp(rng.normal(0.00, 0.35)))
        bbp_scale = float(np.exp(rng.normal(0.00, 0.35)))
        noise_sigma = float(rng.uniform(4.0e-5, 1.3e-4))
    else:
        regime = int(rng.choice([0, 1, 2], p=[0.36, 0.34, 0.30]))
        adg_scale = float(np.exp(rng.normal(0.05, 0.45)))
        bbp_scale = float(np.exp(rng.normal(0.05, 0.45)))
        noise_sigma = float(rng.uniform(4.0e-5, 1.7e-4))
    total = float(np.exp(rng.normal(np.log(0.65 + 0.35 * regime), 0.50 + 0.08 * regime)))
    return regime, total, adg_scale, bbp_scale, noise_sigma


def sample_fraction_level3(regime: int, scenario: str, rng: np.random.Generator) -> np.ndarray:
    if scenario == "clear_feasible":
        return sample_fraction(0, "identifiable", rng)
    if scenario in {"clear_hard_pair", "hard_pair"}:
        return sample_fraction(regime, "hard_pair", rng)
    if scenario in {"clear_ambiguous", "low_margin"}:
        return sample_fraction(regime, "low_margin", rng)
    if scenario in {"transition_hard_negative", "coastal_hard_negative", "external_like_shift"}:
        return sample_fraction(regime, "nuisance_dominated", rng)
    return sample_fraction(regime, "mixed", rng)


def level3_allowed_score(
    regime: int,
    margin: float,
    entropy: float,
    min_sam: float,
    ratio: float,
    cond: float,
    noise_sigma: float,
    adg_scale: float,
    bbp_scale: float,
    pairmate_rrmse: float | None = None,
) -> tuple[float, dict[str, float]]:
    nuisance = float(np.sqrt(adg_scale * bbp_scale))
    z_clear = 0.0
    z_clear += 8.0 * (margin - 0.22)
    z_clear += 2.3 * (0.66 - entropy)
    z_clear += 0.42 * (min_sam - 2.8)
    z_clear += 2.2 * np.tanh((ratio - 0.045) / 0.028)
    z_clear -= 0.42 * np.log10(max(cond, 1.0) / 70.0)
    z_clear -= 2.1 * max(0.0, (noise_sigma - 7.5e-5) / 1e-4)
    z_clear -= 1.35 * max(0.0, nuisance - 1.05)
    if pairmate_rrmse is not None:
        z_clear -= 5.0 * max(0.0, 0.018 - pairmate_rrmse) / 0.018
    clear_feasibility = sigmoid(z_clear)

    # Level-3 decomposition is a clear-water conditional task. Non-clear water
    # may still support recognition/ranking, but not component-level delta-Rrs.
    if regime == 0:
        score = 0.03 + 0.94 * clear_feasibility
        regime_penalty = 0.0
    elif regime == 1:
        score = 0.02 + 0.10 * clear_feasibility
        regime_penalty = 0.88
    else:
        score = 0.005 + 0.04 * clear_feasibility
        regime_penalty = 0.96
    components = {
        "clear_feasibility_raw": float(clear_feasibility),
        "level3_regime_penalty": float(regime_penalty),
        "level3_nuisance_strength": nuisance,
    }
    return float(np.clip(score, 0.0, 1.0)), components


def make_row_level3(
    idx: int,
    qaa: AdaptiveQAAEngine,
    rng: np.random.Generator,
    aph_templates: np.ndarray,
    bbp_templates: np.ndarray,
    scenario: str,
    degenerate_pair_id: str = "",
    forced_frac: np.ndarray | None = None,
    forced_scene: tuple[int, float, float, float, float] | None = None,
    pairmate_rrmse: float | None = None,
) -> dict[str, object]:
    regime, total, adg_scale, bbp_scale, noise_sigma = (
        forced_scene if forced_scene else sample_scene_level3(scenario, rng)
    )
    frac = normalize(forced_frac) if forced_frac is not None else sample_fraction_level3(regime, scenario, rng)
    clean_rrs = forward_model(qaa, frac, total, regime, adg_scale, bbp_scale, aph_templates, bbp_templates)
    rrs = add_noise(clean_rrs, regime, noise_sigma, rng)
    sorted_frac = np.sort(frac)
    margin = float(sorted_frac[-1] - sorted_frac[-2])
    entropy = float(-np.sum(frac * np.log(np.clip(frac, 1e-12, None))) / np.log(len(frac)))
    min_sam = min_active_pair_sam(frac, aph_templates)
    metrics = local_identifiability_metrics(
        qaa,
        frac,
        total,
        regime,
        adg_scale,
        bbp_scale,
        noise_sigma,
        aph_templates,
        bbp_templates,
    )
    score, score_components = level3_allowed_score(
        regime=regime,
        margin=margin,
        entropy=entropy,
        min_sam=min_sam,
        ratio=metrics["composition_vs_nuisance_ratio"],
        cond=metrics["jacobian_condition_number"],
        noise_sigma=noise_sigma,
        adg_scale=adg_scale,
        bbp_scale=bbp_scale,
        pairmate_rrmse=pairmate_rrmse,
    )
    row: dict[str, object] = {
        "sample_uid": f"synthetic_ident_v2_{idx:07d}",
        "sample_type": scenario,
        "degenerate_pair_id": degenerate_pair_id,
        "regime_code": int(regime),
        "regime_name": REGIME_NAMES[int(regime)],
        "total_proxy": float(total),
        "adg_scale": float(adg_scale),
        "bbp_scale": float(bbp_scale),
        "noise_sigma": float(noise_sigma),
        "dominant_pft": PFTS[int(np.argmax(frac))],
        "co_dominant_pft": PFTS[int(np.argsort(frac)[-2])],
        "dominant_margin": margin,
        "fraction_entropy": entropy,
        "min_active_pair_sam_deg": min_sam,
        "level3_decomposition_allowed_soft": score,
        "level3_decomposition_allowed_hard": int(score >= 0.55),
        "decomposition_allowed_soft": score,
        "decomposition_allowed_hard": int(score >= 0.55),
        "degeneracy_risk": float(1.0 - score),
        "clean_rrs_mean": float(np.mean(clean_rrs)),
        "observed_rrs_mean": float(np.mean(rrs)),
        **score_components,
        **metrics,
    }
    for i, pft in enumerate(PFTS):
        row[f"target_fraction_{pft}"] = float(frac[i])
        row[f"target_pft_{pft}"] = float(frac[i] * total)
    for i, wl in enumerate(WAVELENGTHS.astype(int).tolist()):
        row[f"feat_rrs_{wl}"] = float(rrs[i])
        row[f"clean_rrs_{wl}"] = float(clean_rrs[i])
    return row


def paired_degenerate_rows_level3(
    start_idx: int,
    qaa: AdaptiveQAAEngine,
    rng: np.random.Generator,
    aph_templates: np.ndarray,
    bbp_templates: np.ndarray,
    n_pairs: int,
) -> list[dict[str, object]]:
    rows = []
    for pair_idx in range(n_pairs):
        regime, total, adg_scale, bbp_scale, noise_sigma = sample_scene_level3("hard_pair", rng)
        pair = (3, 4) if rng.random() < 0.75 else (6, 7)
        a = rng.uniform(0.22, 0.38)
        f1 = np.full(len(PFTS), 0.006)
        f2 = np.full(len(PFTS), 0.006)
        f1[pair[0]], f1[pair[1]] = a, 1.0 - a
        f2[pair[0]], f2[pair[1]] = 1.0 - a, a
        background = rng.dirichlet(np.ones(len(PFTS))) * 0.04
        f1 = normalize(f1 + background)
        f2 = normalize(f2 + background)
        pair_id = f"degpair_v2_{pair_idx:05d}_{PFTS[pair[0]]}_{PFTS[pair[1]]}"
        scene = (regime, total, adg_scale, bbp_scale, noise_sigma)
        clean_1 = forward_model(qaa, f1, total, regime, adg_scale, bbp_scale, aph_templates, bbp_templates)
        clean_2 = forward_model(qaa, f2, total, regime, adg_scale, bbp_scale, aph_templates, bbp_templates)
        rrmse = relative_rmse(clean_1, clean_2)
        r1 = make_row_level3(
            start_idx + 2 * pair_idx,
            qaa,
            rng,
            aph_templates,
            bbp_templates,
            "degenerate_pair",
            pair_id,
            f1,
            scene,
            pairmate_rrmse=rrmse,
        )
        r2 = make_row_level3(
            start_idx + 2 * pair_idx + 1,
            qaa,
            rng,
            aph_templates,
            bbp_templates,
            "degenerate_pair",
            pair_id,
            f2,
            scene,
            pairmate_rrmse=rrmse,
        )
        comp_l1 = float(np.sum(np.abs(f1 - f2)))
        r1["pairmate_clean_rrs_rrmse"] = rrmse
        r2["pairmate_clean_rrs_rrmse"] = rrmse
        r1["pairmate_composition_l1"] = comp_l1
        r2["pairmate_composition_l1"] = comp_l1
        rows.extend([r1, r2])
    return rows


def plot_summary_v2(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=170)
    for ax, col, title in [
        (axes[0], "composition_vs_nuisance_ratio", "Composition vs nuisance"),
        (axes[1], "decomposition_allowed_soft", "Level-3 allowed label"),
        (axes[2], "level3_regime_penalty", "Level-3 regime penalty"),
    ]:
        for regime_name, sub in df.groupby("regime_name"):
            ax.hist(sub[col], bins=35, alpha=0.45, label=regime_name)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_identifiability_v2_label_distributions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=170)
    sc = ax.scatter(
        df["dominant_margin"],
        df["composition_vs_nuisance_ratio"],
        c=df["decomposition_allowed_soft"],
        s=8,
        cmap="viridis",
        alpha=0.55,
    )
    ax.set_xlabel("Dominant margin")
    ax.set_ylabel("Composition-vs-nuisance ratio")
    ax.set_title("Level-3 conditional decomposition feasibility")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Level-3 allowed soft label")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_identifiability_v2_margin_vs_ratio.png")
    plt.close(fig)


def assign_split_v2(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    split = pd.Series("train", index=df.index, dtype=object)
    pair_ids = sorted([p for p in df["degenerate_pair_id"].dropna().unique().tolist() if str(p)])
    pair_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    pair_rng.shuffle(pair_ids)
    n_pair = len(pair_ids)
    pair_split: dict[str, str] = {}
    for idx, pair_id in enumerate(pair_ids):
        frac = idx / max(n_pair, 1)
        pair_split[pair_id] = "test" if frac < 0.10 else "val" if frac < 0.20 else "train"
    for pair_id, label in pair_split.items():
        split[df["degenerate_pair_id"].astype(str).eq(pair_id)] = label

    unpaired = np.array(df.index[df["degenerate_pair_id"].fillna("").astype(str).eq("")].to_numpy(), copy=True)
    rng.shuffle(unpaired)
    n = len(unpaired)
    test_idx = unpaired[: int(0.10 * n)]
    val_idx = unpaired[int(0.10 * n) : int(0.20 * n)]
    split.loc[test_idx] = "test"
    split.loc[val_idx] = "val"
    return split


def write_readme(out_dir: Path, summary: dict[str, object]) -> None:
    text = f"""# synthetic_identifiability_dataset_v2

This package is a synthetic training and stress-test dataset for the **Level-3 conditional delta-Rrs decomposition feasibility** task.

It is not a replacement for HPLC marker-pigment truth and must not be used to claim real composition accuracy. Its purpose is to train a lightweight physical-axis reject option.

Key design change relative to v1:

- clear-water scenes may be eligible for Level-3 decomposition;
- transition and coastal scenes are mostly hard negatives, even when a local spectral fit is possible;
- labels distinguish recognition/ranking identifiability from component-level delta-Rrs decomposition eligibility.

Rows: {summary["n_rows"]}
Allowed hard rate: {summary["allowed_rate"]:.3f}
Allowed rate by regime: {summary["allowed_rate_by_regime"]}

Main file: `{Path(summary["output_csv"]).name}`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    phys.apply_v52_calibration_bundle(bundle)
    lab_excel = resolve_lab_excel(ROOT, args.lab_excel)
    aph_templates, bbp_templates = build_pft_templates(bundle, lab_excel)
    qaa = AdaptiveQAAEngine(WAVELENGTHS)

    scenario_probs = {
        "clear_feasible": 0.23,
        "clear_ambiguous": 0.15,
        "clear_hard_pair": 0.10,
        "transition_hard_negative": 0.17,
        "coastal_hard_negative": 0.15,
        "external_like_shift": 0.09,
        "low_margin": 0.06,
        "hard_pair": 0.05,
    }
    scenarios = np.array(list(scenario_probs.keys()), dtype=object)
    probs = np.array(list(scenario_probs.values()), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for idx in range(int(args.n_samples)):
        scenario = str(rng.choice(scenarios, p=probs / probs.sum()))
        rows.append(make_row_level3(idx, qaa, rng, aph_templates, bbp_templates, scenario))
        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(json.dumps({"generated": idx + 1, "base_samples": int(args.n_samples)}))
    rows.extend(
        paired_degenerate_rows_level3(
            int(args.n_samples),
            qaa,
            rng,
            aph_templates,
            bbp_templates,
            int(args.n_degenerate_pairs),
        )
    )
    df = pd.DataFrame(rows)
    df["split"] = assign_split_v2(df, rng)

    rrs_cols = [f"feat_rrs_{int(w)}" for w in WAVELENGTHS]
    physics_features = build_physics_feature_frame(df, rrs_cols, WAVELENGTHS.astype(int).tolist())
    df = pd.concat([df, physics_features], axis=1)

    csv_path = out_dir / "synthetic_identifiability_dataset_v2.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "dataset_name": "synthetic_identifiability_dataset_v2",
        "label_semantics": "Level-3 conditional delta-Rrs decomposition feasibility",
        "n_rows": int(len(df)),
        "n_base_samples": int(args.n_samples),
        "n_degenerate_pairs": int(args.n_degenerate_pairs),
        "n_degenerate_rows": int(2 * args.n_degenerate_pairs),
        "allowed_rate": float(df["decomposition_allowed_hard"].mean()),
        "allowed_rate_by_regime": df.groupby("regime_name")["decomposition_allowed_hard"].mean().to_dict(),
        "mean_allowed_soft_by_regime": df.groupby("regime_name")["decomposition_allowed_soft"].mean().to_dict(),
        "allowed_rate_by_type": df.groupby("sample_type")["decomposition_allowed_hard"].mean().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "columns": {
            "rrs": rrs_cols,
            "composition": [f"target_fraction_{p}" for p in PFTS],
            "level3_labels": [
                "level3_decomposition_allowed_soft",
                "level3_decomposition_allowed_hard",
                "decomposition_allowed_soft",
                "decomposition_allowed_hard",
                "degeneracy_risk",
                "clear_feasibility_raw",
                "level3_regime_penalty",
            ],
            "diagnostics": [
                "composition_vs_nuisance_ratio",
                "jacobian_condition_number",
                "min_active_pair_sam_deg",
            ],
        },
        "output_csv": str(csv_path),
    }
    (out_dir / "synthetic_identifiability_dataset_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_summary_v2(df, out_dir)
    write_readme(out_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
