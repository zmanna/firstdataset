from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, STANDARDIZED_FEATURE_COLUMNS, load_qsar_biodegradation
from .week9_validation import run_week9_validation_for_arrays


TIER1_BASELINE_FEATURES = list(STANDARDIZED_FEATURE_COLUMNS)

TIER2_PROXY_FEATURES = [
    "heteroatom_total",
    "heteroatom_fraction",
    "halogen_presence",
    "aromatic_substitution_signal",
    "donor_acceptor_proxy",
    "donor_density",
    "polarity_proxy",
    "mass_topology_proxy",
    "ring_topology_proxy",
    "carbon_hetero_ratio_proxy",
    "nitro_ester_signal",
    "electronic_balance_proxy",
]

TIER3_FUTURE_QUANTUM_FEATURES = [
    "total_energy",
    "dipole_moment",
    "polarizability",
    "HOMO_energy",
    "LUMO_energy",
    "HOMO_LUMO_gap",
    "quadrupole_moment",
    "partial_charge_dispersion",
]


@dataclass(frozen=True)
class Week10FeatureBundle:
    baseline_X: pd.DataFrame
    enhanced_X: pd.DataFrame
    y_binary: np.ndarray


def build_tier2_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    heteroatom_total = frame["nO"] + frame["nN"] + frame["nX"]
    nHM_safe = frame["nHM"].replace(0, 1)

    proxies = pd.DataFrame(
        {
            "heteroatom_total": heteroatom_total,
            "heteroatom_fraction": heteroatom_total / nHM_safe,
            "halogen_presence": (frame["nX"] > 0).astype(int),
            "aromatic_substitution_signal": frame["nCb_minus"] + frame["nArNO2"] + frame["nArCOOR"],
            "donor_acceptor_proxy": frame["nHDon"] + frame["nO"] + frame["nN"],
            "donor_density": frame["nHDon"] / nHM_safe,
            "polarity_proxy": frame["Me"] + frame["Mi"] + frame["SpPosA_B_p"] + frame["SdO"],
            "mass_topology_proxy": frame["HyWi_B_m"] + frame["SpMax_B_m"] + frame["SM6_B_m"],
            "ring_topology_proxy": frame["nCIR"] + frame["nCrt"] + frame["LOC"],
            "carbon_hetero_ratio_proxy": frame["C_percent"] / (heteroatom_total + 1.0),
            "nitro_ester_signal": frame["nArNO2"] + frame["nArCOOR"],
            "electronic_balance_proxy": frame["SpMax_L"] + frame["SpMax_A"] + frame["Psi_i_A"] + frame["Psi_i_1d"],
        }
    )
    return proxies.astype(np.float64)


def build_week10_feature_bundle() -> Week10FeatureBundle:
    bundle = load_qsar_biodegradation(target_as_category=False)
    rename_map = dict(zip(FEATURE_COLUMNS, STANDARDIZED_FEATURE_COLUMNS))
    baseline_X = bundle.X.rename(columns=rename_map).copy()
    tier2 = build_tier2_proxy_features(baseline_X)
    enhanced_X = pd.concat([baseline_X, tier2], axis=1)
    y_binary = (bundle.y.to_numpy(dtype=np.int64) == 2).astype(int)
    return Week10FeatureBundle(
        baseline_X=baseline_X,
        enhanced_X=enhanced_X,
        y_binary=y_binary,
    )


def run_week10_feature_evaluation(*, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = build_week10_feature_bundle()
    diagnostics_base, results_base = run_week9_validation_for_arrays(
        bundle.baseline_X.to_numpy(dtype=np.float64),
        bundle.y_binary,
        random_state=random_state,
    )
    diagnostics_enhanced, results_enhanced = run_week9_validation_for_arrays(
        bundle.enhanced_X.to_numpy(dtype=np.float64),
        bundle.y_binary,
        random_state=random_state,
    )

    diagnostics_base = diagnostics_base.assign(feature_set="tier1_baseline")
    diagnostics_enhanced = diagnostics_enhanced.assign(feature_set="tier1_plus_tier2")
    results_base = results_base.assign(feature_set="tier1_baseline")
    results_enhanced = results_enhanced.assign(feature_set="tier1_plus_tier2")

    diagnostics = pd.concat([diagnostics_base, diagnostics_enhanced], ignore_index=True)
    results = pd.concat([results_base, results_enhanced], ignore_index=True)
    return diagnostics, results


def write_week10_chart(results: pd.DataFrame, output_path: str | Path) -> Path:
    summary = (
        results.groupby(["feature_set", "sampling", "model_name"])[["accuracy", "rb_recall"]]
        .mean()
        .reset_index()
    )
    baseline = summary[summary["feature_set"] == "tier1_baseline"]
    enhanced = summary[summary["feature_set"] == "tier1_plus_tier2"]

    labels = [
        f"{sampling}\n{model}"
        for sampling, model in zip(baseline["sampling"], baseline["model_name"])
    ]
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width / 2, baseline["accuracy"], width=width, label="Tier 1 baseline")
    axes[0].bar(x + width / 2, enhanced["accuracy"], width=width, label="Tier 1 + Tier 2")
    axes[0].set_title("Week 10 Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2, baseline["rb_recall"], width=width, label="Tier 1 baseline")
    axes[1].bar(x + width / 2, enhanced["rb_recall"], width=width, label="Tier 1 + Tier 2")
    axes[1].set_title("Week 10 RB Recall Comparison")
    axes[1].set_ylabel("RB Recall")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].legend()

    fig.tight_layout()
    chart_path = Path(output_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path
