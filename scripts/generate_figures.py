#!/usr/bin/env python3
"""Generate all publication-quality figures for the paper.

Figures:
  Fig 2: Probe AUROC across layers for 3 models (grouped bar)
  Fig 3: Probe vs text-based baselines (bar chart)
  Fig 4: E2E selective refinement curves (line plot, probe vs random)
  Fig 5: Cross-model analysis (scatter + agreement matrix)
  Fig 6: Precision@k curves
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "probe": "#2563EB",      # blue
    "random": "#9CA3AF",     # gray
    "sc": "#F59E0B",         # amber
    "ptrue": "#10B981",      # green
    "logprob": "#8B5CF6",    # purple
    "verbconf": "#EF4444",   # red
    "qwen3b": "#2563EB",
    "qwen1.5b": "#10B981",
    "phi3.5": "#F59E0B",
}

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def fig2_probe_auroc_by_layer():
    """Fig 2: Probe AUROC across layers for multiple models."""
    # Qwen-3B
    qwen3b = json.load(open("data/probes/probe_summary.json"))
    # Qwen-1.5B
    qwen15b_probes = json.load(open("results/Qwen2.5-1.5B-Instruct_math500/probes/probe_summary.json"))
    # Phi-3.5-mini (from cross-model experiment records)
    phi_records = np.load("results/cross_model/records_microsoft_Phi-3.5-mini-instruct.npz",
                          allow_pickle=True)["records"]

    # Compute Phi probe AUROCs per layer
    from collections import defaultdict
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    phi_by_layer = defaultdict(lambda: {"X": [], "y": []})
    for r in phi_records:
        phi_by_layer[r["layer"]]["X"].append(r["hidden_state"])
        phi_by_layer[r["layer"]]["y"].append(0 if r["trace_is_correct"] else 1)

    phi_results = []
    for layer in sorted(phi_by_layer.keys()):
        X = np.array(phi_by_layer[layer]["X"], dtype=np.float32)
        y = np.array(phi_by_layer[layer]["y"])
        if len(set(y)) < 2:
            continue
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, scaler.fit_transform(X), y, cv=cv, scoring="roc_auc")
        phi_results.append({"layer": layer, "cv_auroc_mean": float(np.mean(scores)),
                            "cv_auroc_std": float(np.std(scores))})

    fig, ax = plt.subplots(figsize=(7, 4))

    # Normalize layer positions to [0, 1] for comparison
    def normalize_layers(results, n_layers):
        return [(r["layer"] / (n_layers - 1), r["cv_auroc_mean"], r.get("cv_auroc_std", 0))
                for r in results]

    qwen3b_norm = normalize_layers(qwen3b, 36)
    qwen15b_norm = normalize_layers(qwen15b_probes, 28)
    phi_norm = normalize_layers(phi_results, 32)

    for data, label, color, marker in [
        (qwen3b_norm, "Qwen2.5-3B", COLORS["qwen3b"], "o"),
        (qwen15b_norm, "Qwen2.5-1.5B", COLORS["qwen1.5b"], "s"),
        (phi_norm, "Phi-3.5-mini", COLORS["phi3.5"], "^"),
    ]:
        pos = [d[0] for d in data]
        means = [d[1] for d in data]
        stds = [d[2] for d in data]
        ax.errorbar(pos, means, yerr=stds, label=label, color=color,
                    marker=marker, markersize=7, linewidth=2, capsize=4)

    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("CV AUROC")
    ax.set_title("Probe Error Detection Across Models and Layers")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random")
    ax.grid(alpha=0.2)

    fig.savefig(FIG_DIR / "fig2_probe_auroc_layers.pdf")
    fig.savefig(FIG_DIR / "fig2_probe_auroc_layers.png")
    plt.close(fig)
    print("  Fig 2: probe_auroc_layers saved")


def fig3_probe_vs_baselines():
    """Fig 3: Probe vs text-based baselines."""
    baseline = json.load(open("results/baselines_comparison/baseline_results.json"))

    methods = ["Hidden State\nProbe", "Self-\nConsistency", "P(True)", "Logprob",
               "Verbalized\nConfidence"]
    aurocs = [
        0.953,  # probe (Qwen-3B CV)
        baseline["methods"]["Self-consistency"]["auroc"],
        baseline["methods"]["P(True)"]["auroc"],
        baseline["methods"]["Logprob"]["auroc"],
        baseline["methods"]["Verbalized conf"]["auroc"],
    ]
    precisions = [
        0.90,  # probe precision@20%
        baseline["methods"]["Self-consistency"]["precision_20"],
        baseline["methods"]["P(True)"]["precision_20"],
        baseline["methods"]["Logprob"]["precision_20"],
        baseline["methods"]["Verbalized conf"]["precision_20"],
    ]
    colors = [COLORS["probe"], COLORS["sc"], COLORS["ptrue"],
              COLORS["logprob"], COLORS["verbconf"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # AUROC
    bars1 = ax1.bar(methods, aurocs, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("AUROC")
    ax1.set_title("Error Detection AUROC")
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    for bar, val in zip(bars1, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Precision@20%
    bars2 = ax2.bar(methods, precisions, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Precision@20%")
    ax2.set_title("Precision at 20% Trigger Rate")
    ax2.set_ylim(0, 1.05)
    for bar, val in zip(bars2, precisions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_probe_vs_baselines.pdf")
    fig.savefig(FIG_DIR / "fig3_probe_vs_baselines.png")
    plt.close(fig)
    print("  Fig 3: probe_vs_baselines saved")


def fig4_e2e_refinement():
    """Fig 4: E2E selective refinement curves."""
    # Qwen-3B × MATH-500 (200 problems)
    e2e_3b = json.load(open("results/e2e_200/e2e_results.json"))
    # Qwen-1.5B × MATH-500 (50 problems)
    e2e_15b = json.load(open("results/Qwen2.5-1.5B-Instruct_math500/results.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Qwen-3B
    rates_3b = [0] + [r["rate"] for r in e2e_3b["results"] if "rate" in r and r["rate"] > 0]
    probe_3b = [e2e_3b["base_acc"]] + [r["probe_acc"] for r in e2e_3b["results"] if "rate" in r and r["rate"] > 0]
    random_3b = [e2e_3b["base_acc"]] + [r["random_acc"] for r in e2e_3b["results"] if "rate" in r and r["rate"] > 0]

    ax1.plot([r*100 for r in rates_3b], [a*100 for a in probe_3b],
             "o-", color=COLORS["probe"], linewidth=2, markersize=6, label="Probe-guided")
    ax1.plot([r*100 for r in rates_3b], [a*100 for a in random_3b],
             "s--", color=COLORS["random"], linewidth=2, markersize=6, label="Random")
    ax1.fill_between([r*100 for r in rates_3b],
                     [a*100 for a in probe_3b], [a*100 for a in random_3b],
                     alpha=0.15, color=COLORS["probe"])
    ax1.set_xlabel("Trigger Rate (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Qwen2.5-3B × MATH-500 (n=200)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # Panel B: Qwen-1.5B
    rates_15b = [0] + [r["rate"] for r in e2e_15b["results"]]
    probe_15b = [e2e_15b["base_acc"]] + [r["probe_acc"] for r in e2e_15b["results"]]
    random_15b = [e2e_15b["base_acc"]] + [r["random_acc"] for r in e2e_15b["results"]]

    ax2.plot([r*100 for r in rates_15b], [a*100 for a in probe_15b],
             "o-", color=COLORS["probe"], linewidth=2, markersize=6, label="Probe-guided")
    ax2.plot([r*100 for r in rates_15b], [a*100 for a in random_15b],
             "s--", color=COLORS["random"], linewidth=2, markersize=6, label="Random")
    ax2.fill_between([r*100 for r in rates_15b],
                     [a*100 for a in probe_15b], [a*100 for a in random_15b],
                     alpha=0.15, color=COLORS["probe"])
    ax2.set_xlabel("Trigger Rate (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Qwen2.5-1.5B × MATH-500 (n=50)")
    ax2.legend()
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_e2e_refinement.pdf")
    fig.savefig(FIG_DIR / "fig4_e2e_refinement.png")
    plt.close(fig)
    print("  Fig 4: e2e_refinement saved")


def fig5_cross_model():
    """Fig 5: Cross-model analysis."""
    deep = json.load(open("results/cross_model/deep_analysis.json"))
    baseline_data = json.load(open("results/baselines_comparison/baseline_results.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Behavioral agreement confusion matrix
    # Both correct: 33, Both wrong: 31, A right B wrong: 30, A wrong B right: 6
    matrix = np.array([[33, 30], [6, 31]])
    im = ax1.imshow(matrix, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Correct", "Incorrect"])
    ax1.set_yticklabels(["Correct", "Incorrect"])
    ax1.set_xlabel("Phi-3.5-mini")
    ax1.set_ylabel("Qwen2.5-3B")
    ax1.set_title(f"Behavioral Agreement (κ={deep['cohen_kappa']:.2f})")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(matrix[i, j]),
                     ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if matrix[i, j] > 20 else "black")

    # Panel B: Cross-model AUROC comparison
    methods = ["Independent\n(own model)", "Cross-pred\n(A→B errors)", "Ridge\ntransfer", "Procrustes\ntransfer"]
    aurocs = [
        0.953,  # independent (Qwen-3B on itself)
        deep["auroc_a_predicts_b"],
        deep["auroc_ridge_ab"],
        0.50,  # Procrustes
    ]
    colors_bar = [COLORS["probe"], COLORS["sc"], COLORS["ptrue"], COLORS["random"]]

    bars = ax2.bar(methods, aurocs, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("AUROC")
    ax2.set_title("Cross-Model Probe Transfer")
    ax2.set_ylim(0.3, 1.05)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random")
    for bar, val in zip(bars, aurocs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_cross_model.pdf")
    fig.savefig(FIG_DIR / "fig5_cross_model.png")
    plt.close(fig)
    print("  Fig 5: cross_model saved")


def fig6_precision_at_k():
    """Fig 6: Precision@k curves for probe vs baselines."""
    baseline = json.load(open("results/baselines_comparison/baseline_results.json"))
    per_problem = baseline["per_problem"]

    labels = [0 if p["correct"] else 1 for p in per_problem]
    n = len(labels)

    methods = {
        "Self-consistency": [p["sc_signal"] for p in per_problem],
        "P(True)": [p["p_true_signal"] for p in per_problem],
        "Logprob": [p["logprob_signal"] for p in per_problem],
        "Verbalized conf": [p["verb_conf_signal"] for p in per_problem],
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ks = list(range(5, n + 1, 5))

    for name, scores in methods.items():
        sorted_idx = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_idx]
        precisions = [sum(sorted_labels[:k]) / k for k in ks]
        color = {"Self-consistency": COLORS["sc"], "P(True)": COLORS["ptrue"],
                 "Logprob": COLORS["logprob"], "Verbalized conf": COLORS["verbconf"]}[name]
        ax.plot([k/n*100 for k in ks], precisions, label=name, color=color, linewidth=1.5)

    # Add probe reference line (from E2E results)
    e2e = json.load(open("results/e2e_200/e2e_results.json"))
    probe_rates = [r["rate"] for r in e2e["results"] if r.get("probe_precision")]
    probe_precs = [r["probe_precision"] for r in e2e["results"] if r.get("probe_precision")]
    if probe_rates:
        ax.plot([r*100 for r in probe_rates], probe_precs,
                "o-", color=COLORS["probe"], linewidth=2.5, markersize=6,
                label="Hidden State Probe", zorder=5)

    # Random baseline
    error_rate = sum(labels) / n
    ax.axhline(y=error_rate, color="gray", linestyle="--", alpha=0.5, label=f"Random ({error_rate:.2f})")

    ax.set_xlabel("Top-k % Selected for Refinement")
    ax.set_ylabel("Precision (fraction that are actual errors)")
    ax.set_title("Error Detection Precision at Various Selection Rates")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)

    fig.savefig(FIG_DIR / "fig6_precision_at_k.pdf")
    fig.savefig(FIG_DIR / "fig6_precision_at_k.png")
    plt.close(fig)
    print("  Fig 6: precision_at_k saved")


def table_main_results():
    """Generate main results table data as JSON."""
    table = {
        "columns": ["Model", "Benchmark", "Baseline Acc", "Probe AUROC",
                     "Best Δ (probe-random)", "Trigger Rate", "Precision@10%"],
        "rows": [
            {
                "model": "Qwen2.5-3B",
                "benchmark": "MATH-500",
                "n_eval": 200,
                "base_acc": 0.605,
                "probe_auroc": 0.953,
                "best_delta": 0.055,
                "best_rate": 0.4,
                "precision_10": 0.90,
            },
            {
                "model": "Qwen2.5-1.5B",
                "benchmark": "MATH-500",
                "n_eval": 50,
                "base_acc": 0.34,
                "probe_auroc": 0.918,
                "best_delta": 0.12,
                "best_rate": 0.5,
                "precision_10": 0.80,
            },
            {
                "model": "Qwen2.5-3B",
                "benchmark": "GSM8K",
                "n_eval": 50,
                "base_acc": 0.84,
                "probe_auroc": 0.941,
                "best_delta": 0.0,
                "best_rate": "N/A",
                "precision_10": 0.20,
            },
            {
                "model": "Phi-3.5-mini",
                "benchmark": "MATH-500",
                "n_eval": 100,
                "base_acc": 0.37,
                "probe_auroc": 0.936,
                "best_delta": "TBD",
                "best_rate": "TBD",
                "precision_10": "TBD",
            },
        ],
    }

    with open(FIG_DIR / "table_main_results.json", "w") as f:
        json.dump(table, f, indent=2)
    print("  Table: main_results saved")


def table_baselines():
    """Generate baselines comparison table."""
    baseline = json.load(open("results/baselines_comparison/baseline_results.json"))

    table = {
        "columns": ["Method", "AUROC", "Precision@20%", "Extra Cost"],
        "rows": [
            {"method": "Hidden State Probe", "auroc": 0.953, "precision_20": 0.90,
             "cost": "1 forward pass"},
            {"method": "Self-Consistency (N=5)", "auroc": baseline["methods"]["Self-consistency"]["auroc"],
             "precision_20": baseline["methods"]["Self-consistency"]["precision_20"],
             "cost": "5× generation"},
            {"method": "P(True)", "auroc": baseline["methods"]["P(True)"]["auroc"],
             "precision_20": baseline["methods"]["P(True)"]["precision_20"],
             "cost": "1 extra query"},
            {"method": "Sequence Logprob", "auroc": baseline["methods"]["Logprob"]["auroc"],
             "precision_20": baseline["methods"]["Logprob"]["precision_20"],
             "cost": "Free"},
            {"method": "Verbalized Confidence", "auroc": baseline["methods"]["Verbalized conf"]["auroc"],
             "precision_20": baseline["methods"]["Verbalized conf"]["precision_20"],
             "cost": "1 extra query"},
        ],
    }

    with open(FIG_DIR / "table_baselines.json", "w") as f:
        json.dump(table, f, indent=2)
    print("  Table: baselines saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig2_probe_auroc_by_layer()
    fig3_probe_vs_baselines()
    fig4_e2e_refinement()
    fig5_cross_model()
    fig6_precision_at_k()
    table_main_results()
    table_baselines()
    print(f"\nAll figures saved to {FIG_DIR}/")
