#!/usr/bin/env python3
"""Generate figures for the solution experiments section of the paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

from pathlib import Path

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


def fig_activation_steering():
    """Figure: Activation steering results for 3B and 7B."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 3B results
    data_3b = json.load(open("results/activation_steering/results.json"))
    alphas_3b = sorted([float(a) for a in data_3b["alphas"].keys()])
    accs_3b = [data_3b["alphas"][str(a)]["accuracy"] for a in alphas_3b]

    ax1.plot(alphas_3b, accs_3b, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax1.axhline(y=data_3b["base_acc"], color='gray', linestyle='--', alpha=0.7, label='Baseline (α=0)')
    ax1.set_xlabel('Steering Strength (α)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Qwen2.5-3B-Instruct')
    ax1.legend()
    ax1.set_ylim(0.40, 0.65)
    ax1.grid(True, alpha=0.3)

    # 7B results
    data_7b = json.load(open("results/activation_steering_7b/results.json"))
    alphas_7b = sorted([float(a) for a in data_7b["alphas"].keys()])
    accs_7b = [data_7b["alphas"][str(a)]["accuracy"] for a in alphas_7b]

    ax2.plot(alphas_7b, accs_7b, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax2.axhline(y=data_7b["base_acc"], color='gray', linestyle='--', alpha=0.7, label='Baseline (α=0)')
    ax2.set_xlabel('Steering Strength (α)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Qwen2.5-7B-Instruct')
    ax2.legend()
    ax2.set_ylim(0.50, 0.70)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_activation_steering.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "fig_activation_steering.png", dpi=150, bbox_inches='tight')
    print("Saved fig_activation_steering")
    plt.close()


def fig_probe_bon():
    """Figure: Probe-guided Best-of-N for both 3B and 7B."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model_name, path in [
        (ax1, 'Qwen2.5-3B', 'results/probe_bon_3b/results.json'),
        (ax2, 'Qwen2.5-7B', 'results/probe_bon_7b/results.json'),
    ]:
        data = json.load(open(path))
        ns = sorted([int(n) for n in data["results"].keys()])
        greedy = [data["results"][str(n)]["greedy"] for n in ns]
        random_acc = [data["results"][str(n)]["random"] for n in ns]
        mv = [data["results"][str(n)]["majority_vote"] for n in ns]
        probe = [data["results"][str(n)]["probe_bon"] for n in ns]
        oracle = [data["results"][str(n)]["oracle"] for n in ns]

        ax.plot(ns, greedy, 's--', color='gray', linewidth=1.5, markersize=7, label='Greedy')
        ax.plot(ns, random_acc, 'o-', color='#9E9E9E', linewidth=1.5, markersize=7, label='Random-of-N')
        ax.plot(ns, mv, 'D-', color='#FF9800', linewidth=2, markersize=7, label='Majority Vote')
        ax.plot(ns, probe, 's-', color='#2196F3', linewidth=2, markersize=8, label='Probe-Guided BoN')
        ax.plot(ns, oracle, '^--', color='#4CAF50', linewidth=1.5, markersize=7, label='Oracle')

        ax.set_xlabel('Number of Samples (N)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Probe-Guided Best-of-N ({model_name})')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.50, 0.95)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_probe_bon.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "fig_probe_bon.png", dpi=150, bbox_inches='tight')
    print("Saved fig_probe_bon")
    plt.close()


def fig_self_correction_7b():
    """Figure: Self-correction results for 7B."""
    data = json.load(open("results/self_correction_7b/results.json"))

    strategies = []
    accs = []
    retries = []
    colors = []

    strategies.append('Greedy')
    accs.append(data["results"]["greedy"]["acc"])
    retries.append(0)
    colors.append('#9E9E9E')

    # Probe-triggered at various thresholds
    for key in sorted(data["results"].keys()):
        if key.startswith("probe_t"):
            t = data["results"][key]["threshold"]
            strategies.append(f'Probe (t={t})')
            accs.append(data["results"][key]["acc"])
            retries.append(data["results"][key]["retries"])
            colors.append('#2196F3')

    strategies.append('Always Retry')
    accs.append(data["results"]["always_retry"]["acc"])
    retries.append(data["results"]["always_retry"]["retries"])
    colors.append('#FF9800')

    strategies.append('Best-of-Two')
    accs.append(data["results"]["best_of_two"]["acc"])
    retries.append(data["results"]["best_of_two"]["retries"])
    colors.append('#E91E63')

    strategies.append('Oracle')
    accs.append(data["results"]["oracle"]["acc"])
    retries.append(data["results"]["oracle"]["retries"])
    colors.append('#4CAF50')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of accuracies
    x = range(len(strategies))
    bars = ax1.bar(x, accs, color=colors, alpha=0.8)
    ax1.axhline(y=data["initial_acc"], color='gray', linestyle='--', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Self-Correction Accuracy (Qwen2.5-7B)')
    ax1.set_ylim(0.60, 0.75)
    ax1.grid(True, alpha=0.3, axis='y')

    # Accuracy vs retries scatter
    for i, (s, a, r) in enumerate(zip(strategies, accs, retries)):
        ax2.scatter(r, a, color=colors[i], s=100, zorder=5)
        ax2.annotate(s, (r, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax2.axhline(y=data["initial_acc"], color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Retries')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Efficiency: Accuracy vs Compute')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_self_correction_7b.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "fig_self_correction_7b.png", dpi=150, bbox_inches='tight')
    print("Saved fig_self_correction_7b")
    plt.close()


def fig_solution_summary():
    """Combined figure: all three solution methods side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Activation Steering
    ax = axes[0]
    data_3b = json.load(open("results/activation_steering/results.json"))
    data_7b = json.load(open("results/activation_steering_7b/results.json"))

    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    deltas_3b = [data_3b["alphas"][str(a)]["accuracy"] - data_3b["base_acc"] for a in alphas]
    deltas_7b = [data_7b["alphas"][str(a)]["accuracy"] - data_7b["base_acc"] for a in alphas]

    ax.plot(alphas, [d * 100 for d in deltas_3b], 'o-', color='#2196F3', linewidth=2, label='3B')
    ax.plot(alphas, [d * 100 for d in deltas_7b], 's-', color='#E91E63', linewidth=2, label='7B')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Δ Accuracy (%)')
    ax.set_title('(a) Activation Steering')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Probe-Guided BoN (both models)
    ax = axes[1]
    ns = [1, 3, 5, 8, 12]
    for label, path, color, marker in [
        ('3B MV', 'results/probe_bon_3b/results.json', '#FF9800', 'D'),
        ('3B Probe', 'results/probe_bon_3b/results.json', '#2196F3', 's'),
        ('7B MV', 'results/probe_bon_7b/results.json', '#FF9800', 'd'),
        ('7B Probe', 'results/probe_bon_7b/results.json', '#E91E63', 'o'),
    ]:
        bon_data = json.load(open(path))
        key = 'majority_vote' if 'MV' in label else 'probe_bon'
        vals = [bon_data["results"][str(n)][key] for n in ns]
        ls = '--' if 'MV' in label else '-'
        ax.plot(ns, [a * 100 for a in vals], marker + ls, color=color, linewidth=1.5, label=label, markersize=6)

    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(b) Probe-Guided Best-of-N')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Self-Correction (both models)
    ax = axes[2]
    sc_3b = json.load(open("results/self_correction_3b/results.json"))
    sc_7b = json.load(open("results/self_correction_7b/results.json"))

    methods = ['Greedy', 'Always\nRetry', 'Best-of\nTwo', 'Oracle']
    x_pos = np.arange(len(methods))
    width = 0.35

    accs_3b = [sc_3b["results"]["greedy"]["acc"], sc_3b["results"]["always_retry"]["acc"],
               sc_3b["results"]["best_of_two"]["acc"], sc_3b["results"]["oracle"]["acc"]]
    accs_7b = [sc_7b["results"]["greedy"]["acc"], sc_7b["results"]["always_retry"]["acc"],
               sc_7b["results"]["best_of_two"]["acc"], sc_7b["results"]["oracle"]["acc"]]

    ax.bar(x_pos - width/2, [a * 100 for a in accs_3b], width, color='#2196F3', alpha=0.8, label='3B')
    ax.bar(x_pos + width/2, [a * 100 for a in accs_7b], width, color='#E91E63', alpha=0.8, label='7B')
    ax.axhline(y=sc_3b["initial_acc"] * 100, color='#2196F3', linestyle='--', alpha=0.4)
    ax.axhline(y=sc_7b["initial_acc"] * 100, color='#E91E63', linestyle='--', alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(c) Self-Correction')
    ax.legend(fontsize=9)
    ax.set_ylim(50, 75)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_solution_summary.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "fig_solution_summary.png", dpi=150, bbox_inches='tight')
    print("Saved fig_solution_summary")
    plt.close()


if __name__ == "__main__":
    fig_activation_steering()
    fig_probe_bon()
    fig_self_correction_7b()
    fig_solution_summary()
    print("\nAll solution figures generated!")
