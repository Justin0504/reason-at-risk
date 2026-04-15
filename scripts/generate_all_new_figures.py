#!/usr/bin/env python3
"""Generate all new figures from supplementary experiments."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'font.family': 'serif',
})

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ===== Figure: Multi-model unified comparison =====
print("Generating multi-model comparison table figure...")

# Consolidate all model data
models_data = {
    "Qwen2.5-3B": {
        "params": "3B",
        "base_acc": 0.530,
        "probe_auroc_cv": 0.953,
        "first_step_auroc": 0.787,
        "max_score_auroc": 0.800,
        "selective_delta": 0.080,
        "selective_rate": 0.60,
    },
    "Qwen2.5-1.5B": {
        "params": "1.5B",
        "base_acc": 0.350,
        "probe_auroc_cv": 0.724,
        "first_step_auroc": 0.504,
        "max_score_auroc": 0.639,
        "selective_delta": 0.100,
        "selective_rate": 0.40,
    },
    "Phi-3.5-mini": {
        "params": "3.8B",
        "base_acc": 0.390,
        "probe_auroc_cv": 0.936,
        "first_step_auroc": None,  # not computed yet
        "max_score_auroc": None,
        "selective_delta": 0.040,
        "selective_rate": 0.40,
    },
}

# ===== Figure 12: Unfaithfulness — Verbalized Confidence vs Probe Score =====
print("Generating unfaithfulness figure...")

conf_data = json.load(open("results/verbalized_confidence/per_problem.json"))
traces = json.load(open("results/online_backtrack_v2/traces.json"))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel A: Scatter — verbalized confidence vs probe score
ax = axes[0]
for r in conf_data:
    t = next((t for t in traces if t["id"] == r["id"]), None)
    if t and r["confidence"]:
        color = '#55A868' if r["correct"] else '#C44E52'
        marker = 'o' if r["correct"] else 'x'
        ax.scatter(r["confidence"] + np.random.normal(0, 0.08),
                   r["max_probe_score"],
                   c=color, alpha=0.5, s=30, marker=marker, linewidths=1)

ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4)
ax.set_xlabel('Verbalized Confidence (1-5)')
ax.set_ylabel('Probe Error Score')
ax.set_title('(a) Text Confidence vs Probe')
# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#55A868', markersize=8, label='Correct'),
    Line2D([0], [0], marker='x', color='#C44E52', markersize=8, label='Wrong', markeredgewidth=2),
]
ax.legend(handles=legend_elements, loc='lower left')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.05, 1.05)

# Annotate unfaithful region
ax.fill_between([3.5, 5.5], 0.8, 1.05, alpha=0.1, color='red')
ax.text(4.5, 0.88, 'Unfaithful\nregion', ha='center', fontsize=8, color='red', style='italic')

# Panel B: Confidence distributions by correctness
ax = axes[1]
correct_conf = [r["confidence"] for r in conf_data if r["correct"] and r["confidence"]]
wrong_conf = [r["confidence"] for r in conf_data if not r["correct"] and r["confidence"]]

bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
ax.hist(correct_conf, bins=bins, alpha=0.6, color='#55A868', label=f'Correct (μ={np.mean(correct_conf):.1f})', edgecolor='black', linewidth=0.5)
ax.hist(wrong_conf, bins=bins, alpha=0.6, color='#C44E52', label=f'Wrong (μ={np.mean(wrong_conf):.1f})', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Verbalized Confidence')
ax.set_ylabel('Count')
ax.set_title('(b) Confidence Distribution')
ax.legend()

# Panel C: First-step AUROC — "Models know from step 1"
ax = axes[2]
# Load first-step data
fs_data = json.load(open("results/first_step_prediction/results.json"))
step_aurocs = fs_data["cumulative_step_aurocs"]

steps = [d["step"] for d in step_aurocs if d["n"] >= 20]
aurocs = [d["auroc"] for d in step_aurocs if d["n"] >= 20]

ax.bar(steps, aurocs, color='#4C72B0', edgecolor='black', linewidth=0.5)
ax.axhline(y=fs_data["max_score_auroc"], color='#DD8452', linestyle='--',
           label=f'Full trace: {fs_data["max_score_auroc"]:.3f}')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Steps Used (cumulative)')
ax.set_ylabel('AUROC')
ax.set_title('(c) Prediction from First k Steps')
ax.legend(loc='lower right')
ax.set_ylim(0.4, 0.9)

plt.tight_layout()
for fmt in ['pdf', 'png']:
    fig.savefig(output_dir / f'fig12_unfaithfulness.{fmt}', bbox_inches='tight')
print("  Saved fig12_unfaithfulness")

# ===== Figure 13: Compute efficiency — Probe vs Always-API =====
print("Generating compute efficiency figure...")

api_data = json.load(open("results/always_api/results.json"))
bt_data = json.load(open("results/online_backtrack_v2/results.json"))

fig, ax = plt.subplots(figsize=(7, 4.5))

# Points: (API fraction, accuracy)
# Baseline: 0% API
# Probe-guided at different thresholds: trigger_rate% API
# Always API: 100% API

api_fracs = [0]  # baseline
accs = [bt_data["base_acc"]]
labels_text = ['Baseline\n(3B only)']
colors = ['#bbb']

for r in bt_data["results"]:
    api_fracs.append(r["trigger_rate"])
    accs.append(r["backtrack_acc"])
    labels_text.append(f'Probe τ={r["threshold"]}\n({r["trigger_rate"]:.0%} API)')
    colors.append('#4C72B0')

api_fracs.append(1.0)
accs.append(api_data["acc_api"])
labels_text.append('Always API\n(100%)')
colors.append('#DD8452')

ax.scatter(api_fracs, accs, c=colors, s=120, zorder=5, edgecolors='black', linewidth=0.5)

for i, txt in enumerate(labels_text):
    offset_y = 0.012 if i > 0 else -0.015
    offset_x = 0
    if i == len(labels_text) - 1:
        offset_y = -0.02
    ax.annotate(txt, (api_fracs[i], accs[i]),
                textcoords="offset points", xytext=(0, 20 if offset_y > 0 else -25),
                ha='center', fontsize=8)

ax.set_xlabel('Fraction of Problems Sent to API')
ax.set_ylabel('Accuracy')
ax.set_title('Compute Efficiency: Probe-Guided vs Always-API')
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(0.48, 0.66)
ax.axhline(y=bt_data["base_acc"], color='gray', linestyle=':', alpha=0.3)

# Draw efficiency frontier
frontier_x = [0] + [r["trigger_rate"] for r in bt_data["results"]]
frontier_y = [bt_data["base_acc"]] + [r["backtrack_acc"] for r in bt_data["results"]]
ax.plot(frontier_x, frontier_y, '--', color='#4C72B0', alpha=0.3)

plt.tight_layout()
for fmt in ['pdf', 'png']:
    fig.savefig(output_dir / f'fig13_compute_efficiency.{fmt}', bbox_inches='tight')
print("  Saved fig13_compute_efficiency")

# ===== Figure 14: Multi-model comparison bar chart =====
print("Generating multi-model bar chart...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

model_names = list(models_data.keys())
x = np.arange(len(model_names))

# Panel A: Probe AUROC comparison
ax = axes[0]
cv_aurocs = [models_data[m]["probe_auroc_cv"] for m in model_names]
bars = ax.bar(x, cv_aurocs, color=['#4C72B0', '#55A868', '#DD8452'],
              edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylabel('AUROC')
ax.set_title('(a) Probe Detection AUROC')
ax.set_ylim(0.5, 1.0)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
for i, v in enumerate(cv_aurocs):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Panel B: Selective refinement delta
ax = axes[1]
deltas = [models_data[m]["selective_delta"] for m in model_names]
base_accs = [models_data[m]["base_acc"] for m in model_names]

width = 0.35
ax.bar(x - width/2, base_accs, width, label='Baseline', color='#bbb',
       edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, [b+d for b, d in zip(base_accs, deltas)], width,
       label='+ Probe Refinement', color=['#4C72B0', '#55A868', '#DD8452'],
       edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('(b) Selective Refinement Improvement')
ax.set_ylim(0, 0.75)
ax.legend(loc='upper right')

for i, (b, d) in enumerate(zip(base_accs, deltas)):
    ax.text(i + width/2, b + d + 0.01, f'+{d:.0%}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
for fmt in ['pdf', 'png']:
    fig.savefig(output_dir / f'fig14_multimodel.{fmt}', bbox_inches='tight')
print("  Saved fig14_multimodel")

plt.close('all')
print("\nAll figures saved!")
