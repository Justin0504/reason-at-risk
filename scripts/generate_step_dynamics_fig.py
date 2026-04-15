#!/usr/bin/env python3
"""Generate step dynamics figure for the paper."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load data
with open("results/step_dynamics_3b/step_dynamics_summary.json") as f:
    d3b = json.load(f)
with open("results/step_dynamics_7b/step_dynamics_summary.json") as f:
    d7b = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

x = np.linspace(0, 1, 10)
labels_x = ["0%", "", "20%", "", "40%", "", "60%", "", "80%", "100%"]

for ax, data, title in [(ax1, d3b, "Qwen2.5-3B"), (ax2, d7b, "Qwen2.5-7B")]:
    c_mean = np.array(data["avg_trajectory_correct"])
    w_mean = np.array(data["avg_trajectory_wrong"])
    c_std = np.array(data["std_trajectory_correct"])
    w_std = np.array(data["std_trajectory_wrong"])

    ax.plot(x, c_mean, "o-", color="#2196F3", label="Correct", markersize=4, linewidth=1.5)
    ax.fill_between(x, c_mean - c_std, c_mean + c_std, alpha=0.15, color="#2196F3")
    ax.plot(x, w_mean, "s-", color="#F44336", label="Wrong", markersize=4, linewidth=1.5)
    ax.fill_between(x, w_mean - w_std, w_mean + w_std, alpha=0.15, color="#F44336")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Relative position in trace", fontsize=9)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=7)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Probe error score", fontsize=9)
ax1.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("paper/figures/fig_step_dynamics.pdf", bbox_inches="tight", dpi=300)
plt.savefig("results/figures/fig_step_dynamics.pdf", bbox_inches="tight", dpi=300)
print("Saved fig_step_dynamics.pdf")
