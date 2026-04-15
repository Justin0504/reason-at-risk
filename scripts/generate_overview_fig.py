#!/usr/bin/env python3
"""Generate a polished two-column overview figure (Figure 1) for the paper.

Layout (left to right, three panels):
  (a) Setup: LLM generates CoT, we extract hidden states
  (b) Finding: Hidden states detect errors; text doesn't
  (c) Implication: Internal interventions fail; routing works
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────
C_BG       = "#FAFAFA"
C_BLUE     = "#1565C0"   # correct / probe
C_RED      = "#C62828"    # wrong / error
C_GREEN    = "#2E7D32"    # success / routing
C_ORANGE   = "#EF6C00"    # warning
C_GRAY     = "#616161"
C_LGRAY    = "#E0E0E0"
C_WHITE    = "#FFFFFF"
C_LBLUE    = "#E3F2FD"
C_LRED     = "#FFEBEE"
C_LGREEN   = "#E8F5E9"
C_LORANGE  = "#FFF3E0"

def rounded_box(ax, xy, w, h, color, text, fontsize=7, text_color="black",
                alpha=1.0, linewidth=0.8, edgecolor=None, bold=False):
    """Draw a rounded rectangle with centered text."""
    ec = edgecolor or color
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=ec,
                         linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=3,
            linespacing=1.3)

def arrow(ax, start, end, color=C_GRAY, style="-|>", lw=1.2, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                        linewidth=lw, connectionstyle=connectionstyle,
                        zorder=4, mutation_scale=10)
    ax.add_patch(a)

def bracket_text(ax, x, y1, y2, text, color=C_GRAY, side="right"):
    """Draw a curly bracket with label."""
    offset = 0.015 if side == "right" else -0.015
    ax.annotate("", xy=(x + offset, y1), xytext=(x + offset, y2),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
    ax.plot([x, x + offset], [y1, y1], color=color, lw=0.8)
    ax.plot([x, x + offset], [y2, y2], color=color, lw=0.8)
    mid = (y1 + y2) / 2
    ax.text(x + offset * 3, mid, text, ha="left" if side == "right" else "right",
            va="center", fontsize=6, color=color, style="italic")


# ── Figure setup ───────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 3.2), facecolor=C_BG)

# Three panels with spacing
# Panel (a): x = 0.02-0.34, Panel (b): 0.36-0.66, Panel (c): 0.68-0.98
ax = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Panel labels
for label, x in [("(a) Setup", 0.16), ("(b) Core Finding", 0.50), ("(c) Implications", 0.83)]:
    ax.text(x, 0.96, label, ha="center", va="top", fontsize=8.5,
            weight="bold", color="#212121")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PANEL (a): Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LLM box
rounded_box(ax, (0.03, 0.68), 0.12, 0.12, C_LBLUE, "LLM\n(Qwen 3B–14B)",
            fontsize=6.5, edgecolor=C_BLUE, text_color=C_BLUE, bold=True)

# Problem input
rounded_box(ax, (0.03, 0.85), 0.12, 0.06, C_LGRAY, "Math Problem",
            fontsize=6, edgecolor=C_GRAY)
arrow(ax, (0.09, 0.85), (0.09, 0.80), color=C_GRAY)

# CoT trace output
rounded_box(ax, (0.175, 0.58), 0.15, 0.28, C_WHITE, "", edgecolor=C_LGRAY, linewidth=0.6)

# Step texts inside CoT box
steps = [
    ("Step 1: Let x = 3...", C_BLUE, 0.82),
    ("Step 2: Then 3×5 = 15", C_BLUE, 0.76),
    ("Step 3: So 15 + 2 = 18", C_RED, 0.70),
    ("Step 4: Therefore...", C_RED, 0.64),
    ("ANSWER: 18  ✗", C_RED, 0.60),
]
for text, color, y in steps:
    ax.text(0.185, y, text, fontsize=5, color=color, va="center", family="monospace")

ax.text(0.25, 0.88, "CoT Trace", ha="center", fontsize=6, color=C_GRAY, style="italic")

arrow(ax, (0.15, 0.74), (0.175, 0.74), color=C_GRAY)

# Hidden states (colored bars next to each step)
for y, color in [(0.82, "#81D4FA"), (0.76, "#81D4FA"), (0.70, "#EF9A9A"), (0.64, "#EF9A9A")]:
    ax.barh(y, 0.02, height=0.04, left=0.33, color=color, edgecolor="white", linewidth=0.3, zorder=3)

ax.text(0.34, 0.88, "Hidden\nStates", ha="center", fontsize=5, color=C_GRAY, va="center")

# Arrow from trace to hidden states
for y in [0.82, 0.76, 0.70, 0.64]:
    ax.annotate("", xy=(0.33, y), xytext=(0.325, y),
                arrowprops=dict(arrowstyle="-|>", color=C_LGRAY, lw=0.5, mutation_scale=5))

# Linear probe
rounded_box(ax, (0.05, 0.38), 0.22, 0.10, C_LORANGE, "Linear Probe\n(Logistic Regression on $\\mathbf{h}_T$)",
            fontsize=5.5, edgecolor=C_ORANGE, text_color="#E65100")
arrow(ax, (0.34, 0.58), (0.22, 0.48), color=C_ORANGE, lw=0.8)

# Probe output
rounded_box(ax, (0.07, 0.22), 0.18, 0.08, C_WHITE,
            "P(error) = 0.82", fontsize=6.5, edgecolor=C_RED, text_color=C_RED, bold=True)
arrow(ax, (0.16, 0.38), (0.16, 0.30), color=C_ORANGE, lw=0.8)

# Verbalized confidence (for contrast)
rounded_box(ax, (0.07, 0.08), 0.18, 0.08, C_WHITE,
            '"I\'m confident: 4.5/5"', fontsize=5.5, edgecolor=C_LGRAY, text_color=C_GRAY)
ax.text(0.16, 0.04, "Text output", ha="center", fontsize=5, color=C_GRAY, style="italic")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PANEL (b): Core Finding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Divider
ax.plot([0.355, 0.355], [0.02, 0.92], color=C_LGRAY, lw=0.5, ls="--")

# Title: "Unfaithful Reasoning"
rounded_box(ax, (0.38, 0.82), 0.22, 0.08, C_LRED,
            "Unfaithful Reasoning", fontsize=7, edgecolor=C_RED, text_color=C_RED, bold=True)

# Two comparison columns
# Left: Hidden State (accurate)
ax.text(0.42, 0.77, "Hidden States", ha="center", fontsize=6, color=C_BLUE, weight="bold")
ax.text(0.42, 0.74, "(accurate)", ha="center", fontsize=5, color=C_BLUE)

# Probe bars
bar_x = 0.39
bar_w = 0.06
# Correct traces
ax.barh(0.67, 0.27 * bar_w / 0.5, height=0.025, left=bar_x, color=C_BLUE, alpha=0.8, zorder=3)
ax.text(bar_x - 0.005, 0.67, "Correct", ha="right", va="center", fontsize=5, color=C_BLUE)
ax.text(bar_x + 0.27 * bar_w / 0.5 + 0.005, 0.67, "0.27", ha="left", va="center", fontsize=5, color=C_BLUE)

# Wrong traces
ax.barh(0.635, 0.74 * bar_w / 0.5, height=0.025, left=bar_x, color=C_RED, alpha=0.8, zorder=3)
ax.text(bar_x - 0.005, 0.635, "Wrong", ha="right", va="center", fontsize=5, color=C_RED)
ax.text(bar_x + 0.74 * bar_w / 0.5 + 0.005, 0.635, "0.74", ha="left", va="center", fontsize=5, color=C_RED)

ax.text(0.42, 0.59, "Gap = 0.47 ✓", ha="center", fontsize=5.5, color=C_GREEN, weight="bold")

# Right: Verbalized Confidence (useless)
ax.text(0.56, 0.77, "Verbalized Text", ha="center", fontsize=6, color=C_GRAY, weight="bold")
ax.text(0.56, 0.74, "(uninformative)", ha="center", fontsize=5, color=C_GRAY)

bar_x2 = 0.52
ax.barh(0.67, 4.87/5 * bar_w / 0.5, height=0.025, left=bar_x2, color=C_BLUE, alpha=0.3, zorder=3)
ax.text(bar_x2 + 4.87/5 * bar_w / 0.5 + 0.005, 0.67, "4.87", ha="left", va="center", fontsize=5, color=C_GRAY)

ax.barh(0.635, 4.55/5 * bar_w / 0.5, height=0.025, left=bar_x2, color=C_RED, alpha=0.3, zorder=3)
ax.text(bar_x2 + 4.55/5 * bar_w / 0.5 + 0.005, 0.635, "4.55", ha="left", va="center", fontsize=5, color=C_GRAY)

ax.text(0.56, 0.59, "Gap = 0.32 ✗", ha="center", fontsize=5.5, color=C_RED, weight="bold")

# Key stats boxes
stats = [
    ("AUROC 0.95", "Error detection\nfrom hidden states", C_LBLUE, C_BLUE),
    ("AUROC 0.79", "Detectable from\nStep 1 alone", C_LBLUE, C_BLUE),
    ("65%", "Traces are\n\"unfaithful\"", C_LRED, C_RED),
]
for i, (num, desc, bg, fg) in enumerate(stats):
    y = 0.47 - i * 0.14
    rounded_box(ax, (0.37, y - 0.01), 0.08, 0.10, bg, num,
                fontsize=7, edgecolor=fg, text_color=fg, bold=True)
    ax.text(0.46, y + 0.04, desc, va="center", fontsize=5, color="#424242", linespacing=1.2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PANEL (c): Implications
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Divider
ax.plot([0.665, 0.665], [0.02, 0.92], color=C_LGRAY, lw=0.5, ls="--")

# Failed interventions (stacked)
ax.text(0.82, 0.88, "Can the model use this signal?", ha="center", fontsize=6,
        color="#424242", style="italic")

fails = [
    ("Activation Steering", "+3%", "marginal"),
    ("Probe Best-of-N", "< MV", "worse than voting"),
    ("Self-Correction", "−3%", "hurts accuracy"),
    ("Activation Patching", "→ 0%", "destroys output"),
]
for i, (name, result, note) in enumerate(fails):
    y = 0.78 - i * 0.095
    rounded_box(ax, (0.69, y), 0.12, 0.065, C_LRED, name, fontsize=5.5,
                edgecolor="#FFCDD2", text_color="#424242")
    ax.text(0.82, y + 0.033, result, ha="center", va="center", fontsize=7,
            color=C_RED, weight="bold")
    ax.text(0.86, y + 0.033, note, ha="left", va="center", fontsize=4.5, color=C_GRAY)

# Big X
ax.text(0.82, 0.39, "✗  Diagnostic, not causal", ha="center", fontsize=6.5,
        color=C_RED, weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_LRED, edgecolor=C_RED, linewidth=0.8))

# Success: Selective Routing
ax.text(0.82, 0.30, "But...", ha="center", fontsize=6, color=C_GRAY, style="italic")

rounded_box(ax, (0.69, 0.13), 0.26, 0.14, C_LGREEN, "",
            edgecolor=C_GREEN, linewidth=1.2)
ax.text(0.82, 0.24, "Selective Routing", ha="center", fontsize=7,
        color=C_GREEN, weight="bold")
ax.text(0.82, 0.20, "Probe flags errors → route to external verifier", ha="center",
        fontsize=5, color="#424242")
ax.text(0.82, 0.155, "+8% accuracy  @  40% API cost", ha="center",
        fontsize=7, color=C_GREEN, weight="bold")

# ── Save ───────────────────────────────────────────────────
plt.savefig("paper/figures/fig_overview.pdf", bbox_inches="tight", dpi=300,
            facecolor=C_BG, edgecolor="none")
plt.savefig("results/figures/fig_overview.pdf", bbox_inches="tight", dpi=300,
            facecolor=C_BG, edgecolor="none")
plt.savefig("paper/figures/fig_overview.png", bbox_inches="tight", dpi=300,
            facecolor=C_BG, edgecolor="none")
print("Saved fig_overview.pdf + fig_overview.png")
