#!/usr/bin/env python3
"""Experiment: Can probe predict final correctness from just the FIRST step?

This is key evidence for "unfaithful reasoning" — if the model's hidden state
at step 1 already knows the trace will fail, it means the model continues
generating wrong reasoning DESPITE internally knowing it's wrong.

Uses existing traces from online_backtrack_v2 (no GPU needed).
Also trains a fresh probe on step-1-only hidden states for proper evaluation.
"""

import json, pickle, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))

output_dir = Path("results/first_step_prediction")
output_dir.mkdir(parents=True, exist_ok=True)

# Load existing traces
traces = json.load(open("results/online_backtrack_v2/traces.json"))

# Analysis 1: Using existing step-level probe scores
print("=== Analysis 1: First-step probe score as predictor ===")

first_scores = []
labels = []
all_scores = []

for t in traces:
    if t["step_scores"]:
        first_scores.append(t["step_scores"][0])
        labels.append(0 if t["correct"] else 1)  # 1 = error
        all_scores.append(t["step_scores"])

first_scores = np.array(first_scores)
labels = np.array(labels)

# AUROC: first step only
auroc_first = roc_auc_score(labels, first_scores)
print(f"  First-step AUROC: {auroc_first:.4f}")

# AUROC: max across all steps (trace-level)
max_scores = np.array([max(s) for s in all_scores])
auroc_max = roc_auc_score(labels, max_scores)
print(f"  Max-score AUROC: {auroc_max:.4f}")

# AUROC: mean across all steps
mean_scores = np.array([np.mean(s) for s in all_scores])
auroc_mean = roc_auc_score(labels, mean_scores)
print(f"  Mean-score AUROC: {auroc_mean:.4f}")

# AUROC: last step only
last_scores = np.array([s[-1] for s in all_scores])
auroc_last = roc_auc_score(labels, last_scores)
print(f"  Last-step AUROC: {auroc_last:.4f}")

# Step-by-step AUROC (how prediction improves with more steps)
print("\n=== Analysis 2: Cumulative step prediction ===")
max_steps = max(len(s) for s in all_scores)
step_aurocs = []
for k in range(1, min(max_steps + 1, 10)):
    # Use max of first k steps
    scores_k = []
    labels_k = []
    for s, l in zip(all_scores, labels):
        if len(s) >= k:
            scores_k.append(max(s[:k]))
            labels_k.append(l)
    if len(set(labels_k)) > 1:
        auc = roc_auc_score(labels_k, scores_k)
        step_aurocs.append({"step": k, "auroc": round(auc, 4), "n": len(labels_k)})
        print(f"  Steps 1-{k}: AUROC = {auc:.4f} (n={len(labels_k)})")

# Analysis 3: First-step score distribution
print("\n=== Analysis 3: Score distributions ===")
correct_first = first_scores[labels == 0]
wrong_first = first_scores[labels == 1]
print(f"  Correct traces - first step score: mean={correct_first.mean():.3f}, std={correct_first.std():.3f}")
print(f"  Wrong traces   - first step score: mean={wrong_first.mean():.3f}, std={wrong_first.std():.3f}")
print(f"  Separation: {wrong_first.mean() - correct_first.mean():.3f}")

# Analysis 4: How many traces are "confidently wrong"?
# = probe score > 0.8 at step 1 AND model continues to wrong answer
print("\n=== Analysis 4: 'Unfaithful' traces ===")
for thresh in [0.6, 0.7, 0.8]:
    high_first = first_scores > thresh
    n_high = high_first.sum()
    n_high_wrong = (high_first & (labels == 1)).sum()
    n_high_correct = (high_first & (labels == 0)).sum()
    low_first = ~high_first
    n_low = low_first.sum()
    n_low_wrong = (low_first & (labels == 1)).sum()

    print(f"  Threshold {thresh}:")
    print(f"    High score at step 1: {n_high} ({n_high}/{len(labels)})")
    print(f"      → Actually wrong: {n_high_wrong}/{n_high} ({n_high_wrong/max(1,n_high):.1%})")
    print(f"      → Actually correct: {n_high_correct}/{n_high} ({n_high_correct/max(1,n_high):.1%})")
    print(f"    Low score at step 1: {n_low}")
    print(f"      → Actually wrong: {n_low_wrong}/{n_low} ({n_low_wrong/max(1,n_low):.1%})")

# Save results
results = {
    "n_traces": len(traces),
    "base_acc": sum(1 for t in traces if t["correct"]) / len(traces),
    "first_step_auroc": round(auroc_first, 4),
    "max_score_auroc": round(auroc_max, 4),
    "mean_score_auroc": round(auroc_mean, 4),
    "last_step_auroc": round(auroc_last, 4),
    "cumulative_step_aurocs": step_aurocs,
    "correct_first_mean": round(float(correct_first.mean()), 4),
    "wrong_first_mean": round(float(wrong_first.mean()), 4),
    "separation": round(float(wrong_first.mean() - correct_first.mean()), 4),
}

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {output_dir}/results.json")
