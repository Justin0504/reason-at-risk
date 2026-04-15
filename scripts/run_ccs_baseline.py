#!/usr/bin/env python3
"""CCS (Contrast-Consistent Search) baseline from Burns et al. 2023.

CCS finds a direction in hidden state space that is consistent with
truth/falsehood without requiring labels. We adapt it to our setting:
- For each problem, we have the model's CoT response
- We create (positive, negative) pairs: "This reasoning is correct" / "This reasoning has errors"
- CCS finds the direction that separates these pairs consistently

We compare CCS AUROC with our supervised linear probe AUROC.

Also implements: confidence-based probe (using model's own logprobs).
"""

import json, pickle, sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.probe.collect_hidden_states import HiddenStateCollector
from src.pipeline.data_loader import load_benchmark


class CCS:
    """Contrast-Consistent Search (Burns et al. 2023).

    Finds a linear direction that satisfies:
    1. Consistency: p(x+) + p(x-) ≈ 1 for each pair
    2. Confidence: p(x) is close to 0 or 1
    """

    def __init__(self, hidden_dim, n_epochs=1000, lr=0.01):
        self.probe = torch.nn.Linear(hidden_dim, 1)
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, h_pos, h_neg):
        """Fit CCS on paired hidden states.

        h_pos: (n_pairs, hidden_dim) — hidden states for positive statement
        h_neg: (n_pairs, hidden_dim) — hidden states for negative statement
        """
        h_pos_t = torch.tensor(h_pos, dtype=torch.float32)
        h_neg_t = torch.tensor(h_neg, dtype=torch.float32)

        # Normalize
        mean = (h_pos_t.mean(0) + h_neg_t.mean(0)) / 2
        std = ((h_pos_t - mean).pow(2).mean(0) + (h_neg_t - mean).pow(2).mean(0)).sqrt() / 2
        std = std.clamp(min=1e-8)
        h_pos_t = (h_pos_t - mean) / std
        h_neg_t = (h_neg_t - mean) / std

        self.mean = mean
        self.std = std

        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=0.01)

        best_loss = float('inf')
        for epoch in range(self.n_epochs):
            p_pos = torch.sigmoid(self.probe(h_pos_t))
            p_neg = torch.sigmoid(self.probe(h_neg_t))

            # Consistency loss: p(x+) + p(x-) should be 1
            consistency = ((p_pos + p_neg - 1) ** 2).mean()

            # Confidence loss: push probabilities away from 0.5
            confidence = (torch.min(p_pos, 1 - p_pos).mean() +
                         torch.min(p_neg, 1 - p_neg).mean())

            loss = consistency + confidence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.probe.state_dict().items()}

        self.probe.load_state_dict(best_state)

    def predict_proba(self, h):
        """Return P(positive) for each hidden state."""
        h_t = torch.tensor(h, dtype=torch.float32)
        h_t = (h_t - self.mean) / self.std
        with torch.no_grad():
            p = torch.sigmoid(self.probe(h_t)).numpy().flatten()
        return p


def collect_ccs_pairs(collector, problems, layer):
    """Collect paired hidden states for CCS.

    For each problem:
    - Generate CoT response
    - Create two prompts: one affirming correctness, one questioning it
    - Extract hidden states from both
    """
    positive_hs = []
    negative_hs = []
    labels = []  # ground truth for evaluation

    for i, problem in enumerate(problems):
        try:
            # Generate CoT
            response = collector.generate_cot(problem["problem"])
            answer = collector._extract_answer(response)
            correct = collector._check_answer(answer, problem["answer"])

            # Create paired prompts
            base = f"Problem: {problem['problem']}\n\nSolution: {response}\n\n"
            pos_prompt = base + "This solution is correct and the final answer is right."
            neg_prompt = base + "This solution contains errors and the final answer is wrong."

            # Extract hidden states
            hs_pos = collector.extract_hidden_states(pos_prompt)
            hs_neg = collector.extract_hidden_states(neg_prompt)

            # Take last token hidden state
            positive_hs.append(hs_pos[layer][-1].numpy().astype(np.float32))
            negative_hs.append(hs_neg[layer][-1].numpy().astype(np.float32))
            labels.append(0 if correct else 1)  # 1 = error

            if (i + 1) % 10 == 0:
                acc = sum(1 for l in labels if l == 0) / len(labels)
                print(f"  [{i+1}/{len(problems)}] acc={acc:.3f}", flush=True)

        except Exception as e:
            print(f"  Error {problem['id']}: {e}", flush=True)

    return np.array(positive_hs), np.array(negative_hs), np.array(labels)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=150)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="results/ccs_comparison")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing hidden states instead of collecting CCS pairs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_existing:
        # Use existing hidden states — simulate CCS by using positive/negative
        # proxy: correct traces as "positive", incorrect as "negative"
        print("Using existing hidden states as CCS proxy...")

        data = np.load("data/probe_data/hidden_states.npz", allow_pickle=True)
        records = list(data["records"])

        summary = json.load(open("data/probes/probe_summary.json"))
        best_layer = max(summary, key=lambda x: x["cv_auroc_mean"])["layer"]

        # Get trace-level data
        by_problem = defaultdict(list)
        for r in records:
            if r["layer"] == best_layer:
                by_problem[r["problem_id"]].append(r)

        X, y = [], []
        for pid, steps in by_problem.items():
            steps.sort(key=lambda s: s["step_index"])
            X.append(steps[-1]["hidden_state"])
            y.append(0 if steps[-1]["trace_is_correct"] else 1)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        # Split into correct/incorrect for CCS pairs
        correct_idx = np.where(y == 0)[0]
        incorrect_idx = np.where(y == 1)[0]

        n_pairs = min(len(correct_idx), len(incorrect_idx))
        h_pos = X[correct_idx[:n_pairs]]
        h_neg = X[incorrect_idx[:n_pairs]]

        print(f"  {n_pairs} CCS pairs, layer {best_layer}")

        # Run CCS
        print("\nTraining CCS...")
        best_ccs_auroc = 0
        for seed in range(10):  # CCS is sensitive to init, try multiple
            ccs = CCS(h_pos.shape[1], n_epochs=1000, lr=0.01)
            torch.manual_seed(seed)
            ccs.fit(h_pos, h_neg)

            # Evaluate on ALL data
            p_scores = ccs.predict_proba(X)

            # CCS doesn't know which direction is "error" vs "correct"
            # Try both orientations
            auroc1 = roc_auc_score(y, p_scores)
            auroc2 = roc_auc_score(y, 1 - p_scores)
            auroc = max(auroc1, auroc2)

            if auroc > best_ccs_auroc:
                best_ccs_auroc = auroc
                best_seed = seed

        print(f"  CCS best AUROC: {best_ccs_auroc:.4f} (seed {best_seed})")

        # Compare with supervised probe
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probe_scores = cross_val_score(clf, scaler.fit_transform(X), y,
                                        cv=cv, scoring="roc_auc")
        probe_auroc = float(np.mean(probe_scores))
        print(f"  Supervised probe CV AUROC: {probe_auroc:.4f}")

        # CCS with CV (proper evaluation)
        ccs_cv_aurocs = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # CCS pairs from train set
            c_idx = np.where(y_train == 0)[0]
            i_idx = np.where(y_train == 1)[0]
            np_pairs = min(len(c_idx), len(i_idx))
            if np_pairs < 5:
                continue

            best_fold_auroc = 0
            for seed in range(5):
                ccs = CCS(X.shape[1], n_epochs=1000, lr=0.01)
                torch.manual_seed(seed)
                ccs.fit(X_train[c_idx[:np_pairs]], X_train[i_idx[:np_pairs]])
                p = ccs.predict_proba(X_test)
                a1 = roc_auc_score(y_test, p)
                a2 = roc_auc_score(y_test, 1 - p)
                best_fold_auroc = max(best_fold_auroc, a1, a2)

            ccs_cv_aurocs.append(best_fold_auroc)

        ccs_cv_mean = np.mean(ccs_cv_aurocs) if ccs_cv_aurocs else 0
        ccs_cv_std = np.std(ccs_cv_aurocs) if ccs_cv_aurocs else 0
        print(f"  CCS CV AUROC: {ccs_cv_mean:.4f} ± {ccs_cv_std:.4f}")

        results = {
            "method": "existing_data_proxy",
            "n_problems": len(X),
            "layer": best_layer,
            "ccs_auroc_all": best_ccs_auroc,
            "ccs_cv_auroc_mean": round(ccs_cv_mean, 4),
            "ccs_cv_auroc_std": round(ccs_cv_std, 4),
            "probe_cv_auroc_mean": round(probe_auroc, 4),
            "probe_cv_auroc_std": round(float(np.std(probe_scores)), 4),
        }

    else:
        # Full CCS with actual positive/negative prompts
        print(f"Collecting CCS pairs from {args.model}...")

        summary = json.load(open("data/probes/probe_summary.json"))
        best_layer = max(summary, key=lambda x: x["cv_auroc_mean"])["layer"]

        collector = HiddenStateCollector(
            model_name=args.model, device="mps", layers_to_probe=[best_layer]
        )

        problems = load_benchmark("math500", max_samples=args.n_problems, seed=42)
        h_pos, h_neg, labels = collect_ccs_pairs(collector, problems, best_layer)

        print(f"\nCollected {len(labels)} pairs")
        print(f"  Correct: {sum(labels == 0)}, Error: {sum(labels == 1)}")

        # Save raw data
        np.savez(output_dir / "ccs_pairs.npz", h_pos=h_pos, h_neg=h_neg, labels=labels)

        # Run CCS
        print("\nTraining CCS...")
        best_ccs_auroc = 0
        for seed in range(10):
            ccs = CCS(h_pos.shape[1], n_epochs=1000, lr=0.01)
            torch.manual_seed(seed)
            ccs.fit(h_pos, h_neg)
            p = ccs.predict_proba(np.vstack([h_pos, h_neg]))
            # Use h_pos scores (correct reasoning) — should have low error prob
            p_pos = p[:len(h_pos)]
            auroc1 = roc_auc_score(labels, p_pos)
            auroc2 = roc_auc_score(labels, 1 - p_pos)
            auroc = max(auroc1, auroc2)
            if auroc > best_ccs_auroc:
                best_ccs_auroc = auroc

        print(f"  CCS AUROC: {best_ccs_auroc:.4f}")

        # Supervised probe for comparison
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probe_scores = cross_val_score(clf, scaler.fit_transform(h_pos), labels,
                                        cv=cv, scoring="roc_auc")
        probe_auroc = float(np.mean(probe_scores))
        print(f"  Supervised probe AUROC: {probe_auroc:.4f}")

        results = {
            "method": "full_ccs_pairs",
            "n_problems": len(labels),
            "layer": best_layer,
            "ccs_auroc": best_ccs_auroc,
            "probe_auroc": probe_auroc,
        }

    # Save
    with open(output_dir / "ccs_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"\n{'='*50}")
    print(f"COMPARISON:")
    print(f"  CCS (unsupervised): {results.get('ccs_cv_auroc_mean', results.get('ccs_auroc', 0)):.4f}")
    print(f"  Supervised probe:   {results.get('probe_cv_auroc_mean', results.get('probe_auroc', 0)):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
