#!/usr/bin/env python3
"""Bootstrap confidence intervals for main AUROC numbers.

Computes 95% CI via 1000 bootstrap resamples for each model's probe AUROC.
Also computes CI for first-step AUROC and verbalized confidence AUROC.
"""

import json, pickle, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")

from sklearn.metrics import roc_auc_score


def bootstrap_auroc(y_true, y_score, n_boot=1000, seed=42):
    """Compute bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return {
        "mean": round(float(np.mean(aucs)), 4),
        "std": round(float(np.std(aucs)), 4),
        "ci_lower": round(float(np.percentile(aucs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(aucs, 97.5)), 4),
        "n_boot": len(aucs),
    }


def load_probe_and_data(results_dir):
    """Load probe predictions from a results directory."""
    results_dir = Path(results_dir)

    # Try to find saved predictions
    for fname in ["predictions.npz", "probe_predictions.npz"]:
        p = results_dir / fname
        if p.exists():
            data = np.load(p)
            return data["y_true"], data["y_score"]

    # Try checkpoint files
    for fname in results_dir.glob("*checkpoint*.json"):
        with open(fname) as f:
            ckpt = json.load(f)
        if "results" in ckpt and isinstance(ckpt["results"], list):
            labels, scores = [], []
            for r in ckpt["results"]:
                if "correct" in r and "probe_score" in r:
                    labels.append(0 if r["correct"] else 1)
                    scores.append(r["probe_score"])
            if labels:
                return np.array(labels), np.array(scores)

    return None, None


def compute_cv_bootstrap(features_file, labels_file, n_boot=1000):
    """Bootstrap from raw features using cross-validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    X = np.load(features_file)
    y = np.load(labels_file)

    rng = np.random.RandomState(42)
    aucs = []

    for _ in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]

        if len(set(y_boot)) < 2 or sum(y_boot) < 3 or sum(1 - y_boot) < 3:
            continue

        # 5-fold CV on bootstrap sample
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for train_idx, val_idx in skf.split(X_boot, y_boot):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_boot[train_idx])
            X_val = scaler.transform(X_boot[val_idx])

            clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
            clf.fit(X_tr, y_boot[train_idx])

            y_val = y_boot[val_idx]
            if len(set(y_val)) < 2:
                continue
            probs = clf.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, probs))

        if fold_aucs:
            aucs.append(np.mean(fold_aucs))

    aucs = np.array(aucs)
    return {
        "mean": round(float(np.mean(aucs)), 4),
        "std": round(float(np.std(aucs)), 4),
        "ci_lower": round(float(np.percentile(aucs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(aucs, 97.5)), 4),
        "n_boot": len(aucs),
    }


def main():
    base = Path("/data/yinian/reason-at-risk")
    output_dir = base / "results" / "bootstrap_ci"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # === 1. Bootstrap from saved features ===
    models = {
        "3B": {
            "features": base / "results" / "hidden_states_3b",
            "probe_dir": base / "results" / "probes_3b",
        },
        "7B": {
            "features": base / "results" / "hidden_states_7b",
            "probe_dir": base / "results" / "probes_7b",
        },
    }

    for model_name, paths in models.items():
        print(f"\n=== {model_name} ===")

        # Look for features
        feat_dir = paths["features"]
        probe_dir = paths["probe_dir"]

        # Try to find numpy arrays of features/labels
        for candidate_dir in [feat_dir, probe_dir,
                              base / f"results/step_dynamics_{model_name.lower()}",
                              base / f"results/cross_benchmark_{model_name.lower()}"]:
            if not candidate_dir.exists():
                continue
            for f in candidate_dir.glob("*.npy"):
                print(f"  Found: {f}")
            for f in candidate_dir.glob("*.npz"):
                print(f"  Found: {f}")
            for f in candidate_dir.glob("*.pkl"):
                print(f"  Found: {f}")

    # === 2. Bootstrap from step dynamics data (has raw probe scores) ===
    for model_tag in ["3b", "7b"]:
        summary_file = base / f"results/step_dynamics_{model_tag}/step_dynamics_summary.json"
        if not summary_file.exists():
            print(f"  Step dynamics {model_tag}: not found")
            continue

        with open(summary_file) as f:
            sd = json.load(f)

        # Extract per-trace labels and final probe scores
        traces_file = base / f"results/step_dynamics_{model_tag}/step_dynamics_checkpoint.json"
        if traces_file.exists():
            with open(traces_file) as f:
                traces = json.load(f)

            labels, scores = [], []
            for t in traces.get("traces", []):
                if "correct" in t and "trajectory" in t and len(t["trajectory"]) > 0:
                    labels.append(0 if t["correct"] else 1)
                    # Use max probe score as the trace-level score
                    scores.append(max(t["trajectory"]))

            if len(labels) > 10 and len(set(labels)) >= 2:
                y_true = np.array(labels)
                y_score = np.array(scores)
                ci = bootstrap_auroc(y_true, y_score)
                results[f"step_dynamics_{model_tag}_max_score"] = ci
                print(f"  Step dynamics {model_tag} (max score): AUROC={ci['mean']} [{ci['ci_lower']}, {ci['ci_upper']}]")

                # Also first-step score
                first_scores = []
                first_labels = []
                for t in traces.get("traces", []):
                    if "correct" in t and "trajectory" in t and len(t["trajectory"]) > 0:
                        first_labels.append(0 if t["correct"] else 1)
                        first_scores.append(t["trajectory"][0])

                if len(first_labels) > 10:
                    ci_first = bootstrap_auroc(np.array(first_labels), np.array(first_scores))
                    results[f"step_dynamics_{model_tag}_first_step"] = ci_first
                    print(f"  Step dynamics {model_tag} (first step): AUROC={ci_first['mean']} [{ci_first['ci_lower']}, {ci_first['ci_upper']}]")

    # === 3. Bootstrap from verbalized confidence ===
    vc_file = base / "results/verbalized_confidence/results.json"
    if vc_file.exists():
        with open(vc_file) as f:
            vc = json.load(f)

        if "per_problem" in vc:
            labels, conf_scores, probe_scores = [], [], []
            for p in vc["per_problem"]:
                if "correct" in p:
                    labels.append(0 if p["correct"] else 1)
                    conf_scores.append(p.get("confidence", 3.0))
                    probe_scores.append(p.get("probe_score", 0.5))

            if len(labels) > 10 and len(set(labels)) >= 2:
                y = np.array(labels)
                # For confidence, lower = more likely error, so negate
                ci_conf = bootstrap_auroc(y, -np.array(conf_scores))
                results["verbalized_confidence_auroc"] = ci_conf
                print(f"\n  Verbalized confidence AUROC: {ci_conf['mean']} [{ci_conf['ci_lower']}, {ci_conf['ci_upper']}]")

                ci_probe = bootstrap_auroc(y, np.array(probe_scores))
                results["probe_vs_confidence"] = ci_probe
                print(f"  Probe score AUROC: {ci_probe['mean']} [{ci_probe['ci_lower']}, {ci_probe['ci_upper']}]")

    # === 4. Bootstrap from first-step prediction ===
    fsp_file = base / "results/first_step_prediction/results.json"
    if fsp_file.exists():
        with open(fsp_file) as f:
            fsp = json.load(f)

        if "per_problem" in fsp:
            labels, scores = [], []
            for p in fsp["per_problem"]:
                if "correct" in p and "first_step_score" in p:
                    labels.append(0 if p["correct"] else 1)
                    scores.append(p["first_step_score"])

            if len(labels) > 10 and len(set(labels)) >= 2:
                ci = bootstrap_auroc(np.array(labels), np.array(scores))
                results["first_step_3b"] = ci
                print(f"\n  First-step 3B: AUROC={ci['mean']} [{ci['ci_lower']}, {ci['ci_upper']}]")

    # === 5. Bootstrap from e2e results ===
    for edir in ["e2e_200", "e2e_probe_api"]:
        efile = base / f"results/{edir}/e2e_results.json"
        if efile.exists():
            with open(efile) as f:
                e2e = json.load(f)
            if "per_problem" in e2e:
                labels, scores = [], []
                for p in e2e["per_problem"]:
                    if "correct" in p and "probe_score" in p:
                        labels.append(0 if p["correct"] else 1)
                        scores.append(p["probe_score"])
                if len(labels) > 10 and len(set(labels)) >= 2:
                    ci = bootstrap_auroc(np.array(labels), np.array(scores))
                    results[f"routing_{edir}"] = ci
                    print(f"\n  Routing {edir}: AUROC={ci['mean']} [{ci['ci_lower']}, {ci['ci_upper']}]")

    # === Save ===
    with open(output_dir / "bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("BOOTSTRAP CI SUMMARY")
    print(f"{'='*60}")
    for key, val in results.items():
        print(f"  {key}: {val['mean']} [{val['ci_lower']}, {val['ci_upper']}] (n_boot={val['n_boot']})")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
