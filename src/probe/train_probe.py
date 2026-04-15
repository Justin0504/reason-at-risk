"""Train a linear probe on hidden states to predict reasoning errors.

Input: collected StepRecords with hidden states and correctness labels.
Output: trained probe weights + evaluation metrics (precision, recall, F1, AUROC).
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle


@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    n_samples: int
    n_positive: int  # error cases
    cv_scores: list[float]


def load_dataset(data_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load hidden states and labels for a specific layer.

    Label: 1 = trace leads to incorrect answer (error), 0 = correct.
    """
    X, y = [], []
    for npz_file in sorted(data_dir.glob("*.npz")):
        data = np.load(npz_file, allow_pickle=True)
        records = data["records"]  # array of dicts
        for rec in records:
            if rec["layer"] == layer:
                X.append(rec["hidden_state"])
                # Label: 1 if trace is INCORRECT (we want to detect errors)
                y.append(0 if rec["trace_is_correct"] else 1)

    return np.array(X), np.array(y)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    C: float = 1.0,
    n_folds: int = 5,
) -> tuple[LogisticRegression, StandardScaler, ProbeResult]:
    """Train logistic regression probe with cross-validation."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42, class_weight="balanced")

    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")

    # Train final model on all data
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    result = ProbeResult(
        layer=layer,
        accuracy=accuracy_score(y, y_pred),
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        f1=f1_score(y, y_pred, zero_division=0),
        auroc=roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0,
        n_samples=len(y),
        n_positive=int(y.sum()),
        cv_scores=cv_scores.tolist(),
    )

    return clf, scaler, result


def train_all_layers(
    data_dir: Path,
    layers: list[int],
    output_dir: Path,
) -> list[ProbeResult]:
    """Train probes for all layers and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        X, y = load_dataset(data_dir, layer)
        print(f"  Samples: {len(y)}, Errors: {y.sum()} ({y.mean()*100:.1f}%)")

        if len(set(y)) < 2:
            print(f"  Skipping: only one class present")
            continue

        clf, scaler, result = train_probe(X, y, layer)
        results.append(result)

        print(f"  Accuracy:  {result.accuracy:.4f}")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1:        {result.f1:.4f}")
        print(f"  AUROC:     {result.auroc:.4f}")
        print(f"  CV AUROC:  {np.mean(result.cv_scores):.4f} ± {np.std(result.cv_scores):.4f}")

        # Save probe
        with open(output_dir / f"probe_layer_{layer}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler, "result": result}, f)

    # Save summary
    summary = [
        {
            "layer": r.layer,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "auroc": r.auroc,
            "cv_auroc_mean": np.mean(r.cv_scores),
            "cv_auroc_std": np.std(r.cv_scores),
            "n_samples": r.n_samples,
            "n_positive": r.n_positive,
        }
        for r in results
    ]
    with open(output_dir / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results
