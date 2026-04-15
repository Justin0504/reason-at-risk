#!/usr/bin/env python3
"""Experiment: Cross-benchmark generalization.

Train probe on MATH-500, evaluate on GSM8K and PrOntoQA.
Also train probes natively on each benchmark for comparison.

This proves the error detection phenomenon is not math-specific.
"""

import argparse, json, pickle, re, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from src.probe.collect_hidden_states import HiddenStateCollector
from src.pipeline.data_loader import load_benchmark


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'\\boxed\{(.+?)\}', text)
    if m: return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

def extract_num(s):
    s = re.sub(r'[,$%]', '', s.strip())
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else None

def check_answer(predicted, reference):
    p, r_ = extract_num(predicted), extract_num(reference)
    if p is not None and r_ is not None:
        return abs(p - r_) < 0.01
    pn = predicted.strip().lower()
    rn = reference.strip().lower()
    if pn == rn: return True
    # For PrOntoQA: TRUE/FALSE matching
    if rn in ("true", "false"):
        return pn.startswith(rn) or rn in pn
    if len(rn) > 3 and (rn in pn or pn in rn): return True
    return False


def collect_hidden_states(collector, problems, best_layer, max_problems=200):
    """Collect last-token hidden states for a set of problems."""
    features, labels, ids = [], [], []
    for i, prob in enumerate(problems):
        try:
            resp = collector.generate_cot(prob["problem"])
            ans = extract_answer(resp)
            correct = check_answer(ans, prob["answer"])

            # Build full text for hidden state extraction
            messages = [{"role": "user", "content": (
                "Solve the following math problem step by step. "
                "Number each step as 'Step 1:', 'Step 2:', etc. "
                f"End with 'ANSWER: <your final answer>'.\n\nProblem: {prob['problem']}\n\nSolution:"
            )}]
            full_text = collector.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + resp

            hs = collector.extract_hidden_states(full_text)
            last_hs = hs[best_layer][-1].numpy().astype(np.float32)
            features.append(last_hs)
            labels.append(0 if correct else 1)
            ids.append(prob["id"])
        except Exception as e:
            print(f"  Error {prob['id']}: {e}")
            continue

        if (i + 1) % 25 == 0:
            acc = 1 - np.mean(labels)
            print(f"  [{i+1}/{len(problems)}] acc={acc:.3f}, n_err={sum(labels)}/{len(labels)}", flush=True)

    return np.array(features), np.array(labels), ids


def train_probe(X, y):
    """Train a logistic regression probe with CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
    clf.fit(X_scaled, y)
    return clf, scaler, float(np.mean(scores))


def evaluate_probe(clf, scaler, X, y):
    """Evaluate a pre-trained probe on new data."""
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    if len(set(y)) < 2:
        return {"auroc": None, "acc": 1 - np.mean(y), "n": len(y)}
    auroc = roc_auc_score(y, probs)
    return {"auroc": round(auroc, 4), "acc": round(1 - np.mean(y), 4), "n": len(y)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/cross_benchmark")
    parser.add_argument("--math-probe-dir", default=None,
                        help="Dir with existing MATH-500 probe to skip retraining")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine best layer
    if args.math_probe_dir:
        probe_dir = Path(args.math_probe_dir)
        summary = json.load(open(probe_dir / "probe_summary.json"))
        best = max(summary, key=lambda x: x["cv_auroc_mean"])
        best_layer = best["layer"]
        with open(probe_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
            data = pickle.load(f)
        math_clf, math_scaler = data["clf"], data["scaler"]
        print(f"Loaded existing MATH probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")
    else:
        # Auto-detect best layer from model
        model_short = args.model.split("/")[-1]
        if "3B" in model_short: best_layer = 27
        elif "7B" in model_short: best_layer = 16
        elif "14B" in model_short: best_layer = 39
        else: best_layer = 27
        math_clf, math_scaler = None, None

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device,
        layers_to_probe=[best_layer]
    )

    results = {}

    # === Phase 1: Collect MATH-500 data (train probe if needed) ===
    print("\n=== MATH-500 ===")
    math_problems = load_benchmark("math500", max_samples=200, seed=42)
    math_train = math_problems[:100]
    math_eval = math_problems[100:]

    if math_clf is None:
        print("Collecting MATH-500 training data...")
        X_math_train, y_math_train, _ = collect_hidden_states(
            collector, math_train, best_layer
        )
        math_clf, math_scaler, math_cv = train_probe(X_math_train, y_math_train)
        print(f"MATH probe: CV AUROC = {math_cv:.4f}")
        results["math500_train"] = {
            "cv_auroc": round(math_cv, 4),
            "n_train": len(X_math_train),
            "acc": round(1 - np.mean(y_math_train), 4),
        }

    print("Evaluating on MATH-500 held-out...")
    X_math_eval, y_math_eval, _ = collect_hidden_states(
        collector, math_eval, best_layer
    )
    math_eval_result = evaluate_probe(math_clf, math_scaler, X_math_eval, y_math_eval)
    results["math500_eval"] = math_eval_result
    print(f"  MATH-500 eval: AUROC={math_eval_result['auroc']}, acc={math_eval_result['acc']}")

    # === Phase 2: GSM8K (easier math — transfer + native) ===
    print("\n=== GSM8K ===")
    gsm_problems = load_benchmark("gsm8k", max_samples=200, seed=42)
    gsm_train = gsm_problems[:100]
    gsm_eval = gsm_problems[100:]

    print("Collecting GSM8K data...")
    X_gsm_all, y_gsm_all, _ = collect_hidden_states(
        collector, gsm_problems, best_layer
    )
    X_gsm_train, y_gsm_train = X_gsm_all[:len(gsm_train)], y_gsm_all[:len(gsm_train)]
    X_gsm_eval, y_gsm_eval = X_gsm_all[len(gsm_train):], y_gsm_all[len(gsm_train):]

    # Transfer: MATH probe → GSM8K
    gsm_transfer = evaluate_probe(math_clf, math_scaler, X_gsm_all, y_gsm_all)
    results["gsm8k_transfer"] = gsm_transfer
    print(f"  GSM8K (MATH probe transfer): AUROC={gsm_transfer['auroc']}")

    # Native: train on GSM8K
    if len(set(y_gsm_train)) >= 2:
        gsm_clf, gsm_scaler, gsm_cv = train_probe(X_gsm_train, y_gsm_train)
        gsm_native = evaluate_probe(gsm_clf, gsm_scaler, X_gsm_eval, y_gsm_eval)
        results["gsm8k_native"] = {"cv_auroc": round(gsm_cv, 4), **gsm_native}
        print(f"  GSM8K (native probe): CV={gsm_cv:.4f}, eval AUROC={gsm_native['auroc']}")
    else:
        print(f"  GSM8K: too few errors ({sum(y_gsm_train)}/{len(y_gsm_train)}), skipping native probe")
        results["gsm8k_native"] = {"note": "too few errors for training"}

    # === Phase 3: PrOntoQA (logical reasoning — non-math transfer) ===
    print("\n=== PrOntoQA (Logical Reasoning) ===")
    logic_problems = load_benchmark("prontoqa", max_samples=200, seed=42)

    print("Collecting PrOntoQA data...")
    X_logic, y_logic, _ = collect_hidden_states(
        collector, logic_problems, best_layer
    )

    # Transfer: MATH probe → Logic
    logic_transfer = evaluate_probe(math_clf, math_scaler, X_logic, y_logic)
    results["prontoqa_transfer"] = logic_transfer
    print(f"  PrOntoQA (MATH probe transfer): AUROC={logic_transfer['auroc']}")

    # Native probe on logic
    mid = len(X_logic) // 2
    X_logic_train, y_logic_train = X_logic[:mid], y_logic[:mid]
    X_logic_eval, y_logic_eval = X_logic[mid:], y_logic[mid:]

    if len(set(y_logic_train)) >= 2:
        logic_clf, logic_scaler, logic_cv = train_probe(X_logic_train, y_logic_train)
        logic_native = evaluate_probe(logic_clf, logic_scaler, X_logic_eval, y_logic_eval)
        results["prontoqa_native"] = {"cv_auroc": round(logic_cv, 4), **logic_native}
        print(f"  PrOntoQA (native probe): CV={logic_cv:.4f}, eval AUROC={logic_native['auroc']}")
    else:
        print(f"  PrOntoQA: too few errors, skipping native probe")
        results["prontoqa_native"] = {"note": "too few errors"}

    # === Save ===
    results["model"] = args.model
    results["best_layer"] = best_layer

    with open(output_dir / "cross_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("CROSS-BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}, Layer: {best_layer}")
    for key in ["math500_eval", "gsm8k_transfer", "gsm8k_native", "prontoqa_transfer", "prontoqa_native"]:
        if key in results:
            r = results[key]
            auroc = r.get("auroc", r.get("cv_auroc", "N/A"))
            print(f"  {key}: AUROC={auroc}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
