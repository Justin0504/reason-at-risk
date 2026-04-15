#!/usr/bin/env python3
"""Probe + Majority Vote hybrid experiment.

Strategy: Use probe to identify likely-wrong traces, then only apply
majority vote (expensive) on those flagged problems. Compare with:
- Pure MV on all problems
- Pure probe routing
- Random selection for MV

This demonstrates compute-efficient error correction without external API.
"""

import argparse, json, re, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
    pn, rn = predicted.strip().lower(), reference.strip().lower()
    if pn == rn: return True
    if len(rn) > 3 and (rn in pn or pn in rn): return True
    return False


def majority_vote(answers):
    """Return the most common answer."""
    from collections import Counter
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="results/probe_mv_hybrid")
    parser.add_argument("--probe-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine best layer
    model_short = args.model.split("/")[-1]
    if "3B" in model_short: best_layer = 27
    elif "7B" in model_short: best_layer = 16
    elif "14B" in model_short: best_layer = 39
    else: best_layer = 27

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device,
        layers_to_probe=[best_layer]
    )

    # Load or train probe
    if args.probe_dir:
        import pickle
        probe_path = Path(args.probe_dir) / f"probe_layer_{best_layer}.pkl"
        with open(probe_path, "rb") as f:
            data = pickle.load(f)
        clf, scaler = data["clf"], data["scaler"]
        if not hasattr(clf, 'multi_class'):
            clf.multi_class = "auto"
        print(f"Loaded probe from {probe_path}")
    else:
        print("Training probe on MATH-500...")
        train_problems = load_benchmark("math500", max_samples=100, seed=42)
        X_train, y_train = [], []
        for prob in train_problems:
            try:
                resp = collector.generate_cot(prob["problem"])
                ans = extract_answer(resp)
                correct = check_answer(ans, prob["answer"])
                messages = [{"role": "user", "content": f"Solve step by step. End with 'ANSWER: <answer>'.\n\nProblem: {prob['problem']}\n\nSolution:"}]
                full_text = collector.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + resp
                hs = collector.extract_hidden_states(full_text)
                X_train.append(hs[best_layer][-1].numpy().astype(np.float32))
                y_train.append(0 if correct else 1)
            except:
                continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        clf.fit(X_scaled, y_train)
        print(f"Probe trained: {len(X_train)} samples, {sum(y_train)} errors")

    # Load evaluation problems
    problems = load_benchmark("math500", max_samples=args.n_problems, seed=123)
    print(f"\n=== Evaluating on {len(problems)} problems, N={args.n_samples} samples ===")

    results_per_problem = []
    from transformers import GenerationConfig

    for i, prob in enumerate(problems):
        try:
            # Generate N samples
            samples = []
            for j in range(args.n_samples):
                temp = 0.7 if j > 0 else 0.0  # First is greedy
                messages = [{"role": "user", "content": f"Solve step by step. End with 'ANSWER: <answer>'.\n\nProblem: {prob['problem']}\n\nSolution:"}]
                input_text = collector.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = collector.tokenizer(input_text, return_tensors="pt").to(collector.device)

                gen_config = GenerationConfig(
                    max_new_tokens=512, do_sample=(temp > 0),
                    temperature=max(temp, 0.01), top_p=0.95,
                    pad_token_id=collector.tokenizer.eos_token_id,
                )
                with torch.no_grad():
                    out = collector.model.generate(**inputs, generation_config=gen_config)
                gen = out[0][inputs.input_ids.shape[1]:]
                resp = collector.tokenizer.decode(gen, skip_special_tokens=True)
                ans = extract_answer(resp)
                correct = check_answer(ans, prob["answer"])

                # Get probe score for greedy trace
                if j == 0:
                    full_text = input_text + resp
                    hs = collector.extract_hidden_states(full_text)
                    last_hs = hs[best_layer][-1].numpy().astype(np.float32).reshape(1, -1)
                    probe_score = float(clf.predict_proba(scaler.transform(last_hs))[0, 1])
                else:
                    probe_score = None

                samples.append({
                    "answer": ans,
                    "correct": correct,
                    "probe_score": probe_score,
                })

            greedy_correct = samples[0]["correct"]
            greedy_probe = samples[0]["probe_score"]
            greedy_ans = samples[0]["answer"]

            # MV over all N
            all_answers = [s["answer"] for s in samples]
            mv_ans = majority_vote(all_answers)
            mv_correct = check_answer(mv_ans, prob["answer"])

            # Oracle: is there any correct answer?
            oracle_correct = any(s["correct"] for s in samples)

            results_per_problem.append({
                "id": prob.get("id", i),
                "greedy_correct": greedy_correct,
                "greedy_probe_score": greedy_probe,
                "mv_correct": mv_correct,
                "oracle_correct": oracle_correct,
                "n_correct_samples": sum(s["correct"] for s in samples),
            })

        except Exception as e:
            print(f"  Error {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            gc = sum(r["greedy_correct"] for r in results_per_problem)
            mc = sum(r["mv_correct"] for r in results_per_problem)
            n = len(results_per_problem)
            print(f"  [{i+1}/{len(problems)}] greedy={gc/n:.3f}, mv={mc/n:.3f}", flush=True)

    # === Compute hybrid strategies ===
    n = len(results_per_problem)
    greedy_acc = sum(r["greedy_correct"] for r in results_per_problem) / n
    mv_acc = sum(r["mv_correct"] for r in results_per_problem) / n
    oracle_acc = sum(r["oracle_correct"] for r in results_per_problem) / n

    print(f"\n{'='*60}")
    print(f"Greedy: {greedy_acc:.3f}")
    print(f"MV@{args.n_samples}: {mv_acc:.3f} (all problems, {args.n_samples}x compute)")
    print(f"Oracle: {oracle_acc:.3f}")

    # Sort by probe score (descending = most likely errors first)
    sorted_results = sorted(results_per_problem, key=lambda r: -r["greedy_probe_score"])

    hybrid_results = []
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        k = int(n * rate)

        # Probe-guided: apply MV to top-k probe-flagged, keep greedy for rest
        probe_flagged = set(r["id"] for r in sorted_results[:k])
        probe_hybrid_correct = 0
        for r in results_per_problem:
            if r["id"] in probe_flagged:
                probe_hybrid_correct += r["mv_correct"]
            else:
                probe_hybrid_correct += r["greedy_correct"]
        probe_hybrid_acc = probe_hybrid_correct / n

        # Random baseline: apply MV to random k problems
        rng = np.random.RandomState(42)
        random_idxs = set(rng.choice(n, k, replace=False))
        random_hybrid_correct = 0
        for j, r in enumerate(results_per_problem):
            if j in random_idxs:
                random_hybrid_correct += r["mv_correct"]
            else:
                random_hybrid_correct += r["greedy_correct"]
        random_hybrid_acc = random_hybrid_correct / n

        # Compute cost: greedy=1x, MV on k problems = 1 + k/n * (N-1)
        effective_cost = 1.0 + rate * (args.n_samples - 1)

        row = {
            "rate": rate,
            "k": k,
            "probe_hybrid_acc": round(probe_hybrid_acc, 4),
            "random_hybrid_acc": round(random_hybrid_acc, 4),
            "mv_all_acc": round(mv_acc, 4),
            "greedy_acc": round(greedy_acc, 4),
            "effective_cost_multiplier": round(effective_cost, 1),
        }
        hybrid_results.append(row)
        delta = probe_hybrid_acc - greedy_acc
        print(f"  Rate={rate:.0%}: probe+MV={probe_hybrid_acc:.3f} (+{delta:.3f}), random+MV={random_hybrid_acc:.3f}, cost={effective_cost:.1f}x")

    summary = {
        "model": args.model,
        "n_problems": n,
        "n_samples": args.n_samples,
        "greedy_acc": round(greedy_acc, 4),
        "mv_all_acc": round(mv_acc, 4),
        "oracle_acc": round(oracle_acc, 4),
        "hybrid_results": hybrid_results,
        "per_problem": results_per_problem,
    }

    with open(output_dir / "probe_mv_hybrid.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
