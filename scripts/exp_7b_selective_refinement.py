#!/usr/bin/env python3
"""Experiment: Selective Refinement for Qwen2.5-7B-Instruct.

Uses the probe to decide WHICH problems to retry, comparing:
  - Probe-guided: sort by probe score (descending), retry the top X%
  - Random baseline: retry a random X% of problems

At each trigger rate (0.1 to 0.9), measure accuracy after selective retry.
The retry is the model itself regenerating with temp=0.7 (no external API).

Pipeline:
1. Load or train 7B probe
2. Generate greedy traces for 100 eval problems, score each with probe
3. For each trigger rate:
   a. Probe-guided: select top-X% by probe score, regenerate those
   b. Random: select random X%, regenerate those
   c. For retried problems, keep the better answer (lower probe score)
4. Compare accuracy curves

Saves to results/selective_refinement_7b/.
"""

import argparse, json, pickle, random, re, sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.probe.collect_hidden_states import HiddenStateCollector
from src.pipeline.data_loader import load_benchmark


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\\boxed\{(.+?)\}', text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def extract_num(s):
    s = re.sub(r'[,$%]', '', s.strip())
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else None


def normalize_answer(s):
    s = s.strip()
    s = re.sub(r'^\$+|^\\\(|\\\)$|\$+$', '', s)
    s = re.sub(r'^\\left\s*', '', s)
    s = re.sub(r'\\right\s*$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def check_answer(predicted, reference):
    p, r = extract_num(predicted), extract_num(reference)
    if p is not None and r is not None:
        return abs(p - r) < 0.01
    pn = normalize_answer(predicted)
    rn = normalize_answer(reference)
    if pn == rn:
        return True
    if len(rn) > 3 and (rn in pn or pn in rn):
        return True
    return False


def get_probe_score(collector, clf, scaler, best_layer, problem_text, response):
    """Get probe error score for a (problem, response) pair."""
    messages = [{"role": "user", "content": (
        "Solve the following math problem step by step. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        "End with 'ANSWER: <your final answer>'.\n\n"
        f"Problem: {problem_text}\n\nSolution:"
    )}]
    full_text = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + response

    hs = collector.extract_hidden_states(full_text)
    last_hs = hs[best_layer][-1].cpu().numpy().astype(np.float32).reshape(1, -1)
    score = float(clf.predict_proba(scaler.transform(last_hs))[0, 1])
    return score


def train_probe_if_needed(collector, best_layer_hint=None):
    """Train a probe from scratch if no saved probe exists. Returns (clf, scaler, best_layer)."""
    from src.probe.collect_hidden_states import HiddenStateCollector
    from sklearn.model_selection import cross_val_score

    print("Training probe from scratch...")
    train_problems = load_benchmark("math500", max_samples=150, seed=42)

    # Determine layers to probe
    n_layers = collector.model.config.num_hidden_layers
    layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    collector.layers = layers
    print(f"  Probing layers: {layers}")

    # Collect hidden states
    X_by_layer = {l: [] for l in layers}
    y_labels = []

    for i, problem in enumerate(train_problems):
        try:
            resp = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(resp)
            correct = check_answer(ans, problem["answer"])

            messages = [{"role": "user", "content": (
                "Solve the following math problem step by step. "
                "Number each step as 'Step 1:', 'Step 2:', etc. "
                "End with 'ANSWER: <your final answer>'.\n\n"
                f"Problem: {problem['problem']}\n\nSolution:"
            )}]
            full_text = collector.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + resp

            hs = collector.extract_hidden_states(full_text)
            for l in layers:
                last_hs = hs[l][-1].cpu().numpy().astype(np.float32)
                X_by_layer[l].append(last_hs)
            y_labels.append(0 if correct else 1)  # 1 = error

        except Exception as e:
            print(f"  Skip {problem['id']}: {e}")
            continue

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(train_problems)}] collected", flush=True)

    y = np.array(y_labels)
    print(f"  Collected {len(y)} samples ({sum(y)} errors, {len(y)-sum(y)} correct)")

    # Train probe for each layer, pick best
    best_layer = layers[0]
    best_auroc = 0.0
    best_clf = None
    best_scaler = None
    probe_results = []

    for l in layers:
        X = np.array(X_by_layer[l])
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        aurocs = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")
        mean_auroc = float(np.mean(aurocs))
        print(f"  Layer {l}: AUROC={mean_auroc:.4f} +/- {np.std(aurocs):.4f}")
        probe_results.append({"layer": l, "cv_auroc_mean": round(mean_auroc, 4)})

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_layer = l
            lr.fit(X_scaled, y)
            best_clf = lr
            best_scaler = sc

    # Save probe
    model_short = collector.model.config._name_or_path.split("/")[-1]
    probe_dir = Path(f"data/probes_{model_short}")
    probe_dir.mkdir(parents=True, exist_ok=True)

    with open(probe_dir / f"probe_layer_{best_layer}.pkl", "wb") as f:
        pickle.dump({"clf": best_clf, "scaler": best_scaler, "layer": best_layer}, f)
    with open(probe_dir / "probe_summary.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"  Best layer: {best_layer}, AUROC={best_auroc:.4f}")
    print(f"  Saved probe to {probe_dir}")

    # Reset collector layers to best only
    collector.layers = [best_layer]

    return best_clf, best_scaler, best_layer


def main():
    parser = argparse.ArgumentParser(
        description="Selective refinement experiment for Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--trigger-rates", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for retry generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/selective_refinement_7b")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or train probe ───────────────────────────────────
    model_short = args.model.split("/")[-1]
    probe_dir = Path(f"data/probes_{model_short}")

    if (probe_dir / "probe_summary.json").exists():
        summary = json.load(open(probe_dir / "probe_summary.json"))
        best = max(summary, key=lambda x: x["cv_auroc_mean"])
        best_layer = best["layer"]
        probe_path = probe_dir / f"probe_layer_{best_layer}.pkl"
        if probe_path.exists():
            with open(probe_path, "rb") as f:
                probe_data = pickle.load(f)
            clf = probe_data["clf"]
            scaler = probe_data["scaler"]
            print(f"Loaded probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")
            need_train = False
        else:
            need_train = True
    else:
        need_train = True

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device,
        layers_to_probe=[best_layer] if not need_train else None
    )

    if need_train:
        clf, scaler, best_layer = train_probe_if_needed(collector)
        collector.layers = [best_layer]

    # ── Load eval problems ────────────────────────────────────
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    N = len(eval_problems)
    print(f"Eval: {N} problems")

    # ── Phase 1: Greedy generation + probe scoring ────────────
    print(f"\n=== Phase 1: Greedy generation + probe scoring ===")
    problem_data = []

    for i, problem in enumerate(eval_problems):
        try:
            resp = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(resp)
            correct = check_answer(ans, problem["answer"])
            score = get_probe_score(
                collector, clf, scaler, best_layer,
                problem["problem"], resp
            )
        except Exception as e:
            print(f"  Error {problem['id']}: {e}")
            resp, ans, correct, score = "", "", False, 1.0

        problem_data.append({
            "id": problem["id"],
            "problem": problem["problem"],
            "reference": problem["answer"],
            "greedy_answer": ans,
            "greedy_correct": correct,
            "greedy_score": score,
            "retry_answer": None,
            "retry_correct": None,
            "retry_score": None,
        })

        if (i + 1) % 10 == 0:
            acc = sum(1 for d in problem_data if d["greedy_correct"]) / len(problem_data)
            avg_score = np.mean([d["greedy_score"] for d in problem_data])
            print(f"  [{i+1}/{N}] acc={acc:.3f} avg_probe={avg_score:.3f}", flush=True)

    greedy_acc = sum(1 for d in problem_data if d["greedy_correct"]) / N
    print(f"Greedy accuracy: {greedy_acc:.4f}")

    # ── Phase 2: Generate retry for ALL problems ──────────────
    # (We generate retries for all so we can simulate any trigger rate)
    print(f"\n=== Phase 2: Generating retries for all problems (temp={args.temperature}) ===")

    for i, d in enumerate(problem_data):
        try:
            resp = collector.generate_cot(d["problem"], temperature=args.temperature)
            ans = extract_answer(resp)
            correct = check_answer(ans, d["reference"])
            score = get_probe_score(
                collector, clf, scaler, best_layer,
                d["problem"], resp
            )
            d["retry_answer"] = ans
            d["retry_correct"] = correct
            d["retry_score"] = score
        except Exception as e:
            print(f"  Retry error {d['id']}: {e}")
            d["retry_answer"] = ""
            d["retry_correct"] = False
            d["retry_score"] = 1.0

        if (i + 1) % 10 == 0:
            retry_acc = sum(1 for d2 in problem_data[:i+1]
                          if d2["retry_correct"] is not None and d2["retry_correct"]) / (i + 1)
            print(f"  [{i+1}/{N}] retry_acc={retry_acc:.3f}", flush=True)

    retry_all_acc = sum(1 for d in problem_data if d["retry_correct"]) / N
    print(f"Retry-all accuracy: {retry_all_acc:.4f}")

    # ── Phase 3: Evaluate at each trigger rate ────────────────
    print(f"\n=== Phase 3: Selective refinement at each trigger rate ===")

    # Sort indices by probe score (descending = most likely error first)
    sorted_indices = sorted(range(N), key=lambda k: -problem_data[k]["greedy_score"])

    results = []

    # Add baselines
    results.append({
        "method": "greedy",
        "trigger_rate": 0.0,
        "k": 0,
        "acc": round(greedy_acc, 4),
        "retries": 0,
    })
    results.append({
        "method": "retry_all",
        "trigger_rate": 1.0,
        "k": N,
        "acc": round(retry_all_acc, 4),
        "retries": N,
    })

    print(f"\n{'Method':<25} {'Rate':>6} {'k':>5} {'Acc':>8} {'Delta':>8}")
    print("-" * 58)
    print(f"{'greedy':<25} {'0%':>6} {0:>5} {greedy_acc:>8.4f} {'+0.0000':>8}")

    for rate in args.trigger_rates:
        k = int(N * rate)

        # ── Probe-guided: pick top-k by probe score ──
        probe_selected = set(sorted_indices[:k])
        probe_correct = 0
        for idx, d in enumerate(problem_data):
            if idx in probe_selected:
                # Use retry if its probe score is lower (more confident)
                if d["retry_score"] is not None and d["retry_score"] < d["greedy_score"]:
                    if d["retry_correct"]:
                        probe_correct += 1
                else:
                    if d["greedy_correct"]:
                        probe_correct += 1
            else:
                if d["greedy_correct"]:
                    probe_correct += 1
        probe_acc = probe_correct / N

        results.append({
            "method": "probe_guided",
            "trigger_rate": rate,
            "k": k,
            "acc": round(probe_acc, 4),
            "retries": k,
        })

        # ── Random baseline: pick random k ──
        random_indices = set(rng.sample(range(N), k))
        random_correct = 0
        for idx, d in enumerate(problem_data):
            if idx in random_indices:
                if d["retry_score"] is not None and d["retry_score"] < d["greedy_score"]:
                    if d["retry_correct"]:
                        random_correct += 1
                else:
                    if d["greedy_correct"]:
                        random_correct += 1
            else:
                if d["greedy_correct"]:
                    random_correct += 1
        random_acc = random_correct / N

        results.append({
            "method": "random_trigger",
            "trigger_rate": rate,
            "k": k,
            "acc": round(random_acc, 4),
            "retries": k,
        })

        probe_delta = probe_acc - greedy_acc
        random_delta = random_acc - greedy_acc
        print(f"{'probe_guided':<25} {rate*100:>5.0f}% {k:>5} {probe_acc:>8.4f} {probe_delta:>+8.4f}")
        print(f"{'random_trigger':<25} {rate*100:>5.0f}% {k:>5} {random_acc:>8.4f} {random_delta:>+8.4f}")

    print(f"{'retry_all':<25} {'100%':>6} {N:>5} {retry_all_acc:>8.4f} "
          f"{retry_all_acc - greedy_acc:>+8.4f}")

    # ── Oracle baseline: only retry when actually wrong ───────
    oracle_correct = sum(1 for d in problem_data
                         if d["greedy_correct"] or d["retry_correct"])
    oracle_acc = oracle_correct / N
    oracle_retries = sum(1 for d in problem_data if not d["greedy_correct"])
    results.append({
        "method": "oracle",
        "trigger_rate": round(oracle_retries / N, 4),
        "k": oracle_retries,
        "acc": round(oracle_acc, 4),
        "retries": oracle_retries,
    })
    print(f"{'oracle':<25} {oracle_retries/N*100:>5.0f}% {oracle_retries:>5} "
          f"{oracle_acc:>8.4f} {oracle_acc - greedy_acc:>+8.4f}")

    # ── Comparison summary ────────────────────────────────────
    print(f"\n=== Probe-Guided vs Random Comparison ===")
    print(f"{'Rate':>6} {'Probe':>8} {'Random':>8} {'Diff':>8} {'Probe wins?'}")
    print("-" * 50)
    for rate in args.trigger_rates:
        probe_r = next(r for r in results
                       if r["method"] == "probe_guided" and r["trigger_rate"] == rate)
        random_r = next(r for r in results
                        if r["method"] == "random_trigger" and r["trigger_rate"] == rate)
        diff = probe_r["acc"] - random_r["acc"]
        wins = "YES" if diff > 0 else ("TIE" if diff == 0 else "NO")
        print(f"{rate*100:>5.0f}% {probe_r['acc']:>8.4f} {random_r['acc']:>8.4f} "
              f"{diff:>+8.4f} {wins}")

    # ── Save results ──────────────────────────────────────────
    summary_out = {
        "model": args.model,
        "probe_layer": best_layer,
        "n_problems": N,
        "temperature": args.temperature,
        "greedy_acc": round(greedy_acc, 4),
        "retry_all_acc": round(retry_all_acc, 4),
        "oracle_acc": round(oracle_acc, 4),
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved summary to {output_dir / 'summary.json'}")

    # Per-problem details
    details = []
    for d in problem_data:
        details.append({
            "id": d["id"],
            "reference": d["reference"],
            "greedy_answer": d["greedy_answer"],
            "greedy_correct": d["greedy_correct"],
            "greedy_score": round(d["greedy_score"], 4),
            "retry_answer": d["retry_answer"],
            "retry_correct": d["retry_correct"],
            "retry_score": round(d["retry_score"], 4) if d["retry_score"] is not None else None,
        })
    with open(output_dir / "per_problem.json", "w") as f:
        json.dump(details, f, indent=2)
    print(f"Saved per-problem details to {output_dir / 'per_problem.json'}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"SELECTIVE REFINEMENT SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"Greedy acc:   {greedy_acc:.4f}")
    print(f"Retry-all:    {retry_all_acc:.4f} ({retry_all_acc - greedy_acc:>+.4f})")
    print(f"Oracle:       {oracle_acc:.4f} ({oracle_acc - greedy_acc:>+.4f})")
    probe_rates = [r for r in results if r["method"] == "probe_guided"]
    if probe_rates:
        best_pr = max(probe_rates, key=lambda r: r["acc"])
        print(f"Best probe:   {best_pr['acc']:.4f} at {best_pr['trigger_rate']*100:.0f}% "
              f"({best_pr['acc'] - greedy_acc:>+.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
