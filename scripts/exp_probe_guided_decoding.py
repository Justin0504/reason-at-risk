#!/usr/bin/env python3
"""Experiment: Probe-Guided Decoding — use probe to select among candidate traces.

Two methods:
1. Best-of-N (BoN): Generate N full traces with temperature>0, select the one
   with LOWEST probe error score. Compare with:
   - Greedy (N=1, temp=0): baseline
   - Random-of-N: pick random trace from N candidates (= majority vote ceiling approximation)
   - Majority vote (MV): pick most common answer among N traces
   - Oracle: pick correct trace if any exists among N (upper bound)

2. Step-level guided: At each reasoning step, generate K continuations,
   score each with probe, keep the best, continue from there.

Key insight: if probe-guided BoN > random BoN / MV, it proves the probe
provides signal BEYOND just sampling diversity. This is a practical method
that requires only 1 probe forward pass per trace (cheap).
"""

import argparse, json, pickle, re, sys, time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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
    """Normalize answer string for comparison."""
    s = s.strip()
    # Remove LaTeX wrappers
    s = re.sub(r'^\$+|^\\\(|\\\)$|\$+$', '', s)
    s = re.sub(r'^\\left\s*', '', s)
    s = re.sub(r'\\right\s*$', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def check_answer(predicted, reference):
    # Try numeric comparison first
    p, r = extract_num(predicted), extract_num(reference)
    if p is not None and r is not None:
        return abs(p - r) < 0.01
    # Normalize and compare
    pn = normalize_answer(predicted)
    rn = normalize_answer(reference)
    if pn == rn:
        return True
    # Check if one contains the other (for truncated references)
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


def majority_vote(answers):
    """Return most common answer."""
    if not answers:
        return ""
    counts = Counter(a.strip().lower() for a in answers if a.strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--n-samples", nargs="+", type=int, default=[1, 3, 5, 8, 12])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", default="results/probe_guided_decoding")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or train probe
    model_short = args.model.split("/")[-1]
    probe_dir = Path(f"data/probes_{model_short}")
    if not (probe_dir / "probe_summary.json").exists():
        probe_dir = Path("data/probes")

    if (probe_dir / "probe_summary.json").exists():
        summary = json.load(open(probe_dir / "probe_summary.json"))
        best = max(summary, key=lambda x: x["cv_auroc_mean"])
        best_layer = best["layer"]
        with open(probe_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
            probe_data = pickle.load(f)
        clf = probe_data["clf"]
        scaler = probe_data["scaler"]
        print(f"Loaded probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")
    else:
        print("ERROR: No probe found. Run probe training first.")
        return

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device, layers_to_probe=[best_layer]
    )

    # Load eval problems
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Eval: {len(eval_problems)} problems, N_samples: {args.n_samples}")

    max_n = max(args.n_samples)

    # Generate max_n traces per problem
    print(f"\n=== Generating {max_n} traces per problem (temp={args.temperature}) ===")
    all_traces = []

    for i, problem in enumerate(eval_problems):
        traces = []

        # Greedy (temp=0) trace
        try:
            greedy_resp = collector.generate_cot(problem["problem"], temperature=0.0)
            greedy_ans = extract_answer(greedy_resp)
            greedy_correct = check_answer(greedy_ans, problem["answer"])
            greedy_score = get_probe_score(
                collector, clf, scaler, best_layer,
                problem["problem"], greedy_resp
            )
        except Exception as e:
            greedy_resp, greedy_ans, greedy_correct, greedy_score = "", "", False, 1.0

        # Sampled traces
        for j in range(max_n):
            try:
                resp = collector.generate_cot(
                    problem["problem"], temperature=args.temperature
                )
                ans = extract_answer(resp)
                correct = check_answer(ans, problem["answer"])
                score = get_probe_score(
                    collector, clf, scaler, best_layer,
                    problem["problem"], resp
                )
                traces.append({
                    "response_len": len(resp),
                    "answer": ans,
                    "correct": correct,
                    "probe_score": score,
                })
            except Exception as e:
                traces.append({
                    "response_len": 0, "answer": "", "correct": False,
                    "probe_score": 1.0,
                })

        all_traces.append({
            "id": problem["id"],
            "reference": problem["answer"],
            "greedy_correct": greedy_correct,
            "greedy_score": greedy_score,
            "traces": traces,
        })

        if (i + 1) % 10 == 0:
            greedy_acc = sum(1 for t in all_traces if t["greedy_correct"]) / len(all_traces)
            print(f"  [{i+1}/{len(eval_problems)}] greedy_acc={greedy_acc:.3f}", flush=True)

    # Evaluate different strategies at each N
    print(f"\n=== Results ===")
    print(f"{'N':>4} {'Greedy':>8} {'Random':>8} {'MajVote':>8} {'ProbeBoN':>8} {'Oracle':>8}")
    print("-" * 50)

    results = {}
    N_total = len(all_traces)
    greedy_acc = sum(1 for t in all_traces if t["greedy_correct"]) / N_total

    for n in args.n_samples:
        random_correct = 0
        mv_correct = 0
        probe_correct = 0
        oracle_correct = 0

        for t in all_traces:
            candidates = t["traces"][:n]
            if not candidates:
                continue

            # Random: pick a random trace
            rand_idx = np.random.randint(len(candidates))
            if candidates[rand_idx]["correct"]:
                random_correct += 1

            # Majority vote: pick most common answer
            answers = [c["answer"] for c in candidates]
            mv_ans = majority_vote(answers)
            if check_answer(mv_ans, t["reference"]):
                mv_correct += 1

            # Probe-guided BoN: pick trace with LOWEST probe score
            best_idx = min(range(len(candidates)), key=lambda k: candidates[k]["probe_score"])
            if candidates[best_idx]["correct"]:
                probe_correct += 1

            # Oracle: any correct trace exists?
            if any(c["correct"] for c in candidates):
                oracle_correct += 1

        random_acc = random_correct / N_total
        mv_acc = mv_correct / N_total
        probe_acc = probe_correct / N_total
        oracle_acc = oracle_correct / N_total

        results[n] = {
            "n": n,
            "greedy": round(greedy_acc, 4),
            "random": round(random_acc, 4),
            "majority_vote": round(mv_acc, 4),
            "probe_bon": round(probe_acc, 4),
            "oracle": round(oracle_acc, 4),
        }

        print(f"{n:>4} {greedy_acc:>8.4f} {random_acc:>8.4f} {mv_acc:>8.4f} "
              f"{probe_acc:>8.4f} {oracle_acc:>8.4f}")

    # Compute deltas
    print(f"\n{'N':>4} {'Δ(Probe-Greedy)':>16} {'Δ(Probe-MV)':>14} {'Δ(Probe-Random)':>16}")
    print("-" * 55)
    for n in args.n_samples:
        r = results[n]
        print(f"{n:>4} {r['probe_bon']-r['greedy']:>+16.4f} "
              f"{r['probe_bon']-r['majority_vote']:>+14.4f} "
              f"{r['probe_bon']-r['random']:>+16.4f}")

    # Save
    summary = {
        "model": args.model,
        "n_problems": N_total,
        "temperature": args.temperature,
        "greedy_acc": round(greedy_acc, 4),
        "results": results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-problem traces (without full text to save space)
    with open(output_dir / "traces.json", "w") as f:
        json.dump(all_traces, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
