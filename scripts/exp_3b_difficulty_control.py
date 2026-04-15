#!/usr/bin/env python3
"""Experiment: Difficulty Control for Qwen2.5-3B-Instruct.

For each eval problem, generate multiple traces with temperature>0 and score
each with the trained probe. Then analyze:
  1. score_std per problem — do "hard" problems yield more variable probe scores?
  2. Within mixed-outcome problems (some correct, some wrong), compare probe
     scores of correct vs wrong generations.
  3. t-test on correct vs wrong probe scores across mixed problems.

This mirrors the 7B difficulty control experiment, applied to the 3B model
using the existing probe at layer 27 (AUROC 0.953).
"""

import argparse, json, pickle, re, sys, time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

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


def main():
    parser = argparse.ArgumentParser(
        description="Difficulty control experiment for Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--n-traces", type=int, default=5,
                        help="Number of sampled traces per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", default="results/difficulty_control_3b")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load probe ────────────────────────────────────────────
    model_short = args.model.split("/")[-1]
    probe_dir = Path(f"data/probes_{model_short}")
    if not (probe_dir / "probe_summary.json").exists():
        probe_dir = Path("data/probes")

    if not (probe_dir / "probe_summary.json").exists():
        print("ERROR: No probe found. Run probe training first.")
        return

    summary = json.load(open(probe_dir / "probe_summary.json"))
    best = max(summary, key=lambda x: x["cv_auroc_mean"])
    best_layer = best["layer"]
    with open(probe_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
        probe_data = pickle.load(f)
    clf = probe_data["clf"]
    scaler = probe_data["scaler"]
    print(f"Loaded probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")

    # ── Load model ────────────────────────────────────────────
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device, layers_to_probe=[best_layer]
    )

    # ── Load eval problems (same split as other experiments) ──
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Eval: {len(eval_problems)} problems, {args.n_traces} traces each "
          f"(temp={args.temperature})")

    # ── Phase 1: Generate traces and score ────────────────────
    print(f"\n=== Phase 1: Generating {args.n_traces} traces per problem ===")
    problem_data = []

    for i, problem in enumerate(eval_problems):
        traces = []
        for j in range(args.n_traces):
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
                    "trace_idx": j,
                    "answer": ans,
                    "correct": correct,
                    "probe_score": score,
                })
            except Exception as e:
                print(f"  Error problem {problem['id']} trace {j}: {e}")
                traces.append({
                    "trace_idx": j,
                    "answer": "",
                    "correct": False,
                    "probe_score": 1.0,
                })

        n_correct = sum(1 for t in traces if t["correct"])
        scores = [t["probe_score"] for t in traces]
        problem_data.append({
            "id": problem["id"],
            "reference": problem["answer"],
            "n_correct": n_correct,
            "n_traces": len(traces),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "traces": traces,
        })

        if (i + 1) % 10 == 0:
            avg_acc = np.mean([d["n_correct"] / d["n_traces"] for d in problem_data])
            avg_std = np.mean([d["score_std"] for d in problem_data])
            print(f"  [{i+1}/{len(eval_problems)}] avg_acc={avg_acc:.3f} "
                  f"avg_score_std={avg_std:.4f}", flush=True)

    # ── Phase 2: Analysis ─────────────────────────────────────
    print(f"\n=== Phase 2: Analysis ===")
    N = len(problem_data)

    # Categorize problems
    all_correct = [d for d in problem_data if d["n_correct"] == d["n_traces"]]
    all_wrong = [d for d in problem_data if d["n_correct"] == 0]
    mixed = [d for d in problem_data
             if 0 < d["n_correct"] < d["n_traces"]]

    print(f"Problem categories:")
    print(f"  All correct:  {len(all_correct)}/{N}")
    print(f"  All wrong:    {len(all_wrong)}/{N}")
    print(f"  Mixed:        {len(mixed)}/{N}")

    # 2a. Score std by category
    if all_correct:
        print(f"\nScore std (all-correct):  mean={np.mean([d['score_std'] for d in all_correct]):.4f}")
    if all_wrong:
        print(f"Score std (all-wrong):    mean={np.mean([d['score_std'] for d in all_wrong]):.4f}")
    if mixed:
        print(f"Score std (mixed):        mean={np.mean([d['score_std'] for d in mixed]):.4f}")

    # 2b. Within mixed problems: correct vs wrong probe scores
    correct_scores = []
    wrong_scores = []
    for d in mixed:
        for t in d["traces"]:
            if t["correct"]:
                correct_scores.append(t["probe_score"])
            else:
                wrong_scores.append(t["probe_score"])

    if correct_scores and wrong_scores:
        print(f"\nWithin mixed problems:")
        print(f"  Correct traces: n={len(correct_scores)}, "
              f"mean_score={np.mean(correct_scores):.4f}, "
              f"std={np.std(correct_scores):.4f}")
        print(f"  Wrong traces:   n={len(wrong_scores)}, "
              f"mean_score={np.mean(wrong_scores):.4f}, "
              f"std={np.std(wrong_scores):.4f}")

        # t-test: correct vs wrong scores
        t_stat, p_value = stats.ttest_ind(correct_scores, wrong_scores,
                                          equal_var=False)
        cohens_d = (np.mean(wrong_scores) - np.mean(correct_scores)) / \
            np.sqrt((np.std(correct_scores)**2 + np.std(wrong_scores)**2) / 2)

        print(f"\n  t-test (correct vs wrong scores):")
        print(f"    t={t_stat:.4f}, p={p_value:.6f}")
        print(f"    Cohen's d = {cohens_d:.4f}")
        print(f"    Significant at 0.05: {'YES' if p_value < 0.05 else 'NO'}")
        print(f"    Significant at 0.01: {'YES' if p_value < 0.01 else 'NO'}")
    else:
        t_stat, p_value, cohens_d = None, None, None
        print("\nNot enough mixed-outcome problems for t-test.")

    # 2c. Correlation: problem difficulty (1 - accuracy) vs mean probe score
    difficulties = [1.0 - d["n_correct"] / d["n_traces"] for d in problem_data]
    mean_scores = [d["score_mean"] for d in problem_data]
    corr, corr_p = stats.pearsonr(difficulties, mean_scores)
    print(f"\nDifficulty vs mean probe score:")
    print(f"  Pearson r = {corr:.4f}, p = {corr_p:.6f}")

    rank_corr, rank_p = stats.spearmanr(difficulties, mean_scores)
    print(f"  Spearman rho = {rank_corr:.4f}, p = {rank_p:.6f}")

    # 2d. Score std vs difficulty correlation
    score_stds = [d["score_std"] for d in problem_data]
    std_corr, std_p = stats.pearsonr(difficulties, score_stds)
    print(f"\nDifficulty vs score std:")
    print(f"  Pearson r = {std_corr:.4f}, p = {std_p:.6f}")

    # ── Save results ──────────────────────────────────────────
    summary_out = {
        "model": args.model,
        "probe_layer": best_layer,
        "probe_auroc": best["cv_auroc_mean"],
        "n_problems": N,
        "n_traces_per_problem": args.n_traces,
        "temperature": args.temperature,
        "categories": {
            "all_correct": len(all_correct),
            "all_wrong": len(all_wrong),
            "mixed": len(mixed),
        },
        "score_std_by_category": {
            "all_correct": round(float(np.mean([d["score_std"] for d in all_correct])), 4) if all_correct else None,
            "all_wrong": round(float(np.mean([d["score_std"] for d in all_wrong])), 4) if all_wrong else None,
            "mixed": round(float(np.mean([d["score_std"] for d in mixed])), 4) if mixed else None,
        },
        "mixed_analysis": {
            "n_correct_traces": len(correct_scores),
            "n_wrong_traces": len(wrong_scores),
            "correct_score_mean": round(float(np.mean(correct_scores)), 4) if correct_scores else None,
            "wrong_score_mean": round(float(np.mean(wrong_scores)), 4) if wrong_scores else None,
            "t_stat": round(float(t_stat), 4) if t_stat is not None else None,
            "p_value": round(float(p_value), 6) if p_value is not None else None,
            "cohens_d": round(float(cohens_d), 4) if cohens_d is not None else None,
            "significant_005": bool(p_value < 0.05) if p_value is not None else None,
            "significant_001": bool(p_value < 0.01) if p_value is not None else None,
        },
        "difficulty_correlation": {
            "pearson_r": round(float(corr), 4),
            "pearson_p": round(float(corr_p), 6),
            "spearman_rho": round(float(rank_corr), 4),
            "spearman_p": round(float(rank_p), 6),
        },
        "difficulty_vs_std": {
            "pearson_r": round(float(std_corr), 4),
            "pearson_p": round(float(std_p), 6),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved summary to {output_dir / 'summary.json'}")

    # Per-problem details (without full response text)
    details = []
    for d in problem_data:
        details.append({
            "id": d["id"],
            "reference": d["reference"],
            "n_correct": d["n_correct"],
            "n_traces": d["n_traces"],
            "accuracy": round(d["n_correct"] / d["n_traces"], 4),
            "score_mean": round(d["score_mean"], 4),
            "score_std": round(d["score_std"], 4),
            "score_min": round(d["score_min"], 4),
            "score_max": round(d["score_max"], 4),
            "traces": [
                {
                    "trace_idx": t["trace_idx"],
                    "answer": t["answer"],
                    "correct": t["correct"],
                    "probe_score": round(t["probe_score"], 4),
                }
                for t in d["traces"]
            ],
        })
    with open(output_dir / "per_problem.json", "w") as f:
        json.dump(details, f, indent=2)
    print(f"Saved per-problem details to {output_dir / 'per_problem.json'}")

    # Print final summary table
    print(f"\n{'='*60}")
    print(f"DIFFICULTY CONTROL SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"Probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")
    print(f"Problems: {N} ({len(all_correct)} easy, {len(mixed)} mixed, {len(all_wrong)} hard)")
    if correct_scores and wrong_scores:
        print(f"Correct trace score: {np.mean(correct_scores):.4f} +/- {np.std(correct_scores):.4f}")
        print(f"Wrong trace score:   {np.mean(wrong_scores):.4f} +/- {np.std(wrong_scores):.4f}")
        print(f"t-test: t={t_stat:.4f}, p={p_value:.6f}, d={cohens_d:.4f}")
    print(f"Difficulty~score: r={corr:.4f} (p={corr_p:.6f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
