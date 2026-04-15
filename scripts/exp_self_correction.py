#!/usr/bin/env python3
"""Experiment: Probe-Guided Self-Correction — use probe to trigger retry.

KEY IDEA: The probe detects when a reasoning trace is likely wrong. Instead of
accepting the first answer, we:
1. Generate a trace (greedy)
2. Score it with the probe
3. If probe_score > threshold (likely error), regenerate with a "verify and correct"
   prompt that includes the original attempt
4. Score the new trace and keep the better one

This tests whether the probe can be used as a SELF-CORRECTION trigger.
The model corrects itself — no external API needed.

Comparison baselines:
- Greedy: single greedy attempt (no correction)
- Always-retry: always regenerate (measures retry benefit without probe)
- Probe-triggered: only retry when probe flags high error risk
- Oracle-triggered: only retry when actually wrong (upper bound)

Key metric: accuracy gain per retry (efficiency). Probe-triggered should achieve
most of the oracle benefit with far fewer retries.
"""

import argparse, json, pickle, re, sys, time
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


def generate_correction(collector, problem_text, original_response, original_answer):
    """Generate a self-correction attempt given the original response."""
    prompt = (
        "You previously attempted this math problem but may have made an error. "
        "Please carefully verify each step of your previous solution, identify any "
        "mistakes, and provide a corrected solution.\n\n"
        f"Problem: {problem_text}\n\n"
        f"Your previous attempt:\n{original_response}\n\n"
        f"Your previous answer: {original_answer}\n\n"
        "Please re-solve this problem step by step, checking each step carefully. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        "End with 'ANSWER: <your final answer>'."
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = collector.tokenizer(input_text, return_tensors="pt").to(collector.device)

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=collector.tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = collector.model.generate(**inputs, generation_config=gen_config)

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return collector.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--output-dir", default="results/self_correction")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe
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

    # Load eval problems (same split as other experiments)
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Eval: {len(eval_problems)} problems")

    # Phase 1: Generate initial traces + probe scores
    print(f"\n=== Phase 1: Initial generation + probe scoring ===")
    problem_data = []

    for i, problem in enumerate(eval_problems):
        try:
            # Initial greedy trace
            resp = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(resp)
            correct = check_answer(ans, problem["answer"])
            score = get_probe_score(
                collector, clf, scaler, best_layer,
                problem["problem"], resp
            )

            problem_data.append({
                "id": problem["id"],
                "problem": problem["problem"],
                "reference": problem["answer"],
                "initial_response": resp,
                "initial_answer": ans,
                "initial_correct": correct,
                "probe_score": score,
                "retry_response": None,
                "retry_answer": None,
                "retry_correct": None,
                "retry_score": None,
            })
        except Exception as e:
            print(f"  Error {problem['id']}: {e}")
            problem_data.append({
                "id": problem["id"],
                "problem": problem["problem"],
                "reference": problem["answer"],
                "initial_response": "",
                "initial_answer": "",
                "initial_correct": False,
                "probe_score": 1.0,
                "retry_response": None,
                "retry_answer": None,
                "retry_correct": None,
                "retry_score": None,
            })

        if (i + 1) % 10 == 0:
            acc = sum(1 for d in problem_data if d["initial_correct"]) / len(problem_data)
            avg_score = np.mean([d["probe_score"] for d in problem_data])
            print(f"  [{i+1}/{len(eval_problems)}] acc={acc:.3f} avg_probe={avg_score:.3f}",
                  flush=True)

    initial_acc = sum(1 for d in problem_data if d["initial_correct"]) / len(problem_data)
    print(f"Initial accuracy: {initial_acc:.4f}")

    # Phase 2: Generate correction for ALL problems (we'll selectively use them)
    print(f"\n=== Phase 2: Generating corrections for all problems ===")

    for i, d in enumerate(problem_data):
        try:
            retry_resp = generate_correction(
                collector, d["problem"], d["initial_response"], d["initial_answer"]
            )
            retry_ans = extract_answer(retry_resp)
            retry_correct = check_answer(retry_ans, d["reference"])
            retry_score = get_probe_score(
                collector, clf, scaler, best_layer,
                d["problem"], retry_resp
            )

            d["retry_response"] = retry_resp
            d["retry_answer"] = retry_ans
            d["retry_correct"] = retry_correct
            d["retry_score"] = retry_score
        except Exception as e:
            print(f"  Retry error {d['id']}: {e}")
            d["retry_response"] = ""
            d["retry_answer"] = ""
            d["retry_correct"] = False
            d["retry_score"] = 1.0

        if (i + 1) % 10 == 0:
            retry_acc = sum(1 for d2 in problem_data[:i+1]
                          if d2["retry_correct"] is not None and d2["retry_correct"]) / (i + 1)
            print(f"  [{i+1}/{len(problem_data)}] retry_acc={retry_acc:.3f}", flush=True)

    # Phase 3: Evaluate strategies at each threshold
    print(f"\n=== Results ===")
    N = len(problem_data)

    # Always-retry baseline
    always_retry_correct = sum(1 for d in problem_data if d["retry_correct"])
    always_retry_acc = always_retry_correct / N

    # Best-of-two: keep whichever (initial or retry) has lower probe score
    best_of_two_correct = 0
    for d in problem_data:
        if d["retry_score"] is not None and d["retry_score"] < d["probe_score"]:
            if d["retry_correct"]:
                best_of_two_correct += 1
        else:
            if d["initial_correct"]:
                best_of_two_correct += 1
    best_of_two_acc = best_of_two_correct / N

    # Oracle-triggered: only use retry when initial is actually wrong
    oracle_correct = 0
    for d in problem_data:
        if d["initial_correct"]:
            oracle_correct += 1
        elif d["retry_correct"]:
            oracle_correct += 1
    oracle_acc = oracle_correct / N

    print(f"{'Strategy':<25} {'Acc':>8} {'Retries':>8} {'Δ':>8}")
    print("-" * 55)
    print(f"{'Greedy (no retry)':<25} {initial_acc:>8.4f} {'0':>8} {'+0.0000':>8}")
    print(f"{'Always retry':<25} {always_retry_acc:>8.4f} {N:>8} "
          f"{always_retry_acc - initial_acc:>+8.4f}")
    print(f"{'Best-of-two (probe)':<25} {best_of_two_acc:>8.4f} {N:>8} "
          f"{best_of_two_acc - initial_acc:>+8.4f}")

    results = {
        "greedy": {"acc": round(initial_acc, 4), "retries": 0},
        "always_retry": {"acc": round(always_retry_acc, 4), "retries": N},
        "best_of_two": {"acc": round(best_of_two_acc, 4), "retries": N},
    }

    # Threshold-based probe-triggered retry
    for threshold in args.thresholds:
        correct = 0
        retries = 0
        for d in problem_data:
            if d["probe_score"] > threshold:
                # Probe flags this as likely wrong — use retry
                retries += 1
                # Pick whichever has lower probe score
                if d["retry_score"] is not None and d["retry_score"] < d["probe_score"]:
                    if d["retry_correct"]:
                        correct += 1
                else:
                    if d["initial_correct"]:
                        correct += 1
            else:
                # Probe says this is fine — keep initial
                if d["initial_correct"]:
                    correct += 1

        acc = correct / N
        results[f"probe_t{threshold}"] = {
            "acc": round(acc, 4),
            "retries": retries,
            "threshold": threshold,
            "retry_rate": round(retries / N, 4),
        }
        print(f"{'Probe (t=' + str(threshold) + ')':<25} {acc:>8.4f} {retries:>8} "
              f"{acc - initial_acc:>+8.4f}")

    # Oracle triggered
    oracle_retries = sum(1 for d in problem_data if not d["initial_correct"])
    print(f"{'Oracle-triggered':<25} {oracle_acc:>8.4f} {oracle_retries:>8} "
          f"{oracle_acc - initial_acc:>+8.4f}")
    results["oracle"] = {
        "acc": round(oracle_acc, 4),
        "retries": oracle_retries,
    }

    # Efficiency analysis
    print(f"\n--- Efficiency Analysis ---")
    print(f"{'Strategy':<25} {'Acc Gain':>10} {'Retries':>8} {'Gain/Retry':>12}")
    print("-" * 60)
    for key, r in results.items():
        if r["retries"] > 0:
            gain = r["acc"] - initial_acc
            efficiency = gain / r["retries"] * N  # normalized gain per retry
            print(f"{key:<25} {gain:>+10.4f} {r['retries']:>8} {efficiency:>+12.4f}")

    # Save
    summary = {
        "model": args.model,
        "n_problems": N,
        "initial_acc": round(initial_acc, 4),
        "results": results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-problem data (without full text)
    traces_out = []
    for d in problem_data:
        traces_out.append({
            "id": d["id"],
            "reference": d["reference"],
            "initial_answer": d["initial_answer"],
            "initial_correct": d["initial_correct"],
            "probe_score": d["probe_score"],
            "retry_answer": d["retry_answer"],
            "retry_correct": d["retry_correct"],
            "retry_score": d["retry_score"],
        })
    with open(output_dir / "traces.json", "w") as f:
        json.dump(traces_out, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
