#!/usr/bin/env python3
"""Compare probe-based error detection against baseline methods.

Baselines:
1. Self-consistency (SC): generate N samples, use agreement as confidence
2. Verbalized confidence: ask model "How confident are you?"
3. Sequence log-probability: use model's own logprob as signal
4. P(True): ask "Is your answer correct? Reply YES or NO"

All baselines produce a scalar "error probability" per trace.
We compare AUROC of each method for predicting actual errors.

This script uses gpt-4o-mini API (no local GPU needed).
"""

import argparse, json, random, re, sys, time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import openai
from src.pipeline.data_loader import load_benchmark
from sklearn.metrics import roc_auc_score


COT_PROMPT = (
    "Solve the following math problem step by step. "
    "Number each step as 'Step 1:', 'Step 2:', etc. "
    "End with 'ANSWER: <your final answer>'.\n\n"
    "Problem: {problem}\n\nSolution:"
)


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


def check_answer(predicted, reference):
    p, r = extract_num(predicted), extract_num(reference)
    if p is not None and r is not None:
        return abs(p - r) < 0.01
    return predicted.strip().lower() == reference.strip().lower()


def api_call(client, messages, temperature=0.0, max_tokens=1024, n=1, logprobs=False):
    """API call with retry."""
    for attempt in range(3):
        try:
            kwargs = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            }
            if logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1
            resp = client.chat.completions.create(**kwargs)
            return resp
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise e


def baseline_cot(client, problem_text):
    """Standard CoT: generate one answer."""
    messages = [{"role": "user", "content": COT_PROMPT.format(problem=problem_text)}]
    resp = api_call(client, messages, temperature=0.0, logprobs=True)
    text = resp.choices[0].message.content or ""
    answer = extract_answer(text)

    # Compute mean logprob as confidence signal
    logprobs_list = []
    if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
        for token_lp in resp.choices[0].logprobs.content:
            logprobs_list.append(token_lp.logprob)

    mean_logprob = np.mean(logprobs_list) if logprobs_list else 0.0
    return text, answer, mean_logprob


def baseline_self_consistency(client, problem_text, n_samples=5):
    """Self-consistency: generate N samples, return disagreement as error signal."""
    messages = [{"role": "user", "content": COT_PROMPT.format(problem=problem_text)}]
    resp = api_call(client, messages, temperature=0.7, max_tokens=1024, n=n_samples)

    answers = []
    for choice in resp.choices:
        text = choice.message.content or ""
        ans = extract_answer(text)
        num = extract_num(ans)
        answers.append(num if num is not None else ans)

    # Compute agreement: fraction of samples that agree with majority
    if all(isinstance(a, float) for a in answers):
        # Group by approximate value
        groups = {}
        for a in answers:
            matched = False
            for g in groups:
                if abs(a - g) < 0.01:
                    groups[g] += 1
                    matched = True
                    break
            if not matched:
                groups[a] = 1
        majority_count = max(groups.values())
        majority_answer = max(groups, key=groups.get)
    else:
        counter = Counter(str(a) for a in answers)
        majority_answer, majority_count = counter.most_common(1)[0]

    agreement = majority_count / n_samples
    # Error signal = 1 - agreement (high disagreement → likely error)
    error_signal = 1.0 - agreement

    # Return majority answer
    if isinstance(majority_answer, float):
        return str(majority_answer), error_signal
    return str(majority_answer), error_signal


def baseline_p_true(client, problem_text, cot_response, answer):
    """P(True): ask model if its answer is correct."""
    messages = [
        {"role": "user", "content": COT_PROMPT.format(problem=problem_text)},
        {"role": "assistant", "content": cot_response},
        {"role": "user", "content": (
            f"Your final answer was: {answer}\n\n"
            "Is this answer correct? Think about it carefully and respond with "
            "only 'YES' or 'NO'."
        )},
    ]
    resp = api_call(client, messages, temperature=0.0, max_tokens=10)
    text = (resp.choices[0].message.content or "").strip().upper()

    # Error signal: 1 if model says NO, 0 if YES
    if "NO" in text:
        return 1.0
    elif "YES" in text:
        return 0.0
    else:
        return 0.5  # uncertain


def baseline_verbalized_confidence(client, problem_text, cot_response, answer):
    """Verbalized confidence: ask model for 0-100 confidence score."""
    messages = [
        {"role": "user", "content": COT_PROMPT.format(problem=problem_text)},
        {"role": "assistant", "content": cot_response},
        {"role": "user", "content": (
            f"Your final answer was: {answer}\n\n"
            "On a scale of 0 to 100, how confident are you that this answer is correct? "
            "Reply with only a number."
        )},
    ]
    resp = api_call(client, messages, temperature=0.0, max_tokens=10)
    text = (resp.choices[0].message.content or "").strip()

    m = re.search(r'\d+', text)
    if m:
        conf = int(m.group())
        # Error signal = 1 - confidence/100
        return 1.0 - conf / 100.0
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sc-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="results/baselines_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI()
    problems = load_benchmark("math500", max_samples=args.n_problems, seed=args.seed)
    print(f"Problems: {len(problems)}")

    results = []
    for i, problem in enumerate(problems):
        try:
            # 1. Standard CoT + logprobs
            cot_text, cot_answer, mean_logprob = baseline_cot(client, problem["problem"])
            correct = check_answer(cot_answer, problem["answer"])

            # 2. Self-consistency
            sc_answer, sc_error_signal = baseline_self_consistency(
                client, problem["problem"], n_samples=args.sc_samples
            )

            # 3. P(True)
            p_true_signal = baseline_p_true(client, problem["problem"], cot_text, cot_answer)

            # 4. Verbalized confidence
            verb_conf_signal = baseline_verbalized_confidence(
                client, problem["problem"], cot_text, cot_answer
            )

            results.append({
                "id": problem["id"],
                "correct": correct,
                "cot_answer": cot_answer,
                "reference": problem["answer"],
                # Error signals (higher = more likely error)
                "logprob_signal": -mean_logprob,  # negate: lower logprob → higher error
                "sc_signal": sc_error_signal,
                "p_true_signal": p_true_signal,
                "verb_conf_signal": verb_conf_signal,
            })

            if (i + 1) % 10 == 0:
                acc = sum(1 for r in results if r["correct"]) / len(results)
                print(f"  [{i+1}/{len(problems)}] acc={acc:.3f}", flush=True)

        except Exception as e:
            print(f"  Error {problem['id']}: {e}", flush=True)

    # Compute AUROCs
    N = len(results)
    acc = sum(1 for r in results if r["correct"]) / N
    labels = [0 if r["correct"] else 1 for r in results]  # 1 = error

    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON (n={N}, acc={acc:.3f})")
    print(f"{'='*60}")

    if len(set(labels)) < 2:
        print("Only one class — can't compute AUROC")
        return

    methods = {
        "Logprob": [r["logprob_signal"] for r in results],
        "Self-consistency": [r["sc_signal"] for r in results],
        "P(True)": [r["p_true_signal"] for r in results],
        "Verbalized conf": [r["verb_conf_signal"] for r in results],
    }

    print(f"\n{'Method':>20} {'AUROC':>8} {'Precision@20%':>15}")
    print("-" * 48)

    summary = {}
    for name, scores in methods.items():
        auroc = roc_auc_score(labels, scores)

        # Precision@20%: of top 20% most suspicious, what fraction are actual errors
        k = max(1, int(N * 0.2))
        sorted_idx = np.argsort(scores)[::-1]  # highest error signal first
        top_k_labels = [labels[i] for i in sorted_idx[:k]]
        precision_20 = sum(top_k_labels) / k

        print(f"{name:>20} {auroc:>8.4f} {precision_20:>15.3f}")
        summary[name] = {"auroc": auroc, "precision_20": precision_20}

    # Save
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump({
            "n": N, "accuracy": acc,
            "methods": summary,
            "per_problem": results,
        }, f, indent=2)

    print(f"\nNote: Probe AUROC (from prior experiments) = 0.953 (CV)")
    print(f"Compare these baselines against the probe to justify the approach.")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
