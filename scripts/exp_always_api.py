#!/usr/bin/env python3
"""Experiment: Always-API upper bound.

What accuracy does gpt-4o-mini achieve when solving ALL problems from scratch?
This is the compute ceiling — the best we could do if we had unlimited API budget.

Comparison:
- Baseline (3B): ~53%
- Probe-guided refinement: ~61% (only 60% of problems sent to API)
- Always API: ??? (100% of problems sent to API)

If always-API = 70%, then probe-guided gets 87% of the API benefit at 60% cost.
"""

import json, re, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import openai

output_dir = Path("results/always_api")
output_dir.mkdir(parents=True, exist_ok=True)

client = openai.OpenAI()

# Load same problems as backtrack v2
traces = json.load(open("results/online_backtrack_v2/traces.json"))
print(f"Loaded {len(traces)} problems")


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


results = []
for i, t in enumerate(traces):
    prompt = (
        f"Solve this math problem step by step. "
        f"End with 'ANSWER: <your final answer>'.\n\n"
        f"Problem: {t['problem']}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=1024,
            )
            api_response = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                api_response = ""

    api_answer = extract_answer(api_response)
    api_correct = check_answer(api_answer, t["reference"])

    results.append({
        "id": t["id"],
        "correct_3b": t["correct"],
        "correct_api": api_correct,
        "api_answer": api_answer,
        "reference": t["reference"],
    })

    if (i + 1) % 20 == 0:
        acc_3b = sum(1 for r in results if r["correct_3b"]) / len(results)
        acc_api = sum(1 for r in results if r["correct_api"]) / len(results)
        print(f"  [{i+1}/{len(traces)}] 3B={acc_3b:.3f} API={acc_api:.3f}", flush=True)

# Summary
N = len(results)
acc_3b = sum(1 for r in results if r["correct_3b"]) / N
acc_api = sum(1 for r in results if r["correct_api"]) / N

# Problems where API fixes 3B's errors
fixed_by_api = sum(1 for r in results if not r["correct_3b"] and r["correct_api"])
broke_by_api = sum(1 for r in results if r["correct_3b"] and not r["correct_api"])

print(f"\n{'='*50}")
print(f"ALWAYS-API UPPER BOUND")
print(f"{'='*50}")
print(f"Baseline (Qwen-3B): {acc_3b:.4f}")
print(f"Always API (gpt-4o-mini): {acc_api:.4f}")
print(f"Delta: {acc_api - acc_3b:+.4f}")
print(f"API fixes 3B errors: {fixed_by_api}/{N}")
print(f"API breaks 3B correct: {broke_by_api}/{N}")

# What fraction of API's improvement does probe-guided get?
bt_acc = 0.61  # from backtrack v2 results
if acc_api > acc_3b:
    efficiency = (bt_acc - acc_3b) / (acc_api - acc_3b)
    print(f"\nProbe-guided gets {efficiency:.0%} of API improvement at 60% API cost")

summary = {
    "n": N,
    "acc_3b": round(acc_3b, 4),
    "acc_api": round(acc_api, 4),
    "delta": round(acc_api - acc_3b, 4),
    "fixed_by_api": fixed_by_api,
    "broke_by_api": broke_by_api,
}

with open(output_dir / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(output_dir / "per_problem.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {output_dir}")
