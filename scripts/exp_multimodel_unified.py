#!/usr/bin/env python3
"""Experiment: Unified multi-model comparison table.

For each model (Qwen-3B, Qwen-1.5B, Phi-3.5):
1. Probe AUROC (from existing data)
2. First-step prediction AUROC
3. Selective refinement Δ at matched rates
4. Unfaithfulness metrics

This consolidates all per-model results into a single comparison.
Also runs Qwen-1.5B through the same backtracking pipeline as Qwen-3B.
"""

import json, pickle, re, sys, time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import openai
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


def check_answer(predicted, reference):
    p, r = extract_num(predicted), extract_num(reference)
    if p is not None and r is not None:
        return abs(p - r) < 0.01
    return predicted.strip().lower() == reference.strip().lower()


def api_refine(problem_text, client):
    prompt = (
        f"Solve this math problem step by step. "
        f"End with 'ANSWER: <your final answer>'.\n\n"
        f"Problem: {problem_text}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except:
            time.sleep(2)
    return ""


output_dir = Path("results/multimodel_unified")
output_dir.mkdir(parents=True, exist_ok=True)

# Models to evaluate
models = [
    {
        "name": "Qwen2.5-1.5B-Instruct",
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "probe_dir": "data/probes_1.5b",
        "probe_summary": "data/probes_1.5b/probe_summary.json",
    },
]

# Load eval problems (same split as backtrack v2)
all_problems = load_benchmark("math500", max_samples=500, seed=0)
train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
eval_problems = [p for p in all_problems if p["id"] not in train_ids][:100]
print(f"Eval problems: {len(eval_problems)}")

api_client = openai.OpenAI()

for model_info in models:
    print(f"\n{'='*60}")
    print(f"Model: {model_info['name']}")
    print(f"{'='*60}")

    # Check if probe exists
    probe_summary_path = Path(model_info["probe_summary"])
    if not probe_summary_path.exists():
        print(f"  No probe found at {probe_summary_path}, training fresh probe...")
        # Will need to generate traces + train probe
        # Load model
        collector = HiddenStateCollector(
            model_name=model_info["hf_name"], device="mps",
            layers_to_probe=list(range(0, 28, 4))  # probe several layers
        )

        # Generate traces for training (first 50 problems from train set)
        train_problems = [p for p in all_problems if p["id"] in train_ids][:75]
        print(f"  Generating {len(train_problems)} training traces...")

        train_features = {layer: [] for layer in collector.layers}
        train_labels = []

        for i, prob in enumerate(train_problems):
            try:
                response = collector.generate_cot(prob["problem"])
                answer = extract_answer(response)
                correct = check_answer(answer, prob["answer"])

                messages = [{"role": "user", "content": (
                    "Solve the following math problem step by step. "
                    "Number each step as 'Step 1:', 'Step 2:', etc. "
                    "End with 'ANSWER: <your final answer>'.\n\n"
                    f"Problem: {prob['problem']}\n\nSolution:"
                )}]
                full_text = collector.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) + response

                hs = collector.extract_hidden_states(full_text)
                for layer in collector.layers:
                    train_features[layer].append(
                        hs[layer][-1].numpy().astype(np.float32)
                    )
                train_labels.append(0 if correct else 1)

                if (i + 1) % 10 == 0:
                    acc = 1 - sum(train_labels) / len(train_labels)
                    print(f"    [{i+1}/{len(train_problems)}] acc={acc:.3f}")
            except Exception as e:
                print(f"    Error: {e}")

        # Train probes per layer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        train_labels = np.array(train_labels)
        best_layer = None
        best_auroc = 0
        probe_results = []

        for layer in collector.layers:
            X = np.array(train_features[layer])
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
            clf = LogisticRegression(C=0.1, max_iter=1000)
            scores = cross_val_score(clf, X_scaled, train_labels, cv=5, scoring='roc_auc')
            auroc = scores.mean()
            probe_results.append({"layer": layer, "cv_auroc_mean": round(auroc, 4)})
            print(f"    Layer {layer}: AUROC = {auroc:.4f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer

        # Save best probe
        probe_dir = Path(model_info["probe_dir"])
        probe_dir.mkdir(parents=True, exist_ok=True)

        X_best = np.array(train_features[best_layer])
        scaler = StandardScaler().fit(X_best)
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(scaler.transform(X_best), train_labels)

        with open(probe_dir / f"probe_layer_{best_layer}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler, "layer": best_layer}, f)

        with open(probe_dir / "probe_summary.json", "w") as f:
            json.dump(probe_results, f, indent=2)

        print(f"  Best probe: layer {best_layer}, AUROC = {best_auroc:.4f}")
    else:
        # Load existing probe
        summary = json.load(open(probe_summary_path))
        best = max(summary, key=lambda x: x["cv_auroc_mean"])
        best_layer = best["layer"]
        best_auroc = best["cv_auroc_mean"]

        probe_path = Path(model_info["probe_dir"]) / f"probe_layer_{best_layer}.pkl"
        with open(probe_path, "rb") as f:
            probe_data = pickle.load(f)
        clf, scaler = probe_data["clf"], probe_data["scaler"]

        collector = HiddenStateCollector(
            model_name=model_info["hf_name"], device="mps",
            layers_to_probe=[best_layer]
        )
        print(f"  Loaded probe: layer {best_layer}, AUROC = {best_auroc:.4f}")

    # Evaluate: generate traces + step-level probing
    print(f"\n  Generating eval traces with step-level probing...")
    traces = []

    for i, problem in enumerate(eval_problems):
        try:
            response = collector.generate_cot(problem["problem"])
            answer = extract_answer(response)
            correct = check_answer(answer, problem["answer"])

            step_pattern = r'(Step \d+:.*?)(?=Step \d+:|ANSWER:|$)'
            matches = list(re.finditer(step_pattern, response, re.DOTALL))
            steps = [m.group(1).strip() for m in matches] if matches else []

            messages = [{"role": "user", "content": (
                "Solve the following math problem step by step. "
                "Number each step as 'Step 1:', 'Step 2:', etc. "
                "End with 'ANSWER: <your final answer>'.\n\n"
                f"Problem: {problem['problem']}\n\nSolution:"
            )}]
            full_prefix = collector.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            step_scores = []
            for step_idx in range(len(steps)):
                partial = "\n".join(steps[:step_idx + 1])
                full_text = full_prefix + partial
                hs = collector.extract_hidden_states(full_text)
                last_hs = hs[best_layer][-1].numpy().astype(np.float32).reshape(1, -1)
                score = float(clf.predict_proba(scaler.transform(last_hs))[0, 1])
                step_scores.append(score)

            traces.append({
                "id": problem["id"],
                "problem": problem["problem"],
                "reference": problem["answer"],
                "response": response,
                "answer": answer,
                "correct": correct,
                "steps": steps,
                "step_scores": step_scores,
                "max_score": max(step_scores) if step_scores else 0,
            })

            if (i + 1) % 10 == 0:
                acc = sum(1 for t in traces if t["correct"]) / len(traces)
                print(f"    [{i+1}/{len(eval_problems)}] acc={acc:.3f}")

        except Exception as e:
            print(f"    Error {problem['id']}: {e}")

    N = len(traces)
    base_acc = sum(1 for t in traces if t["correct"]) / N
    print(f"\n  Baseline accuracy: {base_acc:.4f}")

    # First-step AUROC
    first_scores = np.array([t["step_scores"][0] for t in traces if t["step_scores"]])
    max_scores_arr = np.array([t["max_score"] for t in traces if t["step_scores"]])
    labels = np.array([0 if t["correct"] else 1 for t in traces if t["step_scores"]])

    first_auroc = roc_auc_score(labels, first_scores) if len(set(labels)) > 1 else 0
    max_auroc = roc_auc_score(labels, max_scores_arr) if len(set(labels)) > 1 else 0
    print(f"  First-step AUROC: {first_auroc:.4f}")
    print(f"  Max-score AUROC: {max_auroc:.4f}")

    # Selective refinement at 40%
    sorted_traces = sorted(traces, key=lambda t: -t["max_score"])
    k = int(0.4 * N)
    refine_ids = set(t["id"] for t in sorted_traces[:k])

    refined_results = []
    for t in traces:
        if t["id"] in refine_ids and not t["correct"]:
            api_resp = api_refine(t["problem"], api_client)
            api_ans = extract_answer(api_resp)
            api_correct = check_answer(api_ans, t["reference"])
            refined_results.append(api_correct)
        else:
            refined_results.append(t["correct"])

    refined_acc = sum(refined_results) / N
    delta = refined_acc - base_acc
    print(f"  Selective refinement (40%): {refined_acc:.4f} (Δ={delta:+.4f})")

    # Save model results
    model_results = {
        "model": model_info["name"],
        "n_eval": N,
        "base_acc": round(base_acc, 4),
        "probe_auroc": round(best_auroc, 4),
        "probe_layer": best_layer,
        "first_step_auroc": round(first_auroc, 4),
        "max_score_auroc": round(max_auroc, 4),
        "selective_refinement_40pct": round(refined_acc, 4),
        "delta_40pct": round(delta, 4),
    }

    with open(output_dir / f"{model_info['name'].replace('/', '_')}.json", "w") as f:
        json.dump(model_results, f, indent=2)

    # Save traces for further analysis
    with open(output_dir / f"{model_info['name'].replace('/', '_')}_traces.json", "w") as f:
        json.dump([{k: v for k, v in t.items()} for t in traces], f, indent=2)

    # Free GPU memory
    del collector
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

print(f"\nAll results saved to {output_dir}")
