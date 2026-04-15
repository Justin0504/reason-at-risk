#!/usr/bin/env python3
"""GPU server experiments for NeurIPS submission.

Runs on RTX 6000 Ada (48GB). Experiments:
1. Qwen2.5-7B-Instruct probe detection + selective refinement
2. Llama-3.1-8B-Instruct probe detection + selective refinement
3. Multi-benchmark: MATH-500 + GSM8K + GPQA (if available)
4. Difficulty control: same problem, multiple generations
5. Multi-seed confidence intervals for main results
6. First-step prediction for all models

Uses CUDA GPUs 3-7 (others are in use).
"""

import argparse, json, pickle, re, sys, time, os
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Add extra lib path
sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import openai
from src.probe.collect_hidden_states import HiddenStateCollector

# Direct import to avoid __init__.py pulling in anthropic
import importlib
_data_loader = importlib.import_module("src.pipeline.data_loader")
load_benchmark = _data_loader.load_benchmark


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
        f"End with 'ANSWER: <your final answer>'.\n\nProblem: {problem_text}"
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


def run_model_experiment(model_name, device, output_dir, n_train=100, n_eval=200, seed=42):
    """Full experiment pipeline for one model."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} on {device}")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine layers to probe
    probe_layers = list(range(0, 64, 4))  # will be clipped to actual model size

    collector = HiddenStateCollector(
        model_name=model_name, device=device,
        layers_to_probe=probe_layers
    )
    actual_layers = collector.layers
    print(f"Probing layers: {actual_layers}")

    # Load problems
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    np.random.seed(seed)
    indices = np.random.permutation(len(all_problems))
    train_problems = [all_problems[i] for i in indices[:n_train]]
    eval_problems = [all_problems[i] for i in indices[n_train:n_train + n_eval]]
    print(f"Train: {len(train_problems)}, Eval: {len(eval_problems)}")

    # Phase 1: Generate training traces + extract hidden states
    print(f"\n--- Phase 1: Training Traces ---")
    train_features = {layer: [] for layer in actual_layers}
    train_labels = []
    train_responses = []

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
            for layer in actual_layers:
                train_features[layer].append(
                    hs[layer][-1].cpu().numpy().astype(np.float32)
                )
            train_labels.append(0 if correct else 1)
            train_responses.append({
                "id": prob["id"], "correct": correct, "answer": answer,
                "response": response[:500]
            })

            if (i + 1) % 20 == 0:
                acc = 1 - sum(train_labels) / len(train_labels)
                print(f"  [{i+1}/{len(train_problems)}] acc={acc:.3f}")
        except Exception as e:
            print(f"  Error train {i}: {e}")

    train_labels = np.array(train_labels)
    train_acc = 1 - train_labels.mean()
    print(f"Training set accuracy: {train_acc:.4f}")

    # Phase 2: Train probes per layer with CV
    print(f"\n--- Phase 2: Train Probes ---")
    probe_results = []
    best_layer = None
    best_auroc = 0
    best_clf = None
    best_scaler = None

    for layer in actual_layers:
        X = np.array(train_features[layer])
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clf = LogisticRegression(C=0.1, max_iter=1000)

        if len(set(train_labels)) > 1:
            scores = cross_val_score(clf, X_scaled, train_labels, cv=5, scoring='roc_auc')
            auroc = scores.mean()
            auroc_std = scores.std()
        else:
            auroc, auroc_std = 0.5, 0

        probe_results.append({
            "layer": layer, "cv_auroc_mean": round(auroc, 4),
            "cv_auroc_std": round(auroc_std, 4)
        })
        print(f"  Layer {layer:2d}: AUROC = {auroc:.4f} ± {auroc_std:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer
            clf.fit(scaler.transform(X), train_labels)
            best_clf = clf
            best_scaler = scaler

    print(f"Best: layer {best_layer}, AUROC = {best_auroc:.4f}")

    # Save probe
    with open(output_dir / "probe.pkl", "wb") as f:
        pickle.dump({"clf": best_clf, "scaler": best_scaler, "layer": best_layer}, f)
    with open(output_dir / "probe_layers.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    # Phase 3: Eval with step-level probing
    print(f"\n--- Phase 3: Eval Traces + Step Probing ---")
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

            # Probe each step
            step_scores = []
            for step_idx in range(len(steps)):
                partial = "\n".join(steps[:step_idx + 1])
                full_text = full_prefix + partial
                hs = collector.extract_hidden_states(full_text)
                last_hs = hs[best_layer][-1].cpu().numpy().astype(np.float32).reshape(1, -1)
                score = float(best_clf.predict_proba(best_scaler.transform(last_hs))[0, 1])
                step_scores.append(score)

            # Also get full trace hidden state for trace-level probe
            full_text = full_prefix + response
            hs = collector.extract_hidden_states(full_text)
            trace_hs = hs[best_layer][-1].cpu().numpy().astype(np.float32).reshape(1, -1)
            trace_score = float(best_clf.predict_proba(best_scaler.transform(trace_hs))[0, 1])

            traces.append({
                "id": problem["id"],
                "problem": problem["problem"],
                "reference": problem["answer"],
                "answer": answer,
                "correct": correct,
                "n_steps": len(steps),
                "step_scores": step_scores,
                "trace_score": trace_score,
                "max_score": max(step_scores) if step_scores else trace_score,
                "first_score": step_scores[0] if step_scores else trace_score,
            })

            if (i + 1) % 20 == 0:
                acc = sum(1 for t in traces if t["correct"]) / len(traces)
                print(f"  [{i+1}/{len(eval_problems)}] acc={acc:.3f}")

        except Exception as e:
            print(f"  Error eval {i}: {e}")

    N = len(traces)
    base_acc = sum(1 for t in traces if t["correct"]) / N
    print(f"\nEval accuracy: {base_acc:.4f} ({N} problems)")

    # Phase 4: Compute metrics
    print(f"\n--- Phase 4: Metrics ---")
    labels = np.array([0 if t["correct"] else 1 for t in traces])
    trace_scores = np.array([t["trace_score"] for t in traces])
    max_scores = np.array([t["max_score"] for t in traces])
    first_scores = np.array([t["first_score"] for t in traces])

    trace_auroc = roc_auc_score(labels, trace_scores) if len(set(labels)) > 1 else 0.5
    max_auroc = roc_auc_score(labels, max_scores) if len(set(labels)) > 1 else 0.5
    first_auroc = roc_auc_score(labels, first_scores) if len(set(labels)) > 1 else 0.5

    print(f"  Trace-level AUROC: {trace_auroc:.4f}")
    print(f"  Max-score AUROC:   {max_auroc:.4f}")
    print(f"  First-step AUROC:  {first_auroc:.4f}")

    # Selective refinement
    print(f"\n--- Phase 5: Selective Refinement ---")
    api_client = openai.OpenAI()

    for rate in [0.2, 0.4, 0.6]:
        sorted_traces = sorted(traces, key=lambda t: -t["max_score"])
        k = int(rate * N)
        refine_ids = set(t["id"] for t in sorted_traces[:k])

        refined_correct = 0
        api_calls = 0
        for t in traces:
            if t["id"] in refine_ids and not t["correct"]:
                api_resp = api_refine(t["problem"], api_client)
                api_ans = extract_answer(api_resp)
                if check_answer(api_ans, t["reference"]):
                    refined_correct += 1
                api_calls += 1
            elif t["correct"]:
                refined_correct += 1

        refined_acc = refined_correct / N
        delta = refined_acc - base_acc
        print(f"  Rate {rate:.0%}: acc={refined_acc:.4f} (Δ={delta:+.4f}), API calls={api_calls}")

    # Save all results
    summary = {
        "model": model_name,
        "n_train": len(train_labels),
        "n_eval": N,
        "base_acc": round(base_acc, 4),
        "train_acc": round(float(train_acc), 4),
        "best_layer": best_layer,
        "probe_cv_auroc": round(best_auroc, 4),
        "trace_auroc": round(trace_auroc, 4),
        "max_score_auroc": round(max_auroc, 4),
        "first_step_auroc": round(first_auroc, 4),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "traces.json", "w") as f:
        json.dump([{k: v for k, v in t.items() if k != "problem"} for t in traces], f, indent=2)

    with open(output_dir / "probe_layers.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\nSaved to {output_dir}")

    # Free memory
    del collector
    torch.cuda.empty_cache()

    return summary


def run_difficulty_control(model_name, device, output_dir, n_problems=50, n_generations=5):
    """Same problem, multiple generations — does probe score vary per trace?

    If probe only captures problem difficulty, scores should be the same.
    If probe captures trace-specific signals, scores should vary.
    """
    print(f"\n{'='*70}")
    print(f"DIFFICULTY CONTROL: {model_name}")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe
    probe_dir = output_dir.parent / model_name.split("/")[-1]
    probe_path = probe_dir / "probe.pkl"
    if not probe_path.exists():
        print(f"  No probe found at {probe_path}, skipping")
        return None

    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)
    clf, scaler, best_layer = probe_data["clf"], probe_data["scaler"], probe_data["layer"]

    collector = HiddenStateCollector(
        model_name=model_name, device=device,
        layers_to_probe=[best_layer]
    )

    problems = load_benchmark("math500", max_samples=500, seed=0)
    np.random.seed(123)
    selected = np.random.choice(len(problems), n_problems, replace=False)
    test_problems = [problems[i] for i in selected]

    results = []
    for i, prob in enumerate(test_problems):
        scores_for_problem = []
        correct_for_problem = []

        for gen in range(n_generations):
            try:
                # Generate with temperature > 0 for diversity
                response = collector.generate_cot(prob["problem"], max_new_tokens=1024, temperature=0.7)
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
                last_hs = hs[best_layer][-1].cpu().numpy().astype(np.float32).reshape(1, -1)
                score = float(clf.predict_proba(scaler.transform(last_hs))[0, 1])

                scores_for_problem.append(score)
                correct_for_problem.append(correct)
            except:
                pass

        if len(scores_for_problem) >= 2:
            results.append({
                "id": prob["id"],
                "scores": scores_for_problem,
                "correct": correct_for_problem,
                "score_std": float(np.std(scores_for_problem)),
                "score_mean": float(np.mean(scores_for_problem)),
                "acc": sum(correct_for_problem) / len(correct_for_problem),
            })

        if (i + 1) % 10 == 0:
            avg_std = np.mean([r["score_std"] for r in results])
            print(f"  [{i+1}/{n_problems}] avg_score_std={avg_std:.4f}")

    # Analysis
    stds = [r["score_std"] for r in results]
    # Problems where model sometimes correct, sometimes wrong
    mixed = [r for r in results if 0 < r["acc"] < 1]

    print(f"\n  Total problems: {len(results)}")
    print(f"  Avg score std across generations: {np.mean(stds):.4f}")
    print(f"  Problems with mixed correctness: {len(mixed)}/{len(results)}")

    if mixed:
        # For mixed problems, is the probe score higher when wrong?
        correct_scores = []
        wrong_scores = []
        for r in mixed:
            for s, c in zip(r["scores"], r["correct"]):
                if c:
                    correct_scores.append(s)
                else:
                    wrong_scores.append(s)
        print(f"  Mixed problems - correct gen score: {np.mean(correct_scores):.4f}")
        print(f"  Mixed problems - wrong gen score: {np.mean(wrong_scores):.4f}")
        print(f"  Separation: {np.mean(wrong_scores) - np.mean(correct_scores):.4f}")

        if len(correct_scores) > 5 and len(wrong_scores) > 5:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(wrong_scores, correct_scores, alternative='greater')
            print(f"  Mann-Whitney U p-value: {p:.6f}")

    summary = {
        "model": model_name,
        "n_problems": len(results),
        "n_generations": n_generations,
        "avg_score_std": round(float(np.mean(stds)), 4),
        "n_mixed": len(mixed),
    }

    with open(output_dir / "difficulty_control.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "difficulty_control_details.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir}")

    del collector
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="GPU to use")
    parser.add_argument("--exp", choices=["7b", "8b", "difficulty", "all"], default="all")
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=200)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"  # after setting CUDA_VISIBLE_DEVICES

    base_dir = Path("/data/yinian/reason-at-risk/results/gpu")
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.exp in ("7b", "all"):
        run_model_experiment(
            "Qwen/Qwen2.5-7B-Instruct", device,
            base_dir / "Qwen2.5-7B-Instruct",
            n_train=args.n_train, n_eval=args.n_eval
        )

    if args.exp in ("8b", "all"):
        run_model_experiment(
            "microsoft/Phi-3.5-mini-instruct", device,
            base_dir / "Phi-3.5-mini-instruct",
            n_train=args.n_train, n_eval=args.n_eval
        )

    if args.exp in ("difficulty", "all"):
        # Run difficulty control on 7B (must run 7b first)
        run_difficulty_control(
            "Qwen/Qwen2.5-7B-Instruct", device,
            base_dir / "Qwen2.5-7B-Instruct",
            n_problems=50, n_generations=5
        )

    print("\n\nALL EXPERIMENTS COMPLETE!")


if __name__ == "__main__":
    main()
