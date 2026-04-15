#!/usr/bin/env python3
"""Full experiment pipeline for Qwen2.5-14B-Instruct.

Stages:
1. Train hidden-state linear probe for error detection
2. Evaluate probe AUROC (per-layer + best)
3. Selective refinement: probe-guided re-solve of flagged traces
4. First-step prediction analysis
5. Difficulty control: per-problem generation variance

Run:
    PYTHONPATH="/data/yinian/pylibs" python3 scripts/exp_14b_full.py \
        --device cuda:0 --output-dir results/14b_full
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
from sklearn.metrics import roc_auc_score

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.probe.collect_hidden_states import HiddenStateCollector
from src.pipeline.data_loader import load_benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
N_LAYERS = 40
HIDDEN_DIM = 5120
CANDIDATE_LAYERS = sorted(set(
    [l * 4 for l in range(10) if l * 4 < N_LAYERS] + [N_LAYERS - 1]
))  # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]

TRAIN_SIZE = 80
EVAL_SIZE = 100
DIFFICULTY_SIZE = 50
DIFFICULTY_GENS = 5


# ---------------------------------------------------------------------------
# Answer helpers (same as exp_probe_guided_decoding.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def get_last_hidden(collector, layer, problem_text, response):
    """Extract last-token hidden state for a (problem, response) pair."""
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
    return hs[layer][-1].cpu().numpy().astype(np.float32)


def get_probe_score(collector, clf, scaler, best_layer, problem_text, response):
    """Get probe P(error) score for a (problem, response) pair."""
    vec = get_last_hidden(collector, best_layer, problem_text, response).reshape(1, -1)
    return float(clf.predict_proba(scaler.transform(vec))[0, 1])


def get_step_hidden_states(collector, layer, problem_text, response):
    """Extract hidden states at each step boundary.

    Returns list of numpy arrays (one per step) and the step texts.
    """
    messages = [{"role": "user", "content": (
        "Solve the following math problem step by step. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        "End with 'ANSWER: <your final answer>'.\n\n"
        f"Problem: {problem_text}\n\nSolution:"
    )}]
    full_input = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = full_input + response

    hs_dict = collector.extract_hidden_states(full_text)
    hs_tensor = hs_dict[layer]  # (seq_len, hidden_dim)

    # Find step boundaries
    steps = collector.find_step_boundaries(response)
    if not steps:
        # Fallback: just use last token
        return [hs_tensor[-1].cpu().numpy().astype(np.float32)], ["full_trace"]

    encoding = collector.tokenizer(
        full_text, return_offsets_mapping=True, truncation=True, max_length=2048
    )
    offsets = encoding["offset_mapping"]
    prompt_len = len(full_input)

    step_vecs = []
    step_texts = []
    for char_start, char_end, step_text in steps:
        abs_start = prompt_len + char_start
        abs_end = prompt_len + char_end
        last_tok = None
        for tok_idx, (ts, te) in enumerate(offsets):
            if ts >= abs_start and te <= abs_end:
                last_tok = tok_idx
        if last_tok is not None and last_tok < hs_tensor.shape[0]:
            step_vecs.append(hs_tensor[last_tok].cpu().numpy().astype(np.float32))
            step_texts.append(step_text[:200])

    if not step_vecs:
        step_vecs = [hs_tensor[-1].cpu().numpy().astype(np.float32)]
        step_texts = ["full_trace"]

    return step_vecs, step_texts


# ---------------------------------------------------------------------------
# Stage 1 & 2: Train probe + evaluate AUROC
# ---------------------------------------------------------------------------
def stage_probe_train(collector, train_problems, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 1-2: Probe Training & AUROC Evaluation")
    print("=" * 60)

    # Collect hidden states for training problems
    print(f"Collecting hidden states for {len(train_problems)} training problems...")
    train_data = {layer: [] for layer in CANDIDATE_LAYERS}
    train_labels = []

    for i, problem in enumerate(train_problems):
        try:
            response = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(response)
            correct = check_answer(ans, problem["answer"])
            label = 0 if correct else 1  # 1 = error

            messages = [{"role": "user", "content": (
                "Solve the following math problem step by step. "
                "Number each step as 'Step 1:', 'Step 2:', etc. "
                "End with 'ANSWER: <your final answer>'.\n\n"
                f"Problem: {problem['problem']}\n\nSolution:"
            )}]
            full_text = collector.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + response

            hs = collector.extract_hidden_states(full_text)
            for layer in CANDIDATE_LAYERS:
                vec = hs[layer][-1].cpu().numpy().astype(np.float32)
                train_data[layer].append(vec)
            train_labels.append(label)

        except Exception as e:
            print(f"  [WARN] Problem {problem['id']} failed: {e}")
            continue

        if (i + 1) % 10 == 0:
            n_err = sum(train_labels)
            print(f"  [{i+1}/{len(train_problems)}] collected, "
                  f"errors={n_err}/{len(train_labels)}", flush=True)

    train_labels = np.array(train_labels)
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    print(f"Training data: {len(train_labels)} samples, {n_pos} errors, {n_neg} correct")

    if n_pos < 3 or n_neg < 3:
        print("ERROR: Not enough positive/negative samples for probe training.")
        return None, None, None, None, []

    # Train probe at each candidate layer
    probe_results = []
    best_auroc = 0
    best_layer = CANDIDATE_LAYERS[-1]
    best_clf = None
    best_scaler = None

    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    for layer in CANDIDATE_LAYERS:
        X = np.stack(train_data[layer])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(C=0.1, max_iter=1000)

        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            print(f"  Layer {layer:2d}: skipped (not enough samples per class for CV)")
            continue

        scores = cross_val_score(clf, X_scaled, train_labels,
                                 cv=n_splits, scoring='roc_auc')
        mean_auroc = float(scores.mean())
        std_auroc = float(scores.std())

        # Refit on all data
        clf.fit(X_scaled, train_labels)

        probe_results.append({
            "layer": layer,
            "cv_auroc_mean": round(mean_auroc, 4),
            "cv_auroc_std": round(std_auroc, 4),
        })

        # Save probe
        with open(probe_dir / f"probe_layer_{layer}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler, "layer": layer}, f)

        print(f"  Layer {layer:2d}: CV AUROC = {mean_auroc:.4f} +/- {std_auroc:.4f}")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_layer = layer
            best_clf = clf
            best_scaler = scaler

    # Save summary
    with open(probe_dir / "probe_summary.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\nBest probe: layer {best_layer}, AUROC = {best_auroc:.4f}")

    return best_clf, best_scaler, best_layer, best_auroc, probe_results


# ---------------------------------------------------------------------------
# Stage 3: Selective Refinement
# ---------------------------------------------------------------------------
def stage_selective_refinement(collector, clf, scaler, best_layer,
                               eval_problems, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 3: Selective Refinement (probe-guided)")
    print("=" * 60)

    # First, generate greedy traces and score them
    print(f"Generating greedy traces for {len(eval_problems)} eval problems...")
    problem_data = []

    for i, problem in enumerate(eval_problems):
        try:
            response = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(response)
            correct = check_answer(ans, problem["answer"])
            score = get_probe_score(
                collector, clf, scaler, best_layer,
                problem["problem"], response
            )
        except Exception as e:
            print(f"  [WARN] Problem {problem['id']} failed: {e}")
            response, ans, correct, score = "", "", False, 1.0

        problem_data.append({
            "id": problem["id"],
            "reference": problem["answer"],
            "greedy_answer": ans,
            "greedy_correct": correct,
            "probe_score": score,
        })

        if (i + 1) % 10 == 0:
            acc = sum(1 for p in problem_data if p["greedy_correct"]) / len(problem_data)
            print(f"  [{i+1}/{len(eval_problems)}] greedy_acc={acc:.3f}", flush=True)

    baseline_acc = sum(1 for p in problem_data if p["greedy_correct"]) / len(problem_data)
    print(f"Baseline greedy accuracy: {baseline_acc:.4f}")

    # Sort by probe score descending (most likely wrong first)
    sorted_data = sorted(problem_data, key=lambda x: x["probe_score"], reverse=True)

    # Selective refinement at various rates
    rates = [round(r, 1) for r in np.arange(0.1, 1.0, 0.1)]
    refinement_results = []

    print(f"\nRunning selective refinement at rates: {rates}")
    print(f"{'Rate':>6} {'#Refine':>8} {'Acc':>8} {'Delta':>8}")
    print("-" * 35)

    # Cache refined answers so we only regenerate each problem once
    refined_cache = {}

    for rate in rates:
        n_refine = int(len(sorted_data) * rate)
        refine_ids = set(p["id"] for p in sorted_data[:n_refine])

        # Regenerate flagged problems with temp=0.7
        for pid in refine_ids:
            if pid not in refined_cache:
                prob = next(p for p in eval_problems if p["id"] == pid)
                try:
                    resp = collector.generate_cot(prob["problem"], temperature=0.7)
                    ans = extract_answer(resp)
                    correct = check_answer(ans, prob["answer"])
                except Exception:
                    ans, correct = "", False
                refined_cache[pid] = {"answer": ans, "correct": correct}

        # Compute accuracy: refined answer for flagged, greedy for rest
        n_correct = 0
        for p in problem_data:
            if p["id"] in refine_ids:
                n_correct += int(refined_cache[p["id"]]["correct"])
            else:
                n_correct += int(p["greedy_correct"])

        acc = n_correct / len(problem_data)
        delta = acc - baseline_acc

        refinement_results.append({
            "rate": rate,
            "n_refined": n_refine,
            "accuracy": round(acc, 4),
            "delta": round(delta, 4),
        })

        print(f"{rate:>6.1f} {n_refine:>8} {acc:>8.4f} {delta:>+8.4f}")

    # Save
    summary = {
        "n_problems": len(eval_problems),
        "baseline_accuracy": round(baseline_acc, 4),
        "refinement_results": refinement_results,
        "problem_data": problem_data,
    }
    with open(output_dir / "selective_refinement.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved selective refinement results.")
    return problem_data


# ---------------------------------------------------------------------------
# Stage 4: First-Step Prediction Analysis
# ---------------------------------------------------------------------------
def stage_first_step_prediction(collector, clf, scaler, best_layer,
                                 eval_problems, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 4: First-Step Prediction Analysis")
    print("=" * 60)

    print(f"Collecting step-level hidden states for {len(eval_problems)} problems...")
    first_scores = []
    last_scores = []
    all_step_scores = []
    labels = []

    for i, problem in enumerate(eval_problems):
        try:
            response = collector.generate_cot(problem["problem"], temperature=0.0)
            ans = extract_answer(response)
            correct = check_answer(ans, problem["answer"])
            label = 0 if correct else 1

            step_vecs, step_texts = get_step_hidden_states(
                collector, best_layer, problem["problem"], response
            )

            # Score each step
            step_scores = []
            for vec in step_vecs:
                v = vec.reshape(1, -1)
                s = float(clf.predict_proba(scaler.transform(v))[0, 1])
                step_scores.append(s)

            first_scores.append(step_scores[0])
            last_scores.append(step_scores[-1])
            all_step_scores.append(step_scores)
            labels.append(label)

        except Exception as e:
            print(f"  [WARN] Problem {problem['id']} failed: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_problems)}] collected", flush=True)

    first_scores = np.array(first_scores)
    last_scores = np.array(last_scores)
    labels = np.array(labels)

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    print(f"Collected: {len(labels)} problems, {n_pos} errors, {n_neg} correct")

    if n_pos < 2 or n_neg < 2:
        print("Not enough variety for AUROC computation. Skipping.")
        return

    # AUROC from different aggregations
    auroc_first = roc_auc_score(labels, first_scores)
    auroc_last = roc_auc_score(labels, last_scores)

    max_scores = np.array([max(s) for s in all_step_scores])
    auroc_max = roc_auc_score(labels, max_scores)

    mean_scores = np.array([np.mean(s) for s in all_step_scores])
    auroc_mean = roc_auc_score(labels, mean_scores)

    print(f"\n  First-step AUROC:  {auroc_first:.4f}")
    print(f"  Last-step AUROC:   {auroc_last:.4f}")
    print(f"  Max-score AUROC:   {auroc_max:.4f}")
    print(f"  Mean-score AUROC:  {auroc_mean:.4f}")

    # Cumulative step prediction
    print("\n  Cumulative step AUROC:")
    max_steps = max(len(s) for s in all_step_scores)
    cumulative = []
    for k in range(1, min(max_steps + 1, 10)):
        scores_k = []
        labels_k = []
        for s, l in zip(all_step_scores, labels):
            if len(s) >= k:
                scores_k.append(max(s[:k]))
                labels_k.append(l)
        if len(set(labels_k)) > 1:
            auc = roc_auc_score(labels_k, scores_k)
            cumulative.append({"step": k, "auroc": round(auc, 4), "n": len(labels_k)})
            print(f"    Steps 1-{k}: AUROC = {auc:.4f} (n={len(labels_k)})")

    # Score distributions
    correct_first = first_scores[labels == 0]
    wrong_first = first_scores[labels == 1]
    separation = float(wrong_first.mean() - correct_first.mean())

    print(f"\n  Correct traces first-step score: mean={correct_first.mean():.3f}, std={correct_first.std():.3f}")
    print(f"  Wrong traces first-step score:   mean={wrong_first.mean():.3f}, std={wrong_first.std():.3f}")
    print(f"  Separation: {separation:.3f}")

    # Unfaithful reasoning analysis
    print("\n  'Unfaithful' traces (high probe score at step 1, continues anyway):")
    unfaithful = {}
    for thresh in [0.6, 0.7, 0.8]:
        high = first_scores > thresh
        n_high = int(high.sum())
        n_high_wrong = int((high & (labels == 1)).sum())
        n_high_correct = int((high & (labels == 0)).sum())
        unfaithful[str(thresh)] = {
            "n_flagged": n_high,
            "n_wrong": n_high_wrong,
            "n_correct": n_high_correct,
            "precision": round(n_high_wrong / max(1, n_high), 4),
        }
        print(f"    Threshold {thresh}: flagged={n_high}, "
              f"wrong={n_high_wrong}/{n_high} "
              f"({n_high_wrong/max(1,n_high):.1%})")

    results = {
        "n_problems": len(labels),
        "n_errors": int(n_pos),
        "first_step_auroc": round(auroc_first, 4),
        "last_step_auroc": round(auroc_last, 4),
        "max_score_auroc": round(auroc_max, 4),
        "mean_score_auroc": round(auroc_mean, 4),
        "cumulative_step_aurocs": cumulative,
        "correct_first_mean": round(float(correct_first.mean()), 4),
        "wrong_first_mean": round(float(wrong_first.mean()), 4),
        "separation": round(separation, 4),
        "unfaithful_analysis": unfaithful,
    }
    with open(output_dir / "first_step_prediction.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved first-step prediction results.")


# ---------------------------------------------------------------------------
# Stage 5: Difficulty Control
# ---------------------------------------------------------------------------
def stage_difficulty_control(collector, clf, scaler, best_layer,
                              eval_problems, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 5: Difficulty Control (generation variance)")
    print("=" * 60)

    problems = eval_problems[:DIFFICULTY_SIZE]
    print(f"Generating {DIFFICULTY_GENS} traces each for {len(problems)} problems "
          f"(temp=0.7)...")

    problem_results = []

    for i, problem in enumerate(problems):
        gen_scores = []
        gen_correct = []

        for g in range(DIFFICULTY_GENS):
            try:
                resp = collector.generate_cot(problem["problem"], temperature=0.7)
                ans = extract_answer(resp)
                correct = check_answer(ans, problem["answer"])
                score = get_probe_score(
                    collector, clf, scaler, best_layer,
                    problem["problem"], resp
                )
                gen_scores.append(score)
                gen_correct.append(int(correct))
            except Exception:
                gen_scores.append(1.0)
                gen_correct.append(0)

        score_mean = float(np.mean(gen_scores))
        score_std = float(np.std(gen_scores))
        acc = float(np.mean(gen_correct))

        problem_results.append({
            "id": problem["id"],
            "reference": problem["answer"],
            "n_gens": DIFFICULTY_GENS,
            "accuracy": round(acc, 4),
            "probe_scores": [round(s, 4) for s in gen_scores],
            "probe_score_mean": round(score_mean, 4),
            "probe_score_std": round(score_std, 4),
            "correct_flags": gen_correct,
        })

        if (i + 1) % 10 == 0:
            avg_acc = np.mean([p["accuracy"] for p in problem_results])
            avg_std = np.mean([p["probe_score_std"] for p in problem_results])
            print(f"  [{i+1}/{len(problems)}] avg_acc={avg_acc:.3f}, "
                  f"avg_score_std={avg_std:.3f}", flush=True)

    # Aggregate analysis
    all_accs = [p["accuracy"] for p in problem_results]
    all_stds = [p["probe_score_std"] for p in problem_results]

    # Split into easy (acc=1.0), hard (acc=0.0), and medium
    easy = [p for p in problem_results if p["accuracy"] == 1.0]
    hard = [p for p in problem_results if p["accuracy"] == 0.0]
    medium = [p for p in problem_results if 0 < p["accuracy"] < 1.0]

    easy_std = np.mean([p["probe_score_std"] for p in easy]) if easy else 0
    hard_std = np.mean([p["probe_score_std"] for p in hard]) if hard else 0
    medium_std = np.mean([p["probe_score_std"] for p in medium]) if medium else 0

    easy_score = np.mean([p["probe_score_mean"] for p in easy]) if easy else 0
    hard_score = np.mean([p["probe_score_mean"] for p in hard]) if hard else 0
    medium_score = np.mean([p["probe_score_mean"] for p in medium]) if medium else 0

    print(f"\n  Overall: {len(problem_results)} problems")
    print(f"  Easy (acc=1.0):   {len(easy):3d} problems, "
          f"score_mean={easy_score:.3f}, score_std={easy_std:.3f}")
    print(f"  Medium (0<acc<1): {len(medium):3d} problems, "
          f"score_mean={medium_score:.3f}, score_std={medium_std:.3f}")
    print(f"  Hard (acc=0.0):   {len(hard):3d} problems, "
          f"score_mean={hard_score:.3f}, score_std={hard_std:.3f}")

    # Correlation between accuracy and probe score
    accs_arr = np.array(all_accs)
    stds_arr = np.array(all_stds)
    means_arr = np.array([p["probe_score_mean"] for p in problem_results])

    corr_acc_score = float(np.corrcoef(accs_arr, means_arr)[0, 1]) if len(accs_arr) > 1 else 0
    corr_acc_std = float(np.corrcoef(accs_arr, stds_arr)[0, 1]) if len(accs_arr) > 1 else 0

    print(f"\n  Correlation(accuracy, probe_score_mean) = {corr_acc_score:.4f}")
    print(f"  Correlation(accuracy, probe_score_std)  = {corr_acc_std:.4f}")

    summary = {
        "n_problems": len(problem_results),
        "n_gens_per_problem": DIFFICULTY_GENS,
        "temperature": 0.7,
        "overall_accuracy": round(float(np.mean(all_accs)), 4),
        "easy_count": len(easy),
        "medium_count": len(medium),
        "hard_count": len(hard),
        "easy_score_mean": round(easy_score, 4),
        "medium_score_mean": round(medium_score, 4),
        "hard_score_mean": round(hard_score, 4),
        "easy_score_std": round(easy_std, 4),
        "medium_score_std": round(medium_std, 4),
        "hard_score_std": round(hard_std, 4),
        "corr_accuracy_score_mean": round(corr_acc_score, 4),
        "corr_accuracy_score_std": round(corr_acc_std, 4),
        "per_problem": problem_results,
    }
    with open(output_dir / "difficulty_control.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved difficulty control results.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full experiment pipeline for Qwen2.5-14B-Instruct"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/14b_full")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data splits
    print("Loading benchmark data...")
    train_problems = load_benchmark("math500", max_samples=150, seed=42)
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in train_problems)
    eval_pool = [p for p in all_problems if p["id"] not in train_ids]

    # Take subsets according to spec
    train_subset = train_problems[:TRAIN_SIZE]
    eval_subset = eval_pool[:EVAL_SIZE]

    print(f"Train: {len(train_subset)} problems")
    print(f"Eval:  {len(eval_subset)} problems")

    # Load model with ALL candidate layers
    print(f"\nLoading {MODEL_NAME} on {args.device}...")
    collector = HiddenStateCollector(
        model_name=MODEL_NAME,
        device=args.device,
        layers_to_probe=CANDIDATE_LAYERS,
    )

    # ---- Stage 1 & 2: Probe training + AUROC ----
    best_clf, best_scaler, best_layer, best_auroc, probe_results = \
        stage_probe_train(collector, train_subset, output_dir)

    if best_clf is None:
        print("Probe training failed. Exiting.")
        return

    # Reload collector with only the best layer to save memory on forward passes
    print(f"\nReloading collector with best layer only (layer {best_layer})...")
    del collector
    torch.cuda.empty_cache()
    collector = HiddenStateCollector(
        model_name=MODEL_NAME,
        device=args.device,
        layers_to_probe=[best_layer],
    )

    # ---- Stage 3: Selective Refinement ----
    problem_data = stage_selective_refinement(
        collector, best_clf, best_scaler, best_layer, eval_subset, output_dir
    )

    # ---- Stage 4: First-Step Prediction ----
    # Reload collector with best layer (need step-level extraction)
    stage_first_step_prediction(
        collector, best_clf, best_scaler, best_layer, eval_subset, output_dir
    )

    # ---- Stage 5: Difficulty Control ----
    stage_difficulty_control(
        collector, best_clf, best_scaler, best_layer, eval_subset, output_dir
    )

    # ---- Final summary ----
    elapsed = time.time() - t0
    final_summary = {
        "model": MODEL_NAME,
        "device": args.device,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "candidate_layers": CANDIDATE_LAYERS,
        "best_probe_layer": best_layer,
        "best_probe_auroc": best_auroc,
        "train_size": len(train_subset),
        "eval_size": len(eval_subset),
        "difficulty_size": DIFFICULTY_SIZE,
        "difficulty_gens": DIFFICULTY_GENS,
        "probe_results": probe_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"ALL STAGES COMPLETE  ({elapsed/60:.1f} min)")
    print(f"Results saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
