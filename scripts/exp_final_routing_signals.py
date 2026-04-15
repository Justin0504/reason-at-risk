#!/usr/bin/env python3
"""FINAL EXPERIMENT: Routing Signal Comparison + Multi-Model Routing.

For the same set of problems, compare routing using different error signals:
  1. Hidden state probe (ours)
  2. Sequence log-probability
  3. Verbalized confidence
  4. Self-consistency (majority vote agreement)

For each signal, route the top-r% (highest error score) to GPT-4o-mini.
Compare accuracy at each trigger rate.

Also runs on multiple models (3B, 7B) to show cross-model routing.

PHASE 1 (GPU server): Generate traces + extract all signals
PHASE 2 (local): Route to GPT-4o-mini + compute comparison + bootstrap CIs

Usage:
  # Phase 1 - on GPU server:
  python exp_final_routing_signals.py --phase generate --model Qwen/Qwen2.5-3B-Instruct --n-eval 200 --n-sc 5
  python exp_final_routing_signals.py --phase generate --model Qwen/Qwen2.5-7B-Instruct --n-eval 200 --n-sc 5

  # Phase 2 - locally (needs OPENAI_API_KEY):
  python exp_final_routing_signals.py --phase route --results-dir results/final_routing/Qwen2.5-3B-Instruct
  python exp_final_routing_signals.py --phase route --results-dir results/final_routing/Qwen2.5-7B-Instruct

  # Phase 3 - locally: analyze + produce tables
  python exp_final_routing_signals.py --phase analyze --results-dir results/final_routing
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
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


def check_answer(predicted, reference):
    p, r_ = extract_num(predicted), extract_num(reference)
    if p is not None and r_ is not None:
        return abs(p - r_) < 0.01
    return predicted.strip().lower() == reference.strip().lower()


# ---------------------------------------------------------------------------
# PHASE 1: Generate traces + signals (runs on GPU)
# ---------------------------------------------------------------------------

def phase_generate(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pipeline.data_loader import load_benchmark
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model_name = args.model
    short_name = model_name.split("/")[-1]
    out_dir = Path(args.output_dir or f"results/final_routing/{short_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_eval = args.n_eval
    n_sc = args.n_sc

    # --- Load data ---
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    # Training split: first 100 (seed=42 with max_samples=150 was used before,
    # but let's be explicit about the split)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids]
    train_problems = [p for p in all_problems if p["id"] in train_ids]

    if len(eval_problems) > n_eval:
        eval_problems = random.Random(123).sample(eval_problems, n_eval)

    print(f"Model: {model_name}")
    print(f"Train: {len(train_problems)}, Eval: {len(eval_problems)}")

    # --- Load model ---
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.output_hidden_states = True
    model.eval()

    n_layers = model.config.num_hidden_layers
    # Probe at 5 layers in upper half
    probe_layers = [int(n_layers * f) for f in [0.5, 0.6, 0.7, 0.8, 0.9]]
    probe_layers = sorted(set(min(l, n_layers - 1) for l in probe_layers))
    print(f"Layers: {n_layers}, probing: {probe_layers}")

    COT_PROMPT = (
        "Solve the following math problem step by step. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        "End with 'ANSWER: <your final answer>'.\n\n"
        "Problem: {problem}\n\nSolution:"
    )
    CONF_PROMPT = (
        "Rate your confidence in the above answer on a scale from 1 "
        "(very uncertain) to 5 (very confident). Reply with just the number."
    )

    def generate_trace(problem_text, temperature=0.0, extract_hidden=True, max_new_tokens=1024):
        """Generate a CoT trace. Returns (text, logprob, hidden_states_dict)."""
        prompt = COT_PROMPT.format(problem=problem_text)
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        # Move to first device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        generated_ids = out.sequences[0, input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute sequence log-probability
        logprob = 0.0
        if out.scores:
            for i, score in enumerate(out.scores):
                if i < len(generated_ids):
                    token_id = generated_ids[i].item()
                    log_probs = torch.log_softmax(score[0], dim=-1)
                    logprob += log_probs[token_id].item()
            logprob /= max(len(generated_ids), 1)  # normalize by length

        # Extract hidden states at probe layers (last token)
        hidden_dict = {}
        if extract_hidden:
            full_input = tokenizer(
                input_text + text, return_tensors="pt", truncation=True, max_length=2048
            )
            full_input = {k: v.to(device) for k, v in full_input.items()}
            with torch.no_grad():
                full_out = model(**full_input, output_hidden_states=True)
            # full_out.hidden_states is tuple of (n_layers+1, ) each (1, seq_len, hidden_dim)
            for layer in probe_layers:
                hs = full_out.hidden_states[layer + 1][0, -1, :].cpu().float().numpy()
                hidden_dict[layer] = hs
            del full_out
            torch.cuda.empty_cache()

        return text, logprob, hidden_dict

    def get_verbalized_confidence(problem_text, cot_text):
        """Get verbalized confidence score (1-5)."""
        messages = [
            {"role": "user", "content": COT_PROMPT.format(problem=problem_text)},
            {"role": "assistant", "content": cot_text},
            {"role": "user", "content": CONF_PROMPT},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        m = re.search(r'[1-5]', resp)
        return int(m.group()) if m else 5  # default high confidence

    # -----------------------------------------------------------------------
    # Step 1: Train probe on training problems
    # -----------------------------------------------------------------------
    print("\n=== Training probe ===")
    train_hidden = {l: [] for l in probe_layers}
    train_labels = []

    t_start = time.time()
    for i, p in enumerate(train_problems):
        text, _, hidden_dict = generate_trace(p["problem"], temperature=0.0, extract_hidden=True)
        answer = extract_answer(text)
        correct = check_answer(answer, p["answer"])
        train_labels.append(0 if correct else 1)
        for l in probe_layers:
            train_hidden[l].append(hidden_dict[l])
        if (i + 1) % 5 == 0:
            acc = sum(1 for x in train_labels if x == 0) / len(train_labels)
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(train_problems) - i - 1)
            print(f"  [{i+1}/{len(train_problems)}] train acc={acc:.3f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    train_labels = np.array(train_labels)

    # Train probes at each layer, select best by CV AUROC
    best_layer, best_auroc, best_probe, best_scaler = None, 0, None, None
    layer_results = {}

    for l in probe_layers:
        X = np.array(train_hidden[l])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, train_labels, cv=cv, scoring="roc_auc")
        mean_auroc = scores.mean()

        layer_results[l] = {"cv_auroc": mean_auroc, "cv_std": scores.std()}
        print(f"  Layer {l}: CV AUROC = {mean_auroc:.4f} ± {scores.std():.4f}")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_layer = l
            clf.fit(X_scaled, train_labels)
            best_probe = clf
            best_scaler = scaler

    print(f"\nBest layer: {best_layer} (AUROC = {best_auroc:.4f})")

    # Save probe
    with open(out_dir / "probe.pkl", "wb") as f:
        pickle.dump({"probe": best_probe, "scaler": best_scaler, "layer": best_layer}, f)

    # -----------------------------------------------------------------------
    # Step 2: Evaluate on eval problems — collect ALL signals
    # -----------------------------------------------------------------------
    print(f"\n=== Evaluating {len(eval_problems)} problems ===")
    results = []
    eval_t0 = time.time()

    for i, p in enumerate(eval_problems):
        entry = {"id": p["id"], "problem": p["problem"], "reference": p["answer"]}

        # 1. Greedy trace + probe score + logprob
        text, logprob, hidden_dict = generate_trace(p["problem"], temperature=0.0, extract_hidden=True)
        answer = extract_answer(text)
        correct = check_answer(answer, p["answer"])

        # Probe score
        hs = hidden_dict[best_layer].reshape(1, -1)
        hs_scaled = best_scaler.transform(hs)
        probe_score = best_probe.predict_proba(hs_scaled)[0, 1]  # P(error)

        entry["greedy_text"] = text
        entry["greedy_answer"] = answer
        entry["correct"] = correct
        entry["probe_score"] = float(probe_score)
        entry["logprob"] = float(logprob)  # more negative = less confident

        # 2. Verbalized confidence
        conf = get_verbalized_confidence(p["problem"], text)
        entry["verbalized_confidence"] = conf

        # 3. Self-consistency (N samples)
        sc_answers = [answer]  # include greedy
        for j in range(n_sc - 1):
            sc_text, _, _ = generate_trace(p["problem"], temperature=0.7, extract_hidden=False)
            sc_ans = extract_answer(sc_text)
            sc_answers.append(sc_ans)

        # SC agreement: fraction of samples agreeing with majority
        from collections import Counter
        ans_counts = Counter()
        for a in sc_answers:
            n = extract_num(a)
            key = str(n) if n is not None else a.strip().lower()
            ans_counts[key] += 1
        majority_ans, majority_count = ans_counts.most_common(1)[0]
        sc_agreement = majority_count / len(sc_answers)
        entry["sc_agreement"] = float(sc_agreement)
        entry["sc_answers"] = sc_answers

        # SC majority vote answer
        # Find the actual answer string that maps to majority key
        for a in sc_answers:
            n = extract_num(a)
            key = str(n) if n is not None else a.strip().lower()
            if key == majority_ans:
                entry["sc_majority_answer"] = a
                break
        entry["sc_majority_correct"] = check_answer(entry.get("sc_majority_answer", ""), p["answer"])

        results.append(entry)

        acc = sum(1 for r in results if r["correct"]) / len(results)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - eval_t0
            eta = elapsed / (i + 1) * (len(eval_problems) - i - 1)
            print(f"  [{i+1}/{len(eval_problems)}] acc={acc:.3f} "
                  f"elapsed={elapsed:.0f}s eta={eta/60:.1f}min")

    # Save results
    with open(out_dir / "eval_signals.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    acc = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\nDone! {len(results)} problems, accuracy={acc:.3f}")
    print(f"Saved to {out_dir}/eval_signals.json")
    print(f"Layer results: {json.dumps(layer_results, indent=2)}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "model": model_name,
            "n_train": len(train_problems),
            "n_eval": len(results),
            "accuracy": acc,
            "best_layer": best_layer,
            "best_cv_auroc": best_auroc,
            "layer_results": {str(k): v for k, v in layer_results.items()},
        }, f, indent=2)


# ---------------------------------------------------------------------------
# PHASE 2: Route to GPT-4o-mini (runs locally, needs OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def phase_route(args):
    from openai import OpenAI

    results_dir = Path(args.results_dir)
    signals_file = results_dir / "eval_signals.json"

    with open(signals_file) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems from {signals_file}")

    client = OpenAI()

    # Get API answers for ALL problems (we'll cache them)
    cache_file = results_dir / "api_answers.json"
    if cache_file.exists():
        with open(cache_file) as f:
            api_cache = json.load(f)
        print(f"Loaded {len(api_cache)} cached API answers")
    else:
        api_cache = {}

    n_new = 0
    for entry in data:
        pid = entry["id"]
        if pid not in api_cache:
            prompt = (
                f"Solve this math problem step by step. "
                f"End with 'ANSWER: <your final answer>'.\n\n"
                f"Problem: {entry['problem']}"
            )
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    api_text = resp.choices[0].message.content or ""
                    api_ans = extract_answer(api_text)
                    api_correct = check_answer(api_ans, entry["reference"])
                    api_cache[pid] = {
                        "answer": api_ans,
                        "correct": api_correct,
                    }
                    n_new += 1
                    break
                except Exception as e:
                    print(f"  API error for {pid}: {e}, retrying...")
                    time.sleep(2)

            if n_new % 20 == 0 and n_new > 0:
                # Save periodically
                with open(cache_file, "w") as f:
                    json.dump(api_cache, f, indent=2)
                print(f"  Cached {n_new} new API answers...")

    # Final save
    with open(cache_file, "w") as f:
        json.dump(api_cache, f, indent=2)
    print(f"Total API answers: {len(api_cache)} ({n_new} new)")

    # --- Compute routing results for each signal ---
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    n = len(data)

    # Define signal extractors (higher = more likely error)
    signals = {
        "probe": [d["probe_score"] for d in data],
        "logprob": [-d["logprob"] for d in data],  # negate: lower logprob = more error
        "confidence": [-(d["verbalized_confidence"] / 5.0) for d in data],  # negate: lower conf = more error
        "sc_disagree": [1.0 - d["sc_agreement"] for d in data],  # higher disagreement = more error
    }

    # Ground truth
    correct = [d["correct"] for d in data]
    base_acc = sum(correct) / n

    # Always-API accuracy
    api_acc = sum(1 for d in data if api_cache.get(d["id"], {}).get("correct", False)) / n

    routing_results = {"base_acc": base_acc, "api_acc": api_acc, "n": n, "signals": {}}

    for sig_name, sig_scores in signals.items():
        sig_results = []

        # Sort by signal (descending = most likely errors first)
        ranked = sorted(range(n), key=lambda i: sig_scores[i], reverse=True)

        for rate in rates:
            k = int(n * rate)
            routed_set = set(ranked[:k])

            # Compute accuracy: routed problems use API answer, rest use greedy
            n_correct = 0
            for i, d in enumerate(data):
                if i in routed_set:
                    n_correct += int(api_cache.get(d["id"], {}).get("correct", False))
                else:
                    n_correct += int(d["correct"])

            acc = n_correct / n

            # Precision of routing: what fraction of routed problems were actually wrong?
            n_routed_wrong = sum(1 for i in routed_set if not correct[i])
            precision = n_routed_wrong / k if k > 0 else 0

            # Bootstrap CI
            accs_boot = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                idx = rng.choice(n, size=n, replace=True)
                # Re-rank within bootstrap sample
                boot_ranked = sorted(idx, key=lambda i: sig_scores[i], reverse=True)
                boot_k = int(len(idx) * rate)
                boot_routed = set(boot_ranked[:boot_k])
                boot_correct = 0
                for j, i in enumerate(idx):
                    if j in boot_routed:
                        boot_correct += int(api_cache.get(data[i]["id"], {}).get("correct", False))
                    else:
                        boot_correct += int(data[i]["correct"])
                accs_boot.append(boot_correct / len(idx))

            ci_lo, ci_hi = np.percentile(accs_boot, [2.5, 97.5])

            sig_results.append({
                "rate": rate,
                "k": k,
                "acc": acc,
                "delta": acc - base_acc,
                "precision": precision,
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
            })

        routing_results["signals"][sig_name] = sig_results

    # Random routing baseline (averaged over 100 seeds)
    random_results = []
    for rate in rates:
        k = int(n * rate)
        accs = []
        for seed in range(100):
            rng = random.Random(seed)
            routed = set(rng.sample(range(n), k))
            nc = 0
            for i, d in enumerate(data):
                if i in routed:
                    nc += int(api_cache.get(d["id"], {}).get("correct", False))
                else:
                    nc += int(d["correct"])
            accs.append(nc / n)
        random_results.append({
            "rate": rate,
            "k": k,
            "acc": float(np.mean(accs)),
            "delta": float(np.mean(accs) - base_acc),
            "ci_lo": float(np.percentile(accs, 2.5)),
            "ci_hi": float(np.percentile(accs, 97.5)),
        })
    routing_results["signals"]["random"] = random_results

    # Save
    with open(results_dir / "routing_comparison.json", "w") as f:
        json.dump(routing_results, f, indent=2)

    # Print table
    print(f"\n{'='*80}")
    print(f"Routing Signal Comparison (n={n}, base_acc={base_acc:.3f}, api_acc={api_acc:.3f})")
    print(f"{'='*80}")
    print(f"{'Rate':>6} {'Probe':>12} {'Logprob':>12} {'Confidence':>12} {'SC-disagree':>12} {'Random':>12}")
    print("-" * 80)

    for ri in range(len(rates)):
        rate = rates[ri]
        row = f"{rate:>5.0%}"
        for sig in ["probe", "logprob", "confidence", "sc_disagree", "random"]:
            r = routing_results["signals"][sig][ri]
            row += f"  {r['acc']:.3f}({r['delta']:+.3f})"
        print(row)

    print(f"\nSaved to {results_dir}/routing_comparison.json")


# ---------------------------------------------------------------------------
# PHASE 3: Analyze + produce paper-ready tables
# ---------------------------------------------------------------------------

def phase_analyze(args):
    results_dir = Path(args.results_dir)

    # Collect results from all models
    model_dirs = sorted(results_dir.glob("Qwen*")) + sorted(results_dir.glob("Phi*"))

    print("=" * 90)
    print("PAPER TABLE: Routing Signal Comparison")
    print("=" * 90)

    for mdir in model_dirs:
        routing_file = mdir / "routing_comparison.json"
        if not routing_file.exists():
            print(f"  Skipping {mdir.name}: no routing_comparison.json")
            continue

        with open(routing_file) as f:
            data = json.load(f)

        print(f"\n--- {mdir.name} (n={data['n']}, base={data['base_acc']:.3f}, API={data['api_acc']:.3f}) ---")
        print(f"{'Rate':>6} | {'Probe':>16} | {'Logprob':>16} | {'Verbalized':>16} | {'SC-disagree':>16} | {'Random':>16}")
        print("-" * 100)

        sigs = data["signals"]
        for ri in range(len(sigs.get("probe", []))):
            rate = sigs["probe"][ri]["rate"]
            row = f"{rate:>5.0%} |"
            for sig in ["probe", "logprob", "confidence", "sc_disagree", "random"]:
                if sig in sigs and ri < len(sigs[sig]):
                    r = sigs[sig][ri]
                    row += f" {r['acc']:.3f} [{r.get('ci_lo',0):.2f},{r.get('ci_hi',0):.2f}] |"
                else:
                    row += f" {'---':>16} |"
            print(row)

    # Detection AUROC comparison
    print("\n" + "=" * 90)
    print("DETECTION AUROC by Signal")
    print("=" * 90)

    for mdir in model_dirs:
        signals_file = mdir / "eval_signals.json"
        if not signals_file.exists():
            continue

        with open(signals_file) as f:
            edata = json.load(f)

        from sklearn.metrics import roc_auc_score

        labels = [0 if d["correct"] else 1 for d in edata]
        if len(set(labels)) < 2:
            print(f"  {mdir.name}: all same label, skipping")
            continue

        aurocs = {}
        aurocs["probe"] = roc_auc_score(labels, [d["probe_score"] for d in edata])
        aurocs["logprob"] = roc_auc_score(labels, [-d["logprob"] for d in edata])
        aurocs["confidence"] = roc_auc_score(labels, [-d["verbalized_confidence"] for d in edata])
        aurocs["sc_disagree"] = roc_auc_score(labels, [1 - d["sc_agreement"] for d in edata])

        print(f"\n{mdir.name}:")
        for sig, auc in sorted(aurocs.items(), key=lambda x: -x[1]):
            print(f"  {sig:>15}: {auc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "route", "analyze"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-sc", type=int, default=5, help="Number of SC samples")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-dir", default="results/final_routing")
    args = parser.parse_args()

    if args.phase == "generate":
        phase_generate(args)
    elif args.phase == "route":
        phase_route(args)
    elif args.phase == "analyze":
        phase_analyze(args)
