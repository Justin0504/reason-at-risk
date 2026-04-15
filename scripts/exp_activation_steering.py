#!/usr/bin/env python3
"""Experiment: Activation Steering — use probe direction to correct reasoning.

KEY IDEA: The linear probe's weight vector defines the "error direction" in
hidden state space. During generation, we subtract this direction from the
model's hidden states, steering the model AWAY from error-associated representations.

If this improves accuracy, it proves:
1. CAUSAL evidence: the probe direction causally influences reasoning quality
2. PRACTICAL method: self-correction without external API, zero extra compute

This is the core METHOD contribution for the paper.

Approach:
- Extract probe weight vector w (the "error direction")
- Register a forward hook at the probe's best layer
- During generation, modify: h' = h - α * (h · w_norm) * w_norm
  (project out the error component with strength α)
- Compare accuracy: baseline vs steered at various α values
"""

import argparse, json, pickle, re, sys, time
import numpy as np
import torch
from pathlib import Path
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))
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


def check_answer(predicted, reference):
    p, r = extract_num(predicted), extract_num(reference)
    if p is not None and r is not None:
        return abs(p - r) < 0.01
    return predicted.strip().lower() == reference.strip().lower()


class ActivationSteerer:
    """Hooks into model to steer hidden states away from error direction."""

    def __init__(self, model, tokenizer, layer_idx, error_direction, device="mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.alpha = 0.0  # steering strength (set before generation)
        self.hook = None

        # Normalize error direction
        self.error_dir = torch.tensor(
            error_direction, dtype=torch.float16
        ).to(device)
        self.error_dir = self.error_dir / self.error_dir.norm()

    def _steering_hook(self, module, input, output):
        """Forward hook: project out error direction from hidden states."""
        if self.alpha == 0:
            return output

        # output is a tuple; hidden states are the first element
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Project out error direction: h' = h - α * (h · d) * d
        # Only steer the LAST token (being generated) to avoid corrupting context
        proj = torch.sum(hidden_states[:, -1:, :] * self.error_dir, dim=-1, keepdim=True)
        hidden_states = hidden_states.clone()
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - self.alpha * proj * self.error_dir

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register_hook(self):
        """Register the steering hook on the target layer."""
        # Access the transformer layer
        layers = self.model.model.layers  # Qwen/Llama architecture
        target_layer = layers[self.layer_idx]
        self.hook = target_layer.register_forward_hook(self._steering_hook)

    def remove_hook(self):
        if self.hook:
            self.hook.remove()
            self.hook = None

    def generate(self, problem_text, alpha, max_new_tokens=1024):
        """Generate with activation steering at given strength."""
        self.alpha = alpha

        prompt = (
            "Solve the following math problem step by step. "
            "Number each step as 'Step 1:', 'Step 2:', etc. "
            "End with 'ANSWER: <your final answer>'.\n\n"
            f"Problem: {problem_text}\n\nSolution:"
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    parser.add_argument("--output-dir", default="results/activation_steering")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine probe directory based on model
    model_short = args.model.split("/")[-1]
    probe_dir = Path(f"data/probes_{model_short}")
    probe_summary_path = probe_dir / "probe_summary.json"

    # Only fallback to default if model IS 3B
    if not probe_summary_path.exists() and "3B" in model_short:
        probe_dir = Path("data/probes")
        probe_summary_path = probe_dir / "probe_summary.json"

    # Check if probe exists for THIS model; if not, train one
    if probe_summary_path.exists():
        summary = json.load(open(probe_summary_path))
        best = max(summary, key=lambda x: x["cv_auroc_mean"])
        best_layer = best["layer"]
        with open(probe_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
            probe_data = pickle.load(f)
        clf = probe_data["clf"]
        scaler = probe_data["scaler"]
        print(f"Loaded probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")
    else:
        print(f"No probe found for {model_short}, training one...")
        # Need to train a probe for this model first
        probe_dir = Path(f"data/probes_{model_short}")
        probe_dir.mkdir(parents=True, exist_ok=True)

        # Load model for training
        train_collector = HiddenStateCollector(
            model_name=args.model, device=args.device
        )
        candidate_layers = train_collector.layers

        train_problems = load_benchmark("math500", max_samples=150, seed=42)[:80]
        train_features = {l: [] for l in candidate_layers}
        train_labels = []

        for i, prob in enumerate(train_problems):
            try:
                resp = train_collector.generate_cot(prob["problem"])
                ans = extract_answer(resp)
                correct = check_answer(ans, prob["answer"])
                messages = [{"role": "user", "content": (
                    "Solve the following math problem step by step. "
                    "Number each step as 'Step 1:', 'Step 2:', etc. "
                    f"End with 'ANSWER: <your final answer>'.\n\nProblem: {prob['problem']}\n\nSolution:"
                )}]
                full_text = train_collector.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) + resp
                hs = train_collector.extract_hidden_states(full_text)
                for l in candidate_layers:
                    train_features[l].append(hs[l][-1].cpu().numpy().astype(np.float32))
                train_labels.append(0 if correct else 1)
                if (i+1) % 20 == 0:
                    print(f"  Train [{i+1}/{len(train_problems)}] acc={1-np.mean(train_labels):.3f}")
            except Exception as e:
                print(f"  Train error: {e}")

        train_labels = np.array(train_labels)
        best_layer, best_auroc, clf, scaler = None, 0, None, None
        probe_results = []
        for l in candidate_layers:
            X = np.array(train_features[l])
            sc = StandardScaler().fit(X)
            c = LogisticRegression(C=0.1, max_iter=1000)
            if len(set(train_labels)) > 1 and len(X) > 10:
                scores = cross_val_score(c, sc.transform(X), train_labels, cv=5, scoring='roc_auc')
                auroc = scores.mean()
            else:
                auroc = 0.5
            probe_results.append({"layer": l, "cv_auroc_mean": round(auroc, 4)})
            print(f"  Layer {l}: AUROC={auroc:.4f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = l
                c.fit(sc.transform(X), train_labels)
                clf, scaler = c, sc

        with open(probe_dir / f"probe_layer_{best_layer}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler, "layer": best_layer}, f)
        with open(probe_dir / "probe_summary.json", "w") as f:
            json.dump(probe_results, f, indent=2)
        print(f"  Best: layer {best_layer}, AUROC={best_auroc:.4f}")

        del train_collector
        torch.cuda.empty_cache()

    # Error direction = probe weight vector (after scaling)
    w = clf.coef_[0]  # (hidden_dim,)
    error_direction = w / scaler.scale_
    print(f"Probe: layer {best_layer}, error direction dim={len(error_direction)}, norm={np.linalg.norm(error_direction):.4f}")

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device, layers_to_probe=[best_layer]
    )

    # Create steerer
    steerer = ActivationSteerer(
        collector.model, collector.tokenizer,
        best_layer, error_direction, device=args.device
    )
    steerer.register_hook()

    # Load eval problems
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Eval problems: {len(eval_problems)}")

    # Run at each alpha
    all_results = {}

    for alpha in args.alphas:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")
        print(f"{'='*50}")

        results = []
        for i, problem in enumerate(eval_problems):
            try:
                response = steerer.generate(problem["problem"], alpha=alpha)
                answer = extract_answer(response)
                correct = check_answer(answer, problem["answer"])
                results.append({
                    "id": problem["id"],
                    "correct": correct,
                    "answer": answer,
                    "response_len": len(response),
                })

                if (i + 1) % 20 == 0:
                    acc = sum(1 for r in results if r["correct"]) / len(results)
                    print(f"  [{i+1}/{len(eval_problems)}] acc={acc:.3f}")

            except Exception as e:
                print(f"  Error {problem['id']}: {e}")
                results.append({
                    "id": problem["id"], "correct": False,
                    "answer": "", "response_len": 0, "error": str(e)
                })

        N = len(results)
        acc = sum(1 for r in results if r["correct"]) / N
        all_results[alpha] = {
            "alpha": alpha,
            "accuracy": round(acc, 4),
            "n": N,
        }
        print(f"  Final: acc={acc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ACTIVATION STEERING RESULTS")
    print(f"{'='*60}")
    print(f"{'Alpha':>8} {'Acc':>8} {'Δ':>8}")
    print("-" * 30)
    base_acc = all_results[0.0]["accuracy"]
    for alpha in args.alphas:
        r = all_results[alpha]
        delta = r["accuracy"] - base_acc
        print(f"{alpha:>8.1f} {r['accuracy']:>8.4f} {delta:>+8.4f}")
        all_results[alpha]["delta"] = round(delta, 4)

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "model": args.model,
            "layer": best_layer,
            "base_acc": base_acc,
            "alphas": {str(a): all_results[a] for a in args.alphas},
        }, f, indent=2)

    print(f"\nSaved to {output_dir}")

    steerer.remove_hook()


if __name__ == "__main__":
    main()
