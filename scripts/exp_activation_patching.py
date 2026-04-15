#!/usr/bin/env python3
"""Experiment: Activation patching for causal analysis.

For problems with mixed outcomes (some traces correct, some wrong),
patch hidden states from correct traces into wrong traces to test
whether the error signal is causal.

Key question: if we replace the hidden state at the critical layer
of a wrong trace with the hidden state from a correct trace (same problem),
does the model produce a better answer?

Results:
  - If patching fixes errors → error signal is causal, model "has" correct computation
  - If patching doesn't help → error signal is diagnostic readout, not causal driver
"""

import argparse, json, pickle, re, sys
import numpy as np
import torch
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe.collect_hidden_states import HiddenStateCollector, COT_PROMPT
from src.pipeline.data_loader import load_benchmark


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'\\boxed\{(.+?)\}', text)
    if m: return m.group(1).strip()
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
    pn, rn = predicted.strip().lower(), reference.strip().lower()
    if pn == rn: return True
    if len(rn) > 3 and (rn in pn or pn in rn): return True
    return False


class ActivationPatcher:
    """Hook-based activation patching at a specified layer."""

    def __init__(self, model, target_layer_idx):
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.hook_handle = None
        self.patch_vector = None  # (hidden_dim,) or None
        self.patch_mode = "replace_last"  # or "add_direction"
        self.patch_alpha = 1.0

    @contextmanager
    def patch(self, patch_vector, mode="replace_last", alpha=1.0):
        """Context manager for activation patching during generation."""
        self.patch_vector = patch_vector
        self.patch_mode = mode
        self.patch_alpha = alpha

        # Register hook on the target layer
        layer_module = self.model.model.layers[self.target_layer_idx]
        self.hook_handle = layer_module.register_forward_hook(self._hook_fn)
        try:
            yield
        finally:
            self.hook_handle.remove()
            self.hook_handle = None
            self.patch_vector = None

    def _hook_fn(self, module, input, output):
        """Modify hidden states at the target layer."""
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if self.patch_vector is None:
            return output

        patch = self.patch_vector.to(hidden_states.device, dtype=hidden_states.dtype)

        if self.patch_mode == "replace_last":
            # Replace the last token's hidden state with the patch
            hidden_states[:, -1, :] = patch
        elif self.patch_mode == "add_direction":
            # Add a direction vector scaled by alpha to all positions
            hidden_states = hidden_states + self.patch_alpha * patch.unsqueeze(0).unsqueeze(0)
        elif self.patch_mode == "blend":
            # Blend the last token: h' = (1-alpha)*h + alpha*patch
            hidden_states[:, -1, :] = (
                (1 - self.patch_alpha) * hidden_states[:, -1, :] + self.patch_alpha * patch
            )

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def generate_with_patch(collector, patcher, problem_text, patch_vector,
                        mode="replace_last", alpha=1.0, temperature=0.0):
    """Generate CoT with activation patching at the critical layer."""
    prompt = COT_PROMPT.format(problem=problem_text)
    messages = [{"role": "user", "content": prompt}]
    input_text = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = collector.tokenizer(input_text, return_tensors="pt").to(collector.device)

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1.0) if temperature == 0 else temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        pad_token_id=collector.tokenizer.eos_token_id,
    )

    with patcher.patch(patch_vector, mode=mode, alpha=alpha):
        with torch.no_grad():
            outputs = collector.model.generate(**inputs, generation_config=gen_config)

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return collector.tokenizer.decode(generated, skip_special_tokens=True)


def get_last_hidden_state(collector, best_layer, problem_text, response):
    """Extract last-token hidden state for a given response."""
    messages = [{"role": "user", "content": COT_PROMPT.format(problem=problem_text)}]
    full_text = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + response
    hs = collector.extract_hidden_states(full_text)
    return hs[best_layer][-1]  # (hidden_dim,) tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", default="results/activation_patching")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--n-traces", type=int, default=5,
                        help="Traces per problem for finding mixed-outcome problems")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir)

    # Load probe
    summary = json.load(open(probe_dir / "probe_summary.json"))
    best = max(summary, key=lambda x: x["cv_auroc_mean"])
    best_layer = best["layer"]
    with open(probe_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
        data = pickle.load(f)
    clf, scaler = data["clf"], data["scaler"]
    print(f"Loaded probe: layer {best_layer}, AUROC={best['cv_auroc_mean']:.4f}")

    # Load model
    collector = HiddenStateCollector(
        model_name=args.model, device=args.device,
        layers_to_probe=[best_layer]
    )
    patcher = ActivationPatcher(collector.model, best_layer)

    # Load eval problems
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Finding mixed-outcome problems from {len(eval_problems)} candidates...")

    # === Phase 1: Find mixed-outcome problems ===
    checkpoint_path = output_dir / "patching_checkpoint.json"
    if checkpoint_path.exists():
        all_data = json.load(open(checkpoint_path))
        mixed_problems = all_data.get("mixed_problems", [])
        patching_results = all_data.get("patching_results", [])
        done_phase1 = all_data.get("phase1_done", False)
        print(f"Resumed: {len(mixed_problems)} mixed problems, {len(patching_results)} patching results")
    else:
        mixed_problems = []
        patching_results = []
        done_phase1 = False

    if not done_phase1:
        for i, prob in enumerate(eval_problems):
            trace_data = []
            for j in range(args.n_traces):
                try:
                    resp = collector.generate_cot(prob["problem"], temperature=0.7)
                    ans = extract_answer(resp)
                    correct = check_answer(ans, prob["answer"])
                    hs = get_last_hidden_state(collector, best_layer, prob["problem"], resp)
                    trace_data.append({
                        "response": resp,
                        "answer": ans,
                        "correct": correct,
                        "hidden_state": hs.numpy().tolist(),  # for serialization
                    })
                except Exception as e:
                    print(f"  Error {prob['id']} trace {j}: {e}")

            # Check if mixed
            corrects = [t["correct"] for t in trace_data]
            if any(corrects) and not all(corrects):
                correct_traces = [t for t in trace_data if t["correct"]]
                wrong_traces = [t for t in trace_data if not t["correct"]]
                mixed_problems.append({
                    "id": prob["id"],
                    "problem": prob["problem"],
                    "answer": prob["answer"],
                    "correct_hs": correct_traces[0]["hidden_state"],
                    "wrong_hs": wrong_traces[0]["hidden_state"],
                    "correct_response": correct_traces[0]["response"],
                    "wrong_response": wrong_traces[0]["response"],
                    "n_correct": sum(corrects),
                    "n_wrong": len(corrects) - sum(corrects),
                })

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(eval_problems)}] found {len(mixed_problems)} mixed problems", flush=True)
                # Save checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump({"mixed_problems": mixed_problems, "patching_results": patching_results,
                               "phase1_done": False}, f)

        done_phase1 = True
        with open(checkpoint_path, "w") as f:
            json.dump({"mixed_problems": mixed_problems, "patching_results": patching_results,
                       "phase1_done": True}, f)

    print(f"\nFound {len(mixed_problems)} mixed-outcome problems")

    # === Phase 2: Activation patching experiments ===
    print("\n=== Activation Patching ===")
    done_ids = {r["id"] for r in patching_results}

    for mi, mp in enumerate(mixed_problems):
        if mp["id"] in done_ids:
            continue

        correct_hs = torch.tensor(mp["correct_hs"], dtype=torch.float16)
        wrong_hs = torch.tensor(mp["wrong_hs"], dtype=torch.float16)

        # Compute error direction for this problem
        error_direction = wrong_hs - correct_hs
        error_direction = error_direction / (error_direction.norm() + 1e-8)

        result = {"id": mp["id"], "experiments": {}}

        # Exp 1: Replace last-token HS of a new generation with correct trace's HS
        try:
            patched_resp = generate_with_patch(
                collector, patcher, mp["problem"], correct_hs,
                mode="replace_last", alpha=1.0
            )
            patched_ans = extract_answer(patched_resp)
            patched_correct = check_answer(patched_ans, mp["answer"])
            result["experiments"]["replace_with_correct"] = {
                "correct": patched_correct,
                "answer": patched_ans[:50],
            }
        except Exception as e:
            result["experiments"]["replace_with_correct"] = {"error": str(e)}

        # Exp 2: Subtract error direction (various alphas)
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            try:
                patched_resp = generate_with_patch(
                    collector, patcher, mp["problem"],
                    -alpha * error_direction,  # subtract error direction
                    mode="add_direction", alpha=1.0
                )
                patched_ans = extract_answer(patched_resp)
                patched_correct = check_answer(patched_ans, mp["answer"])
                result["experiments"][f"subtract_error_alpha_{alpha}"] = {
                    "correct": patched_correct,
                    "answer": patched_ans[:50],
                }
            except Exception as e:
                result["experiments"][f"subtract_error_alpha_{alpha}"] = {"error": str(e)}

        # Exp 3: Blend with correct HS
        for alpha in [0.3, 0.5, 0.7]:
            try:
                patched_resp = generate_with_patch(
                    collector, patcher, mp["problem"], correct_hs,
                    mode="blend", alpha=alpha
                )
                patched_ans = extract_answer(patched_resp)
                patched_correct = check_answer(patched_ans, mp["answer"])
                result["experiments"][f"blend_correct_alpha_{alpha}"] = {
                    "correct": patched_correct,
                    "answer": patched_ans[:50],
                }
            except Exception as e:
                result["experiments"][f"blend_correct_alpha_{alpha}"] = {"error": str(e)}

        # Exp 4: Control — no patching, greedy baseline
        try:
            baseline_resp = collector.generate_cot(mp["problem"])
            baseline_ans = extract_answer(baseline_resp)
            baseline_correct = check_answer(baseline_ans, mp["answer"])
            result["experiments"]["baseline_greedy"] = {
                "correct": baseline_correct,
                "answer": baseline_ans[:50],
            }
        except Exception as e:
            result["experiments"]["baseline_greedy"] = {"error": str(e)}

        patching_results.append(result)

        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({"mixed_problems": mixed_problems, "patching_results": patching_results,
                       "phase1_done": True}, f)

        if (mi + 1) % 5 == 0:
            print(f"  [{mi+1}/{len(mixed_problems)}] patching experiments done", flush=True)

    # === Analysis ===
    print(f"\n{'='*60}")
    print("ACTIVATION PATCHING RESULTS")
    print(f"{'='*60}")

    exp_names = set()
    for r in patching_results:
        exp_names.update(r["experiments"].keys())

    summary_data = {
        "model": args.model,
        "best_layer": best_layer,
        "n_mixed_problems": len(mixed_problems),
        "n_patching_experiments": len(patching_results),
        "results_by_method": {},
    }

    for exp_name in sorted(exp_names):
        correct_count = 0
        total = 0
        for r in patching_results:
            if exp_name in r["experiments"]:
                exp = r["experiments"][exp_name]
                if "correct" in exp:
                    total += 1
                    if exp["correct"]:
                        correct_count += 1
        if total > 0:
            acc = correct_count / total
            summary_data["results_by_method"][exp_name] = {
                "accuracy": round(acc, 4),
                "correct": correct_count,
                "total": total,
            }
            print(f"  {exp_name}: {correct_count}/{total} = {acc:.3f}")

    with open(output_dir / "patching_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(output_dir / "patching_details.json", "w") as f:
        json.dump(patching_results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
