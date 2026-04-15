#!/usr/bin/env python3
"""Experiment: Step-level probe score dynamics.

For each reasoning trace, compute the probe error score at every step boundary.
This reveals WHEN the model internally detects its error — before or after
the visible mistake appears in text.

Outputs:
  - Per-trace step-by-step probe scores
  - Average trajectories for correct vs wrong traces
  - "Detection lead time": how many steps before the actual error does the probe fire
"""

import argparse, json, pickle, re, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe.collect_hidden_states import HiddenStateCollector
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


def get_step_scores(collector, clf, scaler, best_layer, problem_text, response):
    """Get probe error score at each step boundary in the response."""
    # Build full text
    messages = [{"role": "user", "content": (
        "Solve the following math problem step by step. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        f"End with 'ANSWER: <your final answer>'.\n\nProblem: {problem_text}\n\nSolution:"
    )}]
    full_input = collector.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = full_input + response

    # Find step boundaries
    steps = collector.find_step_boundaries(response)
    if not steps:
        return []

    # Extract hidden states for entire sequence
    hs_dict = collector.extract_hidden_states(full_text)

    # Map step boundaries to token positions
    encoding = collector.tokenizer(full_text, return_offsets_mapping=True,
                                    truncation=True, max_length=2048)
    offsets = encoding["offset_mapping"]
    prompt_len = len(full_input)

    step_scores = []
    for step_idx, (char_start, char_end, step_text) in enumerate(steps):
        abs_start = prompt_len + char_start
        abs_end = prompt_len + char_end

        # Find last token of this step
        step_last_token = None
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= abs_start and tok_end <= abs_end:
                step_last_token = tok_idx

        if step_last_token is None or step_last_token >= hs_dict[best_layer].shape[0]:
            continue

        # Get probe score at this step
        hs = hs_dict[best_layer][step_last_token].numpy().astype(np.float32).reshape(1, -1)
        score = float(clf.predict_proba(scaler.transform(hs))[0, 1])
        step_scores.append({
            "step": step_idx,
            "score": round(score, 4),
            "text": step_text[:100],
        })

    return step_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-dir", required=True,
                        help="Directory containing probe_summary.json and probe pkl files")
    parser.add_argument("--output-dir", default="results/step_dynamics")
    parser.add_argument("--n-problems", type=int, default=100)
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

    # Load eval problems (different seed from training)
    all_problems = load_benchmark("math500", max_samples=500, seed=0)
    train_ids = set(p["id"] for p in load_benchmark("math500", max_samples=150, seed=42))
    eval_problems = [p for p in all_problems if p["id"] not in train_ids][:args.n_problems]
    print(f"Analyzing {len(eval_problems)} problems")

    # Collect step-by-step scores
    traces = []
    checkpoint_path = output_dir / "step_dynamics_checkpoint.json"

    # Resume from checkpoint
    if checkpoint_path.exists():
        traces = json.load(open(checkpoint_path))
        done_ids = {t["id"] for t in traces}
        print(f"Resuming from checkpoint: {len(traces)} done")
    else:
        done_ids = set()

    for i, prob in enumerate(eval_problems):
        if prob["id"] in done_ids:
            continue

        try:
            resp = collector.generate_cot(prob["problem"])
            ans = extract_answer(resp)
            correct = check_answer(ans, prob["answer"])
            step_scores = get_step_scores(
                collector, clf, scaler, best_layer, prob["problem"], resp
            )

            if step_scores:
                traces.append({
                    "id": prob["id"],
                    "correct": correct,
                    "n_steps": len(step_scores),
                    "step_scores": step_scores,
                    "final_score": step_scores[-1]["score"],
                    "first_score": step_scores[0]["score"],
                    "max_score": max(s["score"] for s in step_scores),
                })
        except Exception as e:
            print(f"  Error {prob['id']}: {e}")
            continue

        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(traces, f, indent=2)

        if (len(traces)) % 20 == 0:
            n_correct = sum(1 for t in traces if t["correct"])
            print(f"  [{len(traces)}/{len(eval_problems)}] "
                  f"acc={n_correct/len(traces):.3f}", flush=True)

    # === Analysis ===
    correct_traces = [t for t in traces if t["correct"]]
    wrong_traces = [t for t in traces if not t["correct"]]
    print(f"\nTotal: {len(traces)} traces ({len(correct_traces)} correct, {len(wrong_traces)} wrong)")

    # Average trajectories (normalized to same length)
    def normalize_trajectory(step_scores, n_bins=10):
        """Resample step scores to fixed number of bins."""
        scores = [s["score"] for s in step_scores]
        if len(scores) <= 1:
            return scores * n_bins
        indices = np.linspace(0, len(scores) - 1, n_bins)
        return [float(np.interp(idx, range(len(scores)), scores)) for idx in indices]

    n_bins = 10
    correct_trajectories = [normalize_trajectory(t["step_scores"], n_bins) for t in correct_traces if t["n_steps"] >= 2]
    wrong_trajectories = [normalize_trajectory(t["step_scores"], n_bins) for t in wrong_traces if t["n_steps"] >= 2]

    if correct_trajectories and wrong_trajectories:
        avg_correct = np.mean(correct_trajectories, axis=0).tolist()
        avg_wrong = np.mean(wrong_trajectories, axis=0).tolist()
        std_correct = np.std(correct_trajectories, axis=0).tolist()
        std_wrong = np.std(wrong_trajectories, axis=0).tolist()
    else:
        avg_correct = avg_wrong = std_correct = std_wrong = []

    # Score at each relative position
    analysis = {
        "model": args.model,
        "best_layer": best_layer,
        "n_traces": len(traces),
        "n_correct": len(correct_traces),
        "n_wrong": len(wrong_traces),
        "avg_steps_correct": round(np.mean([t["n_steps"] for t in correct_traces]), 2) if correct_traces else 0,
        "avg_steps_wrong": round(np.mean([t["n_steps"] for t in wrong_traces]), 2) if wrong_traces else 0,
        # First step stats
        "first_score_correct": round(np.mean([t["first_score"] for t in correct_traces]), 4) if correct_traces else 0,
        "first_score_wrong": round(np.mean([t["first_score"] for t in wrong_traces]), 4) if wrong_traces else 0,
        # Final step stats
        "final_score_correct": round(np.mean([t["final_score"] for t in correct_traces]), 4) if correct_traces else 0,
        "final_score_wrong": round(np.mean([t["final_score"] for t in wrong_traces]), 4) if wrong_traces else 0,
        # Max score stats
        "max_score_correct": round(np.mean([t["max_score"] for t in correct_traces]), 4) if correct_traces else 0,
        "max_score_wrong": round(np.mean([t["max_score"] for t in wrong_traces]), 4) if wrong_traces else 0,
        # Normalized trajectories
        "trajectory_bins": n_bins,
        "avg_trajectory_correct": [round(x, 4) for x in avg_correct],
        "avg_trajectory_wrong": [round(x, 4) for x in avg_wrong],
        "std_trajectory_correct": [round(x, 4) for x in std_correct],
        "std_trajectory_wrong": [round(x, 4) for x in std_wrong],
    }

    # Separation at each position
    if avg_correct and avg_wrong:
        separations = [round(w - c, 4) for c, w in zip(avg_correct, avg_wrong)]
        analysis["trajectory_separation"] = separations
        analysis["max_separation_position"] = float(np.argmax(separations)) / (n_bins - 1)

    with open(output_dir / "step_dynamics_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    with open(output_dir / "step_dynamics_traces.json", "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\n{'='*60}")
    print("STEP DYNAMICS SUMMARY")
    print(f"{'='*60}")
    print(f"First step:  correct={analysis['first_score_correct']:.3f}, wrong={analysis['first_score_wrong']:.3f}")
    print(f"Final step:  correct={analysis['final_score_correct']:.3f}, wrong={analysis['final_score_wrong']:.3f}")
    print(f"Max score:   correct={analysis['max_score_correct']:.3f}, wrong={analysis['max_score_wrong']:.3f}")
    if avg_correct and avg_wrong:
        print(f"Max separation at position: {analysis['max_separation_position']:.1%} of trace")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
