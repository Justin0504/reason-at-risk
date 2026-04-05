#!/usr/bin/env python3
"""Main experiment runner for Week 1: baselines + oracle monitor."""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.pipeline.llm_backend import OpenAIBackend, AnthropicBackend
from src.pipeline.reasoning_pipeline import ReasoningPipeline
from src.pipeline.data_loader import load_benchmark
from src.monitor.oracle_monitor import OracleConflictMonitor
from src.monitor.refine_strategy import RefineStrategy
from src.eval.evaluator import Evaluator
from src.eval.metrics import compute_metrics


METHOD_MAP = {
    "no_cot": "run_no_cot",
    "cot": "run_cot",
    "self_consistency": "run_self_consistency",
    "confidence_adaptive": "run_confidence_adaptive",
    "oracle_monitor": "run_cot_with_monitor",
}


def run_single(pipeline, method, problem, cfg):
    fn_name = METHOD_MAP[method]
    fn = getattr(pipeline, fn_name)
    if method == "self_consistency":
        return fn(problem, k=cfg.get("self_consistency", {}).get("k", 5))
    elif method == "confidence_adaptive":
        return fn(problem, threshold=cfg.get("confidence_adaptive", {}).get("threshold", -0.5))
    else:
        return fn(problem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmark", type=str, default=None, help="Override benchmark")
    parser.add_argument("--method", type=str, default=None, help="Override method")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Overrides
    benchmarks = [args.benchmark] if args.benchmark else cfg["benchmarks"]
    methods = [args.method] if args.method else cfg["methods"]
    max_samples = args.max_samples or cfg.get("max_samples")
    output_dir = Path(args.output_dir or cfg.get("output_dir", "results/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backend
    provider = cfg["backend"]["provider"]
    model = cfg["backend"]["model"]
    if provider == "openai":
        backend = OpenAIBackend(model=model)
    else:
        backend = AnthropicBackend(model=model)

    # Monitor + strategy (for oracle_monitor method)
    monitor = OracleConflictMonitor()
    strategy = RefineStrategy()
    pipeline = ReasoningPipeline(backend, monitor=monitor, strategy=strategy)

    all_results = []

    for bench_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        data = load_benchmark(bench_name, max_samples=max_samples, seed=cfg.get("seed", 42))
        evaluator = Evaluator(bench_name)
        print(f"Loaded {len(data)} problems")

        for method in methods:
            # Skip logprob-dependent methods for Anthropic
            if method == "confidence_adaptive" and provider == "anthropic":
                print(f"  Skipping {method} (no logprobs for Anthropic)")
                continue

            print(f"\n  Method: {method}")
            traces = []
            t0 = time.time()

            for i, problem in enumerate(data):
                try:
                    trace = run_single(pipeline, method, problem, cfg)
                    # Evaluate
                    if bench_name in ("gsm8k", "math500"):
                        trace.is_correct = evaluator.check_numeric(
                            trace.final_answer, trace.reference_answer
                        )
                    else:
                        trace.is_correct = evaluator.check(
                            trace.final_answer, trace.reference_answer
                        )
                    traces.append(trace)

                    if (i + 1) % 10 == 0:
                        interim = compute_metrics(traces, method, bench_name)
                        print(f"    [{i+1}/{len(data)}] acc={interim.accuracy:.3f} avg_tok={interim.avg_tokens:.0f}")

                except Exception as e:
                    print(f"    Error on {problem['id']}: {e}")
                    continue

            elapsed = time.time() - t0
            metrics = compute_metrics(traces, method, bench_name)
            print(f"  Done in {elapsed:.1f}s: acc={metrics.accuracy:.4f} avg_tok={metrics.avg_tokens:.1f}")

            # Save results
            result = {
                "metrics": metrics.to_dict(),
                "elapsed_seconds": round(elapsed, 1),
                "config": {"provider": provider, "model": model, "max_samples": max_samples},
            }
            all_results.append(result)

            # Save per-method traces
            trace_file = output_dir / f"{bench_name}_{method}_traces.jsonl"
            with open(trace_file, "w") as f:
                for t in traces:
                    f.write(json.dumps({
                        "id": t.problem_id,
                        "answer": t.final_answer,
                        "reference": t.reference_answer,
                        "correct": t.is_correct,
                        "tokens": t.total_tokens,
                        "extra_tokens": t.total_extra_tokens,
                        "n_steps": len(t.steps),
                        "n_conflicts": t.num_conflicts,
                        "steps": [{"text": s.text, "conflict": s.conflict_detected,
                                   "conflict_type": s.conflict_type} for s in t.steps],
                    }) + "\n")

    # Save summary table
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Benchmark':<12} {'Acc':>8} {'Avg Tok':>10} {'Extra Tok':>10}")
    print(f"{'-'*80}")
    for r in all_results:
        m = r["metrics"]
        print(f"{m['method']:<25} {m['benchmark']:<12} {m['accuracy']:>8.4f} {m['avg_tokens']:>10.1f} {m['avg_extra_tokens']:>10.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
