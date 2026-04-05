#!/usr/bin/env python3
"""Quick sanity check: run 3 GSM8K problems through all methods."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.pipeline.llm_backend import OpenAIBackend
from src.pipeline.reasoning_pipeline import ReasoningPipeline
from src.pipeline.data_loader import load_benchmark
from src.monitor.oracle_monitor import OracleConflictMonitor
from src.monitor.refine_strategy import RefineStrategy
from src.eval.evaluator import Evaluator

backend = OpenAIBackend(model="gpt-4o-mini")
monitor = OracleConflictMonitor()
strategy = RefineStrategy()
pipeline = ReasoningPipeline(backend, monitor=monitor, strategy=strategy)
evaluator = Evaluator("gsm8k")

data = load_benchmark("gsm8k", max_samples=3, seed=42)

methods = [
    ("no_cot", pipeline.run_no_cot),
    ("cot", pipeline.run_cot),
    ("oracle_monitor", pipeline.run_cot_with_monitor),
]

for problem in data:
    print(f"\n{'='*60}")
    print(f"Problem: {problem['problem'][:100]}...")
    print(f"Reference: {problem['answer']}")

    for name, fn in methods:
        trace = fn(problem)
        correct = evaluator.check_numeric(trace.final_answer, problem["answer"])
        print(f"  {name:<20} answer={trace.final_answer:<10} tokens={trace.total_tokens:<6} "
              f"conflicts={trace.num_conflicts} correct={correct}")
        if trace.num_conflicts > 0:
            for s in trace.steps:
                if s.conflict_detected:
                    print(f"    ^ conflict({s.conflict_type}): {s.text[:80]}...")
