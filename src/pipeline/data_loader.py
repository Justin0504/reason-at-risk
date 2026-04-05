"""Unified benchmark data loaders → list[dict] with keys: id, problem, answer, metadata."""

from __future__ import annotations
import random
import re


def load_benchmark(name: str, split: str = "test", max_samples: int | None = None, seed: int = 42) -> list[dict]:
    loaders = {
        "gsm8k": load_gsm8k,
        "math500": load_math500,
        "prontoqa": load_prontoqa,
        "blocksworld": load_blocksworld,
        "humaneval": load_humaneval,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(loaders.keys())}")
    data = loaders[name](split)
    if max_samples and len(data) > max_samples:
        data = random.Random(seed).sample(data, max_samples)
    return data


def load_gsm8k(split="test"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {
            "id": f"gsm8k_{i}",
            "problem": row["question"],
            "answer": row["answer"].split("####")[-1].strip(),
            "metadata": {"full_solution": row["answer"], "benchmark": "gsm8k"},
        }
        for i, row in enumerate(ds)
    ]


def load_math500(split="test"):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split=split)

    results = []
    for i, row in enumerate(ds):
        sol = row.get("solution", row.get("answer", ""))
        m = re.search(r'\\boxed\{(.+?)\}', sol)
        answer = m.group(1) if m else sol.strip()
        results.append({
            "id": f"math500_{i}",
            "problem": row.get("problem", row.get("question", "")),
            "answer": answer,
            "metadata": {
                "subject": row.get("subject", row.get("type", "")),
                "level": row.get("level", ""),
                "benchmark": "math500",
            },
        })
    if len(results) > 500:
        results = random.Random(42).sample(results, 500)
    return results


def load_prontoqa(split="test"):
    from datasets import load_dataset
    try:
        ds = load_dataset("renma/PrOntoQA", split=split)
        return [
            {
                "id": f"prontoqa_{i}",
                "problem": row.get("question", row.get("context", "")),
                "answer": str(row.get("answer", row.get("label", ""))),
                "metadata": {"benchmark": "prontoqa"},
            }
            for i, row in enumerate(ds)
        ]
    except Exception:
        return _synthetic_logic(500)


def _synthetic_logic(n):
    rng = random.Random(42)
    entities = ["cats", "dogs", "birds", "fish", "reptiles",
                "mammals", "animals", "pets", "vertebrates", "creatures"]
    results = []
    for i in range(n):
        a, b, c = rng.sample(entities, 3)
        if rng.random() > 0.5:
            problem = f"All {a} are {b}. All {b} are {c}. Question: Are all {a} {c}?"
            answer = "TRUE"
        else:
            problem = f"All {a} are {b}. All {c} are {b}. Question: Are all {a} {c}?"
            answer = "FALSE"
        results.append({"id": f"prontoqa_{i}", "problem": problem, "answer": answer,
                        "metadata": {"benchmark": "prontoqa", "synthetic": True}})
    return results


def load_blocksworld(split="test"):
    from datasets import load_dataset
    try:
        ds = load_dataset("tomsilver/blocksworld", split=split)
        return [
            {
                "id": f"bw_{i}",
                "problem": row.get("problem", row.get("input", "")),
                "answer": row.get("answer", row.get("output", "")),
                "metadata": {"benchmark": "blocksworld"},
            }
            for i, row in enumerate(ds)
        ]
    except Exception:
        return _synthetic_blocksworld(200)


def _synthetic_blocksworld(n):
    rng = random.Random(42)
    blocks = ["A", "B", "C", "D", "E"]
    results = []
    for i in range(n):
        k = rng.randint(3, 5)
        sel = blocks[:k]
        init = list(sel); rng.shuffle(init)
        goal = list(sel); rng.shuffle(goal)
        problem = (
            f"Initial: {' on '.join(init)} on table.\n"
            f"Goal: {' on '.join(goal)} on table.\n"
            f"Find a valid move sequence."
        )
        results.append({"id": f"bw_{i}", "problem": problem, "answer": " on ".join(goal) + " on table",
                        "metadata": {"benchmark": "blocksworld", "init": init, "goal": goal}})
    return results


def load_humaneval(split="test"):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split=split)
    return [
        {
            "id": row["task_id"],
            "problem": row["prompt"],
            "answer": row["canonical_solution"],
            "metadata": {
                "benchmark": "humaneval",
                "entry_point": row["entry_point"],
                "test": row["test"],
            },
        }
        for row in ds
    ]
