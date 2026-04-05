"""Core reasoning pipeline with pluggable conflict monitor."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .llm_backend import LLMBackend


@dataclass
class ReasoningStep:
    index: int
    text: str
    tokens: int = 0
    conflict_detected: bool = False
    conflict_type: str | None = None  # constraint | state | compute | branch
    extra_reasoning_triggered: bool = False


@dataclass
class ReasoningTrace:
    problem_id: str
    problem_text: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    reference_answer: str = ""
    is_correct: bool | None = None
    total_tokens: int = 0
    total_extra_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def num_conflicts(self) -> int:
        return sum(1 for s in self.steps if s.conflict_detected)

    @property
    def num_extra_reasoning(self) -> int:
        return sum(1 for s in self.steps if s.extra_reasoning_triggered)


class ConflictMonitor:
    """Base class — subclass to implement detection."""
    def check(self, trace_so_far: list[ReasoningStep], problem: dict) -> tuple[bool, str | None]:
        return False, None


class ResponseStrategy:
    """Base class — subclass to implement correction."""
    def respond(self, trace, conflict_step, conflict_type, problem, backend) -> list[ReasoningStep]:
        return []


COT_SYSTEM = (
    "Solve the problem step by step. Show each reasoning step clearly, "
    "numbered as 'Step 1:', 'Step 2:', etc. "
    "After all steps, write your final answer on a new line as 'ANSWER: <answer>'."
)


class ReasoningPipeline:
    def __init__(
        self,
        backend: LLMBackend,
        monitor: ConflictMonitor | None = None,
        strategy: ResponseStrategy | None = None,
    ):
        self.backend = backend
        self.monitor = monitor
        self.strategy = strategy

    # ── Baselines ──────────────────────────────────────────────

    def run_no_cot(self, problem: dict) -> ReasoningTrace:
        messages = [
            {"role": "system", "content": "Answer directly. Output ONLY the final answer, nothing else."},
            {"role": "user", "content": problem["problem"]},
        ]
        resp = self.backend.generate(messages)[0]
        return ReasoningTrace(
            problem_id=problem["id"], problem_text=problem["problem"],
            final_answer=resp.text.strip(), reference_answer=problem.get("answer", ""),
            total_tokens=resp.total_tokens,
        )

    def run_cot(self, problem: dict) -> ReasoningTrace:
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": problem["problem"]},
        ]
        resp = self.backend.generate(messages, max_tokens=4096)[0]
        return ReasoningTrace(
            problem_id=problem["id"], problem_text=problem["problem"],
            steps=self._parse_steps(resp.text),
            final_answer=self._extract_answer(resp.text),
            reference_answer=problem.get("answer", ""),
            total_tokens=resp.total_tokens,
        )

    def run_self_consistency(self, problem: dict, k: int = 5) -> ReasoningTrace:
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": problem["problem"]},
        ]
        responses = self.backend.generate(messages, temperature=0.7, max_tokens=4096, n=k)
        answers = [self._extract_answer(r.text) for r in responses]
        total_tok = sum(r.total_tokens for r in responses)
        vote = Counter(answers).most_common(1)[0][0]
        return ReasoningTrace(
            problem_id=problem["id"], problem_text=problem["problem"],
            final_answer=vote, reference_answer=problem.get("answer", ""),
            total_tokens=total_tok, metadata={"all_answers": answers, "k": k},
        )

    def run_confidence_adaptive(self, problem: dict, threshold: float = -0.5) -> ReasoningTrace:
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": problem["problem"]},
        ]
        resp = self.backend.generate(messages, max_tokens=4096, logprobs=True)[0]
        answer = self._extract_answer(resp.text)
        total_tok = resp.total_tokens
        extra = 0

        avg_lp = 0.0
        if resp.logprobs:
            avg_lp = sum(lp["logprob"] for lp in resp.logprobs) / len(resp.logprobs)

        if avg_lp < threshold:
            re_resps = self.backend.generate(messages, temperature=0.7, max_tokens=4096, n=3)
            # Pick majority
            re_answers = [self._extract_answer(r.text) for r in re_resps]
            answer = Counter(re_answers).most_common(1)[0][0]
            extra = sum(r.total_tokens for r in re_resps)

        return ReasoningTrace(
            problem_id=problem["id"], problem_text=problem["problem"],
            steps=self._parse_steps(resp.text),
            final_answer=answer, reference_answer=problem.get("answer", ""),
            total_tokens=total_tok + extra, total_extra_tokens=extra,
            metadata={"avg_logprob": avg_lp, "re_triggered": extra > 0},
        )

    # ── Core: CoT + Conflict Monitor ──────────────────────────

    def run_cot_with_monitor(self, problem: dict) -> ReasoningTrace:
        if not self.monitor or not self.strategy:
            return self.run_cot(problem)

        trace = self.run_cot(problem)
        extra_tokens = 0

        for i, step in enumerate(trace.steps):
            conflict, ctype = self.monitor.check(trace.steps[:i+1], problem)
            step.conflict_detected = conflict
            step.conflict_type = ctype

            if conflict:
                corrected = self.strategy.respond(
                    trace.steps, i, ctype, problem, self.backend
                )
                if corrected:
                    step.extra_reasoning_triggered = True
                    extra_tokens += sum(s.tokens for s in corrected)
                    trace.steps = trace.steps[:i+1] + corrected
                    trace.final_answer = self._extract_answer(
                        "\n".join(s.text for s in trace.steps)
                    )
                    break

        trace.total_extra_tokens = extra_tokens
        trace.total_tokens += extra_tokens
        return trace

    # ── Helpers ────────────────────────────────────────────────

    def _parse_steps(self, text: str) -> list[ReasoningStep]:
        parts = re.split(r'\n(?=Step \d+[:.:])', text)
        if len(parts) <= 1:
            parts = [l for l in text.split('\n') if l.strip()]
        steps = []
        for i, p in enumerate(parts):
            p = p.strip()
            if p:
                steps.append(ReasoningStep(index=i, text=p, tokens=max(1, len(p) // 4)))
        return steps

    def _extract_answer(self, text: str) -> str:
        m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r'\\boxed\{(.+?)\}', text)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else ""
