"""LLM-based conflict monitor — uses a cheap model to verify each step.

Key design: only flag a step if the verifier is confident there's an actual error.
This should achieve much lower trigger rate than the rule-based oracle while
maintaining or improving precision.
"""

from __future__ import annotations

from ..pipeline.reasoning_pipeline import ConflictMonitor, ReasoningStep
from ..pipeline.llm_backend import LLMBackend


VERIFY_SYSTEM = (
    "You are a math reasoning verifier. You will be given a problem and a sequence of "
    "reasoning steps. Your job is to check the LAST step ONLY for errors.\n\n"
    "Check for:\n"
    "1. COMPUTE: arithmetic or algebraic calculation errors\n"
    "2. STATE: contradicts a value established in a previous step\n"
    "3. CONSTRAINT: violates a condition stated in the problem\n"
    "4. BRANCH: reaches a conclusion that contradicts previous reasoning\n\n"
    "Reply with EXACTLY one line in this format:\n"
    "VERDICT: <CORRECT|CONFLICT> [TYPE] [brief reason]\n\n"
    "Examples:\n"
    "VERDICT: CORRECT\n"
    "VERDICT: CONFLICT COMPUTE 3*7=24 should be 21\n"
    "VERDICT: CONFLICT STATE step 2 said x=5 but now says x=8\n\n"
    "Be conservative — only flag clear, definite errors. If unsure, say CORRECT."
)


class LLMConflictMonitor(ConflictMonitor):
    """Uses a cheap LLM to verify each reasoning step."""

    def __init__(self, verifier_backend: LLMBackend, check_every_n: int = 1):
        self.verifier = verifier_backend
        self.check_every_n = check_every_n
        self.total_verify_tokens = 0

    def check(self, trace_so_far: list[ReasoningStep], problem: dict) -> tuple[bool, str | None]:
        if not trace_so_far:
            return False, None

        # Only check every N steps to save tokens
        if len(trace_so_far) % self.check_every_n != 0:
            return False, None

        current = trace_so_far[-1]
        previous = trace_so_far[:-1]

        # Build verification prompt
        prev_text = "\n".join(f"Step {s.index + 1}: {s.text}" for s in previous) if previous else "(first step)"
        prompt = (
            f"Problem: {problem['problem']}\n\n"
            f"Previous steps:\n{prev_text}\n\n"
            f"Step to verify:\nStep {current.index + 1}: {current.text}\n\n"
            f"Is this step correct?"
        )

        messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        resp = self.verifier.generate(messages, temperature=0.0, max_tokens=60)[0]
        self.total_verify_tokens += resp.total_tokens

        verdict = resp.text.strip().upper()
        if "CONFLICT" in verdict:
            # Extract type
            for ctype in ("COMPUTE", "STATE", "CONSTRAINT", "BRANCH"):
                if ctype in verdict:
                    return True, ctype.lower()
            return True, "unknown"

        return False, None
