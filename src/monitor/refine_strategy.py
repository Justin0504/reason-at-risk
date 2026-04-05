"""Refine strategy: inject conflict info and re-generate from conflict point."""

from __future__ import annotations

from ..pipeline.reasoning_pipeline import ResponseStrategy, ReasoningStep, COT_SYSTEM
from ..pipeline.llm_backend import LLMBackend

CONFLICT_LABELS = {
    "compute": "arithmetic/calculation error",
    "state": "variable value contradicts a previous step",
    "constraint": "violates a constraint stated in the problem",
    "branch": "contradicts a previously established conclusion",
}


class RefineStrategy(ResponseStrategy):
    """On conflict: tell the model what went wrong and ask it to continue from that point."""

    def respond(
        self,
        trace: list[ReasoningStep],
        conflict_step: int,
        conflict_type: str,
        problem: dict,
        backend: LLMBackend,
    ) -> list[ReasoningStep]:
        # Build context: problem + steps up to conflict
        prior_text = "\n".join(s.text for s in trace[:conflict_step])
        conflict_text = trace[conflict_step].text
        conflict_desc = CONFLICT_LABELS.get(conflict_type, "inconsistency")

        prompt = (
            f"You were solving this problem:\n{problem['problem']}\n\n"
            f"Your reasoning so far:\n{prior_text}\n\n"
            f"Your next step was:\n{conflict_text}\n\n"
            f"However, this step contains a {conflict_desc}. "
            f"Please correct this step and continue solving from here. "
            f"End with 'ANSWER: <answer>'."
        )

        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        resp = backend.generate(messages, temperature=0.0, max_tokens=2048)[0]

        # Parse corrected steps
        import re
        lines = [l.strip() for l in resp.text.split('\n') if l.strip()]
        corrected = []
        for i, line in enumerate(lines):
            corrected.append(ReasoningStep(
                index=conflict_step + 1 + i,
                text=line,
                tokens=max(1, resp.total_tokens // max(1, len(lines))),
                extra_reasoning_triggered=True,
            ))
        return corrected
