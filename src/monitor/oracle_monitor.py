"""Oracle (rule-based) conflict monitor — Week 1 upper-bound experiment.

Detects 4 conflict types in reasoning traces using deterministic checks:
  1. compute  — arithmetic errors (sympy verification)
  2. state    — variable value contradictions across steps
  3. constraint — violations of problem-stated constraints
  4. branch   — contradictory conclusions or direction changes
"""

from __future__ import annotations

import re
from ..pipeline.reasoning_pipeline import ConflictMonitor, ReasoningStep


class OracleConflictMonitor(ConflictMonitor):
    """Rule-based conflict detector. Designed for high recall on math/logic."""

    def check(self, trace_so_far: list[ReasoningStep], problem: dict) -> tuple[bool, str | None]:
        if not trace_so_far:
            return False, None

        current = trace_so_far[-1]
        previous = trace_so_far[:-1]

        # 1. Compute errors — check arithmetic expressions in current step
        conflict, ctype = self._check_compute(current.text)
        if conflict:
            return True, ctype

        # 2. State contradictions — compare variable assignments
        conflict, ctype = self._check_state(current.text, previous)
        if conflict:
            return True, ctype

        # 3. Constraint violations — check against problem constraints
        conflict, ctype = self._check_constraint(current.text, problem)
        if conflict:
            return True, ctype

        # 4. Branch conflicts — contradictory conclusions
        conflict, ctype = self._check_branch(current.text, previous)
        if conflict:
            return True, ctype

        return False, None

    def _check_compute(self, text: str) -> tuple[bool, str | None]:
        """Verify arithmetic expressions using sympy."""
        # Find patterns like "X * Y = Z", "X + Y = Z", "X / Y = Z", "X - Y = Z"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*[\*x×]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
        ]
        ops = [lambda a, b: a * b, lambda a, b: a + b,
               lambda a, b: a - b, lambda a, b: a / b]

        for pattern, op in zip(patterns, ops):
            for match in re.finditer(pattern, text):
                try:
                    a, b, c = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    expected = op(a, b)
                    if abs(expected - c) > 0.01:
                        return True, "compute"
                except (ValueError, ZeroDivisionError):
                    continue
        return False, None

    def _check_state(self, current_text: str, previous: list[ReasoningStep]) -> tuple[bool, str | None]:
        """Detect contradictory variable assignments across steps."""
        # Extract "X = <number>" assignments
        def extract_assignments(text: str) -> dict[str, float]:
            assigns = {}
            for m in re.finditer(r'([a-zA-Z_]\w*)\s*=\s*(\d+(?:\.\d+)?)', text):
                var, val = m.group(1), float(m.group(2))
                assigns[var] = val
            return assigns

        # Extract "there are N <noun>" patterns
        def extract_counts(text: str) -> dict[str, float]:
            counts = {}
            for m in re.finditer(r'(?:there are|we have|total of|equals?)\s+(\d+(?:\.\d+)?)\s+(\w+)', text, re.I):
                counts[m.group(2).lower()] = float(m.group(1))
            return counts

        curr_assigns = extract_assignments(current_text)
        curr_counts = extract_counts(current_text)

        for step in previous:
            prev_assigns = extract_assignments(step.text)
            for var, val in curr_assigns.items():
                if var in prev_assigns and abs(prev_assigns[var] - val) > 0.01:
                    return True, "state"

            prev_counts = extract_counts(step.text)
            for noun, val in curr_counts.items():
                if noun in prev_counts and abs(prev_counts[noun] - val) > 0.01:
                    return True, "state"

        return False, None

    def _check_constraint(self, text: str, problem: dict) -> tuple[bool, str | None]:
        """Check if current step violates explicit constraints from the problem."""
        prob_text = problem.get("problem", "")

        # Extract numeric constraints from problem: "X has 5 apples" → (apples, 5)
        prob_nums = {}
        for m in re.finditer(r'(\d+)\s+(\w+)', prob_text):
            prob_nums[m.group(2).lower()] = float(m.group(1))

        # Check if current step references these quantities with wrong values
        for m in re.finditer(r'(\d+)\s+(\w+)', text):
            noun = m.group(2).lower()
            val = float(m.group(1))
            if noun in prob_nums and abs(prob_nums[noun] - val) > 0.01:
                # Only flag if the step is clearly re-stating the problem quantity (not computing new ones)
                restating = re.search(
                    rf'(?:given|problem says|we know|start with)\s.*{re.escape(noun)}', text, re.I
                )
                if restating:
                    return True, "constraint"

        return False, None

    def _check_branch(self, current_text: str, previous: list[ReasoningStep]) -> tuple[bool, str | None]:
        """Detect contradictory conclusions."""
        # Look for explicit contradiction markers
        contradiction_patterns = [
            r'(?:wait|no|actually|but earlier|this contradicts|that can\'t be)',
            r'(?:on second thought|I made an error|let me reconsider)',
        ]
        for pat in contradiction_patterns:
            if re.search(pat, current_text, re.I):
                return True, "branch"

        # Check for answer flip: step says "answer is X" then later "answer is Y" (Y != X)
        curr_answer = re.search(r'(?:answer|result|total)\s+(?:is|=)\s+(\d+(?:\.\d+)?)', current_text, re.I)
        if curr_answer:
            for step in previous:
                prev_answer = re.search(r'(?:answer|result|total)\s+(?:is|=)\s+(\d+(?:\.\d+)?)', step.text, re.I)
                if prev_answer:
                    if abs(float(curr_answer.group(1)) - float(prev_answer.group(1))) > 0.01:
                        return True, "branch"

        return False, None
