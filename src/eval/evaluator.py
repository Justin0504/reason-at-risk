"""Answer evaluation and normalization per benchmark."""

import re
import math


def _extract_number(text: str) -> str | None:
    """Extract the first number from text."""
    # Try to find a standalone number (possibly negative, decimal, or with commas)
    m = re.search(r'-?\d[\d,]*\.?\d*', text.replace(',', ''))
    return m.group(0) if m else None


def _norm_math(answer: str) -> str:
    ans = answer.strip()
    # Remove common prefixes like "ANSWER:"
    ans = re.sub(r'^(?:ANSWER|answer|Answer)\s*:\s*', '', ans)
    # Remove LaTeX
    ans = re.sub(r'[\$\\]', '', ans)
    ans = re.sub(r'\\text\{(.+?)\}', r'\1', ans)
    ans = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'(\1)/(\2)', ans)
    ans = ans.strip().rstrip('.')

    # Try direct numeric parse
    try:
        val = float(eval(ans.replace(',', '')))
        return str(int(val)) if val == int(val) else f"{val:.6f}".rstrip('0').rstrip('.')
    except Exception:
        pass

    # Fallback: extract first number from the string
    num = _extract_number(ans)
    if num:
        try:
            val = float(num)
            return str(int(val)) if val == int(val) else f"{val:.6f}".rstrip('0').rstrip('.')
        except ValueError:
            pass

    return ans.lower().strip()


def _norm_logic(answer: str) -> str:
    ans = answer.strip().upper()
    mapping = {"TRUE": "TRUE", "YES": "TRUE", "CORRECT": "TRUE",
               "FALSE": "FALSE", "NO": "FALSE", "INCORRECT": "FALSE"}
    return mapping.get(ans, ans)


def _norm_generic(answer: str) -> str:
    return answer.strip().lower()


NORMALIZERS = {
    "gsm8k": _norm_math,
    "math500": _norm_math,
    "prontoqa": _norm_logic,
    "blocksworld": _norm_generic,
    "humaneval": _norm_generic,
}


class Evaluator:
    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        self.norm = NORMALIZERS.get(benchmark, _norm_generic)

    def check(self, predicted: str, reference: str) -> bool:
        return self.norm(predicted) == self.norm(reference)

    def check_numeric(self, predicted: str, reference: str, rtol: float = 1e-3) -> bool:
        try:
            return math.isclose(float(eval(predicted.strip())), float(eval(reference.strip())), rel_tol=rtol)
        except Exception:
            return self.check(predicted, reference)
