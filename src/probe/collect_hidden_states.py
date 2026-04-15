"""Generate CoT traces with a local model and collect per-step hidden states.

For each problem:
1. Generate CoT reasoning using the model
2. Re-encode the full response and extract hidden states at each step boundary
3. Label each step: does the trace from here onward lead to a correct final answer?

Output: a dataset of (hidden_state, label) pairs for probe training.
"""

from __future__ import annotations

import json
import re
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class StepRecord:
    problem_id: str
    step_index: int
    step_text: str
    hidden_state: np.ndarray  # shape: (hidden_dim,)
    layer: int
    trace_is_correct: bool
    # Per-step labels (for more fine-grained training later)
    step_has_error: bool = False


COT_PROMPT = (
    "Solve the following math problem step by step. "
    "Number each step as 'Step 1:', 'Step 2:', etc. "
    "End with 'ANSWER: <your final answer>'.\n\n"
    "Problem: {problem}\n\nSolution:"
)


class HiddenStateCollector:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "mps",
        layers_to_probe: list[int] | None = None,
    ):
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )
        except RuntimeError:
            # Fallback: CPU offload for large models
            print(f"  MPS OOM, falling back to auto device map (CPU+MPS)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            device = "cpu"
        self.model.eval()
        self.device = device

        # Which layers to extract hidden states from
        n_layers = self.model.config.num_hidden_layers
        if layers_to_probe is None:
            # Default: early, middle, late layers
            self.layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        else:
            self.layers = [l for l in layers_to_probe if l < n_layers]

        print(f"Model loaded. {n_layers} layers, probing layers: {self.layers}")

    def generate_cot(self, problem_text: str, max_new_tokens: int = 1024,
                     temperature: float = 0.0) -> str:
        """Generate CoT reasoning for a problem."""
        prompt = COT_PROMPT.format(problem=problem_text)

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        from transformers import GenerationConfig
        if temperature > 0:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,  # must be >0 even if do_sample=False
                top_p=1.0,
                top_k=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode only the generated part
        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def extract_hidden_states(self, full_text: str) -> dict[int, torch.Tensor]:
        """Encode text and extract hidden states at each token position.

        Returns: {layer_idx: tensor of shape (seq_len, hidden_dim)}
        """
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        result = {}
        for layer_idx in self.layers:
            # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
            hs = outputs.hidden_states[layer_idx + 1]  # (1, seq_len, hidden_dim)
            result[layer_idx] = hs.squeeze(0).cpu()  # (seq_len, hidden_dim)

        return result

    def find_step_boundaries(self, text: str) -> list[tuple[int, int, str]]:
        """Find token positions of each reasoning step boundary.

        Returns: [(start_char, end_char, step_text), ...]
        """
        # Split by "Step N:" pattern
        pattern = r'(Step \d+:.*?)(?=Step \d+:|ANSWER:|$)'
        matches = list(re.finditer(pattern, text, re.DOTALL))

        if not matches:
            # Fallback: split by newlines
            lines = text.split('\n')
            boundaries = []
            pos = 0
            for line in lines:
                if line.strip():
                    boundaries.append((pos, pos + len(line), line.strip()))
                pos += len(line) + 1
            return boundaries

        return [(m.start(), m.end(), m.group(1).strip()) for m in matches]

    def collect_for_problem(
        self, problem: dict, reference_answer: str
    ) -> list[StepRecord]:
        """Full pipeline for one problem: generate, extract, label."""

        # 1. Generate CoT
        response = self.generate_cot(problem["problem"])

        # 2. Extract final answer and check correctness
        final_answer = self._extract_answer(response)
        is_correct = self._check_answer(final_answer, reference_answer)

        # 3. Find step boundaries in generated text
        steps = self.find_step_boundaries(response)
        if not steps:
            return []

        # 4. Build the full prompt + response for encoding
        prompt = COT_PROMPT.format(problem=problem["problem"])
        messages = [{"role": "user", "content": prompt}]
        full_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = full_input + response

        # 5. Extract hidden states
        hidden_states = self.extract_hidden_states(full_text)

        # 6. Map step boundaries to token positions
        # Tokenize to get offset mapping
        encoding = self.tokenizer(full_text, return_offsets_mapping=True, truncation=True, max_length=2048)
        offsets = encoding["offset_mapping"]

        prompt_len = len(full_input)
        records = []

        for step_idx, (char_start, char_end, step_text) in enumerate(steps):
            # Adjust char positions to account for the prompt prefix
            abs_start = prompt_len + char_start
            abs_end = prompt_len + char_end

            # Find the last token of this step
            step_last_token = None
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= abs_start and tok_end <= abs_end:
                    step_last_token = tok_idx

            if step_last_token is None or step_last_token >= hidden_states[self.layers[0]].shape[0]:
                continue

            # Extract hidden state at the last token of this step (for each layer)
            for layer_idx in self.layers:
                hs = hidden_states[layer_idx][step_last_token].numpy().astype(np.float32)
                records.append(StepRecord(
                    problem_id=problem["id"],
                    step_index=step_idx,
                    step_text=step_text[:200],  # truncate for storage
                    hidden_state=hs,
                    layer=layer_idx,
                    trace_is_correct=is_correct,
                ))

        return records

    def _extract_answer(self, text: str) -> str:
        m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r'\\boxed\{(.+?)\}', text)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else ""

    def _check_answer(self, predicted: str, reference: str) -> bool:
        """Simple numeric check."""
        def extract_num(s):
            s = re.sub(r'[,$%]', '', s.strip())
            m = re.search(r'-?\d+\.?\d*', s)
            return float(m.group()) if m else None

        p, r = extract_num(predicted), extract_num(reference)
        if p is not None and r is not None:
            return abs(p - r) < 0.01
        return predicted.strip().lower() == reference.strip().lower()
