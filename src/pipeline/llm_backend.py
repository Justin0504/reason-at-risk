"""LLM backend abstraction with token tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import openai
import anthropic


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    logprobs: list = field(default_factory=list)


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        logprobs: bool = False,
        n: int = 1,
    ) -> list[LLMResponse]:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = openai.OpenAI()

    def generate(self, messages, temperature=0.0, max_tokens=2048, logprobs=False, n=1):
        t0 = time.time()
        kwargs = dict(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, n=n,
        )
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        resp = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - t0) * 1000

        results = []
        for choice in resp.choices:
            lp = []
            if choice.logprobs and choice.logprobs.content:
                lp = [{"token": t.token, "logprob": t.logprob} for t in choice.logprobs.content]
            results.append(LLMResponse(
                text=choice.message.content or "",
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                latency_ms=latency, model=self.model, logprobs=lp,
            ))
        return results

    def name(self) -> str:
        return f"openai/{self.model}"


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, messages, temperature=0.0, max_tokens=2048, logprobs=False, n=1):
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        results = []
        for _ in range(n):
            t0 = time.time()
            kwargs = dict(
                model=self.model, messages=chat_messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            if system:
                kwargs["system"] = system
            resp = self.client.messages.create(**kwargs)
            latency = (time.time() - t0) * 1000
            text = resp.content[0].text if resp.content else ""
            results.append(LLMResponse(
                text=text,
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
                latency_ms=latency, model=self.model, logprobs=[],
            ))
        return results

    def name(self) -> str:
        return f"anthropic/{self.model}"
