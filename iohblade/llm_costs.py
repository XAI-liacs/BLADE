"""Token usage and cost bookkeeping for LLM calls.

Providers report the tokens of every call in their API response. BLADE
records that usage (:class:`TokenUsage`) and prices it with a small table
of list prices (:data:`MODEL_PRICES`). No extra API calls are made.

Prices are first-party list prices in USD per million tokens, for the
standard tier and the shortest context tier. Batch, priority, fast-mode,
regional and long-context surcharges are not modelled. Models that are not
in the table (local models, Ollama, models released after
:data:`PRICES_CHECKED`) are logged with cost 0.
"""

import math
import re
from dataclasses import dataclass

#: Date on which the prices below were checked against the providers'
#: pricing pages.
PRICES_CHECKED = "2026-11-26"


@dataclass(frozen=True)
class ModelPrice:
    """List price of a model in USD per million tokens.

    Args:
        input: Uncached input (prompt) tokens.
        output: Output tokens, thinking/reasoning tokens included.
        cache_read: Input tokens read from the prompt cache. ``None`` uses
            the input price.
        cache_write: Input tokens written to the prompt cache (Anthropic).
            ``None`` uses the input price.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


def _claude(input: float, output: float, cache_read: float | None = None):
    # Anthropic: cache reads 0.1x input unless listed otherwise, 5-minute
    # cache writes 1.25x input.
    return ModelPrice(
        input,
        output,
        cache_read if cache_read is not None else input * 0.1,
        input * 1.25,
    )


#: USD per million tokens, checked on :data:`PRICES_CHECKED`.
MODEL_PRICES: dict[str, ModelPrice] = {
    # Anthropic, https://platform.claude.com/docs/en/about-claude/pricing
    "claude-fable-5-2": _claude(8.0, 40.0, 0.20),
    "claude-fable-5-1": _claude(10.0, 50.0),
    "claude-fable-5": _claude(10.0, 50.0),
    "claude-opus-5-5": _claude(4.0, 20.0, 0.20),
    "claude-opus-5": _claude(5.0, 25.0),
    "claude-opus-4-8": _claude(5.0, 25.0),
    "claude-opus-4-7": _claude(5.0, 25.0),
    "claude-opus-4-6": _claude(5.0, 25.0),
    "claude-sonnet-5-5": _claude(2.0, 10.0, 0.10),
    "claude-sonnet-5": _claude(2.0, 10.0),
    "claude-sonnet-4-6": _claude(3.0, 15.0),
    "claude-sonnet-4-5": _claude(3.0, 15.0),
    "claude-haiku-5-5": _claude(0.10, 0.50),
    "claude-haiku-4-5": _claude(1.0, 5.0),
    # OpenAI, https://developers.openai.com/api/docs/pricing
    "gpt-6.2-astra": ModelPrice(8.0, 40.0, 0.40),
    "gpt-6.2-sol": ModelPrice(2.0, 10.0, 0.10),
    "gpt-6.1-sol": ModelPrice(2.0, 10.0, 0.10),
    "gpt-6.1-terra": ModelPrice(1.0, 5.0, 0.05),
    "gpt-6-astra": ModelPrice(10.0, 50.0, 1.0),
    "gpt-6-sol": ModelPrice(2.0, 10.0, 0.20),
    "gpt-6-luna": ModelPrice(0.10, 0.50, 0.01),
    "gpt-5.6-sol": ModelPrice(4.0, 20.0, 0.40),
    "gpt-5.6-terra": ModelPrice(2.0, 12.0, 0.20),
    "gpt-5.6-luna": ModelPrice(0.20, 1.20, 0.02),
    "gpt-5.5": ModelPrice(5.0, 30.0, 0.50),
    "gpt-5.5-pro": ModelPrice(30.0, 180.0),
    "gpt-5.4": ModelPrice(2.50, 15.0, 0.25),
    "gpt-5.4-pro": ModelPrice(30.0, 180.0),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    "gpt-5.2": ModelPrice(1.75, 14.0, 0.175),
    "gpt-5.2-pro": ModelPrice(21.0, 168.0),
    "gpt-5.1": ModelPrice(1.25, 10.0, 0.125),
    "gpt-5": ModelPrice(1.25, 10.0, 0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.0, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.005),
    "gpt-5-pro": ModelPrice(15.0, 120.0),
    "gpt-4.1": ModelPrice(2.0, 8.0, 0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "gpt-4o": ModelPrice(2.50, 10.0, 1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075),
    "o3": ModelPrice(2.0, 8.0, 0.50),
    "o3-pro": ModelPrice(20.0, 80.0),
    # Google, https://ai.google.dev/gemini-api/docs/pricing (paid tier,
    # text input, prompts up to 200k tokens; output includes thinking)
    "gemini-4-carbon": ModelPrice(2.0, 10.0, 0.10),
    "gemini-4-argon": ModelPrice(2.0, 10.0, 0.10),
    "gemini-4-neon": ModelPrice(3.0, 15.0, 0.15),
    "gemini-4-hydrogen": ModelPrice(1.50, 7.50, 0.075),
    "gemini-3.8-flash": ModelPrice(0.75, 3.75, 0.075),
    "gemini-3.6-flash": ModelPrice(0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": ModelPrice(2.0, 12.0, 0.20),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50, 0.025),
    "gemini-3-flash-preview": ModelPrice(0.50, 3.0, 0.05),
    "gemini-2.5-pro": ModelPrice(1.25, 10.0, 0.125),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40, 0.01),
    # DeepSeek, https://api-docs.deepseek.com/quick_start/pricing (peak
    # rates; off-peak requests cost half)
    "deepseek-flash": ModelPrice(0.30, 1.20, 0.006),
    "deepseek-v4-pro": ModelPrice(1.32, 3.96, 0.044),
}

# Prefixes that platforms put before a model ID, and version suffixes
# (dated snapshots) that keep the base model's price.
_PREFIX = re.compile(r"^(?:models/|anthropic\.|openai/|google/|deepseek/)")
_SNAPSHOT = re.compile(r"(?:[-@]\d{4}-?\d{2}-?\d{2}|-latest|-v\d+(?::\d+)?)$")


def model_price(model: str | None) -> ModelPrice | None:
    """The list price of ``model``, or ``None`` if it is not in the table.

    Dated snapshots (``gpt-4o-2024-08-06``, ``claude-opus-4-5@20251101``)
    and platform prefixes (``models/``, ``anthropic.``) are recognised.
    """
    if not model:
        return None
    name = str(model).strip().lower()
    if name in MODEL_PRICES:
        return MODEL_PRICES[name]
    name = _PREFIX.sub("", name)
    while name not in MODEL_PRICES:
        shorter = _SNAPSHOT.sub("", name)
        if shorter == name:
            return None
        name = shorter
    return MODEL_PRICES[name]


@dataclass
class TokenUsage:
    """Tokens of one LLM call, as reported by the provider.

    Args:
        input_tokens: Uncached input tokens (billed at the input price).
        output_tokens: Output tokens, thinking/reasoning tokens included
            when the provider reports them.
        cache_read_tokens: Input tokens served from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache.
        model: The model that answered, if the provider says so.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None

    @property
    def prompt_tokens(self) -> int:
        """All input tokens, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def prompt_cost(self, model: str | None = None) -> float:
        """USD cost of the input tokens (0 for an unpriced model)."""
        price = model_price(self.model or model)
        if price is None:
            return 0.0
        read = price.input if price.cache_read is None else price.cache_read
        write = price.input if price.cache_write is None else price.cache_write
        return (
            self.input_tokens * price.input
            + self.cache_read_tokens * read
            + self.cache_write_tokens * write
        ) / 1e6

    def completion_cost(self, model: str | None = None) -> float:
        """USD cost of the output tokens (0 for an unpriced model)."""
        price = model_price(self.model or model)
        if price is None:
            return 0.0
        return self.output_tokens * price.output / 1e6


def estimate_tokens(text: str | None) -> int:
    """A rough token count of ``text`` (about four characters per token).

    Used only when a provider reports no usage, for example in custom LLM
    classes.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def _get(obj, name, default=None):
    """Attribute or dict item ``name`` of ``obj``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _int(value) -> int:
    """``value`` as a non-negative int; 0 when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def openai_usage(response) -> TokenUsage | None:
    """Usage of an OpenAI-compatible chat completion (OpenAI, DeepSeek, vLLM).

    ``prompt_tokens`` includes cached tokens and ``completion_tokens``
    includes reasoning tokens.
    """
    usage = _get(response, "usage")
    if usage is None:
        return None
    prompt = _int(_get(usage, "prompt_tokens", 0))
    cached = _int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens", 0))
    if not cached:  # DeepSeek reports cache hits separately
        cached = _int(_get(usage, "prompt_cache_hit_tokens", 0))
    cached = min(cached, prompt)
    return TokenUsage(
        input_tokens=prompt - cached,
        output_tokens=_int(_get(usage, "completion_tokens", 0)),
        cache_read_tokens=cached,
        model=_model_name(response),
    )


def anthropic_usage(response) -> TokenUsage | None:
    """Usage of a Claude Messages API response.

    ``input_tokens`` excludes cached tokens; ``output_tokens`` includes
    thinking tokens.
    """
    usage = _get(response, "usage")
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=_int(_get(usage, "input_tokens", 0)),
        output_tokens=_int(_get(usage, "output_tokens", 0)),
        cache_read_tokens=_int(_get(usage, "cache_read_input_tokens", 0)),
        cache_write_tokens=_int(_get(usage, "cache_creation_input_tokens", 0)),
        model=_model_name(response),
    )


def gemini_usage(response) -> TokenUsage | None:
    """Usage of a google-genai response.

    ``candidates_token_count`` excludes thinking, so ``thoughts_token_count``
    is added to the output tokens.
    """
    meta = _get(response, "usage_metadata")
    if meta is None:
        return None
    prompt = _int(_get(meta, "prompt_token_count", 0))
    cached = min(_int(_get(meta, "cached_content_token_count", 0)), prompt)
    output = _int(_get(meta, "candidates_token_count", 0)) + _int(
        _get(meta, "thoughts_token_count", 0)
    )
    return TokenUsage(
        input_tokens=prompt - cached,
        output_tokens=output,
        cache_read_tokens=cached,
        model=_model_name(response, "model_version"),
    )


def ollama_usage(response) -> TokenUsage | None:
    """Usage of an Ollama chat response (``prompt_eval_count``/``eval_count``)."""
    if _get(response, "eval_count") is None:
        return None
    return TokenUsage(
        input_tokens=_int(_get(response, "prompt_eval_count", 0)),
        output_tokens=_int(_get(response, "eval_count", 0)),
        model=_model_name(response),
    )


def _model_name(response, field: str = "model") -> str | None:
    name = _get(response, field)
    return name if isinstance(name, str) and name else None
