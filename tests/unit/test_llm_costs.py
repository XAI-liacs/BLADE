"""Token usage capture and cost computation for LLM calls (#187)."""

import pickle
from types import SimpleNamespace

import pytest

import iohblade.llm as llm_mod
from iohblade.llm import (
    LLM,
    Claude_LLM,
    DeepSeek_LLM,
    Dummy_LLM,
    Gemini_LLM,
    Multi_LLM,
    Ollama_LLM,
    OpenAI_LLM,
)
from iohblade.llm_costs import (
    MODEL_PRICES,
    PRICES_CHECKED,
    TokenUsage,
    anthropic_usage,
    estimate_tokens,
    gemini_usage,
    model_price,
    ollama_usage,
    openai_usage,
)

SESSION = [{"role": "user", "content": "hello"}]


class _Log:
    """Logger stand-in that records log_conversation calls."""

    def __init__(self):
        self.calls = []

    def log_conversation(self, role, content, cost=0.0, tokens=0):
        self.calls.append((role, content, cost, tokens))


def _logged(llm):
    log = _Log()
    llm.set_logger(log)
    return log


# --- price table ----------------------------------------------------------------


def test_price_table_is_dated_and_has_current_claude_models():
    assert PRICES_CHECKED
    for model in (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-2",
        "claude-opus-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        assert model in MODEL_PRICES
    assert MODEL_PRICES["claude-opus-5-5"].input == 4.0
    assert MODEL_PRICES["claude-opus-5-5"].output == 20.0
    assert MODEL_PRICES["claude-fable-5-2"].output == 40.0


@pytest.mark.parametrize(
    "name, key",
    [
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("Claude-Opus-5-5", "claude-opus-5-5"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("claude-haiku-4-5@20251001", "claude-haiku-4-5"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("anthropic.claude-sonnet-4-5-v1:0", "claude-sonnet-4-5"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
    ],
)
def test_model_price_recognises_snapshots_and_prefixes(name, key):
    assert model_price(name) is MODEL_PRICES[key]


@pytest.mark.parametrize(
    "name", [None, "", "llama3.2", "gpt-5.4-ultra", "multi-llm", "DUMMY"]
)
def test_unknown_models_have_no_price(name):
    assert model_price(name) is None


def test_more_specific_model_is_not_priced_as_its_base():
    # gpt-5.4-pro must not fall back to gpt-5.4.
    assert model_price("gpt-5.4-pro") is MODEL_PRICES["gpt-5.4-pro"]
    assert model_price("gpt-5.4-pro").input != model_price("gpt-5.4").input


# --- cost computation ---------------------------------------------------------


def test_cost_from_usage():
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=1_000_000,
    )
    assert usage.prompt_tokens == 4_000_000
    # Opus 5.5: input $4, cache read $0.20, 5-minute cache write 1.25x input.
    assert usage.prompt_cost("claude-opus-5-5") == pytest.approx(4 + 0.4 + 5)
    assert usage.completion_cost("claude-opus-5-5") == pytest.approx(10.0)


def test_cost_uses_the_reported_model_over_the_configured_one():
    usage = TokenUsage(output_tokens=1_000_000, model="claude-haiku-5-5")
    assert usage.completion_cost("claude-opus-5-5") == pytest.approx(0.5)


def test_cost_of_unpriced_model_is_zero():
    usage = TokenUsage(input_tokens=10_000, output_tokens=10_000)
    assert usage.prompt_cost("llama3.2") == 0.0
    assert usage.completion_cost("llama3.2") == 0.0


def test_cache_read_without_listed_price_uses_input_price():
    usage = TokenUsage(cache_read_tokens=1_000_000)
    assert usage.prompt_cost("gpt-5.4-pro") == pytest.approx(30.0)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_token_usage_pickles():
    usage = TokenUsage(1, 2, 3, 4, "claude-opus-5-5")
    assert pickle.loads(pickle.dumps(usage)) == usage


# --- usage parsing per provider ----------------------------------------------


def test_openai_usage_splits_cached_tokens():
    response = SimpleNamespace(
        model="gpt-5.4-2026-03-05",
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=300,  # includes reasoning tokens
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        ),
    )
    assert openai_usage(response) == TokenUsage(
        input_tokens=600,
        output_tokens=300,
        cache_read_tokens=400,
        model="gpt-5.4-2026-03-05",
    )


def test_openai_usage_deepseek_cache_hits():
    response = SimpleNamespace(
        model="deepseek-flash",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
            prompt_cache_hit_tokens=80,
        ),
    )
    usage = openai_usage(response)
    assert (usage.input_tokens, usage.cache_read_tokens) == (20, 80)


def test_openai_usage_missing():
    assert openai_usage(SimpleNamespace(usage=None)) is None


def test_anthropic_usage():
    response = SimpleNamespace(
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2000,  # includes thinking tokens
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=300,
        ),
    )
    assert anthropic_usage(response) == TokenUsage(
        10, 2000, 5000, 300, "claude-opus-5-5"
    )


def test_gemini_usage_adds_thinking_tokens():
    response = SimpleNamespace(
        model_version="gemini-2.5-flash",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1200,
            cached_content_token_count=200,
            candidates_token_count=100,
            thoughts_token_count=900,
        ),
    )
    assert gemini_usage(response) == TokenUsage(
        input_tokens=1000,
        output_tokens=1000,
        cache_read_tokens=200,
        model="gemini-2.5-flash",
    )


def test_ollama_usage_from_dict():
    response = {
        "model": "llama3.2",
        "message": {"content": "OK"},
        "prompt_eval_count": 26,
        "eval_count": 290,
    }
    assert ollama_usage(response) == TokenUsage(26, 290, model="llama3.2")
    assert ollama_usage({"message": {"content": "OK"}}) is None


def test_mock_responses_do_not_produce_token_counts(mocker):
    # Attribute access on a MagicMock returns more mocks, not numbers.
    usage = anthropic_usage(mocker.MagicMock())
    assert usage.prompt_tokens == 0 and usage.output_tokens == 0
    assert usage.model is None


# --- usage capture by the providers -------------------------------------------


def _openai_response(text, prompt, completion, model):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


@pytest.mark.parametrize("cls", [OpenAI_LLM, DeepSeek_LLM])
def test_openai_compatible_query_logs_reported_usage(mocker, cls):
    llm = cls(api_key="fake", model="gpt-5.4")
    create = mocker.patch.object(
        llm.client.chat.completions,
        "create",
        return_value=_openai_response("OK", 1_000_000, 100_000, "gpt-5.4"),
    )
    log = _logged(llm)
    assert llm.query(SESSION) == "OK"
    create.assert_called_once()
    assert log.calls == [
        ("client", "hello", pytest.approx(2.5), 1_000_000),
        ("gpt-5.4", "OK", pytest.approx(1.5), 100_000),
    ]


def test_claude_query_logs_reported_usage(mocker):
    llm = Claude_LLM(api_key="fake")  # claude-opus-5-5
    response = SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="OK")],
        usage=SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    mocker.patch.object(llm.client.messages, "create", return_value=response)
    log = _logged(llm)
    assert llm.query(SESSION) == "OK"
    assert log.calls == [
        ("client", "hello", pytest.approx(4.0), 1_000_000),
        ("claude-opus-5-5", "OK", pytest.approx(20.0), 1_000_000),
    ]


def test_claude_refusal_still_records_usage(mocker):
    llm = Claude_LLM(api_key="fake")
    response = SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason="refusal",
        content=[],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    mocker.patch.object(llm.client.messages, "create", return_value=response)
    assert llm._query(SESSION) == ""
    assert llm.last_usage == TokenUsage(7, 3, model="claude-opus-5-5")


def test_gemini_query_records_usage(mocker):
    llm = Gemini_LLM(api_key="fake", model="gemini-2.5-flash")
    chat = mocker.MagicMock()
    chat.send_message.return_value = SimpleNamespace(
        text="OK",
        model_version="gemini-2.5-flash",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            cached_content_token_count=None,
            candidates_token_count=4,
            thoughts_token_count=6,
        ),
    )
    mocker.patch.object(llm_mod.genai.client.Chats, "create", return_value=chat)
    log = _logged(llm)
    assert llm.query(SESSION) == "OK"
    assert [c[3] for c in log.calls] == [10, 10]


def test_ollama_query_logs_tokens_without_cost(mocker):
    llm = Ollama_LLM(model="llama3.2")
    mocker.patch.object(
        llm.client,
        "chat",
        return_value={
            "model": "llama3.2",
            "message": {"content": "OK"},
            "prompt_eval_count": 26,
            "eval_count": 290,
        },
    )
    log = _logged(llm)
    assert llm.query(SESSION) == "OK"
    assert log.calls == [
        ("client", "hello", 0.0, 26),
        ("llama3.2", "OK", 0.0, 290),
    ]


def test_multi_llm_takes_usage_of_the_chosen_llm(mocker):
    inner = Ollama_LLM(model="llama3.2")
    mocker.patch.object(
        inner.client,
        "chat",
        return_value={"message": {"content": "OK"}, "eval_count": 5},
    )
    multi = Multi_LLM([inner])
    log = _Log()
    multi.logger, multi.log = log, True  # log through the Multi_LLM only
    assert multi.query(SESSION) == "OK"
    assert [c[3] for c in log.calls] == [0, 5]


# --- fallback without reported usage ------------------------------------------


class _Custom(LLM):
    """A user-defined LLM that returns plain strings and reports no usage."""

    def __init__(self, model="custom"):
        super().__init__("", model)

    def _query(self, session, **kwargs):
        return "x" * 40

    def get_config(self):
        return [{"model": self.model, "config": {}}]


def test_custom_llm_falls_back_to_estimates():
    llm = _Custom()
    log = _logged(llm)
    llm.query([{"role": "user", "content": "y" * 400}])
    assert log.calls == [
        ("client", "y" * 400, 0.0, 100),
        ("custom", "x" * 40, 0.0, 10),
    ]


def test_custom_llm_with_priced_model_estimates_cost():
    llm = _Custom(model="claude-opus-5-5")
    log = _logged(llm)
    llm.query([{"role": "user", "content": "y" * 4_000_000}])
    assert log.calls[0][2] == pytest.approx(4.0)
    assert log.calls[1][2] == pytest.approx(10 * 20 / 1e6)


def test_dummy_llm_uses_estimate():
    llm = Dummy_LLM()
    log = _logged(llm)
    reply = llm.query(SESSION)
    assert log.calls[1][3] == estimate_tokens(reply)
    assert llm.last_usage is None


def test_usage_of_a_previous_call_is_not_reused():
    llm = _Custom()
    llm.last_usage = TokenUsage(999, 999)
    log = _logged(llm)
    llm.query(SESSION)
    assert log.calls[1][3] == 10


def test_failed_call_logs_prompt_without_cost():
    class _Failing(_Custom):
        def _query(self, session, **kwargs):
            raise RuntimeError("down")

    llm = _Failing(model="claude-opus-5-5")
    log = _logged(llm)
    with pytest.raises(RuntimeError):
        llm.query(SESSION)
    assert log.calls == [("client", "hello", 0.0, 0)]


def test_no_logger_no_logging():
    assert _Custom().query(SESSION) == "x" * 40


def test_tokencost_is_not_used():
    assert not hasattr(llm_mod, "count_string_tokens")
