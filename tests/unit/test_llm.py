import sys
import copy
import types
import datetime as _dt
import pickle
from unittest.mock import MagicMock, patch

import httpx
import pytest

import iohblade.llm as llm_mod  # the module that defines _query
from iohblade.llm import (
    LLM,
    Claude_LLM,
    Gemini_LLM,
    NoCodeException,
    Ollama_LLM,
    OpenAI_LLM,
    DeepSeek_LLM,
    Dummy_LLM,
    LMStudio_LLM,
    MLX_LM_LLM,
)


class _DummyOpenAI:
    """Stand-in that just records the kwargs used to build it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_openai(monkeypatch):
    """
    Helper that swaps out openai.OpenAI with _DummyOpenAI inside the
    already-imported iohblade.llm module.
    """
    monkeypatch.setattr(llm_mod.openai, "OpenAI", _DummyOpenAI)


class _DummyAnthropic:
    """Stand-in that just records the kwargs used to build it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = MagicMock()


def _patch_anthropic(monkeypatch):
    """Helper that swaps out anthropic.Anthropic with _DummyAnthropic."""
    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", _DummyAnthropic)


def test_openai_llm_getstate_strips_client(monkeypatch):
    _patch_openai(monkeypatch)

    llm = OpenAI_LLM(api_key="sk-test", model="gpt-4-turbo")
    state = llm.__getstate__()

    assert "client" not in state
    # sanity-check something else is still there
    assert state["model"] == "gpt-4-turbo"


def test_openai_llm_deepcopy_restores_client(monkeypatch):
    _patch_openai(monkeypatch)

    original = OpenAI_LLM(api_key="sk-test", model="gpt-4o", temperature=0.17)
    clone = copy.deepcopy(original)

    # new object, equal public state
    assert clone is not original
    assert clone.model == original.model
    assert clone.temperature == original.temperature

    # brand-new client object of the dummy type
    assert isinstance(clone.client, _DummyOpenAI)
    assert clone.client is not original.client
    assert clone.client.kwargs["api_key"] == "sk-test"

    # changing the clone does not leak back
    clone.temperature = 0.99
    assert original.temperature != clone.temperature


def test_openai_llm_pickle_roundtrip(monkeypatch):
    _patch_openai(monkeypatch)

    llm = OpenAI_LLM(api_key="sk-test", model="gpt-3.5-turbo")
    blob = pickle.dumps(llm)
    revived = pickle.loads(blob)

    # revived instance has equivalent state and a fresh client
    assert revived.model == llm.model
    assert isinstance(revived.client, _DummyOpenAI)
    assert revived.client.kwargs["api_key"] == "sk-test"


def test_llm_instantiation():
    # Since LLM is abstract, we'll instantiate a child class
    class DummyLLM(LLM):
        def _query(self, session: list):
            return "Mock response"

    llm = DummyLLM(api_key="fake", model="fake")
    assert llm.api_key == "fake"
    assert llm.model == "fake"


def test_llm_sample_solution_no_code_raises_exception():
    class DummyLLM(LLM):
        def _query(self, session: list):
            return "This has no code block"

    llm = DummyLLM(api_key="x", model="y")
    with pytest.raises(
        Exception
    ):  # uses the fallback `raise Exception("Could not extract...")`
        exec(llm.sample_solution([{"role": "client", "content": "test"}]), {}, {})


def test_llm_sample_solution_good_code():
    class DummyLLM(LLM):
        def _query(self, session: list):
            return "# Description: MyAlgo\n```python\nclass MyAlgo:\n  pass\n```"

    llm = DummyLLM(api_key="x", model="y")
    sol = llm.sample_solution([{"role": "client", "content": "test"}])
    assert sol.name == "MyAlgo"
    assert "class MyAlgo" in sol.code


def test_openai_llm_init():
    # We won't actually call OpenAI's API. Just ensure it can be constructed.
    llm = OpenAI_LLM(api_key="fake_key", model="gpt-3.5-turbo")
    assert llm.model == "gpt-3.5-turbo"


def test_ollama_llm_init():
    llm = Ollama_LLM(model="llama2.0")
    assert llm.model == "llama2.0"


def test_gemini_llm_init():
    llm = Gemini_LLM(api_key="some_key", model="gemini-2.0-flash")
    assert llm.model == "gemini-2.0-flash"


def test_claude_llm_init():
    llm = Claude_LLM(api_key="some_key", model="claude-3-haiku-20240307")
    assert llm.model == "claude-3-haiku-20240307"


def test_deepseek_llm_init(monkeypatch):
    _patch_openai(monkeypatch)
    llm = DeepSeek_LLM(api_key="ds-key")
    assert llm.model == "deepseek-chat"
    assert llm.client.kwargs.get("base_url") == "https://api.deepseek.com"


def _resource_exhausted(delay_secs: int = 2) -> Exception:
    """
    Build a faux `ResourceExhausted`-style exception carrying a `retry_delay`
    attr that the retry logic recognises.
    """
    err = Exception("429 ResourceExhausted")
    err.retry_delay = _dt.timedelta(seconds=delay_secs)
    return err


def _openai_rate_limit(retry_after: int = 2) -> Exception:
    response = httpx.Response(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        request=httpx.Request("POST", "http://test"),
    )
    return llm_mod.openai.RateLimitError("quota", response=response, body=None)


def _ollama_response_error(status: int = 429) -> Exception:
    return llm_mod.ollama.ResponseError("quota", status_code=status)


def test_gemini_llm_retries_then_succeeds(monkeypatch):
    """_query should sleep, retry once, then return the model reply."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")

    # -- stub out time.sleep so the test is instant
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    # First chats.create → chat.send_message raises; second returns text
    chat_fail = MagicMock()
    chat_fail.send_message.side_effect = _resource_exhausted(2)

    chat_ok = MagicMock()
    chat_ok.send_message.return_value = type("R", (), {"text": "OK-DONE"})

    fake_client = MagicMock()
    fake_client.chats.create.side_effect = [chat_fail, chat_ok]
    llm.client = fake_client

    reply = llm._query([{"role": "user", "content": "hello"}], max_retries=3)

    assert reply == "OK-DONE"
    assert fake_client.chats.create.call_count == 2  # 1 failure + 1 success
    slept.assert_called_once_with(3)  # 2 s + 1 s safety buffer


def test_gemini_llm_gives_up_after_max_retries(monkeypatch):
    """_query should bubble the error once max_retries is exceeded."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    chat_fail = MagicMock()
    chat_fail.send_message.side_effect = _resource_exhausted(1)

    fake_client = MagicMock()
    fake_client.chats.create.return_value = chat_fail
    llm.client = fake_client

    with pytest.raises(Exception):
        llm._query([{"role": "user", "content": "boom"}], max_retries=2)

    # It sleeps exactly `max_retries` times (raises on the next attempt)
    assert slept.call_count == 2


def test_openai_llm_retries_then_succeeds(monkeypatch):
    llm = OpenAI_LLM(api_key="fake", model="gpt-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="DONE"))]
    llm.client.chat.completions.create = MagicMock(
        side_effect=[_openai_rate_limit(2), ok]
    )

    reply = llm._query([{"role": "user", "content": "hi"}], max_retries=2)
    assert reply == "DONE"
    assert llm.client.chat.completions.create.call_count == 2
    slept.assert_called_once_with(2)


def test_openai_llm_gives_up(monkeypatch):
    llm = OpenAI_LLM(api_key="fake", model="gpt-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    llm.client.chat.completions.create = MagicMock(
        side_effect=[_openai_rate_limit(1), _openai_rate_limit(1)]
    )

    with pytest.raises(llm_mod.openai.RateLimitError):
        llm._query([{"role": "user", "content": "boom"}], max_retries=1)
    slept.assert_called_once_with(1)


def _anthropic_rate_limit(retry_after: int = 2) -> Exception:
    response = httpx.Response(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        request=httpx.Request("POST", "http://test"),
    )
    return llm_mod.anthropic.RateLimitError("quota", response=response, body=None)


def test_claude_llm_getstate_strips_client(monkeypatch):
    _patch_anthropic(monkeypatch)

    llm = Claude_LLM(api_key="sk-test", model="claude-test")
    state = llm.__getstate__()

    assert "client" not in state
    assert state["model"] == "claude-test"


def test_claude_llm_deepcopy_restores_client(monkeypatch):
    _patch_anthropic(monkeypatch)

    original = Claude_LLM(api_key="sk-test", model="claude-test", temperature=0.3)
    clone = copy.deepcopy(original)

    assert clone is not original
    assert clone.model == original.model
    assert clone.temperature == original.temperature
    assert isinstance(clone.client, _DummyAnthropic)
    assert clone.client is not original.client
    assert clone.client.kwargs["api_key"] == "sk-test"

    clone.temperature = 0.99
    assert original.temperature != clone.temperature


def test_claude_llm_pickle_roundtrip(monkeypatch):
    _patch_anthropic(monkeypatch)

    llm = Claude_LLM(api_key="sk-test", model="claude-test")
    blob = pickle.dumps(llm)
    revived = pickle.loads(blob)

    assert revived.model == llm.model
    assert isinstance(revived.client, _DummyAnthropic)
    assert revived.client.kwargs["api_key"] == "sk-test"


def test_claude_llm_retries_then_succeeds(monkeypatch):
    llm = Claude_LLM(api_key="fake", model="claude-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    ok = MagicMock()
    ok.content = [{"text": "DONE"}]
    llm.client.messages.create = MagicMock(side_effect=[_anthropic_rate_limit(2), ok])

    reply = llm._query([{"role": "user", "content": "hi"}], max_retries=2)
    assert reply == "DONE"
    assert llm.client.messages.create.call_count == 2
    slept.assert_called_once_with(2)


def test_claude_llm_gives_up(monkeypatch):
    llm = Claude_LLM(api_key="fake", model="claude-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    llm.client.messages.create = MagicMock(
        side_effect=[_anthropic_rate_limit(1), _anthropic_rate_limit(1)]
    )

    with pytest.raises(llm_mod.anthropic.RateLimitError):
        llm._query([{"role": "user", "content": "boom"}], max_retries=1)
    slept.assert_called_once_with(1)


def test_ollama_llm_retries_then_succeeds(monkeypatch):
    llm = Ollama_LLM(model="llama-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    monkeypatch.setattr(
        llm_mod.ollama.Client,
        "chat",
        MagicMock(
            side_effect=[_ollama_response_error(429), {"message": {"content": "OK"}}]
        ),
    )

    reply = llm._query([{"role": "u", "content": "hi"}], max_retries=2)
    assert reply == "OK"
    llm_mod.ollama.Client.chat.assert_called_with(
        model=llm.model,
        messages=[{"role": "user", "content": "hi\n"}],
        options = {}    # Options feature added.
    )
    slept.assert_called_once_with(10)


def test_ollama_llm_gives_up(monkeypatch):
    llm = Ollama_LLM(model="llama-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    monkeypatch.setattr(
        llm_mod.ollama.Client,
        "chat",
        MagicMock(side_effect=[_ollama_response_error(), _ollama_response_error()]),
    )

    with pytest.raises(llm_mod.ollama.ResponseError):
        llm._query([{"role": "u", "content": "boom"}], max_retries=1)
    slept.assert_called_once_with(10)


def test_dummy_llm():
    llm = Dummy_LLM(model="dummy-model")
    assert llm.model == "dummy-model"
    response = llm._query([{"role": "user", "content": "test"}])
    assert (
        len(response) == 919
    ), "Dummy_LLM should return a 919-character string, returned length: {}".format(
        len(response)
    )


def test_ollama_llm_query_forwards_kwargs(monkeypatch):
    llm = Ollama_LLM(model="llama-test")

    # Mock the underlying chat function
    mocked_chat = MagicMock(return_value={"message": {"content": "OK"}})
    monkeypatch.setattr(llm_mod.ollama.Client, "chat", mocked_chat)

    session = [{"role": "user", "content": "Hello"}]
    session_out = copy.deepcopy(session)    # Factoring long message combiner.
    session_out[0]["content"] += "\n"
    # Example kwargs to forward
    extra_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 512
    }

    # Call _query with session and kwargs
    result = llm._query(session, **extra_kwargs)

    # Assert the return value is as expected
    assert result == "OK"

    # Assert the underlying chat function was called with correct arguments
    mocked_chat.assert_called_once_with(
        model=llm.model,
        messages=session_out,
        options=extra_kwargs
    )

def test_gemini_query_forwards_kwargs(monkeypatch):
    llm = Gemini_LLM(api_key="none", model="llamea-test")
    base_config = copy.deepcopy(llm.generation_config)

    # Prepare mocks
    mocked_chat = MagicMock()
    mocked_chat.send_message.return_value.text = "OK"

    monkeypatch.setattr(llm_mod.genai.client.Chats , "create", MagicMock(return_value=mocked_chat))

    session = [{"role": "user", "content": "Hello"}]
    extra_kwargs = {"temperature": 0.7, "top_p": 0.9, "top_k": 12}

    # Expected config after merge
    expected_config = copy.deepcopy(base_config)
    expected_config.update(extra_kwargs)

    # Call method
    result = llm._query(session, **extra_kwargs)

    # Assertions
    assert result == "OK"
    llm.client.chats.create.assert_called_once_with(
        model="llamea-test",
        history=[],
        config=expected_config
    )
    mocked_chat.send_message.assert_called_once_with("Hello")

def test_openai_query_forwards_kwargs(monkeypatch):
    llm = OpenAI_LLM(api_key="whatup", model="llamea-test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OK"
    mocked_chat = MagicMock(return_value=mock_response)
    monkeypatch.setattr(llm.client.chat.completions , "create", mocked_chat)

    session = [{"role": "user", "content": "Hello"}]
    extra_kwargs = {"temperature": 0.7, "top_p": 0.9, "top_k": 12}

    result = llm._query(session, **extra_kwargs)
    local_temperature = copy.deepcopy(llm.temperature)

    assert result == "OK"
    temperature = llm.temperature
    if "temperature" in extra_kwargs:
        temperature = extra_kwargs["temperature"]
        extra_kwargs.pop("temperature")

    llm.client.chat.completions.create.assert_called_once_with(
        model = "llamea-test",
        messages=session,
        temperature=temperature,
        **extra_kwargs
    )

    assert llm.temperature == local_temperature

def sample_solution_passes_kwargs(monkeypatch):
    class LLM_mock(LLM):
        pass

    obj = LLM_mock(api_key="fake", model="llamea_test")
    obj.query = MagicMock(return_value="mocked message")

    session_messages = [{"role": "user", "content": "hello"}]
    kwargs = {"temperature": 0.7, "max_tokens": 500}

    obj.sample_solution(session_messages, **kwargs)

    obj.query.assert_called_once_with(session_messages, **kwargs)

def test_ollama_deep_copies(monkeypatch):
    llm1 = Ollama_LLM(model="llama-test")
    llm2 = copy.deepcopy(llm1)

    assert llm1.__getstate__() == llm2.__getstate__()

    mocked_chat1 = MagicMock(return_value={"message": {"content": "OK"}})
    monkeypatch.setattr(llm1.client, "chat", mocked_chat1)

    mocked_chat2 = MagicMock(return_value={"message": {"content": "OK"}})
    monkeypatch.setattr(llm2.client, "chat", mocked_chat2)

    session = [{"role": "user", "content": "Hello"}]
    session_out = copy.deepcopy(session)    # Factoring long message combiner.
    session_out[0]["content"] += "\n"

    extra_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 512
    }

    # Call _query with session and kwargs
    result = llm1._query(session, **extra_kwargs)

    # Assert the return value is as expected
    assert result == "OK"

    # Assert the underlying chat function was called with correct arguments
    mocked_chat1.assert_called_once_with(
        model=llm1.model,
        messages=session_out,
        options=extra_kwargs
    )

    mocked_chat2.assert_not_called()

@pytest.fixture
def fake_mlx_load(monkeypatch):
    llm = object()
    tokenizer = object()
    calls = []

    def fake_load(model, model_config=None):
        calls.append((model, model_config))
        print(calls)
        return llm, tokenizer

    monkeypatch.setattr(
        "iohblade.llm.load",
        fake_load
    )

    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "calls": calls,
    }

class AlwaysFailLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("LLM failure")

class FakeTokenier:
    def apply_chat_template(self, session, add_generation_prompt:bool):
        return session


def test_query_fails_only_after_max_tries(fake_mlx_load, monkeypatch):

    mock_tokenizer = FakeTokenier()
    failing_llm = AlwaysFailLLM()
    # Replace the llm returned by load()
    fake_mlx_load["llm"] = failing_llm

    llm = MLX_LM_LLM(
        model="dummy-model",
        max_tokens=10,
    )

    # monkeypatch the internal llm and dependencies
    monkeypatch.setattr('iohblade.llm.generate', failing_llm.generate)
    llm.tokenizer = mock_tokenizer

    result = llm._query([{"client": "hello"}], max_tries=3, add_generation_prompt=True)

    assert result == ""
    assert failing_llm.calls == 3

def test_deepcopy_shares_llm_and_tokenizer(fake_mlx_load):
    llm = MLX_LM_LLM(
        model="dummy-model",
        config={"foo": "bar"},
        max_tokens=123,
    )

    llm_copy = copy.deepcopy(llm)

    # Shared heavy objects
    assert llm.llm is llm_copy.llm
    assert llm.tokenizer is llm_copy.tokenizer

    # But normal attributes are copied
    assert llm is not llm_copy
    assert llm.config == llm_copy.config
    assert llm.max_tokens == llm_copy.max_tokens

def test_getstate_setstate_restores_llm_and_tokenizer(fake_mlx_load):
    llm = MLX_LM_LLM(
        model="dummy-model",
        config={"alpha": 0.5},
        max_tokens=42,
    )

    # Serialize + deserialize
    blob = pickle.dumps(llm)
    restored = pickle.loads(blob)

    # load() must be called again
    assert len(fake_mlx_load["calls"]) == 2

    # Config and parameters restored
    assert restored.model == llm.model
    assert restored.config == llm.config
    assert restored.max_tokens == llm.max_tokens

    # Heavy objects exist
    assert hasattr(restored, "llm")
    assert hasattr(restored, "tokenizer")

    assert restored.llm is fake_mlx_load["llm"]
    assert restored.tokenizer is fake_mlx_load["tokenizer"]

class AlwaysFailLMStudio:
    def __init__(self):
        self.calls = 0

    def respond(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("LMStudio failure")


class AlwaysSucceedLMStudio:
    def __init__(self, response="OK"):
        self.calls = 0
        self.response = response

    def respond(self, *args, **kwargs):
        self.calls += 1
        return self.response

@pytest.fixture
def fake_lms_llm(monkeypatch):
    calls = []

    lms = types.ModuleType("llamea.llm.lms")

    def factory(model):
        client = AlwaysFailLMStudio()
        calls.append((model, client))
        return client

    lms.llm = factory

    monkeypatch.setitem(sys.modules, "llamea.llm.lms", lms)

    import iohblade.llm
    monkeypatch.setattr(iohblade.llm, "lms", lms, raising=False)

    return {"calls": calls}

def test_lms_query_fails_only_after_max_tries(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config=None,
    )

    # Replace llm with failing instance explicitly
    failing = AlwaysFailLMStudio()
    llm.llm = failing

    result = llm._query(
        [{"role": "user", "content": "hello"}],
        max_tries=3,
    )

    assert result == ""
    assert failing.calls == 3

def test_lms_query_success_and_think_stripped(fake_lms_llm):
    llm = LMStudio_LLM(model="dummy", config=None)

    llm.llm = AlwaysSucceedLMStudio(
        response="<think>This client dare only greet me??? Fine, I'll be kind for now, *hmph*.</think>Hello!"
    )

    result = llm._query(
        [{"role": "user", "content": "hi"}],
        max_tries=2,
    )

    assert result == "Hello!"

def test_lms_deepcopy_shares_llm_instance(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config={"temperature": 0.2},
    )

    llm_copy = copy.deepcopy(llm)

    # Shared heavy object
    assert llm.llm is llm_copy.llm

    # Independent wrapper
    assert llm is not llm_copy
    assert llm.model == llm_copy.model
    assert llm.config == llm_copy.config

def test_lms_getstate_setstate_reloads_llm(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config={"top_p": 0.9},
    )

    # One load from __init__
    assert len(fake_lms_llm["calls"]) == 1

    blob = pickle.dumps(llm)
    restored = pickle.loads(blob)

    # Second load from __setstate__
    assert len(fake_lms_llm["calls"]) == 2

    assert restored.model == llm.model
    assert restored.config == llm.config
    assert hasattr(restored, "llm")