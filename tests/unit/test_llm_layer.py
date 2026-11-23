"""Regression tests for #205: LLM response parsing, providers and retries."""

import pickle
from types import SimpleNamespace

import httpx
import openai
import pytest

import iohblade.llm as llm_module
from iohblade.llm import LLM, Dummy_LLM, Gemini_LLM, OpenAI_LLM
from iohblade.methods.random_search import RandomSearch
from iohblade.problem import Problem
from iohblade.solution import Solution

NO_CODE = "Could not extract generated code"


class _Recording(LLM):
    """An LLM whose replies are scripted and whose calls are recorded."""

    def __init__(self, replies=("",), **kwargs):
        super().__init__("", "recording", **kwargs)
        self.replies = list(replies)
        self.calls = []

    def _query(self, session, **kwargs):
        self.calls.append((session, kwargs))
        return self.replies.pop(0) if self.replies else ""

    def get_config(self):
        return [{"model": self.model, "config": {}}]


# --- code and class extraction ---------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Run:\n```bash\npip install x\n```\nHere is the algorithm:\n```python\nclass A:\n    pass\n```",
        "```py\nclass A:\n    pass\n```",
        "```python \nclass A:\n    pass\n```",
        "```python\r\nclass A:\r\n    pass\r\n```\r\n",
        "Some text\n```\nclass A:\n    pass\n```",
        "```Python3\nclass A:\n    pass\n```",
    ],
)
def test_code_extraction_handles_common_formats(message):
    code = _Recording().extract_algorithm_code(message)
    assert code == "class A:\n    pass"


def test_code_extraction_prefers_python_over_untagged_block():
    message = "```\nsome output\n```\n```python\nclass A:\n    pass\n```"
    assert _Recording().extract_algorithm_code(message) == "class A:\n    pass"


def test_code_extraction_without_code_block():
    code = _Recording().extract_algorithm_code("I cannot do that.")
    assert NO_CODE in code


def test_code_extraction_inline_fence_fallback():
    message = "Code: ```python\nclass A:\n    pass\n```"
    assert _Recording().extract_algorithm_code(message) == "class A:\n    pass"


@pytest.mark.parametrize(
    "code, expected",
    [
        ("class A:\n    pass\n", "A"),
        (
            "class Particle:\n    pass\n\nclass PSO:\n"
            "    def __call__(self, f):\n        p = Particle()\n",
            "PSO",
        ),
        (
            "class PSO:\n    def __call__(self, f):\n        p = Particle()\n\n"
            "class Particle:\n    pass\n",
            "PSO",
        ),
        (
            "# helper subclass: see below\nclass Optimizer(object):\n    pass\n",
            "Optimizer",
        ),
        ("def f():\n    return 1\n", ""),
        ("class Broken(:\n  pass", "Broken"),
        ("class Broken:\n  def x(:\n", "Broken"),
    ],
)
def test_class_name_extraction(code, expected):
    assert _Recording().extract_classname(code) == expected


def test_custom_patterns_are_used():
    llm = _Recording(
        code_pattern=r"<code>(.*?)</code>",
        name_pattern=r"name=(\w+)",
        desc_pattern=r"DESC: (.*)",
    )
    assert llm.extract_algorithm_code("<code>x = 1</code>") == "x = 1"
    assert llm.extract_classname("name=Foo") == "Foo"
    assert llm.extract_algorithm_description("DESC: fast") == "fast"
    assert llm.to_dict()["code_pattern"] == r"<code>(.*?)</code>"


def test_sample_solution_names_the_main_class():
    reply = (
        "# Description: PSO\n```python\nclass Particle:\n    pass\n\n"
        "class PSO:\n    def __call__(self, f):\n        return Particle()\n```"
    )
    solution = _Recording([reply]).sample_solution([{"role": "user", "content": "x"}])
    assert solution.name == "PSO"
    assert solution.description == "PSO"


# --- query --------------------------------------------------------------------


def test_sample_solution_passes_kwargs_to_the_provider():
    llm = _Recording(["```python\nclass A: pass\n```"])
    llm.sample_solution([{"role": "user", "content": "x"}], temperature=0.2)
    assert llm.calls[0][1] == {"temperature": 0.2}


def test_query_without_kwargs_supports_old_style_providers():
    # Dummy_LLM._query takes no keyword arguments.
    assert "dummy" in Dummy_LLM().query([{"role": "user", "content": "x"}]).lower()


def test_prompt_tokens_are_counted(monkeypatch):
    logged = []
    logger = SimpleNamespace(
        log_conversation=lambda role, msg, cost, tokens: logged.append(tokens)
    )
    monkeypatch.setattr(
        llm_module, "count_string_tokens", lambda prompt, model: len(prompt)
    )
    llm = _Recording(["reply"])
    llm.set_logger(logger)
    llm.query([{"role": "user", "content": "hello"}])
    assert logged == [len("hello"), len("reply")]


def test_diff_mode_warns(monkeypatch):
    monkeypatch.setattr(LLM, "_warned_diff_mode", False, raising=False)
    with pytest.warns(UserWarning, match="diff_mode"):
        _Recording(["```python\nclass A: pass\n```"]).sample_solution(
            [{"role": "user", "content": "x"}], diff_mode=True
        )


# --- methods send valid roles -------------------------------------------------


class _OneShot(Problem):
    def __init__(self):
        super().__init__(name="one")
        self.task_prompt = "task"

    def __call__(self, solution, logger=None):
        return solution.set_scores(1.0)

    def evaluate(self, solution):
        return solution

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


def test_random_search_sends_user_role():
    llm = _Recording(["```python\nclass A: pass\n```"])
    RandomSearch(llm, budget=1)(_OneShot())
    session, _ = llm.calls[0]
    assert {m["role"] for m in session} == {"user"}


# --- OpenAI -------------------------------------------------------------------


def _openai_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("error", response=response, body=None)


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


def _openai_with(outcomes, monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: sleeps.append(s))
    llm = OpenAI_LLM("key", model="gpt-4o", temperature=0.8)
    completions = _FakeCompletions(outcomes)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions, sleeps


@pytest.mark.parametrize(
    "cls, status",
    [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.NotFoundError, 404),
    ],
)
def test_openai_does_not_retry_client_errors(cls, status, monkeypatch):
    llm, completions, sleeps = _openai_with([_openai_error(cls, status)], monkeypatch)
    with pytest.raises(cls):
        llm._query([{"role": "user", "content": "x"}])
    assert len(completions.calls) == 1 and sleeps == []


def test_openai_retries_keep_temperature_and_parse_retry_after(monkeypatch):
    outcomes = [
        _openai_error(openai.RateLimitError, 429, {"Retry-After": "1.5"}),
        _openai_error(openai.InternalServerError, 500),
        "done",
    ]
    llm, completions, sleeps = _openai_with(outcomes, monkeypatch)
    assert llm._query([{"role": "user", "content": "x"}], temperature=0.1) == "done"
    assert [c["temperature"] for c in completions.calls] == [0.1, 0.1, 0.1]
    assert sleeps[0] == 1.5


# --- Gemini -------------------------------------------------------------------


class _FakeChats:
    def __init__(self, error=None):
        self.error = error
        self.history = None

    def create(self, model, history, config):
        self.history = history
        if self.error is not None:
            raise self.error
        return SimpleNamespace(send_message=lambda m: SimpleNamespace(text="ok"))


def _gemini_with(chats, monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: sleeps.append(s))
    llm = Gemini_LLM("key", model="gemini-2.0-flash")
    llm.client = SimpleNamespace(chats=chats)
    return llm, sleeps


def test_gemini_history_uses_the_sdk_format(monkeypatch):
    from google.genai import types

    chats = _FakeChats()
    llm, _ = _gemini_with(chats, monkeypatch)
    session = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert llm._query(session) == "ok"
    assert [h["role"] for h in chats.history] == ["user", "user", "model"]
    for item in chats.history:  # validates against the SDK's own model
        types.Content.model_validate(item)


def test_gemini_does_not_retry_invalid_requests(monkeypatch):
    from google.genai import errors

    chats = _FakeChats(error=errors.ClientError(400, {"error": {"message": "bad"}}))
    llm, sleeps = _gemini_with(chats, monkeypatch)
    with pytest.raises(errors.ClientError):
        llm._query([{"role": "user", "content": "x"}])
    assert sleeps == []


def test_gemini_retries_rate_limits(monkeypatch):
    from google.genai import errors

    chats = _FakeChats(error=errors.ClientError(429, {"error": {"message": "slow"}}))
    llm, sleeps = _gemini_with(chats, monkeypatch)
    with pytest.raises(errors.ClientError):
        llm._query([{"role": "user", "content": "x"}], max_retries=2)
    assert len(sleeps) == 2


# --- pickling of local-model LLMs -----------------------------------------------


def test_mlx_getstate_does_not_strip_the_live_model():
    obj = llm_module.MLX_LM_LLM.__new__(llm_module.MLX_LM_LLM)
    obj.__dict__.update(model="m", config=None, llm="LLM", tokenizer="TOK")
    state = obj.__getstate__()
    assert "llm" not in state and "tokenizer" not in state
    assert obj.llm == "LLM" and obj.tokenizer == "TOK"


def test_lmstudio_setstate_without_lmstudio(monkeypatch):
    monkeypatch.setattr(llm_module, "lms", object)
    obj = llm_module.LMStudio_LLM.__new__(llm_module.LMStudio_LLM)
    obj.__setstate__({"model": "m", "config": None})
    assert obj.llm is None
