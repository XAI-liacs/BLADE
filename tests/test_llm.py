import pytest
from unittest.mock import MagicMock
from iohblade.llm import LLM, OpenAI_LLM, Ollama_LLM, Gemini_LLM, NoCodeException

def test_llm_instantiation():
    # Since LLM is abstract, we'll instantiate a child class
    class DummyLLM(LLM):
        def query(self, session: list):
            return "Mock response"

    llm = DummyLLM(api_key="fake", model="fake")
    assert llm.api_key == "fake"
    assert llm.model == "fake"

def test_llm_sample_solution_no_code_raises_exception():
    class DummyLLM(LLM):
        def query(self, session: list):
            return "This has no code block"

    llm = DummyLLM(api_key="x", model="y")
    with pytest.raises(Exception):  # uses the fallback `raise Exception("Could not extract...")`
        exec(llm.sample_solution([{"role": "client", "content": "test"}]), {}, {})

def test_llm_sample_solution_good_code():
    class DummyLLM(LLM):
        def query(self, session: list):
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
