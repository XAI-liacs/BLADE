"""Tests for HookedLLM on a tiny randomly initialised model (no downloads)."""

import copy
import json
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from iohblade.llm import HookedLLM, _find_decoder_layers
from iohblade.loggers.base import RunLogger

N_LAYERS = 3
D_MODEL = 32
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _tiny_tokenizer():
    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    vocab = {"[UNK]": 0, "[PAD]": 1}
    vocab.update({c: i + 2 for i, c in enumerate(chars)})
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("."), behavior="isolated"
    )
    tok.decoder = tokenizers.decoders.Fuse()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]"
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def _tiny_model(vocab_size):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    model = transformers.LlamaForCausalLM(config)
    model.generation_config.eos_token_id = None  # always generate max tokens
    return model


@pytest.fixture(scope="module", autouse=True)
def _hide_mocked_trackio():
    # test_trackio_logger installs a MagicMock "trackio" module at import
    # time; transformers' lazy imports probe it with find_spec and fail.
    import sys
    from unittest.mock import MagicMock

    fake = sys.modules.get("trackio")
    if isinstance(fake, MagicMock):
        del sys.modules["trackio"]
    yield
    if isinstance(fake, MagicMock):
        sys.modules["trackio"] = fake


@pytest.fixture(scope="module")
def tiny(_hide_mocked_trackio):
    tok = _tiny_tokenizer()
    return _tiny_model(len(tok)), tok


def _make(tiny, **kwargs):
    model, tok = tiny
    kwargs.setdefault("max_new_tokens", 6)
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("dtype", "float32")
    return HookedLLM("tiny-llama", hf_model=model, tokenizer=tok, **kwargs)


SESSION = [{"role": "user", "content": "write code"}]


def test_find_decoder_layers(tiny):
    model, _ = tiny
    assert len(_find_decoder_layers(model)) == N_LAYERS
    with pytest.raises(ValueError):
        _find_decoder_layers(torch.nn.Linear(2, 2))


def test_query_captures_all_layers(tiny):
    llm = _make(tiny)
    try:
        text = llm.query(SESSION)
        assert isinstance(text, str)
        rec = llm.records[-1]
        assert rec["solution_id"] is None
        assert rec["n_gen_tokens"] == 6
        assert rec["layers"] == list(range(N_LAYERS))
        for key in HookedLLM.ARRAY_KEYS:
            assert rec["arrays"][key].shape == (N_LAYERS, D_MODEL)
            assert rec["arrays"][key].dtype == np.float16
            assert np.isfinite(rec["arrays"][key]).all()
        assert rec["mean_entropy"] > 0
        assert rec["mean_logprob"] <= 0
        assert rec["prompt"] == SESSION and rec["response"] == text
    finally:
        _remove_hooks(llm)


def test_capture_layer_subset_and_disable(tiny):
    llm = _make(tiny, capture_layers=[1])
    try:
        llm.query(SESSION)
        assert llm.records[-1]["arrays"]["last"].shape == (1, D_MODEL)
        llm.capture = False
        llm.query(SESSION)
        assert "arrays" not in llm.records[-1]
    finally:
        _remove_hooks(llm)


def test_sample_solution_links_and_writes_to_run_dir(tiny, tmp_path):
    llm = _make(tiny)
    try:
        llm.set_logger(RunLogger(name="t", root_dir=str(tmp_path)))
        sol = llm.sample_solution(SESSION, parent_ids=["p1"])
        rec_id = sol.metadata["mi_record_id"]
        act_dir = tmp_path / "run-t" / "mi_activations"
        lines = (act_dir / "index.jsonl").read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["record_id"] == rec_id
        assert entry["solution_id"] == sol.id
        assert entry["parent_ids"] == ["p1"]
        assert "arrays" not in entry
        arrays = np.load(act_dir / entry["file"])
        assert set(arrays.files) == set(HookedLLM.ARRAY_KEYS)
        assert arrays["last"].shape == (N_LAYERS, D_MODEL)
        assert llm.records == []  # written to disk, not kept in memory
    finally:
        _remove_hooks(llm)


def test_activation_dir_without_logger(tiny, tmp_path):
    llm = _make(tiny, activation_dir=str(tmp_path / "acts"))
    try:
        llm.query(SESSION)
        assert (tmp_path / "acts" / "index.jsonl").exists()
    finally:
        _remove_hooks(llm)


def test_replay_matches_captured_states(tiny):
    llm = _make(tiny)
    try:
        llm.query(SESSION)
        rec = llm.records[-1]
        rep = llm.replay(rec["prompt"], rec["response"])
        n = rep["n_prompt_tokens"]
        assert n == rec["n_prompt_tokens"]
        assert rep["hidden"].shape[:2] == (N_LAYERS, len(rep["tokens"]))
        np.testing.assert_allclose(
            rep["hidden"][:, n - 1].astype(np.float32),
            rec["arrays"]["last"].astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        # gen_mean covers every generated token except the last one.
        gen = rep["hidden"][:, n : n + rec["n_gen_tokens"] - 1].astype(np.float32)
        np.testing.assert_allclose(
            gen.mean(1),
            rec["arrays"]["gen_mean"].astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    finally:
        _remove_hooks(llm)


def test_steering_changes_and_restores(tiny):
    llm = _make(tiny)
    try:
        llm.query(SESSION)
        base = llm.records[-1]
        llm.set_steering({1: np.ones(D_MODEL) * 50.0}, alpha=1.0)
        llm.query(SESSION)
        steered = llm.records[-1]
        assert steered["steering"]["layers"] == [1]
        # Generation-only steering leaves the prompt states untouched ...
        np.testing.assert_array_equal(steered["arrays"]["last"], base["arrays"]["last"])
        # ... but changes the states while generating.
        assert not np.allclose(
            steered["arrays"]["gen_mean"][1], base["arrays"]["gen_mean"][1]
        )
        llm.set_steering({1: np.ones(D_MODEL) * 50.0}, positions="all")
        llm.query(SESSION)
        assert not np.allclose(
            llm.records[-1]["arrays"]["last"][1], base["arrays"]["last"][1]
        )
        llm.clear_steering()
        llm.query(SESSION)
        assert llm.records[-1]["response"] == base["response"]
        assert llm.records[-1]["steering"] is None
        with pytest.raises(ValueError):
            llm.set_steering({1: np.ones(D_MODEL)}, positions="prompt")
    finally:
        _remove_hooks(llm)


def test_deepcopy_shares_model_and_getstate_drops_it(tiny):
    llm = _make(tiny)
    try:
        clone = copy.deepcopy(llm)
        assert clone.hf_model is llm.hf_model
        assert clone._lock is llm._lock
        clone.query(SESSION)
        assert len(clone.records) == 1 and llm.records == []
        state = llm.__getstate__()
        for key in HookedLLM._HEAVY:
            assert key not in state
        pickle.dumps(state)
    finally:
        _remove_hooks(llm)


def test_config_and_dict(tiny):
    llm = _make(tiny)
    try:
        cfg = llm.get_config()[0]
        assert cfg["model"] == "tiny-llama"
        assert cfg["config"]["capture_layers"] == list(range(N_LAYERS))
        assert llm.to_dict()["temperature"] == 0
    finally:
        _remove_hooks(llm)


def _remove_hooks(llm):
    # Each test builds a new HookedLLM on the shared module-scoped model.
    for h in llm._hook_handles:
        h.remove()
