"""LLaMEA's feature-guided mutation needs xgboost, optional since LLaMEA 1.5.1."""

import importlib.util

import pytest

from iohblade.config import ConfigError, load_plan
from iohblade.llm import Dummy_LLM
from iohblade.methods import LLaMEA

REAL_FIND_SPEC = importlib.util.find_spec


def _without_xgboost(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "xgboost" else REAL_FIND_SPEC(name, *a, **k)
        ),
    )


def test_llamea_works_without_xgboost(monkeypatch):
    _without_xgboost(monkeypatch)
    method = LLaMEA(Dummy_LLM("d"), budget=2)
    assert "feature_guided_mutation" not in method.kwargs


def test_feature_guided_mutation_without_xgboost_fails_early(monkeypatch):
    _without_xgboost(monkeypatch)
    with pytest.raises(ImportError, match="--group feature-guidance"):
        LLaMEA(Dummy_LLM("d"), budget=2, feature_guided_mutation=True)


def test_feature_guided_mutation_with_xgboost(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            object() if name == "xgboost" else REAL_FIND_SPEC(name, *a, **k)
        ),
    )
    method = LLaMEA(Dummy_LLM("d"), budget=2, feature_guided_mutation=True)
    assert method.kwargs["feature_guided_mutation"] is True


def test_validate_reports_missing_xgboost(monkeypatch, tmp_path):
    _without_xgboost(monkeypatch)
    config = tmp_path / "experiment.yml"
    config.write_text("""
version: 1
experiment: {output_dir: out, runs: 1, budget: 2}
llms: {d: {provider: dummy}}
methods:
  - {type: llamea, llm: d, params: {feature_guided_mutation: true}}
problems: [{type: circle_packing}]
""")
    with pytest.raises(ConfigError, match="feature-guidance"):
        load_plan(str(config), env={})
