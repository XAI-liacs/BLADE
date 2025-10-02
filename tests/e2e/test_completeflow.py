import os
import shutil
from unittest.mock import MagicMock, patch
import shutil
import json
import pytest

from iohblade.experiment import Experiment, MA_BBOB_Experiment
from iohblade.llm import LLM, Gemini_LLM, OpenAI_LLM, Ollama_LLM, Dummy_LLM, Claude_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.problems import BBOB_SBOX, MA_BBOB, AutoML
from iohblade.methods import LLaMEA, EoH, RandomSearch, ReEvo
import ioh


@pytest.fixture
def cleanup_tmp_dir():
    # Creates a temporary directory for tests, yields its name, then cleans up
    dirname = "test_results"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    yield dirname
    # Cleanup
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


DUMMY_REPLY_OPT = """# Description: test algo
# Code: 
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=100, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        f_opt = np.inf
        x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < f_opt:
                f_opt = f
                x_opt = x
        return f_opt, x_opt
```
"""


LLM_CLASSES = [Gemini_LLM, OpenAI_LLM, Ollama_LLM, Dummy_LLM, Claude_LLM]
METHOD_CLASSES = [LLaMEA, EoH, ReEvo, RandomSearch]


def _make_llm(llm_class):
    # Supply api_key only when required by __init__
    kwargs = {"model": "test-" + str(llm_class)}
    if "api_key" in llm_class.__init__.__code__.co_varnames:
        kwargs["api_key"] = "test-key"
    return llm_class(**kwargs)


def test_full_matrix_experiment_optimization(cleanup_tmp_dir, monkeypatch):
    # Patch the provider-specific network call with a universal fake
    def fake_query(self, session_messages, **kwargs):
        return DUMMY_REPLY_OPT

    llm_class = Dummy_LLM
    budget = 6
    monkeypatch.setattr(llm_class, "_query", fake_query)
    llm = _make_llm(llm_class)

    # List containing function IDs we consider
    training_instances = [(1, 1)]
    test_instances = [(1, 2)]

    logger = ExperimentLogger(os.path.join(cleanup_tmp_dir, f"e2etest-results"))

    problems = []
    problems.append(
        BBOB_SBOX(
            training_instances=training_instances,
            test_instances=test_instances,
            dims=[2],
            budget_factor=100,
            eval_timeout=120,
            name=f"BBOB-f1",
            problem_type=ioh.ProblemClass.BBOB,
            full_ioh_log=False,
        )
    )
    problems.append(
        MA_BBOB(
            training_instances=[1],
            test_instances=[2],
            dims=[2],
            budget_factor=100,
            eval_timeout=120,
            name=f"MABBOB-i1",
        )
    )

    methods = []
    for method_class in METHOD_CLASSES:
        methods.append(method_class(llm=llm, budget=budget))
    # Now add LLaMEA with all LLMs
    for llm_class in LLM_CLASSES:
        monkeypatch.setattr(llm_class, "_query", fake_query)
        llm = _make_llm(llm_class)
        methods.append(
            LLaMEA(llm=llm, budget=budget, name=f"LLaMEA-{llm_class.__name__}")
        )
    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=2,
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=2,
    )  # normal run using 2 parallel jobs

    experiment()  # run the experiment

    # perform checks, we can check both the logger and the file system
    for p in problems:
        for m in methods:
            for seed in range(2):
                entry = logger._get_run_entry(m.name, p.name, seed)

                assert (
                    entry is not None
                ), f"Progress entry not created for {m.name}, {p.name}, seed {seed}"
                assert (
                    entry["evaluations"] == budget
                ), f"Entry {m.name}, {p.name}, seed {seed} did not use the full budget."
                assert (
                    entry["end_time"] is not None
                ), f"Entry {m.name}, {p.name}, seed {seed} did not finish."

    # check files
    exp_log_file = os.path.join(logger.dirname, "experimentlog.jsonl")
    progress_log_file = os.path.join(logger.dirname, "progress.json")
    with open(exp_log_file, "r") as f:
        exp_lines = [json.loads(l) for l in f]
    # check inner experiments
    for exp in exp_lines:
        log_dir = exp["log_dir"]
        assert os.path.isdir(os.path.join(logger.dirname, log_dir))
        log_dir_path = os.path.join(logger.dirname, log_dir, "log.jsonl")
        assert os.path.exists(log_dir_path)
        with open(log_dir_path, "r") as f2:
            exp_lines_log = [json.loads(l) for l in f2]
        assert (
            len(exp_lines_log) == budget
        ), f"Log {log_dir}, dit not contain {budget} solutions."
    with open(progress_log_file, "r") as f:
        progress = json.load(f)
    num_exps = len(problems) * len(methods) * 2  # 2 runs
    assert len(exp_lines) == num_exps
    assert progress["current"] == num_exps
    assert progress["total"] == num_exps
