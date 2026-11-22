import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from iohblade import envs
from iohblade.problem import _eval_pythonpath

from .harness_problems import GoodProblem


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IOHBLADE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("IOHBLADE_ENV_CACHE", raising=False)
    return tmp_path / "cache" / "envs"


def test_cache_root_and_switch(cache_dir, monkeypatch):
    assert envs.cache_root() == cache_dir
    assert envs.cache_enabled()
    for value in ("0", "false", "No", "off"):
        monkeypatch.setenv("IOHBLADE_ENV_CACHE", value)
        assert not envs.cache_enabled()


def test_cache_root_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("IOHBLADE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    if os.name != "nt":
        assert envs.cache_root() == tmp_path / "iohblade" / "envs"


def test_env_key():
    a = envs.env_key(["numpy", "scipy>=1"], False)
    assert a == envs.env_key(["scipy>=1", "numpy "], False)  # order, spaces
    assert a != envs.env_key(["numpy"], False)
    assert a != envs.env_key(["numpy", "scipy>=1"], True)


def test_local_path_dependencies_are_not_cached(tmp_path):
    (tmp_path / "pkg").mkdir()
    assert envs.cacheable(["numpy", "scipy>=1.10"])
    assert not envs.cacheable([str(tmp_path / "pkg")])
    assert not envs.cacheable(["./pkg"])
    assert not envs.cacheable(["mypkg @ file:///x/y"])
    assert not envs.cacheable(["-e ./pkg"])
    assert not envs.cacheable(["--editable=pkg"])
    assert not envs.cacheable(["numpy", "-r requirements.txt"])


def test_cached_env_is_reused(cache_dir):
    first = envs.get_env([], cache=True)
    assert first.cached and not first.reused
    assert first.path.parent == cache_dir
    assert (first.path / envs.MARKER).is_file() and first.python.exists()
    second = envs.get_env([], cache=True)
    assert second.reused and second.path == first.path
    meta = json.loads((first.path / envs.MARKER).read_text())
    assert meta["dependencies"] == [] and "last_used" in meta

    listed = envs.list_envs()
    assert [e["key"] for e in listed] == [first.path.name]
    assert listed[0]["complete"] and listed[0]["size"] > 0
    assert envs.clear_cache() == 1
    assert envs.list_envs() == []
    assert envs.clear_cache() == 0


def test_incomplete_env_is_rebuilt(cache_dir):
    key = envs.env_key([], False)
    broken = cache_dir / key
    broken.mkdir(parents=True)
    (broken / "junk").write_text("half-built")
    info = envs.get_env([], cache=True)
    assert not info.reused and not (broken / "junk").exists()
    assert (broken / envs.MARKER).is_file()


def test_parallel_requests_build_once(cache_dir):
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(envs.get_env([], cache=True)))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.path for r in results}) == 1
    assert sum(not r.reused for r in results) == 1


def test_uncached_env_is_temporary(cache_dir, monkeypatch):
    monkeypatch.setenv("IOHBLADE_ENV_CACHE", "0")
    info = envs.get_env([], cache=True)
    assert not info.cached and cache_dir not in info.path.parents
    import shutil

    shutil.rmtree(info.path)


def test_problem_cleanup_keeps_cached_env_and_removes_private_one(cache_dir):
    cached = GoodProblem()
    cached._ensure_env()
    cached.cleanup()
    assert cached._env_path.exists()

    private = GoodProblem()
    private.env_cache = False
    private._ensure_env()
    assert cache_dir not in private._env_path.parents
    private.cleanup()
    assert not private._env_path.exists()


def _completed(code, out=""):
    return subprocess.CompletedProcess([], code, stdout=out, stderr=out)


def test_install_falls_back_to_pip(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(1, "uv broke") if cmd[0] == "/fake/uv" else _completed(0)

    monkeypatch.setattr(envs.shutil, "which", lambda name: "/fake/uv")
    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    envs._install(tmp_path / "python", ["numpy"])
    assert calls[0][:3] == ["/fake/uv", "pip", "install"]
    assert calls[1][1:4] == ["-m", "pip", "install"]


def test_build_tools_are_installed_first_and_on_path(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, env=None, **kwargs):
        calls.append((cmd, env["PATH"]))
        return _completed(0)

    monkeypatch.setattr(envs.shutil, "which", lambda name: None)
    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    python = tmp_path / "env" / "bin" / "python"
    envs._install(python, ["ioh==0.3.22", "SWIG", "smac==2.3.1"])
    assert [c[0][4:] for c in calls] == [["SWIG"], ["ioh==0.3.22", "smac==2.3.1"]]
    assert all(path.split(os.pathsep)[0] == str(python.parent) for _, path in calls)


def test_smac_benchmarks_declare_swig():
    from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX
    from iohblade.benchmarks.BBOB.mabbob import MA_BBOB

    for problem in (BBOB_SBOX(dims=[2]), MA_BBOB(dims=[2])):
        deps = problem.dependencies
        assert "swig" in deps and deps.index("swig") < len(deps)


def test_install_failure_reports_both_installers(monkeypatch, tmp_path):
    monkeypatch.setattr(envs.shutil, "which", lambda name: "/fake/uv")
    monkeypatch.setattr(
        envs.subprocess, "run", lambda cmd, **kw: _completed(1, f"{cmd[0]} failed")
    )
    with pytest.raises(envs.EnvironmentSetupError) as exc:
        envs._install(tmp_path / "python", ["nope"])
    text = str(exc.value)
    assert text.startswith("Failed to install the dependencies ['nope']")
    assert "uv pip install failed" in text and "pip install failed" in text
    assert isinstance(exc.value, RuntimeError)  # backward compatible


def test_install_without_uv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(envs.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        envs.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _completed(0)
    )
    envs._install(tmp_path / "python", ["numpy"])
    envs._install(tmp_path / "python", [])
    assert len(calls) == 1 and calls[0][1:3] == ["-m", "pip"]


def test_failed_cached_build_leaves_no_env(cache_dir, monkeypatch):
    def broken_install(python, deps):
        raise envs.EnvironmentSetupError("Failed to install the dependencies x")

    monkeypatch.setattr(envs, "_install", broken_install)
    with pytest.raises(envs.EnvironmentSetupError):
        envs.get_env(["something"], cache=True)
    assert not (cache_dir / envs.env_key(["something"], False)).exists()


def test_isolated_pythonpath(cache_dir):
    shim = envs.iohblade_only_path()
    assert (shim / "iohblade" / "__init__.py").exists()
    assert sorted(p.name for p in shim.iterdir()) == ["iohblade"]
    assert envs.iohblade_only_path() == shim  # reused

    problem = SimpleNamespace(isolated_pythonpath=True, extra_pythonpath=["/x"])
    parts = _eval_pythonpath(problem, "/host/site-packages").split(os.pathsep)
    assert parts[0] == str(shim)
    assert "/host/site-packages" not in parts
    assert os.path.abspath("/x") in parts


def test_isolated_problem_still_evaluates(cache_dir, tmp_path, monkeypatch):
    """Only iohblade and extra_pythonpath are importable in isolated mode."""
    import sys
    import textwrap

    from iohblade.solution import Solution

    module_dir = tmp_path / "code"
    module_dir.mkdir()
    (module_dir / "blade_isolated_mod.py").write_text(textwrap.dedent("""
            from iohblade.problem import Problem


            class IsolatedProblem(Problem):
                def evaluate(self, solution):
                    try:
                        import pytest  # installed for BLADE, not for evaluation
                        solution.set_scores(1.0)
                    except ImportError:
                        solution.set_scores(5.0)
                    return solution

                def test(self, solution):
                    return solution

                def to_dict(self):
                    return {}

                def get_config(self):
                    return {}
            """))
    monkeypatch.syspath_prepend(str(module_dir))
    import blade_isolated_mod

    try:
        problem = blade_isolated_mod.IsolatedProblem()
        problem.extra_pythonpath = [str(module_dir)]
        problem.eval_timeout = 120
        problem.isolated_pythonpath = True
        assert problem(Solution(code="x = 1", name="s")).fitness == 5.0

        # A problem from a module that is not on extra_pythonpath cannot be
        # imported in isolated mode: reported as a harness error.
        test_problem = GoodProblem()
        test_problem.isolated_pythonpath = True
        result = test_problem(Solution(code="x = 1", name="s"))
        assert "No module named 'tests'" in result.error
    finally:
        sys.modules.pop("blade_isolated_mod", None)
