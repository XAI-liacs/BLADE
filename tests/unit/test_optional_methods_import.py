import subprocess
import sys


def test_methods_import_without_the_methods_group():
    """MoEH needs codebleu (the optional `methods` group). Without
    them, importing iohblade.methods used to fail, so no method (not even
    LLaMEA) could be used after a plain `pip install iohblade`."""
    code = (
        "import sys\n"
        "for name in ('codebleu', 'eoh', 'reevo'):\n"
        "    sys.modules[name] = None  # make the import fail\n"
        "import iohblade.methods as m\n"
        "assert m.MoEH_Method is None and m.MutationType is None\n"
        "assert m.LLaMEA is not None and m.FunSearch is not None\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_psutil_is_a_declared_dependency():
    """iohblade.llm imports psutil, which used to come only with optional
    packages (smac, distributed)."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip() == "psutil" for d in deps)
