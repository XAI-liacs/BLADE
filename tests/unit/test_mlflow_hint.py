import os
import subprocess
import sys


def _env_after_import(env):
    code = "import iohblade, os; " "print(os.environ.get('MLFLOW_DISABLE_AGENT_HINT'))"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_import_silences_the_mlflow_agent_hint():
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_DISABLE_AGENT_HINT"}
    assert _env_after_import(env) == "1"


def test_user_setting_is_kept():
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="0")
    assert _env_after_import(env) == "0"
