"""Configuration for experiment 2 (is the feedback used?)."""

EXPERIMENT = "exp02_feedback_attribution"
SOURCE_EXPERIMENT = "exp01_probing"  # recorded LLaMEA runs to analyse


def load_source_config():
    """Experiment 1's ``exp_config`` (both modules are named ``exp_config``)."""
    from importlib import util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / SOURCE_EXPERIMENT / "exp_config.py"
    spec = util.spec_from_file_location("source_exp_config", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
