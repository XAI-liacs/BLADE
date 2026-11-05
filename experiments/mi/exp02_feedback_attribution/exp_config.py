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


def cf_source(tag: str) -> dict:
    """Source campaign of a counterfactual campaign ``tag``.

    ``03_prepare_counterfactuals.py`` writes ``cf_meta.json`` with the
    experiment 1 campaign the queries come from (``source_tag``), its model
    key and the steering layer. Campaigns prepared before that file existed
    (``cf``) used Qwen3-8B ``main``.
    """
    import json
    from pathlib import Path

    from common.paths import data_dir

    path = Path(data_dir(EXPERIMENT, "processed", tag)) / "cf_meta.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"source_tag": "main", "model": "qwen3-8b"}
