import pytest

from iohblade.problems import HLP


def test_hlp_valid_feature_file():
    problem = HLP(specific_high_level_features=["Basins"], add_info_to_prompt=True)
    assert problem.function_file.endswith("Basins.jsonl")
    assert "Basins" in problem.get_prompt()


def test_hlp_invalid_feature():
    with pytest.raises(ValueError):
        HLP(specific_high_level_features=["NotAFeature"])
