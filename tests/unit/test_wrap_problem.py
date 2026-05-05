from iohblade import Solution, wrap_problem


def dummy_eval(problem, solution):
    solution.set_scores(1.23, "ok")
    return solution


def test_wrap_problem_creates_instance():
    p = wrap_problem(
        dummy_eval,
        name="Wrapped",
        eval_timeout=5,
        training_instances=[1],
        test_instances=[2],
        dependencies=["numpy"],
        imports="import numpy as np",
        task_prompt="Do task",
        example_prompt="Example code",
    )
    s = Solution()
    res = p.evaluate(s)
    assert res.fitness == 1.23
    assert p.name == "Wrapped"
    assert p.task_prompt == "Do task"
    assert p.example_prompt == "Example code"
    d = p.to_dict()
    assert d["name"] == "Wrapped"
