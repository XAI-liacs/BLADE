from iohblade.mcts_node import MCTS_Node


class MCTS_Prompts:
    """
    An extension of `iohblade.Problem` instanced that adds necessary prompt mutation in algorithm including
    `{i1, e1, e2, m1, m2, s1}`.
    """

    @classmethod
    def get_desctiption_prompt(cls, task_prompt: str, solution: MCTS_Node) -> str:
        prompt_content = (
            task_prompt
            + "\n"
            + "Following is the a Code implementing a heuristic algorithm with function name "
            + solution.name
            + " to solve the above mentioned problem.\n"
        )
        prompt_content += "\n\nCode:\n" + solution.code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    @classmethod
    def get_prompt_refine(cls, task_prompt, solution: MCTS_Node):
        prompt_content = (
            task_prompt
            + "\n"
            + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '"
            + solution.name
            + "' for implementing the heuristic algorithm.\n"
        )
        prompt_content += "\nDesign Idea:\n" + solution.description
        prompt_content += "\n\nCode:\n" + solution.code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    @classmethod
    def get_prompt_i1(cls, task_prompt, example_prompt, format_prompt):
        return "\n".join([task_prompt, example_prompt, format_prompt])

    @classmethod
    def get_prompt_e1(
        cls, task_prompt, example_prompt, format_prompt, indivs: list[MCTS_Node]
    ):
        prompt_indiv = (
            "\n".join(
                list(
                    map(
                        lambda ind: f"```python {ind.code}```\nObjective Value={ind.fitness}\n",
                        indivs,
                    )
                )
            )
            + "\n"
        )

        prompt_content = (
            "\n"
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n\n"
            + prompt_indiv
            + "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value, will adhering to following contract. \n"
            + task_prompt
            + example_prompt
            + format_prompt
        )

        return prompt_content

    @classmethod
    def get_prompt_e2(
        cls, task_prompt, example_prompt, format_prompt, indivs: list[MCTS_Node]
    ):
        prompt_indiv = "\n".join(
            list(
                map(
                    lambda item: f"""
No. {item[0] + 1} with description: {item[1].description}
and code
```python
{item[1].code}
```
Has objective Value {item[1].fitness}.
""",
                    enumerate(indivs),
                )
            )
        )

        prompt_content = (
            "\n"
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes and objective values as follows: \n\n"
            + prompt_indiv
            + f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n"
            f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, reply with a response adhering to following contract. Make sure only the final code is in code block. \
'"
            + task_prompt
            + example_prompt
            + format_prompt
        )
        return prompt_content

    @classmethod
    def get_prompt_m1(
        cls, task_prompt, example_prompt, format_prompt, indiv: MCTS_Node
    ):
        prompt_content = (
            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: "
            + indiv.description
            + "\n\
Code:\n\
"
            + indiv.code
            + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n"
            "Respond in adherance to following contract:"
        )
        prompt_content += "\n".join([task_prompt, example_prompt, format_prompt])
        return prompt_content

    @classmethod
    def get_prompt_m2(
        cls, task_prompt, example_prompt, format_prompt, indiv: MCTS_Node
    ):
        prompt_content = (
            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: "
            + indiv.description
            + "\n\
Code:\n\
"
            + indiv.code
            + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n"
            "Respond in adherance to following contract:"
        )
        prompt_content += "\n".join([task_prompt, example_prompt, format_prompt])
        return prompt_content

    @classmethod
    def get_prompt_s1(
        cls, task_prompt, example_prompt, format_prompt, indivs: list[MCTS_Node]
    ):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code and its objective value are: \n"
                + indivs[i].description
                + "\n"
                + indivs[i].code
                + "\n"
                + f"Objective value: {indivs[i].fitness}"
                + "\n\n"
            )

        prompt_content = (
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes and objective values as follows: \n\n"
            + prompt_indiv
            + f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n"
            "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence.\n"
            "Respond in adherance to following contract: "
        )
        prompt_content += "\n".join([task_prompt, example_prompt, format_prompt])
        return prompt_content
