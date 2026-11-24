import copy
import inspect
import math
import random
import traceback
from typing import Any, Iterable, Optional

from iohblade.fitness import is_better
from iohblade.llm import LLM
from iohblade.mcts_node import MCTS_Node
from iohblade.method import Method
from iohblade.methods.mcts_ahd.prompts import MCTS_Prompts
from iohblade.methods.prompt_utils import task_context
from iohblade.problem import Problem
from iohblade.solution import Solution


# region Helper Functions:
def safe_max(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return max float ignoring None; None if all are None."""
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else None


def safe_min(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return min float ignoring None; None if all are None."""
    filtered = [v for v in values if v is not None]
    return min(filtered) if filtered else None


# endregion


class MCTS:
    def __init__(
        self,
        llm: LLM,
        problem: Problem,
        budget: int,
        lambda_0: float = 0.1,
        alpha: float = 0.5,
        maximisation: bool = True,
        max_children: int = 10,
        expansion_factor: int = 2,  # Referred to as k in (https://arxiv.org/pdf/2501.08603) algorithm 1.
    ):
        """
        MCTS method for solving a given `Problem` using LLMs.

        ## Args:
        `llm:iohblade.LLM` Any LLM model from `iohblade.llm.py`.\\
        `problem: iohblade.Problem`: An iohblade problem instance with appropriate prompts, and evaluation function for solving the problem.\\
        `buget: int`: Number of evaluations allowed for the method.\\
        `lambda_0: float`: A constant λ_0 used in UCT calculation.\\
        `alpha: float`: Intended as the progressive-widening coefficient of
        the paper. **Currently unused**: it is stored and logged, but
        progressive widening is decided by `MCTS_Node.is_fully_expanded`
        (`max_children` and the children's Q values), not by `alpha`.\\
        `maximisation: bool`: The direction of optimisation, setting it to false will lead to arg max(f), else arg min(f).\\
        `max_children: int`: A limit to maximum number of children any given node can have.\\
        `expansion_factor: int` Number of m1 and m2 mutations allowed during the expansion phase.

        ## Budget:
        `run()` makes at most `budget` calls to `problem(...)`: no node is
        generated (LLM call) or simulated once the remaining budget is used.
        """
        self.llm = llm
        self.problem = problem
        self.maximisation = maximisation
        self.lambda_0 = lambda_0
        # Not used by the search (see docstring); kept for configs and logs.
        self.alpha = alpha
        self.expansion_factor = expansion_factor
        self.eval_remain = budget

        self.budget = budget

        # Prefedined parameters.
        self.max_depth = 10
        self.epsilon = 1e-10
        self.discount_factor = 1
        self.q_min = None
        self.q_max = None
        self.rank_list = []
        self.e2_candidates: list[MCTS_Node] = []
        self.max_children = max_children

        # Instantiate the root node, with empty solution.
        solution = Solution()
        self.root = MCTS_Node(solution, approach="root")
        if maximisation:
            self.root.fitness = float("-inf")
        else:
            self.root.fitness = float("inf")

        self.best_solution: MCTS_Node = (
            self.root
        )  # Best solution node used as reference for e2 expansion.

    # region Node Generators.

    def _get_new_node(
        self, approach: str, relevant_nodes: list[MCTS_Node], depth: int
    ) -> MCTS_Node:
        """
        Given a generation, approcach in {i1, e1, e2, m1, m2, s1}, get a mcts node.
        ## Note
        Diffrerent approaces require different set of relevant nodes:\\
        `i1`: Needs empty list as relevant node, (initialisation method).\\
        `e1`: Needs sibling nodes of the root node, can only be used to generate root's children.\\
        `e2`: Needs a parent and a reference (Elite) node.\\
        `m1 and m2`: Needs the parent node.\\
        `s1`: Needs all the parent node, i.e. trace for root node to leaf node.

        ## Args:
            `approach: str`: Asserted to be one of the following {i1, e1, e2, m1, m2, s1}.
            `relevant_nodes: [MCTS_Node]`: A list of relevant `MCTS_Node`s, that can are in relationship with returning nodes as decribed in notes above.
            `depth: int`: Depth at which the current node is supposed to be added.

        ## Returns:
            `MCTS_Node`: Generate with LLM, a node with the code, and re-gererated description.

        ## Raises:
            `ValueError` raised if the approach string is not in [i1, e1, e2, m1, m2, s1].
            `NoCodeException` raised when LLM fails to return code in expected format (llm.sample_solution failure.)
            `Exception` All other interaction failures with LLM.
        """
        prompt = ""
        # The task with the context the problem adds in get_prompt(); the
        # example and format are added by the prompts themselves.
        task_prompt = task_context(self.problem)
        example_prompt = self.problem.example_prompt
        format_prompt = self.problem.format_prompt

        match approach:
            case "i1":
                prompt = MCTS_Prompts.get_prompt_i1(
                    task_prompt, example_prompt, format_prompt
                )
            case "e1":
                prompt = MCTS_Prompts.get_prompt_e1(
                    task_prompt,
                    example_prompt,
                    format_prompt,
                    relevant_nodes,
                    maximisation=self.maximisation,
                )
            case "e2":
                prompt = MCTS_Prompts.get_prompt_e2(
                    task_prompt,
                    example_prompt,
                    format_prompt,
                    relevant_nodes,
                    maximisation=self.maximisation,
                )
            case "m1":
                relevant_node = relevant_nodes[-1]
                prompt = MCTS_Prompts.get_prompt_m1(
                    task_prompt, example_prompt, format_prompt, relevant_node
                )
            case "m2":
                relevant_node = relevant_nodes[-1]
                prompt = MCTS_Prompts.get_prompt_m2(
                    task_prompt, example_prompt, format_prompt, relevant_node
                )
            case "s1":
                prompt = MCTS_Prompts.get_prompt_s1(
                    task_prompt,
                    example_prompt,
                    format_prompt,
                    relevant_nodes,
                    maximisation=self.maximisation,
                )
            case _:
                error_msg = f"Error enconutered {approach} method, which is not in expected list [i1, m1, m2, e1, e2, s1]."
                raise ValueError(error_msg)
        message = [{"role": "user", "content": prompt}]

        solution = None
        for i in range(5):  # Try upto 5 times.
            try:
                solution = self.llm.sample_solution(message)
                break
            except Exception as e:
                if i == 4:
                    raise e  # Forward error.
        if solution:
            mcts_node = MCTS_Node(solution, approach, depth=depth)
            refine_description_prompt = MCTS_Prompts.get_desctiption_prompt(
                task_prompt, mcts_node
            )
            message = [{"role": "user", "content": refine_description_prompt}]
            description = self.llm.query(message)
            mcts_node.description = description
            return mcts_node
        return MCTS_Node(Solution("error"), "error")

    def _get_m1_nodes(self, as_child_of_node: MCTS_Node) -> list[MCTS_Node]:
        """
        Gathers relevant nodes for permforming M1 Mutation, which requires just the parent.
        Adheres to returning [MCTS_Node] standard.

        ## Args:
            `as_child_of_node: MCTS_Node`: A node in the tree (below root), for which m1 nodes is being added as child.\\

        ## Returns:
            `[MCTS_Node]` A list of m1 relevant nodes, in thi case the parent.
        ## Raises:
            `ValueError`: If the `as_child_of_node` is root.
        """
        if as_child_of_node.is_root:
            raise ValueError("M1 cannot be used to generate a node at depth [0, 1].")
        return [as_child_of_node]

    def _get_m2_nodes(self, as_child_of_node: MCTS_Node) -> list[MCTS_Node]:
        """
        Gathers relevant nodes for permforming M2 Mutation, which requires just the parent.
        Adheres to returning [MCTS_Node] standard.

        ## Args:
            `as_child_of_node: MCTS_Node`: A node in the tree (below root), for which m2 nodes is being added as child.\\

        ## Raises:
            `ValueError`: If the `as_child_of_node` is root.
        """
        try:
            return self._get_m1_nodes(as_child_of_node)
        except ValueError:
            raise ValueError("M2 cannot be used to generate a node at depth [0, 1].")

    def _get_s1_nodes(self, as_child_of_node: MCTS_Node) -> list[MCTS_Node]:
        """
        Gathers relevant nodes for permforming S1 Mutation which is defined as a trace from root to as_child_of_node, specifically $(root, as_child_of_node]$.
        Adheres to returning [MCTS_Node] standard.

        ## Args:
            `as_child_of_node: MCTS_Node`: A node in the tree (below root), whose s1 relevant nodes need to be returned.\\

        ## Raises:
            `ValueError`: If the `as_child_of_node` is root or one of root's children.
        """
        if as_child_of_node.is_root:
            raise ValueError("S1 cannot be used to generate children of root node.")

        return_nodes = []
        current = as_child_of_node
        while not current.is_root:
            return_nodes.append(current)
            if current.parent:
                current = current.parent
        return return_nodes[::-1]  # Return trace from root to current node.

    def _get_e1_nodes(self, as_child_of_node: MCTS_Node) -> list[MCTS_Node]:
        """
        Return relevant nodes for e1 mutation, adheres to returning list[MCTS_Node]. The `for_node` must be a child of root node.
        Returns the sibling of the `for_node`.

        ## Args:
            `as_child_of_node: MCTS_Node`: Always the root node.

        ## Returns:
            `list[MCTS_Node]`: Children of the root node, which are going to be sibling to the node being generated.
        ## Raises:
            ValueError: If `as_child_of_node != root`.
        """
        if not as_child_of_node.is_root:
            raise ValueError(
                "E1 Mutation is only applicable on depth = 1, i.e. for root node."
            )
        return as_child_of_node.children

    def _get_e2_nodes(self, as_child_of_node: MCTS_Node) -> list[MCTS_Node]:
        """
        Return relevant nodes for adding E2 mutation node as child to `as_child_of_node`, where `for_node` is a MCTS_Node with min depth = 1.
        If there is no best solution yet (due to evaluation errors), reverts back to M1.

        ## Args:
        `as_child_of_node: MCTS_Node`: MCTS_Node in the tree at a minimum depth of 1.

        ## Returns:
        `[MCTS_Node]`: Relevant nodes for e2 mutation child if best_solution is exists: [as_child_of_node, best_solution], else returns [as_child_of_node].

        ## Raises:
        `ValueError`: If `for_node` is below depth 2.
        """
        if as_child_of_node.is_root:
            raise ValueError("E2 cannot be used to generate child of root node.")
        choice_pool = list(
            set(
                self.e2_candidates
                + ([self.best_solution] if not self.best_solution.is_root else [])
            )
        )
        relevant_nodes = random.sample(choice_pool, k=min(5, len(choice_pool)))
        return relevant_nodes

    # endregion

    # region MCTS Methods.
    def initialise(self, initial_node_count: int = 3):
        """
        Initialises the algorithm, by appending predefined number of nodes to the root node.
        Generate 1 i1 node, and n - 1 e1 node based on

        ## Args:
            `initial_node_count: int = 3` Number of initial nodes to be added to the tree.

        ## Returns:
            `None`: Inline algorithm, changes the data-structure, but returns nothing.
        """
        if initial_node_count < 1:
            return
        initial_node = self._get_new_node("i1", [], depth=1)
        self.root.add_child(initial_node)

        for _ in range(1, initial_node_count):
            node = self._get_new_node("e1", [initial_node], depth=1)
            self.root.add_child(node)

    def simulate(self, node: MCTS_Node):
        """
        Evaluate the node, and set it's fitness value based on the performance of the algorithm.

        ## Args:
        `node: MCTS_Node`: A node that is being simulated; to evaluate the performance.

        ## Returns:
            `MCTS_Node`: Inline algorithm, changes the data-structure, but returns nothing.

        """
        if self.eval_remain <= 0:
            # Safety net: never evaluate beyond the budget. The node keeps
            # Q = None, so the selection treats it as least attractive.
            return node
        self.eval_remain -= 1
        # Evaluate a copy without tree links: pickling a node carries its
        # parent and children, so the whole tree would otherwise be sent to
        # the evaluation subprocess. copy_attributes restores the links.
        detached = copy.copy(node)
        detached.parent = None
        detached.children = []
        new_node = self.problem(detached)
        if new_node is detached:
            # Evaluated in place: carry the results back to the tree node.
            for key, value in vars(detached).items():
                if key not in ("parent", "children"):
                    setattr(node, key, value)
            new_node = node
        else:
            new_node.copy_attributes(node)

        if math.isnan(new_node.fitness) or math.isinf(new_node.fitness):
            new_node.Q = None
            return new_node
        new_node.Q = new_node.fitness
        self.q_min = safe_min([self.q_min, new_node.Q])
        self.q_max = safe_max([self.q_max, new_node.Q])
        if is_better(
            new_node.fitness, self.best_solution.fitness, not self.maximisation
        ):
            self.best_solution = new_node
        return new_node

    def selection(
        self, max_new_nodes: int | None = None
    ) -> tuple[list[MCTS_Node], MCTS_Node]:
        """
        Iteratively pick fittest child from root node, to the leaf node, while adhering to progressive widening.

        ## Args:
            `max_new_nodes: int | None`: At most this many progressive
            widening nodes are generated (each one costs an evaluation);
            `None` means no limit.

        ## Returns:
        The function returns a tuple of [NCTS_Node], and NCTS_Node, which is to be interpreted as:\\
            `expanded_node : [MCTS_Node]`: A list of nodes added to tree adhering to `Progressive Widening`.\\
            `selected_node : MCTS_Node`: A leaf node that is selected for expansion.
        """
        current = self.root
        expanded_nodes = []
        while not current.is_leaf:
            # Expand node.
            if (
                max_new_nodes is None or len(expanded_nodes) < max_new_nodes
            ) and not current.is_fully_expanded(
                self.max_children, maximisation=self.maximisation
            ):
                if current.is_root:
                    relevant_nodes = self._get_e1_nodes(current)
                    node = self._get_new_node("e1", relevant_nodes, depth=1)
                else:
                    relevant_nodes = self._get_e2_nodes(current)
                    node = self._get_new_node(
                        "e2", relevant_nodes, depth=current.depth + 1
                    )
                current.add_child(node)
                expanded_nodes.append(node)

            # Find best child: uct() is higher-is-better in both directions.
            best_child = max(current.children, key=self.uct)

            current = best_child
        return expanded_nodes, current

    def expansion(self, on_node: MCTS_Node, max_nodes: int | None = None):
        """
        Impelements the expansion phase of the MCTS_AHD. "Apply expansion e2, s1, m1 (k times), m2 (k times), a total of 2k+2 new nodes added.
        Only implemented on leaf nodes, that is not root.

        ## Args:
        `on_node: MCTS_Node`: A MCTS_Node instance that is a leaf node, (non root) on which expansion is to be performed.
        `max_nodes: int | None`: Generate at most this many nodes (each one
        costs an evaluation), in the order m1, m2 (k times), s1, e2; `None`
        generates all 2k+2.

        ## Returns:
        `None`: Inline implementation that updates underlying Data Structure. Nothing to return.

        ## Raises:
        ValueError: if `on_node` is not leaf node or is a root node.
        """
        if (not on_node.is_leaf) or on_node.is_root:
            print(f"self.is_root {on_node.is_root}, self.is_leaf {on_node.is_leaf}")
            raise ValueError("Expansion only works on non-root leaf node.")

        approaches = ["m1", "m2"] * self.expansion_factor + ["s1", "e2"]
        if max_nodes is not None:
            approaches = approaches[: max(max_nodes, 0)]
        node_getters = {
            "m1": self._get_m1_nodes,
            "m2": self._get_m2_nodes,
            "s1": self._get_s1_nodes,
            "e2": self._get_e2_nodes,
        }
        for approach in approaches:
            relevant_nodes = node_getters[approach](on_node)
            node = self._get_new_node(approach, relevant_nodes, on_node.depth + 1)
            on_node.add_child(node)

    def backpropogate(self, node: MCTS_Node):
        """
        Backpropagate the subtree fitness from leaf node to root node, for determining next expoloration.

        ## Args:
        `node: MCTS_Node` : A node to iteratively start score back-propagation from.

        ## Returns:
            `None`: Function is an inplace mutation on MCTS_Node objects, and returns/throws nothing.
        """
        if node.Q is not None and node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()

        parent = node.parent
        while parent is not None:
            child_Qs = (child.Q for child in parent.children)
            best_child_Q = (
                safe_max(child_Qs) if self.maximisation else safe_min(child_Qs)
            )
            if parent.Q is not None and best_child_Q is not None:
                parent.Q = (
                    parent.Q * (1 - self.discount_factor)
                    + best_child_Q * self.discount_factor
                )
            parent.visit += 1
            if node.depth in [1, 2]:
                self.e2_candidates.append(node)
            parent = parent.parent

    # endregion
    # region UCT

    def uct(self, node: MCTS_Node) -> float:
        """
        Scores the provided node with a score, determining how likely it is to better optima on visiting current
        node again.

        ## Args:
            `node: MCTS_Node`: A non-root node that needs to be scored.\\
            `eval_remains: int`: Number of evaluation remaining for current optimisation process.

        ## Returns:
            `float`: The UCT score; higher is more promising in both
            optimisation directions (``-inf`` for nodes without a valid Q).
        """
        uct = 0
        exploration_constant = self.lambda_0 * max(self.eval_remain, 0)
        if node.Q is None or self.q_max is None or self.q_min is None:
            # Unevaluated or failed nodes are the least attractive; the
            # selection always takes the highest UCT.
            return float("-inf")
        if self.q_max > self.q_min:
            # Normalised exploitation term, higher is better in both
            # optimisation directions.
            if self.maximisation:
                uct = (node.Q - self.q_min) / (self.q_max - self.q_min)
            else:
                uct = (self.q_max - node.Q) / (self.q_max - self.q_min)
        if node.parent:
            return uct + (
                exploration_constant
                * (math.log(node.parent.visit + 1) ** 0.5)
                / node.visit
            )
        return 0

    # endregion

    # region Tree Printer.
    def print_tree(
        self,
        root,
        get_label=lambda n: f"({n.approach}) Node(id:{n.id}, Q={n.Q}, N={n.visit}, depth={n.depth})",
        prefix="",
    ):
        """
        Recursively print the MCTS tree to the console in a readable text format.
        """
        print(
            prefix + "└── " + get_label(root),
            f"(uct:{self.uct(root)})",
            f"{'*' if root.is_root else ''}",
        )
        child_count = len(root.children)
        for i, child in enumerate(root.children):
            is_last = i == child_count - 1
            branch = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.print_tree(child, get_label, new_prefix)

    # endregion

    # region Runner.
    def run(self):
        print("Started MCTS-AHD solver.")

        # Never generate more initial nodes than can be evaluated.
        self.initialise(min(3, self.eval_remain))

        print(f"Initialised with {len(self.root.children)} nodes.")

        for i, child in enumerate(self.root.children):
            print(f"\tEvaluating {child.id} node.")

            child = self.simulate(child)
            child.parent = self.root
            self.root.children[i] = child

            print(f"\t\tFitness {child.fitness}")
            print(f"\t\tFeedback {child.feedback}")

        for child in self.root.children:
            self.backpropogate(child)

        iteration = 1
        while self.eval_remain > 0:
            print(f"Q_min = {self.q_min}, Q_max = {self.q_max}.")
            self.print_tree(self.root)

            print(f"Iteratrion # {iteration}.")

            # Every generated node is evaluated once, so the number of new
            # nodes is capped at the remaining budget.
            progressive_widening_nodes, selected_node = self.selection(
                max_new_nodes=self.eval_remain
            )

            print(f"Selected Node: {selected_node}")

            remaining = self.eval_remain - len(progressive_widening_nodes)
            if remaining > 0:
                self.expansion(selected_node, max_nodes=remaining)
            expanded_nodes = selected_node.children

            print(
                f"Generating {len(progressive_widening_nodes)} progressive widening nodes, {len(expanded_nodes)} leaf nodes."
            )

            node_lists = [progressive_widening_nodes, expanded_nodes]
            for nodes in node_lists:
                for i, node in enumerate(nodes):
                    print(f"\tEvaluating {node.id} node.")

                    new_node = self.simulate(node)

                    # update the list we're iterating
                    if new_node is not node:
                        nodes[i] = new_node

                        # keep parent pointers consistent too
                        if node.parent is not None:
                            p = node.parent
                            try:
                                idx = p.children.index(node)
                            except ValueError:
                                # in case it was already replaced earlier
                                idx = (
                                    p.children.index(nodes[i])
                                    if nodes[i] in p.children
                                    else None
                                )

                            if idx is not None:
                                p.children[idx] = new_node
                            new_node.parent = p

                    print(f"\t\tFitness {nodes[i].fitness}.")
                    print(f"\t\tFeedback {nodes[i].feedback}")

            for node in (
                expanded_nodes + progressive_widening_nodes
            ):  # Make sure progressive widening nodes are handeled after expanded nodes.
                print(f"\tBackpropogating from node #{node.id}.")

                self.backpropogate(node)

            print(f"\tBudget remaining {self.eval_remain}.")

            iteration += 1

        print(f"Total iterations: {iteration}.")
        print(
            "--------------------------------------------------------------------------------------------------------------"
        )
        return self.best_solution

    # endregion


# region iohblade Wrapper.
class MCTS_Method(Method):
    def __init__(
        self,
        llm: LLM,
        budget: int,
        lambda_0: float = 0.1,
        alpha: float = 0.5,
        maximisation: bool = True,
        max_children: int = 5,
        expansion_factor: int = 2,
    ):
        """
        MCTS method wrapper for adding it to iohblade/methods.

        ## Args:
        `llm:iohblade.LLM` Any LLM model from `iohblade.llm.py`.\\
        `buget: int`: Number of evaluations allowed for the method.\\
        `lambda_0: float`: A constant λ_0 used in UCT calculation.\\
        `alpha: float`: Intended progressive-widening coefficient; currently
        unused (see `MCTS`), kept for configuration compatibility.\\
        `maximisation: bool`: The direction of optimisation, setting it to false will lead to arg max(f), else arg min(f).\\
        `max_children: int`: A limit to maximum number of children any given node can have.\\
        `expansion_factor: int` Number of m1 and m2 mutations allowed during the expansion phase.
        """
        super().__init__(llm, budget, name="MCTS_AHD")
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.maximisation = maximisation
        self.max_children = max_children
        self.expansion_factor = expansion_factor
        sig = inspect.signature(self.__init__)
        self.init_params = {
            k: getattr(self, k)
            for k in sig.parameters
            if k not in ("self", "name", "budget", "llm")
        }

    def __call__(self, problem: Problem):
        """
        Executes search using MCTS_AHD optimiser.

        Returns:
            Solution: The best solution found.
        """

        # Get optimisation direction/ set default as maximisaiton.
        minimisation = getattr(problem, "minimisation", False)
        self.maximisation = not minimisation

        self.mcts_instance = MCTS(
            self.llm,
            problem,
            self.budget,
            self.lambda_0,
            self.alpha,
            self.maximisation,
            self.max_children,
            self.expansion_factor,
        )
        try:
            return self.mcts_instance.run()
        except Exception as e:
            print("Some error occured, best solution till now V")
            print(traceback.print_exc(), e, "-----------", sep="\n")
            return self.mcts_instance.best_solution

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        kwargs = dict(self.init_params)
        return {
            "method_name": self.name if self.name != None else "MCTS_AHD",
            "budget": self.budget,
            "kwargs": kwargs,
        }

    # endregion

    # region Database Logger:
    def get_config(self) -> dict[str, Any]:
        config = {
            "lambda_0": self.lambda_0,
            "alpha": self.alpha,
            "max_children": self.max_children,
            "expansion_factor": self.expansion_factor,
            "budget": self.budget,
            "minimisation": not self.maximisation,
        }

        return {
            "name": "MCTS_AHD",
            "source": "https://github.com/XAI-liacs/BLADE/tree/main/iohblade/methods/mcts_ahd",
            "config": config,
        }


# endregion

if __name__ == "__main__":
    from iohblade.llm import Dummy_LLM

    llm = Dummy_LLM()
    mcts_ahd = MCTS_Method(
        llm,
        10,
    )
    config = mcts_ahd.get_config()
    for key, value in config.items():
        print(
            f"=============================={key}============================================="
        )
        print(value)
