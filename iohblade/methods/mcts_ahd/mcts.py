import math
import random
import inspect

from .mcts_node import MCTS_Node
from .prompts import MCTS_Prompts

from iohblade import Solution, Problem, LLM
from iohblade.method import Method

class MCTS:
    def __init__(self,
                 llm: LLM,
                 problem: Problem,
                 budget: int,
                 lambda_0: float = 0.1,
                 alpha: float = 0.5,
                 maximisation: bool = True,
                 max_children: int = 5,
                 expansion_factor: int = 2           #Referred to as k in (https://arxiv.org/pdf/2501.08603) algorithm 1.
    ):
        """
        MCTS method for solving a given `Problem` using LLMs.

        ## Args:
        `llm:iohblade.LLM` Any LLM model from `iohblade.llm.py`.\\
        `problem: iohblade.Problem`: An iohblade problem instance with appropriate prompts, and evaluation function for solving the problem.\\
        `buget: int`: Number of evaluations allowed for the method.\\
        `lambda_0: float`: A constant λ_0 used in UCT calculation.\\
        `alpha: float`: Expansion coefficient for progressive widening the tree.\\
        `maximisation: bool`: The direction of optimisation, setting it to false will lead to arg max(f), else arg min(f).\\
        `max_children: int`: A limit to maximum number of children any given node can have.\\
        `expansion_factor: int` Number of m1 and m2 mutations allowed during the expansion phase.
        """
        self.llm = llm
        self.problem = problem
        self.maximisation = maximisation
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.expansion_factor = expansion_factor
        self.eval_remain = budget
        
        self.budget = budget
        
        #Prefedined parameters.
        self.max_depth = 10
        self.epsilon = 1e-10
        self.discount_factor = 1
        self.q_min = 0 if maximisation else -1e4
        self.q_max = -1e4 if maximisation else 0      #-10,000
        self.rank_list = []
        self.e1_candidates: list[MCTS_Node] = []
        self.max_children = max_children

        #Instantiate the root node, with empty solution.
        solution = Solution()
        self.root = MCTS_Node(solution, approach='root')

        self.best_solution: MCTS_Node = self.root          #Best solution node used as reference for e2 expansion.

    
    def _get_new_node(self, approach: str, relevant_nodes: list[MCTS_Node], depth: int) -> MCTS_Node:
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
        task_prompt = self.problem.task_prompt
        example_prompt = self.problem.example_prompt
        format_prompt = self.problem.format_prompt

        match approach:
            case "i1":
                prompt = MCTS_Prompts.get_prompt_i1(task_prompt, example_prompt, format_prompt)
            case "e1":
                prompt = MCTS_Prompts.get_prompt_e1(task_prompt, example_prompt, format_prompt, relevant_nodes)
            case "e2":
                prompt = MCTS_Prompts.get_prompt_e2(task_prompt, example_prompt, format_prompt, relevant_nodes)
            case "m1":
                relevant_node = relevant_nodes[-1]
                prompt = MCTS_Prompts.get_prompt_m1(task_prompt, example_prompt, format_prompt, relevant_node)
            case "m2":
                relevant_node = relevant_nodes[-1]
                prompt = MCTS_Prompts.get_prompt_m2(task_prompt, example_prompt, format_prompt, relevant_node)
            case "s1":
                prompt = MCTS_Prompts.get_prompt_s1(task_prompt, example_prompt, format_prompt, relevant_nodes)
            case _:
                error_msg = f"Error enconutered {approach} method, which is not in expected list [i1, m1, m2, e1, e2, s1]."
                raise ValueError(error_msg)
        message = [{"role": "client", "content": prompt}]
        
        solution = None
        for i in range(5):      #Try upto 5 times.
            try:
                solution = self.llm.sample_solution(message)
                break
            except Exception as e:
                if i == 4:
                    raise e     # Forward error.
        if solution:
            mcts_node = MCTS_Node(solution, approach, depth=depth)
            refine_description_prompt = MCTS_Prompts.get_desctiption_prompt(task_prompt, mcts_node)
            message = [{"role": "client", "content": refine_description_prompt}]
            descrpition = self.llm.query(message)
            mcts_node.description = descrpition
            return mcts_node
        return MCTS_Node(Solution("error"), 'error')


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
                raise ValueError("M1 cannot be used to generate a node at depth 1 or below.")
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
            raise ValueError("M2 cannot be used to generate a node at depth 1 or below.")

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
            else:
                break                   #Extra safety.
        return return_nodes[::-1]       #Return trace from root to current node.

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
            raise ValueError("E1 Mutation is only applicable on depth = 1, i.e. for root node.")
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
        relevant_nodes = random.sample(self.e1_candidates, k=min(5, len(self.e1_candidates)))
        if not self.best_solution.is_root:
            relevant_nodes.append(self.best_solution)
        return []       # Never runs.


    def initialise(self, initial_node_count:int = 3):
        """
        Initialises the algorithm, by appending predefined number of nodes to the root node.
        Generate 1 i1 node, and n - 1 e1 node based on 
        
        ## Args:
            `initial_node_count: int = 3` Number of initial nodes to be added to the tree.

        ## Returns:
            `None`: Inline algorithm, changes the data-structure, but returns nothing.
        """
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
        """
        self.eval_remain -= 1
        self.problem.evaluate(node)
        if not self.maximisation:           # The paper only refers to maximisation problem.
            node.fitness *= -1
            node.Q = node.fitness
        if self.best_solution.fitness < node.fitness:
            self.best_solution = node

    def selection(self) -> tuple[list[MCTS_Node], MCTS_Node]:
        """
        Iteratively pick fittest child from root node, to the leaf node, while adhering to progressive widening.

        ## Args:
            `None`: No arguements are required.

        ## Returns:
        The function returns a tuple of [NCTS_Node], and NCTS_Node, which is to be interpreted as:\\
            `expanded_node : [MCTS_Node]`: A list of nodes added to tree adhering to `Progressive Widening`.\\
            `selected_node : MCTS_Node`: A leaf node that is selected for expansion.
        """
        current = self.root
        expanded_nodes = []
        while not current.is_leaf:
            #Expand node.
            if not current.is_fully_expanded(self.max_children):
                if current.is_root:
                    relevant_nodes = self._get_e1_nodes(current)
                    node = self._get_new_node("e1", relevant_nodes, depth=1)
                else:
                    relevant_nodes = self._get_e2_nodes(current)
                    node = self._get_new_node("e2", relevant_nodes, depth=current.depth + 1)
                current.add_child(node)
                expanded_nodes.append(node)
            
            #Find best child.
            best_child = current.children[0]
            for child in current.children:
                if self.uct(best_child) < self.uct(child):
                    best_child = child
            
            current = best_child
        return expanded_nodes, current

    def expansion(self, on_node: MCTS_Node):
        """
        Impelements the expansion phase of the MCTS_AHD. "Apply expansion e2, s1, m1 (k times), m2 (k times), a total of 2k+2 new nodes added.
        Only implemented on leaf nodes, that is not root.

        ## Args:
        `on_node: MCTS_Node`: A MCTS_Node instance that is a leaf node, (non root) on which expansion is to be performed.

        ## Returns:
        `None`: Inline implementation that updates underlying Data Structure. Nothing to return.

        ## Raises:
        ValueError: if `on_node` is not leaf node or is a root node.
        """
        if (not on_node.is_leaf) or on_node.is_root:
            print(f"self.is_root {on_node.is_root}, self.is_leaf {on_node.is_leaf}")
            raise ValueError("Expansion only works on non-root leaf node.")
        
        for _ in range(self.expansion_factor):
            relevant_nodes = self._get_m1_nodes(on_node)
            node = self._get_new_node('m1', relevant_nodes, on_node.depth + 1)
            on_node.add_child(node)

            relevant_nodes = self._get_m2_nodes(on_node)
            node = self._get_new_node('m2', relevant_nodes, on_node.depth + 1)
            on_node.add_child(node)

        relevant_nodes = self._get_s1_nodes(on_node)
        node = self._get_new_node('s1', relevant_nodes, on_node.depth + 1)
        on_node.add_child(node)

        relevant_nodes = self._get_e2_nodes(on_node)
        node = self._get_new_node('e2', relevant_nodes, on_node.depth + 1)
        on_node.add_child(node)
    
    def backpropogate(self, node: MCTS_Node):
        """
        Backpropagate the subtree fitness from leaf node to root node, for determining next expoloration.

        ## Args:
        `node: MCTS_Node` : A node to iteratively start score back-propagation from.

        ## Returns:
            `None`: Function is an inplace mutation on MCTS_Node objects, and returns/throws nothing.
        """
        if node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.Q)
        self.q_max = max(self.q_max, node.Q)
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = parent.Q * (1 - self.discount_factor) + best_child_Q * self.discount_factor
            parent.visit += 1
            if node.depth in [1, 2]:              #No Idea why this exist, commented till I figure out.
                self.e1_candidates.append(node)
            parent = parent.parent

    def uct(self, node: MCTS_Node) -> float:
        """
        Scores the provided node with a score, determining how likely it is to better optima on visiting current 
        node again.

        ## Args:
            `node: MCTS_Node`: A non-root node that needs to be scored.\\
            `eval_remains: int`: Number of evaluation remaining for current optimisation process.

        ## Returns:
            `None`: Inplace mutation function, which retuns or throws nothing.
        """
        exploration_constant = self.lambda_0 * self.eval_remain
        if node.parent:
            return (node.Q - self.q_min) / (self.q_max - self.q_min) + exploration_constant * (math.log(node.parent.visit + 1)) ** 0.5 / node.visit
        return 0

    def print_tree(self, root, get_label=lambda n: f"Node(id:{n.id}, pid={n.parent.id if n.parent else None}, Q={n.Q}, depth={n.depth})", prefix=""):
        """
        Recursively print the MCTS tree to the console in a readable text format.
        """
        print(prefix + "└── " + get_label(root))
        child_count = len(root.children)
        for i, child in enumerate(root.children):
            is_last = (i == child_count - 1)
            branch = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            # Only recurse, don't print twice
            self.print_tree(child, get_label, new_prefix)
    
    def run(self):
        print("Started MCTS-AHD solver.")
        self.initialise()
        print(f"Initialised with {len(self.root.children)} nodes.")
        for child in self.root.children:
            print(f"\tEvaluating {child.id} node.")
            self.simulate(child)
            print(f"\t\tFitness {child.fitness}")
            print(f"\t\tFeedback {child.feedback}")

        iteration = 1
        while self.eval_remain > 0:
            self.print_tree(self.root)
            print(f"Iteratrion # {iteration}.")
            progressive_widening_nodes, selected_node = self.selection()
            print(f"Selected Node: {selected_node}")
            self.expansion(selected_node)
            expanded_nodes = selected_node.children
            print(f"Generating {len(progressive_widening_nodes)} progressive widening nodes, {len(expanded_nodes)} leaf nodes.")
            for node in progressive_widening_nodes + expanded_nodes:
                print(f"\tEvaluating {node.id} node.")
                self.simulate(node)
                print(f"\t\tFitness {node.fitness}.")
                print(f"\t\tFeedback {node.feedback}")
            
            for node in expanded_nodes + progressive_widening_nodes:    #Make sure progressive widening nodes are handeled after expanded nodes.
                print(f"\tBackpropogating from node #{node.id}.")
                self.backpropogate(node)
            print(f"\tBudget remaining {self.eval_remain}.")
            iteration += 1
        print(f"Total iterations: {iteration}.")
        
        return self.best_solution
    
class MCTS_Method(Method):
    def __init__(self, 
                 llm: LLM,
                 budget: int,
                 lambda_0: float = 0.1,
                 alpha: float = 0.5,
                 maximisation:bool = True,
                 max_children:int = 5,
                 expansion_factor:int = 2,):
        """
        MCTS method wrapper for adding it to iohblade/methods.

        ## Args:
        `llm:iohblade.LLM` Any LLM model from `iohblade.llm.py`.\\
        `buget: int`: Number of evaluations allowed for the method.\\
        `lambda_0: float`: A constant λ_0 used in UCT calculation.\\
        `alpha: float`: Expansion coefficient for progressive widening the tree.\\
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
        self.init_params = {k: getattr(self, k) for k in sig.parameters if k not in ('self', 'name', 'budget')}

    def __call__(self, problem: Problem):
        """
        Executes search using MCTS_AHD optimiser.

        Returns:
            Solution: The best solution found.
        """
        self.mcts_instance = MCTS(
            self.llm,
            problem,
            self.budget,
            self.lambda_0,
            self.alpha,
            self.maximisation,
            self.max_children,
            self.expansion_factor
        )
        return self.mcts_instance.run()

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "MCTS_AHD",
            "budget": self.budget,
            "kwargs": self.init_params,
        }
