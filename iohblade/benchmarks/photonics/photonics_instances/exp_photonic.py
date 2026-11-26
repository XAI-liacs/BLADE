import os
import sys

import ioh
import numpy as np

from .problems.brag_mirror import brag_mirror
from .problems.ellipsometry_inverse import ellipsometry
from .problems.grating2D import grating2D
from .problems.plasmonic_nanostructure import plasmonic_nanostructure
from .problems.sophisticated_antireflection_design import (
    sophisticated_antireflection_design,
)

# Seed of the random reference structure of the ellipsometry problem. It is
# fixed so that every (re)built instance has the same target spectrum.
ELLIPSOMETRY_REFERENCE_SEED = 42

# Descriptions of the instances that get_photonic_instance builds. They
# describe the physics only: no benchmark or instance names and no known
# optimal structure (the hints are in algorithmic_insights).
problem_descriptions = {
    "bragg": "The task is to maximize the reflectivity at a wavelength of 600 nm of a multilayer structure of 10 layers, in air and on a substrate with refractive index 1.8, that alternate between two materials with refractive indices 1.4 and 1.8; func returns 1 - reflectivity, which is minimized. The variables are the 10 layer thicknesses, each between 0 and about 214 nm (half the wavelength in the material with index 1.4). The landscape has many local minima. ",
    "ellipsometry": "The task is to retrieve the permittivity and the thickness of a single unknown layer on a gold substrate, in air, from its known ellipsometric response. The optimization minimizes the difference between the calculated and the reference ellipsometric ratio r_p / r_s (the sum of the absolute differences) for 100 wavelengths between 400 and 800 nm at a fixed angle of incidence of 40 degrees. The 2 variables are the permittivity of the layer (1.1 to 3) and its thickness (50 to 150 nm). ",
    "photovoltaic": "The task is to design an antireflective coating of 10 layers that alternate between materials with permittivities 2 and 3, on a 30,000 nm thick amorphous silicon layer, to maximize the short-circuit current of the solar cell (the absorption in the silicon) for wavelengths from 375 to 750 nm under the solar spectrum; func returns 1 - efficiency, which is minimized. The variables are the 10 layer thicknesses, each between 30 and 250 nm. ",
}
algorithmic_insights = {
    "bragg": "For this problem, the optimization landscape contains multiple local minima due to the wave nature of the problem. And periodic solutions are known to provide near-optimal results, suggesting the importance of leveraging constructive interference principles. Here are some suggestions for designing algorithms: 1. Use global optimization algorithms like Differential Evolution (DE) or Genetic Algorithms (GA) to explore the parameter space broadly. 2. Symmetric initialization strategies (e.g., Quasi-Oppositional DE) can improve exploration by evenly sampling the search space. 3. Algorithms should preserve modular characteristics in solutions, as multilayer designs often benefit from distinct functional blocks. 4. Combine global methods with local optimization (e.g., BFGS) to fine-tune solutions near promising regions. 5. Encourage periodicity in solutions via tailored cost functions or constraints. ",
    "ellipsometry": "This problem has small parameter space with fewer variables (thickness and refractive index), and the cost function is smooth and relatively free of noise, making it amenable to local optimization methods. Here are suggestions for designing algorithms: 1. Use local optimization algorithms like BFGS or Nelder-Mead, as they perform well in low-dimensional, smooth landscapes. 2. Uniform sampling across the parameter space ensures sufficient coverage for initial guesses. 3. Utilize fast convergence algorithms that can quickly exploit the smooth cost function landscape. 4. Iteratively adjust bounds and constraints to improve parameter estimates once initial solutions are obtained. ",
    "photovoltaic": "This problem is a challenging high-dimensional optimization problem with noisy cost functions due to the realistic solar spectrum, and it requires maximizing absorption while addressing trade-offs between reflectance and interference effects. Here are the suggestions for designing algorithms: 1. Combine global methods (e.g., DE, CMA-ES) for exploration with local optimization for refinement. 2. Use consistent benchmarking and convergence analysis to allocate computational resources effectively. 3. Encourage algorithms to detect and preserve modular structures (e.g., layers with specific roles like anti-reflective or coupling layers). 4. Gradually increase the number of layers during optimization to balance problem complexity and computational cost. 5. Integrate robustness metrics into the cost function to ensure the optimized design tolerates small perturbations in layer parameters. ",
}


def get_photonic_instance(
    problem_name="bragg", reference_seed=ELLIPSOMETRY_REFERENCE_SEED
):
    """Build the IOH problem for a photonic benchmark.

    Args:
        problem_name (str): "bragg", "ellipsometry" or "photovoltaic".
        reference_seed (int): Seed of the random reference structure of the
            ellipsometry problem (unused by the other problems).

    Returns:
        The wrapped IOH problem.
    """
    if problem_name == "bragg":
        # ------- define "mini-bragg" optimization problem
        nb_layers = 10  # number of layers of full stack
        target_wl = 600.0  # nm
        mat_env = 1.0  # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(
            prob, name="brag_mirror", optimization_type=ioh.OptimizationType.MIN
        )
        problem = ioh.get_problem("brag_mirror", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
    elif problem_name == "ellipsometry":
        # ------- define "ellipsometry" optimization problem
        mat_env = 1.0
        mat_substrate = "Gold"
        nb_layers = 1
        min_thick = 50  # nm
        max_thick = 150
        min_eps = 1.1  # permittivity
        max_eps = 3
        wavelengths = np.linspace(400, 800, 100)  # nm
        angle = 40 * np.pi / 180  # rad
        prob = ellipsometry(
            mat_env,
            mat_substrate,
            nb_layers,
            min_thick,
            max_thick,
            min_eps,
            max_eps,
            wavelengths,
            angle,
            seed=reference_seed,
        )
        ioh.problem.wrap_real_problem(
            prob,
            name="ellipsometry",
            optimization_type=ioh.OptimizationType.MIN,
        )
        problem = ioh.get_problem("ellipsometry", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
    elif problem_name == "photovoltaic":
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 10
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(
            nb_layers, min_thick, max_thick, wl_min, wl_max
        )
        ioh.problem.wrap_real_problem(
            prob,
            name="sophisticated_antireflection_design",
            optimization_type=ioh.OptimizationType.MIN,
        )
        problem = ioh.get_problem(
            "sophisticated_antireflection_design", dimension=prob.n
        )
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
    return problem
