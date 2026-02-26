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

problem_descriptions = {
    "bragg": "The Bragg mirror optimization aims to maximize reflectivity at a wavelength of 600 nm using a multilayer structure with alternating refractive indices (1.4 and 1.8). The structure's thicknesses are varied to find the configuration with the highest reflectivity. The problem involves two cases: one with 10 layers (minibragg) and another with 20 layers (bragg), with the latter representing a more complex inverse design problem. The known optimal solution is a periodic Bragg mirror, which achieves the best reflectivity by leveraging constructive interference. This case exemplifies challenges such as multiple local minima in the optimization landscape. ",
    "ellipsometry": "The ellipsometry problem involves retrieving the material and thickness of a reference layer by matching its reflectance properties using a known spectral response. The optimization minimizes the difference between the calculated and measured ellipsometric parameters for wavelengths between 400 and 800 nm and a fixed incidence angle of 40°. The parameters to be optimized include the thickness (30 to 250 nm) and refractive index (1.1 to 3) of the test layer. This relatively straightforward problem models a practical scenario where photonics researchers fine-tune a small number of parameters to achieve a desired spectral fit. ",
    "photovoltaic": "The photovoltaics problem optimizes the design of an antireflective multilayer coating to maximize the absorption in the active silicon layer of a solar cell. The goal is to achieve maximum short-circuit current in the 375 to 750 nm wavelength range. The structure consists of alternating materials with permittivities of 2 and 3, built upon a 30,000 nm thick silicon substrate. Three subcases with increasing complexity are explored, involving 10 layers (photovoltaics), 20 layers (bigphotovoltaics), and 32 layers (hugephotovoltaics). The optimization challenges include balancing high absorption with a low reflectance while addressing the inherent noise and irregularities in the solar spectrum. ",
}
algorithmic_insights = {
    "bragg": "For this problem, the optimization landscape contains multiple local minima due to the wave nature of the problem. And periodic solutions are known to provide near-optimal results, suggesting the importance of leveraging constructive interference principles. Here are some suggestions for designing algorithms: 1. Use global optimization algorithms like Differential Evolution (DE) or Genetic Algorithms (GA) to explore the parameter space broadly. 2. Symmetric initialization strategies (e.g., Quasi-Oppositional DE) can improve exploration by evenly sampling the search space. 3. Algorithms should preserve modular characteristics in solutions, as multilayer designs often benefit from distinct functional blocks. 4. Combine global methods with local optimization (e.g., BFGS) to fine-tune solutions near promising regions. 5. Encourage periodicity in solutions via tailored cost functions or constraints. ",
    "ellipsometry": "This problem has small parameter space with fewer variables (thickness and refractive index), and the cost function is smooth and relatively free of noise, making it amenable to local optimization methods. Here are suggestions for designing algorithms: 1. Use local optimization algorithms like BFGS or Nelder-Mead, as they perform well in low-dimensional, smooth landscapes. 2. Uniform sampling across the parameter space ensures sufficient coverage for initial guesses. 3. Utilize fast convergence algorithms that can quickly exploit the smooth cost function landscape. 4. Iteratively adjust bounds and constraints to improve parameter estimates once initial solutions are obtained. ",
    "photovoltaic": "This problem is a challenging high-dimensional optimization problem with noisy cost functions due to the realistic solar spectrum, and it requires maximizing absorption while addressing trade-offs between reflectance and interference effects. Here are the suggestions for designing algorithms: 1. Combine global methods (e.g., DE, CMA-ES) for exploration with local optimization for refinement. 2. Use consistent benchmarking and convergence analysis to allocate computational resources effectively. 3. Encourage algorithms to detect and preserve modular structures (e.g., layers with specific roles like anti-reflective or coupling layers). 4. Gradually increase the number of layers during optimization to balance problem complexity and computational cost. 5. Integrate robustness metrics into the cost function to ensure the optimized design tolerates small perturbations in layer parameters. ",
}


def get_photonic_instance(problem_name="bragg"):
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
