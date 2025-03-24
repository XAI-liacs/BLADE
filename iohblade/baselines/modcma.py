import numpy as np
from modcma import c_maes


class ModularCMAES:
    """
    Baseline implementation of CMA-ES with active update.
    Can be extended later to cover all Modular CMAES options.
    """

    def __init__(self, budget=10000, dim=10, **kwargs):
        self.budget = budget
        self.dim = dim
        # Instantate a modules object
        self.modules = c_maes.parameters.Modules()
        self.modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
        self.modules.active = True
        # Create a settings object, here also optional parameters such as sigma0 can be specified
        self.settings = c_maes.parameters.Settings(dim, self.modules, **kwargs)
        # Create a parameters object
        self.parameters = c_maes.Parameters(self.settings)
        # Pass the parameters object to the ModularCMAES optimizer class
        self.cma = c_maes.ModularCMAES(self.parameters)

    def __call__(self, func):
        return self.cma.run(func)
