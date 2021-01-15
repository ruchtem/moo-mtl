
from .pareto_mtl import ParetoMTLSolver
from .a_features import AFeaturesSolver

def solver_from_name(name, **kwargs):
    if name == 'ParetoMTL':
        return ParetoMTLSolver(**kwargs)
    elif name == 'proposed':
        return AFeaturesSolver(**kwargs)