import torch

from .pareto_mtl import ParetoMTLSolver
from .a_features import AFeaturesSolver
from .hypernetwork import HypernetSolver

def solver_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLSolver(**kwargs)
    elif method == 'afeature':
        return AFeaturesSolver(**kwargs)
    elif method == 'SingleTask':
        return BaseSolver(**kwargs)
    elif method == 'hyper':
        return HypernetSolver(**kwargs)
    else:
        raise ValueError("Unkown method {}".format(method))


class BaseSolver():

    def __init__(self, model, objectives, task=0, **kwargs):
        self.model = model
        self.objectives = objectives
        self.task = task

    
    def new_point(self, *args):
        pass


    def step(self, batch):

        batch.update(self.model(batch))
        loss = self.objectives[self.task](**batch)
        loss.backward()
    
    def eval_step(self, batch):
        with torch.no_grad():
            return[self.model(batch)]
