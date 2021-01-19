import torch

from .pareto_mtl import ParetoMTLSolver
from .a_features import AFeaturesSolver
from .hypernetwork import HypernetSolver

def solver_from_name(name, **kwargs):
    if name == 'ParetoMTL':
        return ParetoMTLSolver(**kwargs)
    elif name == 'proposed':
        return AFeaturesSolver(**kwargs)
    elif name == 'base':
        return BaseSolver(**kwargs)
    elif name == 'hyper':
        return HypernetSolver(**kwargs)


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
