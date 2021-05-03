import torch
import numpy as np

from pymoo.model.problem import Problem

import multi_objective.utils as utils
from .base import BaseMethod


class ModelWrapper(Problem):

    def __init__(self, model, objectives, task_ids, device):
        self.model = model
        self.objectives = objectives
        self.task_ids = task_ids
        self.device = device
        self.data = None

        self.shapes = [p.shape for p in model.parameters() if p.requires_grad is True]
        self.lenghts = [np.prod(s) for s in self.shapes]

        n_var = sum(self.lenghts)
        super().__init__(
            n_var=n_var,
            n_obj=len(objectives),
            n_constr=0,
            xl=np.full(n_var, -10),
            xu=np.full(n_var, +10),
            elementwise_evaluation=True,)
    

    def _x_to_model(self, x):
        x = torch.split(torch.from_numpy(x), self.lenghts)
        x = [x.view(s) for x, s in zip(x, self.shapes)]
        state_dict = {n: p for n, p in zip(self.model.state_dict().keys(), x)}
        self.model.load_state_dict(state_dict)


    

    def _evaluate(self, x, out, *args, **kwargs):
        with torch.no_grad():
            self._x_to_model(x)

            losses = []

            for batch in self.data:
                batch = utils.dict_to(batch, self.device)
                batch.update(self.model(batch))
                
                if 'scores' in kwargs:
                    task_losses = torch.tensor([kwargs['scores'][t](**batch) for t in self.task_ids])
                else:
                    task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids))
                losses.append(task_losses.cpu().numpy())

            
            losses = np.mean(losses, axis=0).tolist()
            out['F'] = losses
        
    
    def eval(self, X, scores):
        pf = []
        for x in X:
            out = {}
            self._evaluate(x, out, scores=scores)
            pf.append(out['F'])
        return np.array(pf)
        
    

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination



class NSGA2Method(BaseMethod):

    def __init__(self, objectives, model, cfg) -> None:
        super().__init__(objectives, model, cfg)

        # the model is the problem to solve in pymoo terminology
        self.problem = ModelWrapper(model, objectives, self.task_ids, self.device)

        self.algorithm = NSGA2(
            pop_size=cfg.nsga2.pop_size,
            n_offsprings=cfg.nsga2.n_offsprings,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        self.algorithm.setup(self.problem, termination=get_termination("n_gen", cfg.epochs), seed=cfg.seed)

        self.batches = []
    

    def preference_at_inference(self):
        return True

    
    def log(self):
        return {
            'n_eval': self.algorithm.evaluator.n_eval
        }


    def new_epoch(self, e):
        if e > 0:

            self.problem.data = self.batches
            self.algorithm.next()
            self.batches = []
            self.problem.data = None
    

    def step(self, batch):
        self.batches.append(batch)
        return 0
    
    
    def eval_all(self, val_loader, scores):
        if self.algorithm.opt is not None:
            params = self.algorithm.opt.get('X')
            self.problem.data = [batch for batch in val_loader]
            return self.problem.eval(params, scores)
        else:
            return None


