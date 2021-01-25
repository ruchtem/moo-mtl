import torch

from utils import model_from_dataset
from .base import BaseSolver


class SingleTaskSolver(BaseSolver):

    def __init__(self, objectives, **kwargs):
        self.objectives = objectives
        self.task = -1
        self.model = model_from_dataset(method='single_task', **kwargs).cuda()


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()
        if e == 0:
            self.task += 1


    def step(self, batch):
        batch.update(self.model(batch))
        loss = self.objectives[self.task](**batch)
        loss.backward()
        return loss.item()
    

    def log(self):
        return {"task": self.task}


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return[self.model(batch)]
