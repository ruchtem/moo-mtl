import torch

from multi_objective.utils import model_from_dataset
from .base import BaseMethod


class UniformScalingMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.J = len(objectives)

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        batch.update(self.model(batch))
        loss = sum([1/self.J * o(**batch) for o in self.objectives.values()])
        loss.backward()
        return loss.item()


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)