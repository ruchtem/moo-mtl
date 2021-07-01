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
        loss = 0
        for i, t in enumerate(self.task_ids):
            loss += 1/self.J * self.objectives[t](**batch)
        loss.backward()
        return loss.item()


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)