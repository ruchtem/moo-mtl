import torch

from multi_objective.utils import model_from_dataset
from .base import BaseMethod


class UniformScalingMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.J = len(objectives)
        self.normalization = cfg.normalization_type
        self.loss_maxs = cfg.loss_maxs

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        batch.update(self.model(batch))
        loss = 0
        for i, t in enumerate(self.task_ids):
            if self.normalization == 'none':
                loss += 1/self.J * self.objectives[t](**batch)
            elif self.normalization == 'init_loss':
                loss += 1/self.J * (self.objectives[t](**batch) / self.loss_maxs[i])
            else:
                raise ValueError(f"Normalization {self.normalization} not available for uniform")
        loss.backward()
        return loss.item()


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)