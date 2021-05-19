import torch
import numpy as np
from copy import deepcopy
from .base import BaseMethod


class SingleTaskMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.models = [deepcopy(model).cpu() for _ in self.task_ids]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in self.models]

    
    def new_epoch(self, e):
        for m in self.models:
            m.train()


    def step(self, batch):
        losses = []
        for t, optim, model in zip(self.task_ids, self.optimizers, self.models):
            optim.zero_grad()
            model = model.to(self.device)

            result = self._step(batch, model, t)
            optim.step()

            losses.append(result)
        return np.mean(losses).item()


    def _step(self, batch, model, task_id):
        batch.update(model(batch))
        loss = self.objectives[task_id](**batch)
        loss.backward()
        return loss.item()


    def eval_step(self, batch):
        with torch.no_grad():
            for t, m in zip(self.task_ids, self.models):
                m.eval()
                result = m(batch)
                batch[f'logits_{t}'] = result[f'logits_{t}']
        return batch
