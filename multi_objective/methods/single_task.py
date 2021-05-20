import torch
import numpy as np
from copy import deepcopy
from .base import BaseMethod


class SingleTaskMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)

        # Create copies for second to last task. First is handled by main
        self.models = [deepcopy(model) for _ in self.task_ids[1:]]        
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in self.models]

    
    def new_epoch(self, e):
        for m in self.models:
            m.train()


    def step(self, batch):
        losses = []

        for t, optim, model in zip(self.task_ids[1:], self.optimizers, self.models):
            optim.zero_grad()
            result = self._step(batch, model, t)
            optim.step()
            losses.append(result)
        
        # task zero we take the model we got via __init__
        self.model.zero_grad()
        result = self._step(batch, self.model, self.task_ids[0])
        losses.append(result)
        return np.mean(losses).item()


    def _step(self, batch, model, task_id):
        batch.update(model(batch))
        loss = self.objectives[task_id](**batch)
        loss.backward()
        return loss.item()


    def eval_step(self, batch):
        with torch.no_grad():
            for t, m in zip(self.task_ids[1:], self.models):
                m.eval()
                result = m(batch)
                batch[f'logits_{t}'] = result[f'logits_{t}']
            self.model.eval()
            result = self.model(batch)
            t = self.task_ids[0]
            batch[f'logits_{t}'] = result[f'logits_{t}']
        return batch
