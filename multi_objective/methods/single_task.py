import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from .base import BaseMethod
from multi_objective import utils


class SingleTaskMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)

        if cfg.task_id is None:
            # Create copies for second to last task. First is handled by main
            self.models = [deepcopy(model) for _ in self.task_ids[1:]]        
            self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in self.models]
            self.schedulers = [utils.get_lr_scheduler(cfg.lr_scheduler, o, cfg, '') for o in self.optimizers]
            print('num models:', len(self.models))

        self.task_id = cfg.task_id
        

        
    def state_dict(self):
        if self.task_id is None:
            state = OrderedDict()
            for i, (m, o, s) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
                state[f'model.{i}'] = m.state_dict()
                state[f'optimizer.{i}'] = o.state_dict()
                state[f'lr_scheduler.{i}'] = s.state_dict()
            return state

    
    def load_state_dict(self, dict):
        if self.task_id is None:
            for i in range(len(self.models)):
                self.models[i].load_state_dict(dict[f'model.{i}'])
                self.optimizers[i].load_state_dict(dict[f'optimizer.{i}'])
                self.schedulers[i].load_state_dict(dict[f'lr_scheduler.{i}'])



    def new_epoch(self, e):
        if self.task_id is None:
            for m in self.models:
                m.train()
            if e>0:
                for s in self.schedulers:
                    s.step()


    def step(self, batch):
        losses = []
        if self.task_id is None:
            for t, optim, model in zip(self.task_ids[1:], self.optimizers, self.models):
                optim.zero_grad()
                result = self._step(batch, model, t)
                optim.step()
                losses.append(result)
        
        # task zero we take the model we got via __init__
        self.model.zero_grad()
        result = self._step(batch, self.model, self.task_ids[0] if self.task_id is None else self.task_id)
        losses.append(result)
        return np.mean(losses).item()


    def _step(self, batch, model, task_id):
        batch.update(model(batch))
        loss = self.objectives[task_id](**batch)
        loss.backward()
        return loss.item()


    def eval_step(self, batch, task):
        with torch.no_grad():
            if (self.task_id is None and task == 0) or (self.task_id is not None):
                self.model.eval()
                return self.model(batch)
            else:
                model = self.models[task-1]
                model.eval()
                return model(batch)
