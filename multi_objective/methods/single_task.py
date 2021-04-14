import torch

from utils import model_from_dataset
from .base import BaseMethod


class SingleTaskMethod(BaseMethod):

    def __init__(self, objectives, model, **kwargs):
        super().__init__(objectives, model, **kwargs)
        
        if 'task_id' in kwargs:
            self.task_id = kwargs['task_id']
            self.task_ids = None
        else:
            self.task_ids = iter(self.task_ids)

    
    def new_epoch(self, e):
        self.model.train()
        if e == 0 and self.task_ids is not None:
            self.task_id = next(self.task_ids)


    def step(self, batch):
        batch.update(self.model(batch))
        loss = self.objectives[self.task_id](**batch)
        loss.backward()
        return loss.item()
    

    def log(self):
        return {"task": self.task_id}


    def eval_step(self, batch, preference_vector=None):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)
