import torch

from utils import model_from_dataset


class BaseSolver():

    def __init__(self, objectives, task=0, **kwargs):
        self.objectives = objectives
        self.task = task
        self.model = model_from_dataset(method='single_task', **kwargs).cuda()


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        batch.update(self.model(batch))
        loss = self.objectives[self.task](**batch)
        loss.backward()
    

    def eval_step(self, batch):
        with torch.no_grad():
            return[self.model(batch)]
