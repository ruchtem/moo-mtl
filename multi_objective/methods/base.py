import inspect
from abc import abstractmethod


class BaseMethod():


    def __init__(self, objectives, model, device, task_ids=None, **kwargs) -> None:
        super().__init__()
        self.objectives = objectives
        self.model = model
        self.device = device

        if task_ids is not None:
            self.task_ids = task_ids
        else:
            self.task_ids = list(objectives.keys())


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()


    def preference_at_inference(self):
        return False


    @abstractmethod
    def step(self, batch):
        raise NotImplementedError()
    

    def log(self):
        return {}

    @abstractmethod
    def eval_step(self, batch, preference_vector):
        raise NotImplementedError()
