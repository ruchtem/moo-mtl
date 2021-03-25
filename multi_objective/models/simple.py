import torch.nn as nn

from .base import BaseModel


class MultiLeNet(BaseModel):
    """
    An implementation of the LeNet with two task-specific outputs.
    """

    def __init__(self, dim, task_ids, **kwargs):
        super().__init__(task_ids)
        self.first_layer = nn.Conv2d(dim[0], 10, kernel_size=5)
        self.shared =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720 , 50),
            nn.ReLU(),
        )
        self.task_layers = nn.ModuleDict({
            f"logits_{t}": nn.Linear(50, 10) for t in self.task_ids
        })
    

    def forward(self, batch):
        x = batch['data']
        x = self.first_layer(x)
        x = self.shared(x)
        return {
            f"logits_{t}": self.task_layers[f"logits_{t}"](x) for t in self.task_ids
        }


    def change_input_dim(self, dim):
        self.first_layer = nn.Conv2d(dim, 10, kernel_size=5)


class FullyConnected(BaseModel):


    def __init__(self, dim, **kwargs):
        super().__init__()
        self.first_layer = nn.Linear(dim[0], 60)
        self.f = nn.Sequential(
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )


    def forward(self, batch):
        x = batch['data']
        x = self.first_layer(x)
        return dict(logits=self.f(x))
    

    def change_input_dim(self, dim):
        self.first_layer = nn.Linear(dim, 60)
