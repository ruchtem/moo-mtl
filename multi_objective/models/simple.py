import torch.nn as nn

from .base import BaseModel


class MultiLeNet(BaseModel):
    """
    An implementation of the LeNet with two task-specific outputs.
    """

    def __init__(self, dim, task_ids, channel_multiplier=1, **kwargs):
        super().__init__(task_ids)
        self.channel_multiplier = channel_multiplier
        self.first_layer = nn.Conv2d(dim[0], 10*channel_multiplier, kernel_size=5)
        self.shared =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10*channel_multiplier, 20*channel_multiplier, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720*channel_multiplier , 50),
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
        self.first_layer = nn.Conv2d(dim, 10*self.channel_multiplier, kernel_size=5)


class FullyConnected(BaseModel):


    def __init__(self, dim, channel_multiplier, **kwargs):
        super().__init__()
        self.channel_multiplier = channel_multiplier
        self.first_layer = nn.Linear(dim[0], 60*channel_multiplier)
        self.f = nn.Sequential(
            nn.ReLU(),
            nn.Linear(60*channel_multiplier, 25*channel_multiplier),
            nn.ReLU(),
            nn.Linear(25*channel_multiplier, 1),
        )


    def forward(self, batch):
        x = batch['data']
        x = self.first_layer(x)
        return dict(logits=self.f(x))
    

    def change_input_dim(self, dim):
        self.first_layer = nn.Linear(dim, 60*self.channel_multiplier)
