import torch.nn as nn


class MultiLeNet(nn.Module):

    def __init__(self, input_dim=1):
        super().__init__()
        self.f =  nn.Sequential(
            nn.Conv2d(input_dim, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
        )
        self.left = nn.Linear(50, 10)
        self.right = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.f(x)
        return dict(logits_l=self.left(x), logits_r=self.right(x))


    def private_params(self):
        return ['left.weight', 'left.bias', 'right.weight', 'right.bias']

    
    def logits_names(self):
        return ['logits_l', 'logits_r']


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f =  nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 10),
        )
    
    def forward(self, x):
        return dict(logits=self.f(x))
    


class FullyConnected(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, 60),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(60, 25),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        return dict(logits=self.f(x))

TwoParameters = nn.Linear(2, 1, bias=False)