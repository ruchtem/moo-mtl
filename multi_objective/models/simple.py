import torch
import torch.nn as nn


class MultiLeNet(nn.Module):

    def __init__(self, alpha=False):
        super().__init__()
        self.alpha = alpha
        self.f1 =  nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.f2 = nn.Sequential(
            nn.Linear(722 if alpha else 720 , 50),
            # nn.ReLU(),
            # nn.Linear(320 , 50),
            nn.ReLU(),
        )
        # self.f = nn.Sequential(
        #     nn.Linear(36*36+2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 50),
        #     nn.ReLU(),
        # )
        # self.combined = nn.Linear(50, 20)
        self.left = nn.Linear(50, 10)
        self.right = nn.Linear(50, 10)
    
    def forward(self, batch):
        x = batch['data']
        x = self.f1(x)
        if self.alpha:
            
            # b = x.shape[0]
            # x = x.view(b, -1)
            x = torch.hstack((x, batch['alpha_features']))
            # x = self.f(x)
        x = self.f2(x)
        return dict(logits_l=self.left(x), logits_r=self.right(x))
        # return dict(logits_l=x[:, :10], logits_r=x[:, 10:])


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
    
    def forward(self, batch):
        x = batch['data']
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

    def forward(self, batch):
        x = batch['data']
        return dict(logits=self.f(x))
