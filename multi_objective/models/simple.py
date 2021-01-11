import torch.nn as nn


LeNet = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 10),
)

FullyConnected = nn.Sequential(
    nn.Linear(88, 60),
    nn.ReLU(),
    #nn.Dropout(p=0.2),
    nn.Linear(60, 25),
    nn.ReLU(),
    #nn.Dropout(p=0.2),
    nn.Linear(25, 1),
)