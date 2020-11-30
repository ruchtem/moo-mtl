import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils import data
from tensorboardX import SummaryWriter

from loaders.adult_loader import ADULT
from loaders.multi_mnist_loader import MNIST
from datasets import global_transformer
from min_norm_solvers import MinNormSolver, gradient_normalizers

np.set_printoptions(precision=3, suppress=True, linewidth=250)

torch.manual_seed(0)
np.random.seed(0)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def l1_regularization(model, div=1):
    return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1) / div


writer = SummaryWriter(log_dir='runs/{}'.format(datetime.now().strftime("%I:%M:%S on %B %d, %Y")))

# train_set = ADULT("data/adult", split="train")
# test_set = ADULT("data/adult", split="test")
train_set = MNIST("data/mnist", train=True, transform=global_transformer())
test_set = MNIST("data/mnist", train=False, transform=global_transformer())
tasks = [0, 1]

num_workers = 4
train_loader = data.DataLoader(train_set, batch_size=256, num_workers=num_workers)
test_loader = data.DataLoader(test_set, batch_size=len(test_set), num_workers=num_workers)

# model = nn.Sequential(
#     nn.Linear(65, 64),
#     nn.ReLU(),
#     nn.Linear(64, 2),
# )

model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 10),
)

losses = [nn.CrossEntropyLoss(reduction='mean'), l1_regularization]

epochs = 30
lr = 1e-1
loss_scalar = 70
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

gradients = [{}, {}]
pareto_front = []
n_iter = 0

model.cuda()

for e in range(epochs):
    model.train()
    model.apply(weight_reset)

    div = (epochs - e) * loss_scalar
    print(div)
    for _ in range(2):
        for batch in train_loader:
            n_iter += 1
            X = batch[0].cuda()
            ys = batch[1].cuda()

                
            optimizer.zero_grad()

            logits = model(X.float())
            out1 = losses[0](logits, ys)
            out2 = losses[1](model, div)
            output = out1 + out2

            output.backward()
            optimizer.step()
                
            with torch.no_grad():
                y_hat = torch.argmax(logits, dim=1)
                accuracy = sum(y_hat == ys) / len(y_hat)
                writer.add_scalar("acc/train", accuracy, n_iter)
                

            # logging
            params_sum = sum(param.data.abs().sum() for param in model.parameters())
            writer.add_scalar("params_sum", params_sum, n_iter)

            writer.add_scalar("loss/train_0", out1, n_iter)
            writer.add_scalar("loss/train_1", out2, n_iter)
        
    
    model.eval()
    for batch in test_loader:
        n_iter += 1
        X = batch[0].cuda()
        #ys = (batch[1], batch[2])
        ys = [batch[1].cuda()]
        
        for task in tasks:
            if task == 0:
                with torch.no_grad():
                    logits = model(X.float())
                    y_hat = torch.argmax(logits, dim=1)
                    accuracy = sum(y_hat == ys[task]) / len(y_hat)
                    writer.add_scalar("acc/test_{}".format(task), accuracy, n_iter)
                    writer.add_scalar("loss/val", losses[0](logits, ys[task]), n_iter)

    pareto_front.append([params_sum.item(), 1 - accuracy.item()])
    print("Epoch", e, "Sum params, accuracy ({}, {})".format(params_sum, accuracy))
    

pareto = np.array(pareto_front)
np.save("pareto_points", pareto)
plt.plot(pareto[:, 0], pareto[:, 1], 'o')
plt.xlabel("params sum")
plt.ylabel("misclassification rate (test)")
plt.savefig("t.png")