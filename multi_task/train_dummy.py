import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from collections import deque
from torch.utils import data
from tensorboardX import SummaryWriter

from loaders.adult_loader import ADULT
from loaders.multi_mnist_loader import MNIST
from datasets import global_transformer
from min_norm_solvers import MinNormSolver, gradient_normalizers

np.set_printoptions(precision=3, suppress=True, linewidth=250)

torch.manual_seed(0)
np.random.seed(0)

class LossNormalizer():

    def __init__(self, task_ids, buffer_len):
        self.buffer_len = buffer_len
        self.data = {k: deque(maxlen=buffer_len) for k in task_ids}
    
    def append(self, id, data):
        self.data[id].append(data)

    def is_valid(self):
        return all([len(v) >= self.buffer_len for v in self.data.values()])
    
    def get_normalized(self):
        result = {}
        for k, v in self.data.items():
            d = list(v)
            mean = np.mean(d)
            std = np.std(d)
            
            result[k] = (d[-1] - mean) / (std + 1e-8) + 1e-8
            assert result[k] != 0.0
        return result

def calc_angle(g1, g2):
    if g1.ndim == g2.ndim == 1:
        # biases
        g1 = torch.unsqueeze(g1, dim=-1)
        g2 = torch.unsqueeze(g2, dim=-1)
        return torch.nn.functional.cosine_similarity(g1, g2)
    elif g1.ndim == g2.ndim == 2:
        # linear weights
        return torch.nn.functional.cosine_similarity(g1, g2)
    elif g1.ndim == g2.ndim > 2:
        # kernel weights
        return torch.nn.functional.cosine_similarity(g1.view(-1), g2.view(-1), dim=0)
    else:
        raise ValueError()


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

lr = 1e-3
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

gradients = [{}, {}]
pareto_front = []
n_iter = 0
scheduler_values = np.linspace(.01, .99, epochs)
normalizer = LossNormalizer(tasks, buffer_len=10)

model.cuda()

for e in range(epochs):
    model.train()

    importance = scheduler_values[e]
    for batch in train_loader:
        n_iter += 1
        X = batch[0].cuda()
        ys = batch[1].cuda()
        #ys = (batch[1], batch[2])

        loss_data = {}
        
        for task in tasks:
            gradients[task] = {}
            optimizer.zero_grad()

            logits = model(X.float())
            output = losses[0](logits, ys) if task == 0 else losses[1](model)

            output.backward()

            loss_data[task] = output.data.item()
            #normalizer.append(task, output.data.item())

            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("param/{}".format(name), param.data, n_iter)
                    gradients[task][name] = param.grad.data.detach().clone()
            
            with torch.no_grad():
                y_hat = torch.argmax(logits, dim=1)
                accuracy = sum(y_hat == ys) / len(y_hat)
                writer.add_scalar("acc/train_{}".format(task), accuracy, n_iter)
            
        for g1, g2 in zip(gradients[0].items(), gradients[1].items()):
            writer.add_histogram("angle/{}".format(g1[0]), calc_angle(g1[1], g2[1]), n_iter)
            writer.add_scalar("norm1/{}".format(g1[0]), torch.linalg.norm(g1[1]), n_iter)
            writer.add_scalar("norm2/{}".format(g1[0]), torch.linalg.norm(g2[1]), n_iter)

        # Normalize all gradients, this is optional and not included in the paper.
        # gn = gradient_normalizers(gradients, loss_data, "loss+")
        # for t in tasks:
        #     for gr_i in gradients[t]:
        #         gradients[t][gr_i] = gradients[t][gr_i] / gn[t]

        # if normalizer.is_valid():
        #     loss_data = normalizer.get_normalized()
        # else:
        #     continue
        
        # calculate the scaling for MGDA
        alpha = (importance * loss_data[1]) / (loss_data[0] - importance * loss_data[0])
        alpha = torch.Tensor([alpha]).cuda()
        # apply it to the gradients (same as if we would apply it to the losses before calculating the gradient)
        gradients[0] = {k: v*alpha for k, v in gradients[0].items()}


        sol, min_norm = MinNormSolver.find_min_norm_element_FW([[v for k, v in sorted(grads.items())] for grads in gradients])
        #sol, min_norm = MinNormSolver.scipy_impl([[v for k, v in sorted(grads.items())] for grads in gradients])

        # Scaled back-propagation
        for name, param in model.named_parameters():
            param.grad = sum(sol[t] * gradients[t][name] for t in tasks).cuda()
        optimizer.step()


        # logging
        params_sum = sum(param.data.abs().sum() for param in model.parameters())
        
        writer.add_scalar("alpha/0", sol[0], n_iter)
        writer.add_scalar("alpha/1", sol[1], n_iter)
        writer.add_scalar("min_norm", min_norm, n_iter)
        writer.add_scalar("loss/train_0", loss_data[0], n_iter)
        writer.add_scalar("loss/train_1", loss_data[1], n_iter)
        writer.add_scalar("params_sum", params_sum, n_iter)
        
    
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
for i, text in enumerate(range(epochs)):
    plt.annotate(text, (pareto[i,0], pareto[i,1]))
plt.xlabel("params sum")
plt.ylabel("misclassification rate (train)")
plt.savefig("t.png")