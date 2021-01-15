import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from collections import deque
from copy import deepcopy

seed = 1

torch.manual_seed(seed)
np.random.seed(seed)


from loaders.multi_mnist_loader import MNIST
from loaders.adult_loader import ADULT
from loaders.synthetic_loader import Synthetic
from models.simple import LeNet, FullyConnected, TwoParameters
from objectives import (
    CrossEntropyLoss, 
    L1Regularization, 
    L2Regularization, 
    DDPHyperbolicTangentRelaxation, 
    BinaryCrossEntropyLoss, 
    DEOHyperbolicTangentRelaxation,
    MSELoss
)
from utils import calc_devisor, reset_weights, ParetoFront, dict_to_cuda
from solvers.proposed import ProposedSolver
from solvers.pareto_mtl import ParetoMTLSolver
from scores import mcr, DDP, from_objectives

batch_size = 64
lr = .001
epochs = 1
first_epoch = epochs
num_workers = 0
data_path = "data/mnist"
epsilon = 1e-4
warmstart = True
num_pareto_points = 3


objectives = [BinaryCrossEntropyLoss(), DDPHyperbolicTangentRelaxation()]
#objectives = [MSELoss(label_name='labels1'), MSELoss(label_name='labels2'), MSELoss(label_name='labels3'),]
scores = from_objectives(objectives)

# prepare
# train_set = Synthetic(size=1000)
# test_set = Synthetic(size=200)
train_set = ADULT(split="train")
test_set = ADULT(split="test")
# train_set = MNIST(data_path, split='train')
# test_set = MNIST(data_path, split='val')

train_loader = data.DataLoader(train_set, batch_size, num_workers)
test_loader = data.DataLoader(test_set, len(test_set), num_workers)

pareto_front = ParetoFront([s.__class__.__name__ for s in scores])
# pareto_front = ParetoFront(['w1', 'w2'])

# model = LeNet
model = FullyConnected
# model = TwoParameters
model.cuda()

# assumes objectives are scaled in [0, +inf]
divisor = calc_devisor(train_loader, model, objectives)
stop_criteria_buffer = deque(maxlen=3)

# main
solver = ProposedSolver(objectives, divisor, model, num_pareto_points)
# solver = ParetoMTLSolver(objectives, model, num_pareto_points)
# train_set.plot_pareto_loss(loss=solver.ref_vec.cpu())

for j in range(num_pareto_points):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    if not warmstart:
        reset_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr)

    solver.new_point(train_loader, optimizer)

    for e in range(epochs if j>0 else first_epoch):
        model.train()
        for batch in train_loader:
            batch = dict_to_cuda(batch)

            optimizer.zero_grad()
            solver.step(batch)
            optimizer.step()

        
        
        #scheduler.step()
        #print(scheduler.get_last_lr())

        model.eval()
        # for batch in test_loader:
        #     with torch.no_grad():
        #         batch = dict_to_cuda(batch)
                
        #         batch['logits'] = model(batch['data'])
        #         losses = [objective(**batch).item() for objective in objectives]
        #         acc = mcr(**batch)
        #         params_sum = sum(param.data.abs().sum() for param in model.parameters()).item()
        #         print("epoch={:2d}, accuracy={:.4f}, paramsum={:.4f}, loss1={:.4f}, loss2={:.4f}".format(e, acc, params_sum, losses[0], losses[1]))
        #         #pareto_front.append((losses[1], 1-acc))
        batch = dict_to_cuda(test_set.getall())
        batch['logits'] = model(batch['data'])
        score_values = [s(**batch) for s in scores]
        pareto_front.append(score_values)
        #weights = model.weight.data.detach().clone().cpu()[0].numpy()
        #print(weights)
        #pareto_front.append(weights)

        #print("train acc", mcr(model(train_set.X.cuda()), train_set.y.cuda()))
        #print("test acc", mcr(model(test_set.X.cuda()), test_set.y.cuda()))
        #print('fairness', DDP(model(test_set.X.cuda()), test_set.y.cuda(), test_set.s.cuda()))
        #print('fairness proxy', test(model(test_set.X.cuda()), test_set.y.cuda(), None, test_set.s.cuda()).item())

        #train_set.plot_pareto_loss(model.weight.data.detach().clone().cpu())
        
        #train_set.plot_pareto_front()
        pareto_front.plot()
    print()