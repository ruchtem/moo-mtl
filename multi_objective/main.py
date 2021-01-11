import torch
import numpy as np
from torch.utils import data
from collections import deque
from copy import deepcopy

torch.manual_seed(0)
np.random.seed(0)


from loaders.multi_mnist_loader import MNIST
from loaders.adult_loader import ADULT
from models.simple import LeNet, FullyConnected
from objectives import CrossEntropyLoss, L1Regularization, L2Regularization, DDPHyperbolicTangentRelaxation, BinaryCrossEntropyLoss
from utils import calc_devisor, reset_weights, ParetoFront
from solvers.proposed import ProposedSolver
from solvers.pareto_mtl import ParetoMTLSolver
from scores import mcr, DDP

batch_size = 64
lr = 1e-3
epochs = 3
first_epoch = 10
num_workers = 0
data_path = "data/mnist"
epsilon = 1e-4
warmstart = True
num_pareto_points = 10

objectives = [BinaryCrossEntropyLoss(), DDPHyperbolicTangentRelaxation()]
scores = [mcr, DDP]
scheduler_values = np.linspace(.001, .99, num_pareto_points)

# prepare
train_set = ADULT(split="train")
test_set = ADULT(split="test")
# train_set = MNIST(data_path, split='train')
# test_set = MNIST(data_path, split='val')

train_loader = data.DataLoader(train_set, batch_size, num_workers)
test_loader = data.DataLoader(test_set, len(test_set), num_workers)

pareto_front = ParetoFront(["param_sum", "mcr (val)"])

# model = LeNet
model = FullyConnected
model.cuda()

# assumes objectives are scaled in [0, +inf]
divisor = calc_devisor(train_loader, model, objectives)
stop_criteria_buffer = deque(maxlen=3)

# main
solver = ProposedSolver(scheduler_values, objectives, divisor, model)
#solver = ParetoMTLSolver(objectives, model)

for j in range(num_pareto_points):
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

    if not warmstart:
        reset_weights(model)

    solver.new_point(train_loader, optimizer)

    for e in range(epochs if j>0 else first_epoch):
        model.train()
        for batch in train_loader:
            data = batch.data.cuda()
            labels = batch.labels.cuda()
            sensible_attribute = batch.sensible_attribute.cuda() if hasattr(batch, 'sensible_attribute') else None

            optimizer.zero_grad()
            solver.step(data, labels, sensible_attribute)
            optimizer.step()

        
        
        #scheduler.step()
        #print(scheduler.get_last_lr())

        model.eval()
        for batch in test_loader:
            with torch.no_grad():
                data = batch.data.cuda()
                labels = batch.labels.cuda()
                sensible_attribute = batch.sensible_attribute.cuda() if hasattr(batch, 'sensible_attribute') else None
                logits = model(data)
                losses = [objective(logits, labels, model, sensible_attribute).item() for objective in objectives]
                acc = mcr(logits, labels)
                params_sum = sum(param.data.abs().sum() for param in model.parameters()).item()
                print("epoch={:2d}, accuracy={:.4f}, paramsum={:.4f}, loss1={:.4f}, loss2={:.4f}".format(e, acc, params_sum, losses[0], losses[1]))
                #pareto_front.append((losses[1], 1-acc))
        logits = model(test_set.X.cuda())
        score_values = [s(logits, test_set.y.cuda(), test_set.s.cuda()) for s in scores]
        pareto_front.append(score_values)

        print("train acc", mcr(model(train_set.X.cuda()), train_set.y.cuda()))
        print("test acc", mcr(model(test_set.X.cuda()), test_set.y.cuda()))
        print('fairness', DDP(model(test_set.X.cuda()), test_set.y.cuda(), test_set.s.cuda()))
        #print('fairness proxy', test(model(test_set.X.cuda()), test_set.y.cuda(), None, test_set.s.cuda()).item())

        pareto_front.plot()
        print()