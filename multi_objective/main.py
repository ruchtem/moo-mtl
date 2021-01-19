import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from collections import deque
from copy import deepcopy

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

import utils
import settings as s
from loaders.multi_mnist_loader import MNIST
from loaders.adult_loader import ADULT
from loaders.synthetic_loader import Synthetic
from models.simple import LeNet, FullyConnected
from objectives import (
    CrossEntropyLoss, 
    L1Regularization, 
    L2Regularization, 
    DDPHyperbolicTangentRelaxation, 
    BinaryCrossEntropyLoss, 
    DEOHyperbolicTangentRelaxation,
    MSELoss,
    from_name
)

from solvers.proposed import ProposedSolver
from solvers.pareto_mtl import ParetoMTLSolver
from solvers.base import solver_from_name
from scores import mcr, DDP, from_objectives

# settings = s.adult
settings = s.multi_mnist
# settings = s.multi_fashion_mnist
# settings.update(s.afeature)
# settings.update(s.baseSolver)
settings.update(s.hyperSolver)
num_workers = 0
use_scheduler = False



# prepare
train_set = utils.dataset_from_name(settings['dataset'], split='train')
val_set = utils.dataset_from_name(settings['dataset'], split='val')
test_set = utils.dataset_from_name(settings['dataset'], split='test')

train_loader = data.DataLoader(train_set, settings['batch_size'], num_workers)
val_loader = data.DataLoader(val_set, settings['batch_size'], num_workers)
test_loader = data.DataLoader(test_set, len(test_set), num_workers)

model = utils.model_from_dataset(settings['dataset'], settings['name'])

label_names = train_set.label_names() if hasattr(train_set, 'label_names') else None
logits_names = model.logits_names() if hasattr(model, 'logits_names') else None
objectives = from_name(settings.pop('objectives'), label_names, logits_names)
scores = from_objectives(objectives)

pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores])

solver = solver_from_name(objectives=objectives, model=model, **settings)

# main
model.cuda()
for j in range(settings['num_starts']):
    optimizer = torch.optim.Adam(model.parameters(), settings['lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, ])

    if not settings['warmstart']:
        utils.reset_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), settings['lr'])

    # solver.new_point(train_loader, optimizer)

    for e in range(settings['epochs']):
        solver.new_point(train_loader, optimizer)
        model.train()
        for batch in train_loader:
            batch = utils.dict_to_cuda(batch)

            optimizer.zero_grad()
            solver.step(batch)
            optimizer.step()
        
        if use_scheduler:
            scheduler.step()
            print(scheduler.get_last_lr())

        model.eval()
        score_values = np.array([])
        for batch in val_loader:
            batch = utils.dict_to_cuda(batch)
            
            # more than one for some solvers
            s = []
            for l in solver.eval_step(batch):
                batch.update(l)
                s.append([s(**batch) for s in scores])
            if score_values.size == 0:
                score_values = np.array(s)
            else:
                score_values += np.array(s)
        
        score_values /= len(val_loader)
        
        if e < 60:
            pareto_front.points = []
        pareto_front.append(score_values)
        pareto_front.plot()
        print("Epoch {}, val scores={}".format(e, score_values.ravel()))


model.eval()
score_values = np.array([])
for batch in test_loader:
    batch = utils.dict_to_cuda(batch)
    
    # more than one for some solvers
    s = []
    for l in solver.eval_step(batch):
        batch.update(l)
        s.append([s(**batch) for s in scores])
    if score_values.size == 0:
        score_values = np.array(s)
    else:
        score_values += np.array(s)

score_values /= len(test_loader)
print("test scores", score_values)