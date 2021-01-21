import argparse
import torch
import os
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from collections import deque
from copy import deepcopy
from datetime import datetime

seed = 0

np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import utils
import settings as s
from objectives import from_name
from hv import HyperVolume


from solvers.proposed import ProposedSolver
from solvers.pareto_mtl import ParetoMTLSolver
from solvers.base import solver_from_name
from scores import mcr, DDP, from_objectives


def main(settings):
    print("start processig with settings", settings)
    num_workers = 0
    use_scheduler = False

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['dataset'], settings['method'], datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    train_set = utils.dataset_from_name(settings['dataset'], split='train')
    val_set = utils.dataset_from_name(settings['dataset'], split='val')
    test_set = utils.dataset_from_name(settings['dataset'], split='test')

    train_loader = data.DataLoader(train_set, settings['batch_size'], num_workers)
    val_loader = data.DataLoader(val_set, settings['batch_size'], num_workers)
    test_loader = data.DataLoader(test_set, settings['batch_size'], num_workers)

    model = utils.model_from_dataset(**settings)

    objectives = from_name(settings.pop('objectives'), train_set.task_names())
    scores = from_objectives(objectives)

    pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir)

    solver = solver_from_name(objectives=objectives, model=model, **settings)

    # main
    model.cuda()
    for j in range(settings['num_starts']):
        optimizer = torch.optim.Adam(solver.model_params() if hasattr(solver, 'model_params') else model.parameters(), settings['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, ])

        if not settings['warmstart']:
            utils.reset_weights(model)
            optimizer = torch.optim.Adam(solver.model_params() if hasattr(solver, 'model_params') else model.parameters(), settings['lr'])

        #solver.new_point(train_loader, optimizer)

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
            hv = HyperVolume(settings['reference_point'])
            volume = hv.compute(score_values)
            
            pareto_front.points = []
            pareto_front.append(score_values)
            pareto_front.plot()
            print("Epoch {}, hv={}".format(e, volume))


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

    pareto_front.points = []
    pareto_front.append(score_values)
    pareto_front.plot()

    hv = HyperVolume(settings['reference_point'])
    volume = hv.compute(score_values)

    print("test hv={}, scores={}".format(volume, score_values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='celeba')
    parser.add_argument('--method', '-m', default='afeature')
    args = parser.parse_args()

    settings = s.generic

    if args.dataset == 'multi_mnist':
        settings.update(s.multi_mnist)
    elif args.dataset == 'adult':
        settings.update(s.adult)
    elif args.dataset == 'multi_fashion_mnist':
        settings.update(s.multi_fashion_mnist)
    elif args.dataset == 'credit':
        settings.update(s.credit)
    elif args.dataset == 'compas':
        settings.update(s.compas)
    elif args.dataset == 'celeba':
        settings.update(s.celeba)
    
    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
    elif args.method == 'afeature':
        settings.update(s.afeature)
    elif args.method == 'hyper':
        settings.update(s.hyperSolver)

    main(settings)