import torch
import random
import numpy as np

seed = 42

np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import argparse
import os
import pathlib
import time
import json
import matplotlib.pyplot as plt
from torch.utils import data
from collections import deque
from copy import deepcopy
from datetime import datetime


import utils
import settings as s
from objectives import from_name
from hv import HyperVolume


from solvers import HypernetSolver, ParetoMTLSolver, SingleTaskSolver, AFeaturesSolver
from scores import mcr, DDP, from_objectives


def solver_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLSolver(**kwargs)
    elif method == 'afeature':
        return AFeaturesSolver(**kwargs)
    elif method == 'SingleTask':
        return SingleTaskSolver(**kwargs)
    elif method == 'hyper':
        return HypernetSolver(**kwargs)
    else:
        raise ValueError("Unkown method {}".format(method))


def evaluate(solver, scores, data_loader, logdir, reference_point, prefix):

    score_values = np.array([])
    for batch in data_loader:
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
    
    score_values /= len(data_loader)
    hv = HyperVolume(reference_point)
    volume = hv.compute(score_values)
    
    if len(scores) == 2:
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, prefix)
        pareto_front.append(score_values)
        pareto_front.plot()

    result = {
        "scores": score_values.tolist(),
        "hv": volume,
    }
    result.update(solver.log())
    return result



def main(settings):
    print("start processig with settings", settings)

    # create the experiment folders
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else None
    logdir = os.path.join(settings['logdir'], settings['dataset'], settings['method'], slurm_job_id if slurm_job_id else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), train_set.task_names())
    scores = from_objectives(objectives)

    solver = solver_from_name(objectives=objectives, **settings)

    epoch_max = -1
    volume_max = -1
    elapsed_time = 0

    train_results = dict(settings=settings)
    val_results = dict(settings=settings)
    test_results = dict(settings=settings)

    # main
    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(solver.model_params(), settings['lr'])
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'])
        
        for e in range(settings['epochs']):
            tick = time.time()
            solver.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to_cuda(batch)
                optimizer.zero_grad()
                loss = solver.step(batch)
                optimizer.step()
                print("Epoch {:03d}, batch {:03d}, train_loss {:.4f}".format(e, b, loss), end='\r')
            
            if settings['use_scheduler']:
                scheduler.step()
                print("Next lr:", scheduler.get_last_lr())
            
            tock = time.time()
            elapsed_time += (tock - tick)

            # run eval on train set (mainly for debugging)
            if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                train_results[f"start_{j}"][f"epoch_{e}"] = evaluate(solver, scores, 
                    data_loader=train_loader,
                    logdir=logdir,
                    reference_point=settings['reference_point'],
                    prefix=f"train_{e}",
                )

                with open(pathlib.Path(logdir) / "train_results.json", "w") as file:
                    json.dump(train_results, file)

            
            if (e+1) % settings['eval_every'] == 0:
                # Validation results
                result = evaluate(solver, scores, 
                    data_loader=val_loader,
                    logdir=logdir,
                    reference_point=settings['reference_point'],
                    prefix=f"val_{e}",
                )

                if result['hv'] > volume_max:
                    volume_max = result['hv']
                    epoch_max = e
                
                result.update({
                    "max_epoch_so_far": epoch_max,
                    "max_volume_so_far": volume_max,
                    "training_time_so_far": elapsed_time,
                })

                print("Validation: Epoch {:03d}, hv={:.4f}                        ".format(e, result['hv']))
                val_results[f"start_{j}"]["epoch_{}".format(e)] = result

                with open(pathlib.Path(logdir) / "val_results.json", "w") as file:
                    json.dump(val_results, file)

                # test results
                result = evaluate(solver, scores, 
                    data_loader=test_loader,
                    logdir=logdir,
                    reference_point=settings['reference_point'],
                    prefix=f"test_{e}",
                )

                result.update({
                    "training_time_so_far": elapsed_time,
                })

                test_results[f"start_{j}"]["epoch_{}".format(e)] = result

                with open(pathlib.Path(logdir) / "test_results.json", "w") as file:
                                json.dump(test_results, file)

                # Checkpoints
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(solver.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        print("epoch_max={}, val_volume_max={}".format(epoch_max, volume_max))
    return volume_max

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='adult')
    parser.add_argument('--method', '-m', default='pmtl')
    args = parser.parse_args()

    settings = s.generic
    
    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
    elif args.method == 'afeature':
        settings.update(s.afeature)
    elif args.method == 'hyper':
        settings.update(s.hyperSolver)
    elif args.method == 'pmtl':
        settings.update(s.paretoMTL)
    

    if args.dataset == 'mm':
        settings.update(s.multi_mnist)
    elif args.dataset == 'adult':
        settings.update(s.adult)
    elif args.dataset == 'mfm':
        settings.update(s.multi_fashion_mnist)
    elif args.dataset == 'fm':
        settings.update(s.multi_fashion)
    elif args.dataset == 'credit':
        settings.update(s.credit)
    elif args.dataset == 'compas':
        settings.update(s.compas)
    elif args.dataset == 'celeba':
        settings.update(s.celeba)

    main(settings)