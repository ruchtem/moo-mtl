import torch
import random
import numpy as np

import argparse
import os
import pathlib
import time
import json
import itertools
import matplotlib.pyplot as plt
from torch.utils import data
from collections import deque
from copy import deepcopy
from datetime import datetime


import utils
import settings as s
from objectives import from_name
from hv import HyperVolume


from solvers import HypernetSolver, ParetoMTLSolver, SingleTaskSolver, COSMOSSolver
from scores import mcr, DDP, from_objectives


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def solver_from_name(method, **kwargs):
    if method == 'ParetoMTL':
        return ParetoMTLSolver(**kwargs)
    elif 'cosmos' in method:
        return COSMOSSolver(**kwargs)
    elif method == 'SingleTask':
        return SingleTaskSolver(**kwargs)
    elif 'hyper' in method:
        return HypernetSolver(**kwargs)
    else:
        raise ValueError("Unkown method {}".format(method))


epoch_max = -1
volume_max = -1
elapsed_time = 0


def evaluate(j, e, solver, scores, data_loader, logdir, reference_point, split, result_dict):
    assert split in ['train', 'val', 'test']
    global volume_max
    global epoch_max

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

    if score_values.shape[1] > 2:
        # computing hypervolume in high dimensions is expensive
        # Do only pairwise hypervolume and report the average
        n, m = score_values.shape
        volume = 0
        for columns in itertools.combinations(range(m), 2):
            volume += hv.compute(score_values[:, columns])
        volume /= len(list(itertools.combinations(range(m), 2)))
    else:
        volume = hv.compute(score_values)

    if len(scores) == 2:
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, "{}_{:03d}".format(split, e))
        pareto_front.append(score_values)
        pareto_front.plot()

    result = {
        "scores": score_values.tolist(),
        "hv": volume,
    }

    if split == 'val':
        if volume > volume_max:
            volume_max = volume
            epoch_max = e
                    
        result.update({
            "max_epoch_so_far": epoch_max,
            "max_volume_so_far": volume_max,
            "training_time_so_far": elapsed_time,
        })
    elif split == 'test':
        result.update({
            "training_time_so_far": elapsed_time,
        })

    result.update(solver.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict


def main(settings):
    print("start processig with settings", settings)
    set_seed(settings['seed'])

    global elapsed_time

    # create the experiment folders
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], slurm_job_id if slurm_job_id else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))

    # main
    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(solver.model_params(), settings['lr'])
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
        
        for e in range(settings['epochs']):
            print(f"Epoch {e}")
            tick = time.time()
            solver.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to_cuda(batch)
                optimizer.zero_grad()
                loss = solver.step(batch)
                optimizer.step()
                #print("Epoch {:03d}, batch {:03d}, train_loss {:.4f}".format(e, b, loss), end='', flush=True)
            
            tock = time.time()
            elapsed_time += (tock - tick)

            if settings['use_scheduler']:
                val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                scheduler.step()


            # run eval on train set (mainly for debugging)
            if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                train_results = evaluate(j, e, solver, scores, train_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='train',
                    result_dict=train_results)

            
            if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0:
                # Validation results
                val_results = evaluate(j, e, solver, scores, val_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='val',
                    result_dict=val_results)

                # Test results
                test_results = evaluate(j, e, solver, scores, test_loader, logdir, 
                    reference_point=settings['reference_point'],
                    split='test',
                    result_dict=test_results)

            # Checkpoints
            pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
            torch.save(solver.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        print("epoch_max={}, val_volume_max={}".format(epoch_max, volume_max))
    return volume_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='mm')
    parser.add_argument('--method', '-m', default='cosmos_ln')
    args = parser.parse_args()

    settings = s.generic
    
    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
    elif args.method == 'cosmos_ln':
        settings.update(s.cosmos_ln)
    elif args.method == 'cosmos_epo':
        settings.update(s.cosmos_ln)
    elif args.method == 'hyper_ln':
        settings.update(s.hyperSolver_ln)
    elif args.method == 'hyper_epo':
        settings.update(s.hyperSolver_epo)
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

    return settings



if __name__ == "__main__":
    
    settings = parse_args()
    main(settings)