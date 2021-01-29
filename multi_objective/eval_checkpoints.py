import torch
import numpy as np
import os
import pathlib
import torch.utils.data as data

from datetime import datetime

import utils
from main import evaluate, parse_args, set_seed, solver_from_name
from scores import from_objectives
from objectives import from_name
from hv import HyperVolume

def eval(settings):
    set_seed(settings['seed'])
    settings['batch_size'] = 2048

    # create the experiment folders
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], slurm_job_id if slurm_job_id else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    # train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    # train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), val_set.task_names())
    scores = from_objectives(objectives)

    solver = solver_from_name(objectives=objectives, **settings)

    # train_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))

    j = 0
    checkpoint_dir = 'results_paper/cosmos_ln/celeba/2021-01-29_11-06-07/checkpoints'
    checkpoints = pathlib.Path(checkpoint_dir).glob('**/c_*.pth')
    c = list(sorted(checkpoints))[-1]
        
    solver.model.load_state_dict(torch.load(c))

    j, e = c.stem.replace('c_', '').split('-')
    j = int(j)
    e = int(e)

    # run eval on train set (mainly for debugging)
    # if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
    #     train_results = evaluate(j, e, solver, scores, train_loader, logdir, 
    #         reference_point=settings['reference_point'],
    #         split='train',
    #         result_dict=train_results)

    
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


    print()






if __name__ == "__main__":

    settings = parse_args()
    eval(settings)