import torch
import random
import numpy as np

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

import os
import pathlib
import json
import torch.utils.data as data

import utils
from main import parse_args, method_from_name, evaluate
from scores import from_objectives, mcr
from objectives import from_name


# Path to the checkpoint dir. Use the experiment folder.
CHECKPOINT_DIR = 'results/cosmos/cityscapes/cluster'


def eval(settings):
    """
    The full evaluation loop. Generate scores for all checkpoints found in the directory specified above.

    Uses the same ArgumentParser as main.py to determine the method and dataset.
    """
    device = settings['device']
    settings['batch_size'] = 2048

    print("start evaluation with settings", settings)

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    # train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    # train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(**settings)
    scores = from_objectives(objectives, **settings)

    model = utils.model_from_dataset(**settings).to(device)
    method = method_from_name(objectives, model, settings)

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))

    
    task_ids = settings['task_ids'] if settings['method'] == 'SingleTask' else [0]
    for j in task_ids:
        if settings['method'] == 'SingleTask':
            # we ran it in parallel
            checkpoints = pathlib.Path(CHECKPOINT_DIR).glob(f'**/*_{j:03d}/*/c_*.pth')
        else:
            checkpoints = pathlib.Path(CHECKPOINT_DIR).glob('**/c_*.pth')
        
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        # Eval only last checkpoint
        checkpoints = [list(sorted(checkpoints))[-1]]

        for c in sorted(checkpoints):
            print("checkpoint", c)
            _, e = c.stem.replace('c_', '').split('-')

            j = int(j)
            e = int(e)
            
            method.model.load_state_dict(torch.load(c))

            # Validation results
            val_results = evaluate(j, e, method, scores, val_loader,
                    split='val',
                    result_dict=val_results,
                    logdir=logdir,
                    train_time=0,
                    settings=settings,)

            # Test results
            test_results = evaluate(j, e, method, scores, test_loader,
                    split='test',
                    result_dict=test_results,
                    logdir=logdir,
                    train_time=0,
                    settings=settings,)

            # Train results
            # train_results = evaluate(j, e, method, scores, train_loader,
            #         split='train',
            #         result_dict=train_results,
            #         logdir=logdir,
            #         train_time=0,
            #         settings=settings,)


if __name__ == "__main__":

    settings = parse_args()
    eval(settings)