import torch
import random
import numpy as np

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

import argparse
import os
import pathlib
import time
import json
import math
from torch.utils import data

import settings as s
import utils
from objectives import from_name


from methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod
from scores import from_objectives


def method_from_name(objectives, model, settings):
    """
    Initializes the method specified in settings along with its configuration.

    Args:
        objectives (dict): All objectives for the experiment. Structure is
            task_id: Objective.
        model (models.base.BaseModel): A model for the method to learn. It's a
            `torch.nn.Module` with some custom functions required by some MOO methods
        settings (dict): The settings
    
    Returns:
        Method. The configured method instance.
    """
    method = settings['method']
    settings = settings.copy()
    settings.pop('objectives')
    if method == 'ParetoMTL':
        return ParetoMTLMethod(objectives, model, **settings)
    elif 'cosmos' in method:
        return COSMOSMethod(objectives, model, **settings)
    elif method == 'SingleTask':
        return SingleTaskMethod(objectives, model, **settings)
    elif 'hyper' in method:
        return HypernetMethod(objectives, model, **settings)
    elif method == 'mgda':
        return MGDAMethod(objectives, model, **settings)
    elif method == 'uniform':
        return UniformScalingMethod(objectives, model, **settings)
    else:
        raise ValueError("Unkown method {}".format(method))


def evaluate(j, e, method, scores, data_loader, split, result_dict, logdir, train_time, settings):
    """
    Evaluate the method on a given dataset split. Calculates:
    - score for all the scores given in `scores`
    - computes hyper-volume if applicable
    - plots the Pareto front to `logdir` for debugging purposes

    Also stores everything in a json file.

    Args:
        j (int): The index of the run (if there are several starts)
        e (int): Epoch
        method: The method subject to evaluation
        scores (dict): All scores which the method should be evaluated on
        data_loader: The dataloader
        split (str): Split of the evaluation. Used to name log files
        result_dict (dict): Global result dict to store the evaluations for this epoch and run in
        logdir (str): Directory where to store the logs
        train_time (float): The training time elapsed so far, added to the logs
        settings (dict): Settings of the experiment
    
    Returns:
        dict: The updates `result_dict` containing the results of this evaluation
    """
    assert split in ['train', 'val', 'test']
    
    if 'task_ids' in settings:
        J = len(settings['task_ids'])
        task_ids = settings['task_ids']
    else:
        # single output setting
        J = len(settings['objectives'])
        task_ids = list(scores[list(scores)[0]].keys())

    pareto_rays = utils.reference_points(settings['n_partitions'], dim=J)
    n_rays = pareto_rays.shape[0]
    
    # gather the scores
    score_values = {et: utils.EvalResult(J, n_rays, task_ids) for et in scores.keys()}
    for batch in data_loader:
        batch = utils.dict_to(batch, settings['device'])
                
        if method.preference_at_inference():
            data = {et: np.zeros((n_rays, J)) for et in scores.keys()}
            for i, ray in enumerate(pareto_rays):
                logits = method.eval_step(batch, preference_vector=ray)
                batch.update(logits)

                for eval_mode, score in scores.items():

                    data[eval_mode][i] += np.array([score[t](**batch) for t in task_ids])
            
            for eval_mode in scores:
                score_values[eval_mode].update(data[eval_mode], 'pareto_front')
        else:
            # Method gives just a single point
            batch.update(method.eval_step(batch))
            for eval_mode, score in scores.items():
                data = [score[t](**batch) for t in task_ids]
                score_values[eval_mode].update(data, 'single_point')

    # normalize scores and compute hyper-volume
    for v in score_values.values():
        v.normalize()
        if method.preference_at_inference():
            v.compute_hv(settings['reference_point'])
            v.compute_optimal_sol()

    # plot pareto front to pf
    for eval_mode, score in score_values.items():
        if score.pf_available and score.pf.shape[1] == 2:
            pareto_front = utils.ParetoFront(
                ["-".join([str(t), eval_mode]) for t in task_ids], 
                logdir,
                "{}_{}_{:03d}".format(eval_mode, split, e)
            )
            pareto_front.append(score.pf)
            pareto_front.plot()

    result = {k: v.to_dict() for k, v in score_values.items()}
    result.update({"training_time_so_far": train_time,})
    result.update(method.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict


def main(settings):
    print("start processig with settings", settings)
    utils.set_seed(settings['seed'])
    device = settings['device']

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True, num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, len(val_set), shuffle=True, num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, len(test_set), settings['num_workers'])

    objectives = from_name(**settings)
    scores = from_objectives(objectives, with_mcr=False)

    rm1 = utils.RunningMean(400)
    rm2 = utils.RunningMean(400)
    elapsed_time = 0

    model = utils.model_from_dataset(**settings).to(device)
    method = method_from_name(objectives, model, settings)

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(method.model_params()))

    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    # main
    for j in range(settings['num_starts']):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), settings['lr'])
        if settings['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['scheduler_milestones'], gamma=settings['scheduler_gamma'])
        
        for e in range(settings['epochs']):
            print(f"Epoch {e}")
            tick = time.time()
            method.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to(batch, device)
                optimizer.zero_grad()
                stats = method.step(batch)
                optimizer.step()

                loss, sim  = stats if isinstance(stats, tuple) else (stats, 0)
                assert not math.isnan(loss) and not math.isnan(sim)
                print("Epoch {:03d}, batch {:03d}, train_loss {:.4f}, sim {:.4f}, rm train_loss {:.3f}, rm sim {:.3f}".format(e, b, loss, sim, rm1(loss), rm2(sim)))

            tock = time.time()
            elapsed_time += (tock - tick)

            if settings['use_scheduler']:
                val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
                scheduler.step()


            # run eval on train set (mainly for debugging)
            if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
                train_results = evaluate(j, e, method, scores, train_loader,
                    split='train',
                    result_dict=train_results,
                    logdir=logdir,
                    train_time=elapsed_time,
                    settings=settings,)

            
            if settings['eval_every'] > 0 and (e+1) % settings['eval_every'] == 0:
                # Validation results
                val_results = evaluate(j, e, method, scores, val_loader,
                    split='val',
                    result_dict=val_results,
                    logdir=logdir,
                    train_time=elapsed_time,
                    settings=settings,)

                # Test results
                test_results = evaluate(j, e, method, scores, test_loader,
                    split='test',
                    result_dict=test_results,
                    logdir=logdir,
                    train_time=elapsed_time,
                    settings=settings,)

            # Checkpoints
            if settings['checkpoint_every'] > 0 and (e+1) % settings['checkpoint_every'] == 0:
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='celeba', help="The dataset to run on.")
    parser.add_argument('--method', '-m', default='cosmos', help="The method to generate the Pareto front.")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Seed")
    parser.add_argument('--task_id', '-t', default=None, type=int, help='Task id to run single task in parallel. If not set then sequentially.')
    args = parser.parse_args()

    settings = s.generic
    
    if args.method == 'single_task':
        settings.update(s.SingleTaskSolver)
        if args.task_id is not None:
            settings['num_starts'] = 1
            settings['task_id'] = args.task_id
    elif args.method == 'cosmos':
        settings.update(s.cosmos)
    elif args.method == 'hyper_ln':
        settings.update(s.hyperSolver_ln)
    elif args.method == 'hyper_epo':
        settings.update(s.hyperSolver_epo)
    elif args.method == 'pmtl':
        settings.update(s.paretoMTL)
    elif args.method == 'mgda':
        settings.update(s.mgda)
    elif args.method == 'uniform':
        settings.update(s.uniform_scaling)
    
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
    elif args.dataset == 'compass':
        settings.update(s.compass)
    elif args.dataset == 'celeba':
        settings.update(s.celeba)
    
    settings['seed'] = args.seed

    return settings


if __name__ == "__main__":
    
    settings = parse_args()
    main(settings)
