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
import logging
import os
import pathlib
import time
import json
import math

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rtb import log_every_n_seconds, log_first_n, setup_logger

from multi_objective import defaults, utils
from multi_objective.objectives import from_name

from multi_objective.methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod, NSGA2Method
from multi_objective.scores import from_objectives


def method_from_name(method, objectives, model, cfg):
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
    if method == 'pmtl':
        assert cfg.dataset not in ['celeba', 'cityscapes'], f"Not supported"
        return ParetoMTLMethod(objectives, model, cfg)
    elif method == 'cosmos':
        return COSMOSMethod(objectives, model, cfg)
    elif method == 'single_task':
        return SingleTaskMethod(objectives, model, cfg)
    elif 'phn' in method:
        assert cfg.dataset not in ['celeba', 'cityscapes'], f"Not supported"
        return HypernetMethod(objectives, model, cfg)
    elif method == 'mgda':
        return MGDAMethod(objectives, model, cfg)
    elif method == 'nsga2':
        return NSGA2Method(objectives, model, cfg)
    elif method == 'uniform':
        return UniformScalingMethod(objectives, model, **cfg)
    else:
        raise ValueError("Unkown method {}".format(method))


def evaluate(j, e, method, scores, data_loader, split, result_dict, logdir, train_time, cfg, logger):
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
    
    if len(cfg.task_ids) > 0:
        J = len(cfg['task_ids'])
        task_ids = cfg['task_ids']
    else:
        # single output setting
        J = len(cfg['objectives'])
        task_ids = list(scores[list(scores)[0]].keys())

    pareto_rays = utils.reference_points(cfg['n_partitions'], dim=J, min=[0.2, 0.2])
    n_rays = pareto_rays.shape[0]
    
    log_first_n(logging.DEBUG, f"Number of test rays: {n_rays}", n=1)
    
    # gather the scores
    score_values = {et: utils.EvalResult(J, n_rays) for et in scores.keys()}
    if hasattr(method, 'eval_all'):
        # hack to handle nsga2
        score_data = {t: method.eval_all(data_loader, scores[t]) for t in scores.keys()}
        for k, v in score_data.items():
            score_values[k].pf = v
            score_values[k].j = 1
            score_values[k].pf_available = True
    else:
        for b, batch in enumerate(data_loader):
            log_every_n_seconds(logging.INFO, f"Eval {b} of {len(data_loader)}", n=5)
            batch = utils.dict_to(batch, cfg['device'])
                    
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
            v.compute_hv(cfg['reference_point'])
            v.compute_optimal_sol()
            v.compute_dist()


    # plot pareto front to pf
    for eval_mode, score in score_values.items():
        pareto_front = utils.ParetoFront(
            ["-".join([str(t), eval_mode]) for t in task_ids], 
            logdir,
            "{}_{}_{:03d}".format(eval_mode, split, e)
        )
        if score.pf_available:
            pareto_front.plot(score.pf, best_sol_idx=score.optimal_sol_idx)
        else:
            pareto_front.plot(score.center)

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


def setup(rank, world_size):
    assert world_size > 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, method_name, cfg, tag=''):
    cfg.freeze()
    if world_size > 1:
        setup(rank, world_size)
    print(rank, world_size)

    # create the experiment folders
    logdir = os.path.join(cfg['logdir'], method_name, cfg['dataset'], utils.get_runname(cfg) + f'_{tag}')
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(os.path.join(logdir, 'exp.log'), name=__name__, distributed_rank=rank)
    logger.info(f"start experiment with settings: \n{cfg}")
    logger.info(f"\n\n>>>> Running method {method_name} <<<<\n")

    torch.cuda.set_device(rank)

    # prepare
    utils.set_seed(cfg['seed'])
    lr_scheduler = cfg[method_name].lr_scheduler

    objectives = from_name(**cfg)
    scores = from_objectives(objectives, **cfg)

    train_loader, val_loader, test_loader, sampler = utils.loaders_from_name(**cfg)

    rm1 = utils.RunningMean(len(train_loader))
    rm2 = utils.RunningMean(len(train_loader))
    elapsed_time = 0

    model = utils.model_from_dataset(**cfg).to(cfg.device)
    method = method_from_name(method_name, objectives, model, cfg)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    train_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))

    if rank == 0:
        with open(pathlib.Path(logdir) / "settings.json", "w") as file:
            json.dump(train_results, file)

    # main
    num_starts = cfg[method_name].num_starts if 'num_starts' in cfg[method_name] else 1
    for j in range(num_starts):
        if rank == 0:
            train_results[f"start_{j}"] = {}
            val_results[f"start_{j}"] = {}
            test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), cfg[method_name].lr, weight_decay=1e-3)
        # optimizer = torch.optim.SGD(method.model_params(), cfg[method_name].lr, weight_decay=1e-4)
        scheduler = utils.get_lr_scheduler(lr_scheduler, optimizer, cfg, tag)
        
        for e in range(cfg['epochs']):
            if world_size > 1:
                # Keep the gpus in sync, especially after evaluation
                torch.distributed.barrier()

            tick = time.time()
            if world_size > 1:
                sampler.set_epoch(e)
            method.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to(batch, cfg.device)
                optimizer.zero_grad()
                stats = method.step(batch)
                
                optimizer.step()

                loss, sim, norm  = stats if isinstance(stats, tuple) else (stats, 0, 0)
                assert not math.isnan(loss) and not math.isnan(sim)

                log_every_n_seconds(logging.INFO, 
                    f"Epoch {e:03d}, batch {b:03d}, train_loss {loss:.4f}, rm train_loss {rm1(loss):.3f}, rm sim {rm2(sim):.3f}",
                    n=5
                )

            tock = time.time()
            elapsed_time += (tock - tick)

            if rank == 0:
                val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
            scheduler.step()

            # run eval on train set (mainly for debugging)
            if rank == 0:
                if cfg['train_eval_every'] > 0 and (e+1) % cfg['train_eval_every'] == 0 and e > 1:
                    train_results = evaluate(j, e, method, scores, train_loader,
                        split='train',
                        result_dict=train_results,
                        logdir=logdir,
                        train_time=elapsed_time,
                        cfg=cfg,
                        logger=logger,)

                
                if cfg['eval_every'] > 0 and (e+1) % cfg['eval_every'] == 0 and e > 1:
                    # Validation results
                    val_results = evaluate(j, e, method, scores, val_loader,
                        split='val',
                        result_dict=val_results,
                        logdir=logdir,
                        train_time=elapsed_time,
                        cfg=cfg,
                        logger=logger,)

                # Test results
                # test_results = evaluate(j, e, method, scores, test_loader,
                #     split='test',
                #     result_dict=test_results,
                #     logdir=logdir,
                #     train_time=elapsed_time,
                #     settings=settings,)

            # Checkpoints
            if rank == 0:
                if cfg['checkpoint_every'] > 0 and (e+1) % cfg['checkpoint_every'] == 0:
                    pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                    torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))
        if world_size > 1:
            cleanup()
        
        if rank == 0:
            pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
            torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))
    
            epoch_results = val_results['start_0'][f'epoch_{e}']['loss']
            if 'hv' in epoch_results:
                return epoch_results['hv'], epoch_results['dist']
            else:
                # maximize the negative norm, i.e. min norm
                return -np.linalg.norm(epoch_results['center_ray'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Config file.")
    parser.add_argument('--method', default='cosmos', type=str, help="The method to generate the Pareto front.")
    parser.add_argument('--tag', default='', type=str, help="Experiment tag")
    parser.add_argument('--ngpus', default=1, type=int, help="num gpus for distributed training")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    tag = args.tag
    if args.method == 'single_task':
        tag += str(cfg.single_task.task_id)
    return args, tag

    
def get_config(config_file, opts=[]):
    cfg = defaults.get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    return cfg


if __name__ == "__main__":

    args, tag = parse_args()
    cfg = get_config(args.config, args.opts)

    world_size = args.ngpus
    
    if world_size > 1:

        # Rule of thumb to adapt lr as effectivly batch_size * world_size
        print("Adapting learning rate to distributed training.")
        cfg[args.method].lr *= world_size

        # TODO: torch.distributed.launch

        mp.spawn(main,
            args=(world_size, args.method, cfg, tag),
            nprocs=world_size,
            join=True
        )
    else:
        res = main(0, world_size, args.method, cfg, tag)
        print(res)
