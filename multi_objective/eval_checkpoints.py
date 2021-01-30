import torch
import numpy as np
import os
import pathlib
import itertools
import json
import torch.utils.data as data

from datetime import datetime

import utils
from main import parse_args, set_seed, solver_from_name
from scores import from_objectives, mcr
from objectives import from_name
from hv import HyperVolume




def evaluate(j, e, solver, scores, data_loader, logdir, reference_point, split, result_dict):
    assert split in ['train', 'val', 'test']

    # n_test_rays = 25
    n_test_rays = 1

    # sample two columns randomly and vary those
    i = np.random.randint(40, size=2)
    print(i)

    # circle_points = utils.circle_points(n_test_rays, dim=2)
    # test_rays = np.full((n_test_rays, 40), 0.001)
    # test_rays[:, i] = circle_points

    test_rays = np.ones((n_test_rays, 40))

    test_rays /= test_rays.sum(axis=1).reshape(n_test_rays, 1)
    print(test_rays)

    score_values = np.array([])
    
    for k, batch in enumerate(data_loader):
        print(f'eval batch {k+1} of {len(data_loader)}')
        batch = utils.dict_to_cuda(batch)
        
        # more than one for some solvers
        s = []
        for l in solver.eval_step(batch, test_rays):
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
        l = 0
        for columns in itertools.combinations(range(m), 2):
            if columns[0] in i or columns[1] in i:
                volume += hv.compute(score_values[:, columns])

                pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores], logdir, "{}_{:03d}_{}".format(split, e, columns))
                pareto_front.append(score_values[:, columns])
                pareto_front.plot()
                l += 1
        volume /= l
    else:
        volume = hv.compute(score_values)

    result = {
        "scores": score_values.tolist(),
        "hv": volume,
    }

                    
    result.update({
        "max_epoch_so_far": -1,
        "max_volume_so_far": -1,
        "training_time_so_far": -1,
    })

    result.update(solver.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict









def eval(settings):
    settings['batch_size'] = 2048

    print("start evaluation with settings", settings)
    #set_seed(settings['seed'])

    # create the experiment folders
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], slurm_job_id if slurm_job_id else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    # train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    #test_set = utils.dataset_from_name(split='test', **settings)

    # train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    #test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), val_set.task_names())
    scores1 = from_objectives(objectives)
    scores2 = [mcr(o.label_name, o.logits_name) for o in objectives]

    solver = solver_from_name(objectives=objectives, **settings)

    # train_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))

    j = 0
    checkpoint_dir = 'results_celeba/cosmos_ln/celeba/2126915/checkpoints'
    checkpoints = pathlib.Path(checkpoint_dir).glob('**/c_*.pth')
    c = list(sorted(checkpoints))[-1]
    print("cechlpoint", c)
        
    solver.model.load_state_dict(torch.load(c))

    j, e = c.stem.replace('c_', '').split('-')
    j = int(j)
    e = int(e)

    val_results[f"start_{j}"] = {}
    test_results[f"start_{j}"] = {}

    # run eval on train set (mainly for debugging)
    # if settings['train_eval_every'] > 0 and (e+1) % settings['train_eval_every'] == 0:
    #     train_results = evaluate(j, e, solver, scores, train_loader, logdir, 
    #         reference_point=settings['reference_point'],
    #         split='train',
    #         result_dict=train_results)

    
    # Validation results
    val_results = evaluate(j, e, solver, scores2, val_loader, logdir, 
        reference_point=settings['reference_point'],
        split='val',
        result_dict=val_results)

    # Test results
    # test_results = evaluate(j, e, solver, scores, test_loader, logdir, 
    #     reference_point=settings['reference_point'],
    #     split='test',
    #     result_dict=test_results)


    print()






if __name__ == "__main__":

    settings = parse_args()
    eval(settings)