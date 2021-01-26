import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from multi_objective.hv import HyperVolume


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def get_early_stop(epoch_data, key='hv'):
    assert key in ['hv', 'score']
    if key == 'hv':
        last_epoch = epoch_data[natural_sort(epoch_data.keys())[-1]]
        return last_epoch['max_epoch_so_far']
    else:
        min_score = 1e15
        min_epoch = -1
        for e in natural_sort(epoch_data.keys()):

            s = epoch_data[e]['scores'][0]
            s = s[epoch_data[e]['task']]

            if s < min_score:
                min_score = s
                min_epoch = e

        return int(min_epoch.replace('epoch_', ''))


def fix_scores_dim(scores):
    scores = np.array(scores)
    if scores.ndim == 1:
        return np.expand_dims(scores, axis=0).tolist()
    if scores.ndim == 2:
        return scores.tolist()
    if scores.ndim == 3:
        return np.squeeze(scores).tolist()
    raise ValueError()


dirname = 'cluster_results'

datasets = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']#, 'celeba']
methods = ['SingleTask', 'afeature', 'hyper', 'ParetoMTL']

generating_pareto_front = ['afeature', 'hyper']
reference_points = {
    'adult': [1, 1], 'compas': [1, 1], 'credit': [1, 1], 
    'multi_mnist': [2, 2], 'multi_fashion': [2, 2], 'multi_fashion_mnist': [2, 2],
    'celeba': [1 for _ in range(40)]
}

p = Path(dirname)
all_files = list(p.glob('**/*.json'))

results = {}

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        val_file = list(p.glob(f'**/{dataset}/{method}/**/val*.json'))
        test_file = list(p.glob(f'**/{dataset}/{method}/**/test*.json'))
        train_file = list(p.glob(f'**/{dataset}/{method}/**/train*.json'))

        assert len(val_file) == 1
        assert len(test_file) == 1

        with val_file[0].open(mode='r') as json_file:
            data_val = json.load(json_file)
        
        with test_file[0].open(mode='r') as json_file:
            data_test = json.load(json_file)

        result_i = {}
        if method in generating_pareto_front:
            # we have just a single run of the method
            assert len([True for k in data_val.keys() if 'start_' in k]) == 1
            s = 'start_0'
            e = "epoch_{}".format(get_early_stop(data_val[s]))
            val_results = data_val[s][e]
            test_results = data_test[s][e]

            result_i['early_stop_epoch'] = get_early_stop(data_val[s])
            result_i['val_scores'] = val_results['scores']
            result_i['test_scores'] = test_results['scores']
            result_i['val_hv'] = val_results['hv']
            result_i['test_hv'] = test_results['hv']
            result_i['training_time'] = test_results['training_time_so_far']

        else:
            # we need to aggregate results from different runs
            result_i['val_scores'] = []
            result_i['test_scores'] = []
            result_i['early_stop_epoch'] = []
            for s in sorted(data_val.keys()):
                if 'start_' in s:
                    e = "epoch_{}".format(get_early_stop(data_val[s], key='score' if method=='SingleTask' else 'hv'))
                    val_results = data_val[s][e]
                    test_results = data_test[s][e]

                    # the last training time is the correct one, so just override
                    result_i['training_time'] = test_results['training_time_so_far']

                    result_i['early_stop_epoch'].append(int(e.replace('epoch_', '')))

                    if method == 'SingleTask':
                        # we have the task id for the score
                        val_score = val_results['scores'][0][val_results['task']]
                        result_i['val_scores'].append(val_score)
                        test_score = test_results['scores'][0][test_results['task']]
                        result_i['test_scores'].append(test_score)
                    else:
                        # we have no task id
                        result_i['val_scores'].append(val_results['scores'])
                        result_i['test_scores'].append(test_results['scores'])

            result_i['val_scores'] = fix_scores_dim(result_i['val_scores'])
            result_i['test_scores'] = fix_scores_dim(result_i['test_scores'])

            # compute hypervolume
            hv = HyperVolume(reference_points[dataset])
            result_i['val_hv'] = hv.compute(result_i['val_scores'])
            result_i['test_hv'] = hv.compute(result_i['test_scores'])

        results[dataset][method] = result_i


with open('results.json', 'w') as outfile:
    json.dump(results, outfile)


# Generate the plots and tables
for dataset in datasets:
    for method in methods:
        r = results[dataset][method]
        s = np.array(r['test_scores'])
        plt.plot(s[:, 0], s[:, 1], '.', label="{}, hv={:.4f}, e={}".format(method, r['test_hv'], r['early_stop_epoch']))
    
    plt.title(dataset)
    plt.legend()
    plt.savefig(dataset)
    plt.close()


        