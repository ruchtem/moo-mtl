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
    assert key in ['hv', 'score', 'last']
    if key == 'hv':
        last_epoch = epoch_data[natural_sort(epoch_data.keys())[-1]]
        return last_epoch['max_epoch_so_far']
    elif key == 'score':
        min_score = 1e15
        min_epoch = -1
        for e in natural_sort(epoch_data.keys()):
            if 'scores' in epoch_data[e]:
                s = epoch_data[e]['scores'][0]
                s = s[epoch_data[e]['task']]

                if s < min_score:
                    min_score = s
                    min_epoch = e

        return int(min_epoch.replace('epoch_', ''))
    elif key == 'last':
        last_epoch = natural_sort(epoch_data.keys())[-1]
        return int(last_epoch.replace('epoch_', ''))



def fix_scores_dim(scores):
    scores = np.array(scores)
    if scores.ndim == 1:
        return np.expand_dims(scores, axis=0).tolist()
    if scores.ndim == 2:
        return scores.tolist()
    if scores.ndim == 3:
        return np.squeeze(scores).tolist()
    raise ValueError()


def lists_to_tuples(dict):
    for k, v in dict.items():
        if isinstance(v, list):
            dict[k] = tuple(v)
    return dict


def compare_settings(data):
    sets = [set(lists_to_tuples(d['settings']).items()) for d in data]
    diff = set.difference(*sets)
    
    assert len(diff) == 1, f"Runs or not similar apart from seed! {diff}"
    assert 'seed' in dict(diff)


dirname = 'results_plot/results_paper'

datasets = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']#, 'celeba']
methods = ['SingleTask', 'hyper_epo', 'hyper_ln', 'cosmos_ln']#, 'ParetoMTL']

generating_pareto_front = ['cosmos_ln', 'hyper_ln', 'hyper_epo']

stop_key = {
    'SingleTask': 'score', 
    'hyper_epo': 'hv', 
    'hyper_ln': 'hv', 
    'cosmos_ln': 'last', 
    'ParetoMTL': 'hv', 
}

early_stop = ['hyper_ln', 'hyper_epo', 'SingleTask', 'ParetoMTL']

reference_points = {
    'adult': [2, 2], 'compas': [2, 2], 'credit': [2, 2], 
    'multi_mnist': [2, 2], 'multi_fashion': [2, 2], 'multi_fashion_mnist': [2, 2],
    'celeba': [1 for _ in range(40)]
}

ignore_runs = [

]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf']


def load_files(paths):
    contents = []
    for p in paths:
        with p.open(mode='r') as json_file:
            contents.append(json.load(json_file))
    return contents


def mean_and_std(values):
    return (
        np.array(values).mean(axis=0).tolist(),
        np.array(values).std(axis=0).tolist()
    )


def process_non_pareto_front(data_val, data_test):
    result_i = {}
    # we need to aggregate results from different runs
    result_i['val_scores'] = []
    result_i['test_scores'] = []
    result_i['early_stop_epoch'] = []
    for s in sorted(data_val.keys()):
        if 'start_' in s:
            e = "epoch_{}".format(get_early_stop(data_val[s], key=stop_key[method]))
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
    return result_i


p = Path(dirname)
all_files = list(p.glob('**/*.json'))

results = {}

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        # ignore folders that start with underscore
        val_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/val*.json')))
        test_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/test*.json')))
        train_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/train*.json')))

        for r in ignore_runs:
            val_file = [f for f in val_file if str(r) not in f.parts]
            test_file = [f for f in test_file if str(r) not in f.parts]

        assert len(val_file) == len(test_file)

        data_val = load_files(val_file)
        data_test = load_files(test_file)

        if len(val_file) == 0:
            continue
        elif len(val_file) == 1:
            data_val = data_val[0]
            data_test = data_test[0]
        elif len(val_file) > 1:
            compare_settings(data_val)
            compare_settings(data_test)

        result_i = {}
        if method in generating_pareto_front:

            
            s = 'start_0'
            if isinstance(data_val, list):
                # we have multiple runs of the same method
                early_stop_epoch = []
                val_scores = []
                test_scores = []
                val_hv = []
                test_hv = []
                training_time = []
                for val_run, test_run in zip(data_val, data_test):
                    e = "epoch_{}".format(get_early_stop(val_run[s], key=stop_key[method]))
                    val_results = val_run[s][e]
                    test_results = test_run[s][e]

                    early_stop_epoch.append(int(e.replace('epoch_', '')))
                    val_scores.append(val_results['scores'])
                    test_scores.append(test_results['scores'])
                    val_hv.append(val_results['hv'])
                    test_hv.append(test_results['hv'])
                    training_time.append(test_results['training_time_so_far'])
                
                result_i['early_stop_epoch'] = mean_and_std(early_stop_epoch)
                result_i['val_scores'] = mean_and_std(val_scores)
                result_i['test_scores'] = mean_and_std(test_scores)
                result_i['val_hv'] = mean_and_std(val_hv)
                result_i['test_hv'] = mean_and_std(test_hv)
                result_i['training_time'] = mean_and_std(training_time)
            else:
                # we have just a single run of the method
                assert len([True for k in data_val.keys() if 'start_' in k]) == 1
                e = "epoch_{}".format(get_early_stop(data_val[s], key=stop_key[method]))
                val_results = data_val[s][e]
                test_results = data_test[s][e]

                result_i['early_stop_epoch'] = int(e.replace('epoch_', ''))
                result_i['val_scores'] = val_results['scores']
                result_i['test_scores'] = test_results['scores']
                result_i['val_hv'] = val_results['hv']
                result_i['test_hv'] = test_results['hv']
                result_i['training_time'] = test_results['training_time_so_far']

        else:

            if isinstance(data_val, list):
                early_stop_epoch = []
                val_scores = []
                test_scores = []
                val_hv = []
                test_hv = []
                training_time = []
                for val_run, test_run in zip(data_val, data_test):
                    result_i = process_non_pareto_front(val_run, test_run)
                    early_stop_epoch.append(result_i['early_stop_epoch'])
                    val_scores.append(result_i['val_scores'])
                    test_scores.append(result_i['test_scores'])
                    val_hv.append(result_i['val_hv'])
                    test_hv.append(result_i['test_hv'])
                    training_time.append(result_i['training_time'])
                
                result_i['early_stop_epoch'] = mean_and_std(early_stop_epoch)
                result_i['val_scores'] = mean_and_std(val_scores)
                result_i['test_scores'] = mean_and_std(test_scores)
                result_i['val_hv'] = mean_and_std(val_hv)
                result_i['test_hv'] = mean_and_std(test_hv)
                result_i['training_time'] = mean_and_std(training_time)
            else:
                result_i = process_non_pareto_front(data_val, data_test)



        results[dataset][method] = result_i


with open('results.json', 'w') as outfile:
    json.dump(results, outfile)


# Generate the plots and tables

limits = {
    'adult': [.6, .14],
    'compas': [1.5, .35],
    'credit': [.6, .015],
    'multi_mnist': [.5, .5], 
    'multi_fashion': [.75, .75], 
    'multi_fashion_mnist': [.6, .6],
}


for dataset in datasets:
    if dataset not in results:
        continue
    for i, method in enumerate(methods):
        if method not in results[dataset]:
            continue
        r = results[dataset][method]
        if isinstance(r['test_scores'], tuple):
            s = np.array(r['test_scores'][0])   # we take the mean only
        else:
            s = np.array(r['test_scores'])

        plt.plot(s[:, 0], s[:, 1], '.', color=colors[i], label="{}".format(method))
    
    plt.xlim(right=limits[dataset][0])
    plt.ylim(top=limits[dataset][1])
    plt.title(dataset)
    plt.legend()
    plt.savefig(dataset)
    plt.close()


        