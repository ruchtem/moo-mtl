import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pathlib import Path
import re
from pymoo.factory import get_performance_indicator

#
# Helper functions
#

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def get_early_stop(epoch_data, key='hv'):
    assert key in ['hv', 'score', 'last']
    if key == 'hv':
        best_hv = 0
        best_idx = None
        for e in natural_sort(epoch_data.keys()):
            if 'epoch' in e and 'loss' in epoch_data[e]:
                hv = epoch_data[e]['loss']['hv']
                if hv > best_hv:
                    best_hv = hv
                    best_idx = e
        return best_idx
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

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_hv(sol):
    hv = get_performance_indicator("hv", ref_point=np.array([1, 1]))
    return hv.calc(np.array(sol))


#
# Plotting params
#
font_size = 12
figsize=(14, 3.5)

plt.rcParams.update({'font.size': font_size})
plt.tight_layout()

markers = {
    'mgda': 'x', 
    'uniform': '^',
    'phn': '.', 
    'cosmos': 'd', 
    'pmtl': '*'
}

colors = {
    'single_task': '#1f77b4', 
    'mgda': '#ff7f0e', 
    'phn': '#2ca02c',
    'cosmos': '#d62728',
    'pmtl': '#9467bd', 
    'uniform': '#9467bd', 
}

titles = {
    'adult': 'Adult',
    'compas': 'Compass',
    'credit': 'Default', 
    'multi_mnist': "Multi-MNIST", 
    'multi_fashion': 'Multi-Fashion',
    'multi_fashion_mnist': 'Multi-Fashion+MNIST'
}

ax_lables = {
    'adult': ('Binary Cross-Entropy Loss', 'DEO'),
    'compas': ('Binary Cross-Entropy Loss', 'DEO'),
    'credit': ('Binary Cross-Entropy Loss', 'DEO'), 
    'multi_mnist': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
    'multi_fashion': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
    'multi_fashion_mnist': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
}

method_names = {
    'single_task': 'Single Task', 
    'phn': 'PHN',
    'cosmos': 'COSMOS',
    'pmtl': 'ParetoMTL', 
    'uniform': 'Uniform',
    'mgda': 'MGDA'
}

limits_baselines = {
    # dataset: [left, right, bottom, top]
    'adult': [.3, .6, -0.01, .14],
    'compas': [0, 1.5, -.01, .35],
    'credit': [.42, .65, -0.001, .017],
    'multi_mnist': [.2, .5, .2, .5], 
    'multi_fashion': [.35, .75, .4, .75], 
    'multi_fashion_mnist': [.1, .6, .3, .6],
}


#
# Load the data
#

def load_data(
    dirname='results', 
    datasets=['multi_mnist', 'adult', 'compas', 'credit', 'multi_fashion', 'multi_fashion_mnist'],
    methods= ['uniform', 'single_task', 'phn', 'pmtl', 'mgda'],
    ):

    p = Path(dirname)
    results = {}

    for dataset in datasets:
        results[dataset] = {}
        for method in methods:
            val_file = list(sorted(p.glob(f'**/{method}/{dataset}/result_*/val*.json')))
            test_file = list(sorted(p.glob(f'**/{method}/{dataset}/result_*/test*.json')))

            if len(val_file) == 0:
                continue
            assert len(val_file) == len(test_file)

            data_val = load_files(val_file)
            data_test = load_files(test_file)

            test_scores = []
            test_hv = []
            training_time = []

            if method == 'single_task':
                for (val_run_1, val_run_2), (test_run_1, test_run_2) in zip(pairwise(data_val), pairwise(data_test)):
                    e1 = get_early_stop(val_run_1)
                    e2 = get_early_stop(val_run_2)
                    r1 = test_run_1[e1]
                    r2 = test_run_2[e2]

                    sol = [r1['loss']['center_ray'][0], r2['loss']['center_ray'][1]]
                    test_scores.append(sol)
                    test_hv.append(get_hv(sol))
                    training_time.append(r1['training_time_so_far'] + r2['training_time_so_far'])

                    test_run = test_run_1
            else:
                for val_run, test_run in zip(data_val, data_test):
                    e = get_early_stop(val_run)
                    r = test_run[e]
                    test_scores.append(r['loss']['pareto_front'] if 'pareto_front' in r['loss'] else r['loss']['center_ray'])
                    test_hv.append(r['loss']['hv'])
                    training_time.append(r['training_time_so_far'])

            results[dataset][method] = {
                'test_scores': mean_and_std(test_scores),
                'test_hv': mean_and_std(test_hv),
                'train_time': mean_and_std(training_time),
                'num_parameters': test_run['num_parameters'],
            }
        print(f'loaded data for {dataset}')
    return results
        


#
# Generate the plots and tables
#

def plot_row(results, datasets, methods=['cosmos', 'uniform', 'single_task', 'phn', 'pmtl', 'mgda'], prefix=''):
    assert len(datasets) == 3
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for j, dataset in enumerate(datasets):
        if dataset not in results:
            continue
        ax = axes[j]
        lower_limit = None
        for method in methods:
            if method not in results[dataset]:
                continue
            r = results[dataset][method]
            # we take the mean only
            s = np.array(r['test_scores'][0])
            if s.ndim == 1:
                s = np.expand_dims(s, 0)
            if method == 'single_task':
                s = np.squeeze(s)
                ax.axvline(x=s[0], color=colors[method], linestyle='-.')
                ax.axhline(y=s[1], color=colors[method], linestyle='-.', label="{}".format(method_names[method]))
                lower_limit = s
            else:
                print(method)
                ax.plot(
                    s[:, 0], 
                    s[:, 1], 
                    color=colors[method],
                    marker=markers[method],
                    linestyle='--' if method != 'ParetoMTL' else ' ',
                    label="{}".format(method_names[method])
                )
                
                # if dataset == 'multi_fashion' and method == 'cosmos_ln' and prefix == 'cosmos':
                #     axins = zoomed_inset_axes(ax, 7, loc='upper right')
                #     axins.plot(
                #         s[:, 0], 
                #         s[:, 1], 
                #         color=colors[method],
                #         marker=markers[method],
                #         linestyle='--' if method != 'ParetoMTL' else '',
                #         label="{}".format(method_names[method])
                #     )
                #     axins.set_xlim(.4658, .492)
                #     axins.set_ylim(.488, .513)
                #     axins.set_yticklabels([])
                #     axins.set_xticklabels([])
                #     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        lim = limits_baselines[dataset]
        ax.set_xlim(left=lower_limit[0] - .05 if lower_limit is not None else lim[0], right=lim[1])
        ax.set_ylim(bottom=lower_limit[1] - .05 if lower_limit is not None else lim[2], top=lim[3])
        ax.set_title(titles[dataset])
        ax.set_xlabel(ax_lables[dataset][0])
        if j==0:
            ax.set_ylabel(ax_lables[dataset][1])

        if j==2:
            ax.legend(loc='upper right')
    plt.subplots_adjust(wspace=.25)
    fig.savefig(prefix + '_' + '_'.join(datasets) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('success. See', prefix + '_' + '_'.join(datasets) + '.pdf')


# results = load_data(dirname='results_size_2.0', datasets=['multi_mnist', 'multi_fashion', 'multi_fashion_mnist'])
# plot_row(results, datasets=['multi_mnist', 'multi_fashion', 'multi_fashion_mnist'], prefix='size_2')


#
# generating the tables
#

def generate_table(results, datasets, methods, name):
    text = f"""
\\toprule
                & Hyper Vol. & Time (Sec) & \\# Params. \\\\ \\midrule"""
    for dataset in datasets:
        text += f"""
                & \\multicolumn{{3}}{{c}}{{\\bf {titles[dataset]}}} \\\\ \cmidrule{{2-4}}"""
        for method in methods:
            r = results[dataset][method]
            text += f"""
{method_names[method]}    & {r['test_hv'][0]:.2f} $\pm$ {r['test_hv'][1]:.2f}        & {r['training_time'][0]:,.0f}          &  {r['num_parameters']//1000:,d}k \\\\ """

    text += f"""
\\bottomrule"""
    
    with open(f'results_{name}.txt', 'w') as f:
        f.writelines(text)




# datasets1 = ['adult', 'compas', 'credit']
# generate_table(datasets1, methods, name='fairness')

# datasets2 = ['multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
# generate_table(datasets2, methods, name='mnist')
