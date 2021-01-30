import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pathlib import Path
import re
from multi_objective.hv import HyperVolume

from plotting import (
    font_size,
    markers,
    colors,
    natural_sort,
    lists_to_tuples,
    compare_settings,
    stop_key,
    get_early_stop,
    dirname,
    methods,
    ignore_runs,
    figsize,
    titles,
    ax_lables,
    reference_points,
    load_files,
    mean_and_std,
)

datasets = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods = ['cosmos_ln']


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

dirname = "results_ablation/"

p = Path(dirname)
all_files = list(p.glob('**/*.json'))

results = {}

lambdas = [0, .5, .75, 1]
alphas = [.2, .5, 1]

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        results[dataset][method] = {}
        for l in lambdas:
            results[dataset][method][l] = {}
            for a in alphas:

                val_file = list(sorted(p.glob(f'**/{method}/{dataset}/*_{l}_{a}/val*.json')))
                test_file = list(sorted(p.glob(f'**/{method}/{dataset}/*_{l}_{a}/test*.json')))
                train_file = list(sorted(p.glob(f'**/{method}/{dataset}/*_{l}_{a}/train*.json')))

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

                results[dataset][method][l][a] = {}

                s = 'start_0'
                result_i = {}
                if isinstance(data_val, list):
                    result_i['num_parameters'] = data_val[0]['num_parameters']
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
                    result_i['num_parameters'] = data_val['num_parameters']
                    e = "epoch_{}".format(get_early_stop(data_val[s], key=stop_key[method]))
                    val_results = data_val[s][e]
                    test_results = data_test[s][e]

                    result_i['early_stop_epoch'] = int(e.replace('epoch_', ''))
                    result_i['val_scores'] = val_results['scores']
                    result_i['test_scores'] = test_results['scores']
                    result_i['val_hv'] = val_results['hv']
                    result_i['test_hv'] = test_results['hv']
                    result_i['training_time'] = test_results['training_time_so_far']
                
                results[dataset][method][l][a] = result_i


with open('results_ablation.json', 'w') as outfile:
    json.dump(results, outfile)


# Generate the plots and tables
plt.rcParams.update({'font.size': font_size})

limits = {
    # dataset: [left, right, bottom, top]
    'multi_mnist': [.1, 2.5, 0.2, 3], 
    'multi_fashion': [.35, 2.5, .35, 3], 
    'multi_fashion_mnist': [.1, 2, .35, 2],
}

ax_lables = {
    'adult': ('Loss', 'DEO'),
    'compas': ('Loss', 'DEO'),
    'credit': ('Loss', 'DEO'), 
    'multi_mnist': ('Loss Task TL', 'Loss Task BR'), 
    'multi_fashion': ('Loss Task TL', 'Loss Task BR'), 
    'multi_fashion_mnist': ('Loss Task TL', 'Loss Task BR'), 
}

# change size cause of suptitle
figsize = (figsize[0], figsize[1] * .88)

def plot_ablation(datasets, methods, lambdas, alphas):
    for dataset in datasets:
        fig, axes = plt.subplots(1, len(lambdas), figsize=figsize, sharex=True, sharey=True)
        for j, l in enumerate(lambdas):
            ax = axes[j]
            for method in methods:
                if method not in results[dataset]:
                    continue
                
                for a in alphas:

                #color_shades = np.linspace(1.7, .3, len(epochs)).tolist()

                    r = results[dataset][method][l][a]
                    # we take the mean only
                    s = np.array(r['test_scores'][0]) if isinstance(r['test_scores'], tuple) else np.array(r['test_scores'])
                    ax.plot(
                        s[:, 0], 
                        s[:, 1], 
                        #color=adjust_lightness(colors[method], amount=color_shades[i]),
                        #marker=markers[method],
                        marker='.',
                        linestyle='--' if method != 'ParetoMTL' else ' ',
                        label=r"$\alpha = $" + str(a)   # r"$\lambda = $" + str(l) + 
                    )
            if dataset in limits:
                lim = limits[dataset]
                ax.set_xlim(left=lim[0], right=lim[1])
                ax.set_ylim(bottom=lim[2], top=lim[3])

            ax.set_title(r"$\lambda = $" + f"{l}")
            ax.set_xlabel(ax_lables[dataset][0])
            if j==0:
                ax.set_ylabel(ax_lables[dataset][1])
            ax.legend(loc='upper right')
        fig.suptitle(titles[dataset], y=1.09)
        fig.savefig(f'ablation_{dataset}.pdf' , bbox_inches='tight')
        plt.close(fig)

datasets1 = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods1 = ['cosmos_ln']


plot_ablation(datasets1, methods1, lambdas, alphas)
                
