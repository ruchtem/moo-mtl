from main import main
import settings as s 
import json
import os

datasets = ['adult', 'compas', 'credit', 'mm', 'mfm', 'fm']
config_space = {
    'alpha_dir': [0.05, .1, .15, .2, .25, .3, .35, .4,]
}

results = {}
for k, v in config_space.items():
    results[k] = {}
    for u in v:
        results[k][u] = []
        for dataset in datasets:

            settings = s.generic
            settings.update(s.cosmos)

            if dataset == 'adult':
                settings.update(s.adult)
            elif dataset == 'credit':
                settings.update(s.credit)
            elif dataset == 'compas':
                settings.update(s.compas)
            elif dataset == 'mm':
                settings.update(s.multi_mnist)
            elif dataset == 'mfm':
                settings.update(s.multi_fashion_mnist)
            elif dataset == 'fm':
                settings.update(s.multi_fashion)
            
            settings[k] = u
            settings['logdir'] = 'hpo_results'
            settings['train_eval_every'] = 0

            score = main(settings)
            results[k][u].append(score)
    
            with open(os.path.join(settings['logdir'], 'hp_results.json'), 'w') as outfile:
                json.dump(results, outfile)

