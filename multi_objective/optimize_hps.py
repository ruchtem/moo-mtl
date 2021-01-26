from main import main
import settings as s 
import json

datasets = ['adult', 'compas', 'credit']
config_space = {
    'alpha_dir': [.2, .5, .8, 1., 1.2, 1.5, 2., 3., 5., None]
}

results = {}
for k, v in config_space.items():
    results[k] = {}
    for u in v:
        results[k][u] = []
        for dataset in datasets:

            settings = s.generic
            settings.update(s.afeature)

            if dataset == 'adult':
                settings.update(s.adult)
            elif dataset == 'credit':
                settings.update(s.credit)
            elif dataset == 'compas':
                settings.update(s.compas)
            
            settings[k] = u

            score = main(settings)
            results[k][u].append(score)
    
            with open('hp_results.json', 'w') as outfile:
                json.dump(results, outfile)

            
    
