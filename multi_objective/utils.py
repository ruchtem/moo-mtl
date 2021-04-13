import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

from torch.utils.data import DataLoader
from datetime import datetime
from pymoo.factory import get_decomposition, get_reference_directions, get_performance_indicator
from pymoo.visualization.radviz import Radviz

from loaders import adult_loader, compas_loader, multi_mnist_loader, celeba_loader, credit_loader, cityscapes_loader, coco_loader
from models import FullyConnected, MultiLeNet, EfficientNet, ResNet, Pspnet, MaskRCNN

def dataset_from_name(dataset, **kwargs):
    if dataset == 'adult':
        return adult_loader.ADULT(**kwargs)
    elif dataset == 'credit':
        return credit_loader.Credit(**kwargs)
    elif dataset == 'compass':
        return compas_loader.Compas(**kwargs)
    elif dataset == 'multi_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='mnist', **kwargs)
    elif dataset == 'multi_fashion':
        return multi_mnist_loader.MultiMNIST(dataset='fashion', **kwargs)
    elif dataset == 'multi_fashion_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='fashion_and_mnist', **kwargs)
    elif dataset == 'celeba':
        return celeba_loader.CelebA(**kwargs)
    elif dataset == 'cityscapes':
        return cityscapes_loader.CITYSCAPES(**kwargs)
    elif dataset == 'coco':
        return coco_loader.COCO(**kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def loaders_from_name(dataset, **kwargs):
    train = dataset_from_name(dataset, split='train', **kwargs)
    val = dataset_from_name(dataset, split='val', **kwargs)
    test = dataset_from_name(dataset, split='test', **kwargs)
    if dataset in ['adult', 'credit', 'compass', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']:
        val_bs = len(val)
        test_bs = len(test)
    else:
        val_bs = kwargs['batch_size']
        test_bs = kwargs['batch_size']
    return (
        DataLoader(train, kwargs['batch_size'], shuffle=True, num_workers=kwargs['num_workers']),
        DataLoader(val, val_bs, num_workers=kwargs['num_workers']),
        DataLoader(test, test_bs, num_workers=kwargs['num_workers']),
    )


def model_from_dataset(dataset, **kwargs):
    if dataset == 'adult' or dataset == 'credit' or dataset == 'compass':
        return FullyConnected(**kwargs)
    elif dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    elif dataset == 'celeba':
        if 'efficientnet' in kwargs['model_name']:
            return EfficientNet.from_pretrained(**kwargs)
        elif kwargs['model_name'] == 'resnet18':
            return ResNet.from_name(**kwargs)
    elif dataset == 'cityscapes':
        return Pspnet(dim=kwargs['dim'])
    elif dataset == 'coco':
        return MaskRCNN(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


def reference_points(partitions, dim=2):
    """generate evenly distributed preference vector"""
    return get_reference_directions("uniform", dim, n_partitions=partitions)


def optimal_solution(pareto_front, weights=None):
    """Compromise programming from pymoo"""
    if weights:
        assert len(weights) == pareto_front.shape[1]
    else:
        dim = pareto_front.shape[1]
        weights = np.ones(dim) / dim
    
    idx = get_decomposition("asf").do(pareto_front, weights).argmin()
    return idx, pareto_front[idx]


def num_parameters(params):
    if isinstance(params, torch.nn.Module):
        params = params.parameters()
    model_parameters = filter(lambda p: p.requires_grad, params)
    return int(sum([np.prod(p.size()) for p in model_parameters]))


def angle(grads):
    grads = flatten_grads(grads)
    return torch.cosine_similarity(grads[0], grads[1], dim=0)


def flatten_grads(grads):
    result = []
    for grad in grads:
        flatten = torch.cat(([torch.flatten(g) for g in grad.values()]))
        result.append(flatten)
    return result


def reset_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()


def dict_to(dict, device):
    if isinstance(dict, list):
        # we have a list of dicts
        return [dict_to(d, device) for d in dict]
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict.items()}


def normalize_score_dict(d, divisor):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, int) or isinstance(v, float):
            d[k] = v/divisor
        elif isinstance(v, list):
            v = np.array(v) / divisor
            d[k] = v.tolist()
        elif isinstance(v, dict):
            d[k] = normalize_score_dict(v, divisor)
    return d



def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_runname(settings):
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    if slurm_job_id:
        runname = f"{slurm_job_id}"
        if 'ablation' in settings['logdir']:
            runname += f"_{settings['lamda']}_{settings['alpha']}"
    else:
        runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if 'task_id' in settings:
        runname += f"_{settings['task_id']:03d}"
    return runname


def calc_gradients(batch, model, objectives):
    # store gradients and objective values
    gradients = {t: {} for t in objectives}
    obj_values = {t: None for t in objectives}
    for t, objective in objectives.items():
        # zero grad
        model.zero_grad()
        
        logits = model(batch)
        batch.update(logits)

        output = objective(**batch)
        output.backward()
        
        obj_values[t] = output.item()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[t][name] = param.grad.data.detach().clone()
    
    return gradients, obj_values


class EvalResult():

    def __init__(self, J, n_test_rays, task_ids) -> None:
        self.task_ids = task_ids
        self.center = np.zeros(J)
        self.pf = np.zeros((n_test_rays, J))
        self.hv = -1
        self.pf_available = False
        self.i = 0
        self.j = 0
        self.optimal_sol = None

    def update(self, data, mode):
        if mode == 'single_point':
            self.center += data
            self.i += 1
        elif mode == 'pareto_front':
            self.pf_available = True
            self.pf += data
            self.j += 1
        else:
            raise ValueError(f"Unknown eval mode {mode}")

        
    
    def normalize(self):
        if self.i > 0:
            self.center /= self.i
        if self.j > 0:
            self.pf /= self.j

    
    def compute_hv(self, reference_point):
        if self.pf_available:
            hv = get_performance_indicator("hv", ref_point=np.array(reference_point))
            self.hv = hv.calc(self.pf)


    def compute_optimal_sol(self, weights=None):
        if self.pf_available:
            self.optimal_sol_idx, self.optimal_sol = optimal_solution(self.pf, weights)


    def to_dict(self):
        result = {'center_ray': self.center.tolist(),}
        if self.pf_available:
            result.update({
                'pareto_front': self.pf.tolist(),
                'hv': self.hv,
                'optimal_solution': self.optimal_sol.tolist(),
                'optimal_solution_idx': int(self.optimal_sol_idx),
            })

        return result


class RunningMinMaxNormalizer():

    def __init__(self, history_len: int=200) -> None:
        super().__init__()
        self.history = collections.deque(maxlen=history_len)


    def update(self, data):
        data = np.array(data)
        assert data.ndim == 1
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].item()
        
        self.history.append(data)
    

    def normalize(self, data, exploration=.1):
        minimum = np.array(self.history, dtype=np.float).min(axis=0)
        maximum = np.array(self.history, dtype=np.float).max(axis=0)
        diff = maximum - minimum
        diff = diff + 2*exploration*diff    # extend the range for exploration
        result = diff * (data) + minimum
        result[result < 0] = 0
        return result




class RunningMean():

    def __init__(self, len=100) -> None:
        super().__init__()
        self.queue = collections.deque(maxlen=len)


    def append(self, x):
        self.queue.append(x)


    def __call__(self, x=None, axis=None):
        if x is not None:
            self.queue.append(x)
        data = np.array(self.queue)
        return data.mean().item() if axis == None else data.mean(axis=axis)


class ParetoFront():


    def __init__(self, labels, logdir='tmp', prefix=""):
        self.labels = labels
        self.logdir = os.path.join(logdir, 'pf')
        self.prefix = prefix
        self.points = np.array([])

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
    
    
    def plot(self, p, rays=None, best_sol_idx=None):
        if p.ndim == 1:
            p = np.expand_dims(p, 0)

        # plot 2d pf
        if p.shape[1] == 2:
            plt.plot(p[:, 0], p[:, 1], 'o')
            if rays:
                for r in rays:
                    plt.plot([0, r[0]], [0, r[1]], color='black')
            plt.xlabel(self.labels[0])
            plt.ylabel(self.labels[1])
            plt.grid()
            plt.savefig(os.path.join(self.logdir, "pf_{}.png".format(self.prefix)))
            plt.close()

        # plot radviz
        radviz_plot = Radviz().add(p)
        if best_sol_idx is not None:
            radviz_plot.add(p[best_sol_idx], color="red", s=70, label="Solution A")
        radviz_plot.save(os.path.join(self.logdir, "rad_{}.png".format(self.prefix)))
        plt.close()

        norms = np.linalg.norm(p, axis=1)

        plt.plot(norms, 'o')
        if best_sol_idx is not None:
            plt.plot(norms[best_sol_idx], 'ro')
        plt.savefig(os.path.join(self.logdir, "norm_{}.png".format(self.prefix)))
        plt.close()
