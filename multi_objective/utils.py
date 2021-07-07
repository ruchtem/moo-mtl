import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pymoo.factory import get_decomposition, get_reference_directions, get_performance_indicator

from .loaders import multi_mnist_loader
from .models import MultiLeNet

def dataset_from_name(dataset, **kwargs):
    if dataset == 'multi_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='mnist', **kwargs)
    elif dataset == 'multi_fashion':
        return multi_mnist_loader.MultiMNIST(dataset='fashion', **kwargs)
    elif dataset == 'multi_fashion_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='fashion_and_mnist', **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def loaders_from_name(dataset, seed, **kwargs):
    train = dataset_from_name(dataset, split='train', **kwargs)
    val = dataset_from_name(dataset, split='val', **kwargs)
    test = dataset_from_name(dataset, split='test', **kwargs)

    val_bs = kwargs['batch_size']
    test_bs = kwargs['batch_size']
    
    if dist.is_initialized():
        # We are in distributed setting
        sampler = DistributedSampler(train, shuffle=True, seed=seed)
        sampler_val = DistributedSampler(val, shuffle=False, seed=seed)
        sampler_test = DistributedSampler(test, shuffle=False, seed=seed)
    else:
        sampler = None
        sampler_val = None
        sampler_test = None

    return (
        DataLoader(train, kwargs['batch_size'], shuffle=(sampler is None), sampler=sampler, num_workers=kwargs['num_workers']),
        DataLoader(val, val_bs, sampler=sampler_val, num_workers=kwargs['num_workers']),
        DataLoader(test, test_bs, sampler=sampler_test, num_workers=kwargs['num_workers']),
        sampler,
    )


def model_from_dataset(dataset, **kwargs):
    if dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


def get_lr_scheduler(lr_scheduler, optimizer, cfg, tag):
    if lr_scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.)    # does nothing to the lr
    elif lr_scheduler == "CosineAnnealing":
        T_max = cfg['epochs']
        if tag == 'hpo':
            if cfg.dataset == 'multi_mnist' or cfg.dataset == 'multi_fashion' or cfg.dataset == 'multi_fashion_mnist':
                T_max = 100
            else:
                raise ValueError()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    elif lr_scheduler == "MultiStep":
        # if cfg['scheduler_milestones'] is None:
        milestones = [int(.33 * cfg['epochs']), int(.66 * cfg['epochs'])]
        if tag == 'hpo':
            if cfg.dataset == 'multi_mnist' or cfg.dataset == 'multi_fashion' or cfg.dataset == 'multi_fashion_mnist':
                milestones = [33]
            else:
                raise ValueError()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    else:
        raise ValueError(f"Unknown lr scheduler {lr_scheduler}")
    return scheduler


def scale(a, new_max=1, new_min=0, axis=None):
    if np.isscalar(new_max) or len(new_max) == 1:
        new_max = np.repeat(new_max, a.shape[1])
    if np.isscalar(new_min) or len(new_min) == 1:
        new_min = np.repeat(new_min, a.shape[1])
    assert all(m > n for m, n in zip(new_max, new_min)), 'Max > min violated!'
    # scale to 0, 1
    a = (a - a.min(axis=axis)) / (a.max(axis=axis) - a.min(axis=axis))

    new_min = np.array(new_min)
    new_max = np.array(new_max)
    a = a * (new_max - new_min) + new_min
    return a


def reference_points(partitions, dim=2, min=0, max=1, tolerance=0.):
    """generate evenly distributed preference vector"""
    d = get_reference_directions("uniform", dim, n_partitions=partitions)

    if np.isscalar(max) or len(max) == 1:
        max = np.repeat(max, d.shape[1])
    if np.isscalar(min) or len(min) == 1:
        min = np.repeat(min, d.shape[1])

    range = (max - min)
    min = min + tolerance * range

    return scale(d, new_min=min, new_max=max, axis=0)
    


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


def dict_to(dict, device):
    if isinstance(dict, list):
        # we have a list of dicts
        return [dict_to(d, device) for d in dict]
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict.items()}


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

    def __init__(self, J, n_test_rays) -> None:
        self.center = torch.zeros(J).cuda()
        self.pf = torch.zeros((n_test_rays, J)).cuda()
        self.hv = 0
        self.max_angle = None
        self.pf_available = False
        self.i = 0
        self.j = 0
        self.optimal_sol = None

    def update(self, data, mode):
        if mode == 'single_point':
            self.center += torch.from_numpy(data).cuda()
            self.i += 1
        elif mode == 'pareto_front':
            self.pf_available = True
            self.pf += torch.from_numpy(data).cuda()
            self.j += 1
        else:
            raise ValueError(f"Unknown eval mode {mode}")


    def gather(self, world_size):
        assert world_size > 1
        dist.reduce(self.pf, dst=0, op=dist.ReduceOp.SUM)     # collect
        self.pf.data /= world_size                            # average

        dist.reduce(self.center, dst=0, op=dist.ReduceOp.SUM)     # collect
        self.center.data /= world_size                            # average
        
    
    def normalize(self):
        if self.i > 0:
            self.center /= self.i
        if self.j > 0:
            self.pf /= self.j
        
        self.pf = self.pf.cpu().numpy()
        self.center = self.center.cpu().numpy()

    
    def compute_hv(self, reference_point):
        if self.pf_available:
            if self.pf.shape[1] <= 5:
                assert self.pf.shape[1] == len(reference_point)
                hv = get_performance_indicator("hv", ref_point=np.array(reference_point))
                self.hv = hv.calc(self.pf)
        else:
            if self.pf.shape[1] <= 5:
                hv = get_performance_indicator("hv", ref_point=np.array(reference_point))
                self.hv = hv.calc(self.center)


    def compute_optimal_sol(self, weights=None):
        if self.pf_available:
            self.optimal_sol_idx, self.optimal_sol = optimal_solution(self.pf, weights)


    def to_dict(self):
        result = {
            'center_ray': self.center.tolist(),
            'hv': self.hv,
        }
        if self.pf_available:
            result.update({
                'pareto_front': self.pf.tolist(),
                'optimal_solution': self.optimal_sol.tolist(),
                'optimal_solution_idx': int(self.optimal_sol_idx),
            })

        return result


class RunningMean():

    def __init__(self, len=100) -> None:
        super().__init__()
        self.queue = collections.deque(maxlen=len)


    def append(self, x):
        self.queue.append(x)

    
    def std(self, axis=None):
        data = np.array(self.queue)
        return data.std().item() if axis == None else data.std(axis=axis)


    def __call__(self, x=None, axis=None):
        if x is not None:
            self.queue.append(x)
        data = np.array(self.queue)
        return data.mean().item() if axis == None else data.mean(axis=axis)
    

    def __len__(self):
        return len(self.queue)


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
            plt.xlabel(self.labels[0])
            plt.ylabel(self.labels[1])
            plt.grid()
            plt.savefig(os.path.join(self.logdir, "pf_{}.png".format(self.prefix)))
            plt.close()
        else:
            pass # not implemented



import pickle
import random
import numpy as np

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False


def save_checkpoint(savepath, use_torch=False, **kwargs):
    """
    Saves objects to a pickle. Pickles the state_dict of pytorch modules.
    Pickles also the state of random number generators for `random`, `numpy` and, if available, `torch`.
    For loading you can either use
    ```
    with open('path/to/checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    ```
    or to make use of torch's `map_location` in distributed settings:
    ```
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load('path/to/checkpoint.pkl, map_location)
    ```
    This loads the checkpoints into a dict. You can restore torch parameters using `load_state_dict`.
    To update the state of random number generators you can use
    ```
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['np_rng_state'])
    random.setstate(checkpoint['random_rng_state'])
    ```
    Args:
        savepath (str): path for the checkpoint file.
        use_torch (bool): Whether to use torch.save() or the pickle module.
        kwargs: objects to checkpoint, e.g. models, epoch, optimizers, ...
    """
    savedict = {}
    for k, v in kwargs.items():
        if hasattr(v, 'state_dict'):
            savedict[k] = v.state_dict()
        else:
            savedict[k] = v
    if torch_available:
        savedict['torch_rng_state'] = torch.get_rng_state()
    savedict['np_rng_state'] = np.random.get_state()
    savedict['random_rng_state'] = random.getstate()

    if use_torch and not torch_available:
        raise ValueError(f"Asked to use torch.save but torch is not installed.")

    if use_torch:
        torch.save(savedict, savepath)
    else:
        with open(savepath, 'wb') as f:
            pickle.dump(savedict, f)