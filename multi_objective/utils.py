import itertools
import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

import torch.distributed as dist

from numpy.linalg import norm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
from pymoo.factory import get_decomposition, get_reference_directions, get_performance_indicator
from pymoo.visualization.radviz import Radviz


from .loaders import adult_loader, compas_loader, multi_mnist_loader, celeba_loader, credit_loader, cityscapes_loader, coco_loader, movielens_loader
from .models import FullyConnected, MultiLeNet, EfficientNet, ResNet, Pspnet, MultiVAE

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
    elif dataset == 'movielens':
        return movielens_loader.MovieLens(**kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def loaders_from_name(dataset, seed, **kwargs):
    train = dataset_from_name(dataset, split='train', **kwargs)
    val = dataset_from_name(dataset, split='val', **kwargs)
    test = dataset_from_name(dataset, split='test', **kwargs)

    if dataset in ['adult', 'credit', 'compass']:
        val_bs = len(val)
        test_bs = len(test)
    else:
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
    if dataset == 'adult' or dataset == 'credit' or dataset == 'compass':
        return FullyConnected(**kwargs)
    elif dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    elif dataset == 'celeba':
        # if 'efficientnet' in kwargs['model_name']:
        return EfficientNet.from_pretrained(model_name='efficientnet-b4', **kwargs)
        # elif kwargs['model_name'] == 'resnet18':
        #     return ResNet.from_name(**kwargs)
    elif dataset == 'cityscapes':
        return Pspnet(dim=kwargs['dim'])
    elif dataset == 'movielens':
        return MultiVAE(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


def format_list(list, format='.4f'):
    string = ""
    for l in list:
        if string == "":
            string += f"{l:{format}}"
        else:
            string += f", {l:{format}}"
    return string


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
        # else:
        #     milestones = cfg['scheduler_milestones']
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


def find_top_k_binary(values, k):
    """
    taken from https://github.com/swisscom/ai-research-mamo-framework/tree/master/metric
    and adapted
    
    Finds the top k values for each row of a matrix and returns a binary
    mask on their positions.

    The method masks the k input values with the highest numerical value in
    every row of the input 2D numpy array.

    Args:
        values: A PyTorch tensor of values.
        k: An integer that denotes the number of values to obtain from the
            ranking of the values. The method masks the k values with the
            highest scores.

    Returns:
        A binary mask in the form of a 2D Pytorch tensor that outputs the
        top k values per row from the input values.
        For example:

        values = tensor([[0.5, 0.7, 0.3],
                                [0.4, 0.1, 0.7]])
        k = 2
        find_top_k_binary returns:

        tensor([[ True, True, False],
                [ True, False, True]])

    Raises:
        TypeError: An error occured while accessing the arguments -
            one of the arguments is NoneType.
        ValueError: An error occured when checking the dimensions of the
            values argument. It is not a 2D tensor. Or if k is smaller
            than 0.
    """
    if values is None:
        raise TypeError('Argument: values must be set.')
    if k is None:
        raise TypeError('Argument: k must be set.')
    if not isinstance(k, int):
        raise TypeError('Argument: k must be an integer.')
    if not torch.is_tensor(values):
        raise TypeError('Argument: values must be a PyTorch tensor.')
    if values.ndimension() != 2:
        raise ValueError('Argument: values must be a 2D tensor.')
    if k < 1:
        raise ValueError('Argument: k cannot be negative.')
    if k >= values.shape[1]:
        raise ValueError('Argument: k cannot be larger than\
                            values.shape[1]')

    _, idx = torch.topk(values, k=k, dim=1, sorted=False)
    values_binary = torch.zeros_like(values, dtype=torch.bool)
    values_binary = values_binary.scatter(1, idx[:, :k], True)
    values_binary[values <= 0] = False
    return values_binary


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


    def compute_dist(self, normalize=False):
        if self.pf_available:

            n_points, J = self.pf.shape
            rays = get_reference_directions('uniform', n_points=n_points, n_dim=J)

            # normalize the losses
            if normalize:
                min = self.pf.min(axis=0)
                max = self.pf.max(axis=0)
                pf = (self.pf - min) / (max - min + 1e-8)
            else:
                pf = self.pf

            def dist(a, b):
                return norm(a / norm(a) - b / norm(b))

            # compute cosine similarity
            c = [dist(r, p) for r, p in zip(rays, pf)]
            self.dist = np.mean(c)

            # for i, (x, y) in enumerate(pf):
            #     plt.plot(x, y, "ro")
            #     plt.text(x, y, f"{i}")


            # for i, (x, y) in enumerate(rays):
            #     plt.arrow(0, 0, x, y)
            #     plt.text(x, y, f"   {i}: {c[i]:.4f}")
            
            
            # # plt.plot(rays[:, 0], rays[:, 1],  "bo")
            # plt.title(f'l2 norm dist {self.dist}')
            # plt.savefig('dist.png')
            # plt.close()


    def to_dict(self):
        result = {
            'center_ray': self.center.tolist(),
            'hv': self.hv,
        }
        if self.pf_available:
            result.update({
                'pareto_front': self.pf.tolist(),
                'dist': float(self.dist),
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

from colour import Color
import matplotlib as mpl
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
            # if rays is not None:
            #     for r in rays:
            #         plt.plot([0, r[0]], [0, r[1]], color='black')
            plt.xlabel(self.labels[0])
            plt.ylabel(self.labels[1])
            plt.grid()
            plt.savefig(os.path.join(self.logdir, "pf_{}.png".format(self.prefix)))
            plt.close()
        else:
            n_colors = 100
            # plot radviz
            p_normalized = scale(p, axis=0)
            dists = norm(p, axis=1)
            _, bins = np.histogram(dists, bins=n_colors-2)
            color_idx = np.digitize(dists, bins)
            colors = list(Color("red").range_to(Color("blue"), n_colors))

            radviz_plot = CustomRadviz()

            # radviz_plot.add(rays, color='grey', alpha=.5)
            for p, i in zip(p_normalized, color_idx):
                radviz_plot.add(p, color=colors[i].rgb)

            radviz_plot.plot_if_not_done_yet()

            cbaxes = plt.gcf().add_axes([0.8, 0.25, 0.02, 0.5]) 

            cmap= mpl.colors.ListedColormap([c.rgb for c in colors])
            cnorm = mpl.colors.Normalize(vmin=dists.min(), vmax=dists.max())
            cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=cnorm,
                                orientation='vertical')
            cb1.set_label('Distance to origin')

            sort_idx = np.argsort(dists)[:2]
            if len(sort_idx) > 1:
                coordinates = np.squeeze(np.array(radviz_plot.points))
                for (x, y), d in zip(coordinates[sort_idx], dists[sort_idx]):
                    radviz_plot.ax.annotate(f"{d :.4f}", (x, y), xytext=(x+.1, y+.1), arrowprops={'arrowstyle': '->'})
            
            radviz_plot.save(os.path.join(self.logdir, "rad_{}.png".format(self.prefix)))
            plt.close()


from pymoo.visualization.util import plot_circle, plot_radar_line, plot_axis_labels, equal_axis, no_ticks, \
    get_uniform_points_around_circle
class CustomRadviz(Radviz):

    def _do(self):

        # initial a figure with a single plot
        self.init_figure()

        # equal axis length and no ticks
        equal_axis(self.ax)
        no_ticks(self.ax)

        V = get_uniform_points_around_circle(self.n_dim)
        plot_axis_labels(self.ax, V, self.get_labels(), **self.axis_label_style)

        # draw the outer circle and radar lines
        plot_circle(self.ax, **self.axis_style)
        plot_radar_line(self.ax, V, **self.axis_style)

        # draw the endpoints of each objective
        if self.endpoint_style:
            self.ax.scatter(V[:, 0], V[:, 1], **self.endpoint_style)

        # plot all the points
        self.points = []
        for k, (F, kwargs) in enumerate(self.to_plot):
            N = (F[..., None] * V).sum(axis=1) / F.sum(axis=1)[:, None]
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)
            self.points.append(N)



class GradientMonitor():

    means = {}
    stds = {}

    _registered = []


    @staticmethod
    def register_parameters(module, filter=None):
        for n, p in module.named_parameters():
            if filter in n:
                continue
            if p.requires_grad:
                GradientMonitor._registered.append(n)
        
        for n in GradientMonitor._registered:
            GradientMonitor.means[n] = []
            GradientMonitor.stds[n] = []
    

    @staticmethod
    def collect_grads(module):
        for n, p in module.named_parameters():
            if n not in GradientMonitor._registered:
                continue
            
            # GradientMonitor.means[n].append(p.grad.abs().mean().item())
            if p.grad is not None:
                GradientMonitor.means[n].append((p.grad**2).sum().sqrt().item())
                GradientMonitor.stds[n].append(p.grad.abs().std().item())
        
        if len(GradientMonitor.means[n]) % 10 == 0:
            plt.figure(figsize=(20, 20))
            for n in GradientMonitor._registered:
                mean = np.array(GradientMonitor.means[n])
                std = np.array(GradientMonitor.stds[n])
                plt.plot(mean, label=n)
                # plt.fill_between(range(len(std)), mean-std, mean+std, alpha=0.05)
            
            plt.yscale('log')
            plt.legend()
            plt.savefig('grads.png')
            plt.close()


class DebugDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, size=8, replication=1) -> None:
        super().__init__()
        self.replication = replication
        self.size = size
        loader = torch.utils.data.DataLoader(dataset, 1, num_workers=4, collate_fn=lambda x: x)
        self.data = []
        for i, b in enumerate(loader):
            if i >= size:
                break
            self.data.append(b[0])
    

    def __len__(self):
        return len(self.data) * self.replication
    

    def __getitem__(self, index):
        return self.data[index % self.size]
