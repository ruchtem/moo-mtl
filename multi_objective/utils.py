import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain, combinations

from loaders import adult_loader, compas_loader, multi_mnist_loader, celeba_loader, credit_loader
from models import FullyConnected, MultiLeNet, EfficientNet


def dataset_from_name(dataset, **kwargs):
    if dataset == 'adult':
        return adult_loader.ADULT(**kwargs)
    elif dataset == 'credit':
        return credit_loader.Credit(**kwargs)
    elif dataset == 'compas':
        return compas_loader.Compas(**kwargs)
    elif dataset == 'multi_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='mnist', **kwargs)
    elif dataset == 'multi_fashion':
        return multi_mnist_loader.MultiMNIST(dataset='fashion', **kwargs)
    elif dataset == 'multi_fashion_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='fashion_and_mnist', **kwargs)
    elif dataset == 'celeba':
        return celeba_loader.CelebA(**kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def model_from_dataset(dataset, **kwargs):
    if dataset == 'adult':
        return FullyConnected(**kwargs)
    elif dataset == 'credit':
        return FullyConnected(**kwargs)
    elif dataset == 'compas':
        return FullyConnected(**kwargs)
    elif dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    elif dataset == 'celeba':
        # return EfficientNet.from_pretrained(**kwargs)
        return EfficientNet.from_name(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


def circle_points(K, min_angle=0.1, max_angle=np.pi / 2 - 0.1):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def calc_devisor(train_loader, model, objectives):
    values = np.zeros(len(objectives))
    for batch in train_loader:
        batch = dict_to_cuda(batch)
        
        batch['logits'] = model(batch['data'])

        for i, objective in enumerate(objectives):
            values[i] += objective(**batch)

    divisor = values / min(values)
    print("devisor={}".format(divisor))
    return divisor


def num_parameters(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


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
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()


def dict_to_cuda(d):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    p = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return list(reversed(list(p)))


def calc_gradients(batch, model, objectives):
    # store gradients and objective values
    gradients = []
    obj_values = []
    for i, objective in enumerate(objectives):
        # zero grad
        model.zero_grad()
        
        logits = model(batch)
        batch.update(logits)

        output = objective(**batch)
        output.backward()
        
        obj_values.append(output.item())
        gradients.append({})
        
        private_params = model.private_params() if hasattr(model, 'private_params') else []
        for name, param in model.named_parameters():
            if name not in private_params and param.requires_grad and param.grad is not None:
                gradients[i][name] = param.grad.data.detach().clone()
    
    return gradients, obj_values


def is_pareto_efficient(costs, return_mask=True):
        """
        From https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ParetoFront():

    def __init__(self, labels, logdir='tmp', prefix=""):
        self.labels = labels
        self.logdir = os.path.join(logdir, 'pf')
        self.prefix = prefix
        self.points = np.array([])
        self.e = 0

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def append(self, point):
        point = np.array(point)
        if not len(self.points):
            self.points = point
        else:
            self.points = np.vstack((self.points, point))
    
    
    def plot(self):
        p = self.points
        # for e in range(self.e + 1):
        #     idx1 = e * 20
        #     idx2 = (e+1) * 20
        #    plt.plot(p[idx1:idx2, 0], p[idx1:idx2, 1], '-', label="e={}".format(e+1))
        plt.plot(p[:, 0], p[:, 1], 'o')
        # for i, text in enumerate(range(len(self.points))):
        #     plt.annotate(text, (p[i,0], p[i,1]))
        
        #if p.shape[0] >= 3:
        #    front = p[self._is_pareto_efficient(p)]
        #    plt.plot(front[:, 0], front[:, 1], 'ro')

        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
        # plt.legend()
        plt.savefig(os.path.join(self.logdir, "x_{}.png".format(self.prefix)))
        plt.close()
        self.e += 1