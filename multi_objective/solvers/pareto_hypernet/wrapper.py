# code taken from https://github.com/AvivNavon/pareto-hypernetworks/
import torch
import torch.nn as nn
import numpy as np


from utils import num_parameters, circle_points
from solvers.base import BaseSolver

from .models import LeNetPHNHyper, LeNetPHNTargetWrapper
from .solvers import LinearScalarizationSolver, EPOSolver


class HypernetSolver(BaseSolver):

    def __init__(self, objectives, n_test_rays, **kwargs):
        self.objectives = objectives
        self.n_test_rays = n_test_rays
        self.alpha = kwargs['alpha_dir']

        hnet: nn.Module = LeNetPHNHyper([9, 5], ray_hidden_dim=100)
        net: nn.Module = LeNetPHNTargetWrapper([9, 5])

        print("Number of parameters: {}".format(num_parameters(hnet)))

        self.model = hnet.cuda()
        self.net = net.cuda()

        self.solver = LinearScalarizationSolver(n_tasks=len(objectives))


    def step(self, batch):
        if self.alpha > 0:
            ray = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(len(self.objectives))], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            alpha = torch.empty(1, ).uniform_(0., 1.)
            ray = torch.tensor([alpha.item(), 1 - alpha.item()]).cuda()

        img = batch['data']

        weights = self.model(ray)
        batch.update(self.net(img, weights))

        losses = torch.stack([o(**batch) for o in self.objectives])

        ray = ray.squeeze(0)
        loss = self.solver(losses, ray, list(self.model.parameters()))
        loss.backward()

        return loss.item()
    
    def eval_step(self, batch):
        self.model.eval()

        test_rays = circle_points(self.n_test_rays)

        logits = []
        for ray in test_rays:
            ray = torch.from_numpy(ray.astype(np.float32)).cuda()
            ray /= ray.sum()

            weights = self.model(ray)
            logits.append(self.net(batch['data'], weights))
        return logits




