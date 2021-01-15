import torch
import itertools
import numpy as np
from copy import deepcopy

import time

from min_norm_solvers import MinNormSolver
from utils import calc_gradients, powerset


class ProposedSolver():

    def __init__(self, objectives, divisor, model, num_pareto_points):
        self.importances = [.5]#scheduler_values = np.linspace(.01, .99, num_pareto_points)
        self.idx_set = powerset(range(len(objectives)))
        self.objectives = objectives
        self.divisor = divisor
        self.model = model
        self.i = 0
        self.gradient_buffer = []


    def new_point(self, train_loader, optimizer):
        if self.i == 1:
            self.start_weights = {}
            for name, param in self.model.named_parameters():
                self.start_weights[name] = param.detach().clone()
        self.obj_idx = self.idx_set[self.i]
        if self.i > 1:
            with torch.no_grad():
                sd = self.model.state_dict()
                for name, _ in self.model.named_parameters():
                    sd[name] = self.start_weights[name].clone()
                self.model.load_state_dict(sd)
        self.i += 1
        self.gradient_buffer = []
    

    def _update_gradient_buffer(self, gradients):
        if len(self.gradient_buffer) == 0:
            self.gradient_buffer = gradients
        else:
            for i in range(len(self.objectives)):
                for name in self.gradient_buffer[i]:
                    self.gradient_buffer[i][name] += gradients[i][name]


    def step(self, batch):

        gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
        self._update_gradient_buffer(gradients)

        #print(obj_values)
        
        if len(self.objectives) > 1:
            # scale gradients
            #obj_values /= self.divisor
            #alpha = (self.importance * (obj_values[1] + obj_values[2])) / (obj_values[0] - self.importance * obj_values[0])
            #alpha = torch.Tensor([alpha]).cuda()

            scaled_grads = deepcopy(gradients)
            scaled_grads = [scaled_grads[i] for i in self.obj_idx]
            #if self.i > 0:
            for i, grad in enumerate(scaled_grads):
                norm = torch.linalg.norm(torch.cat([torch.flatten(v) for v in grad.values()]))
                grad = {k: v / norm for k, v in grad.items()}
                scaled_grads[i] = grad
            #scaled_grads[0] = {k: v*alpha for k, v in scaled_grads[0].items()}

            # move towards pareto front
            
            sol = np.full(len(self.objectives), .001)
            # sol, min_norm = MinNormSolver.find_min_norm_element([[v for k, v in sorted(grads.items())] for grads in scaled_grads])
            sol_i, min_norm = MinNormSolver.scipy_impl([[v for k, v in sorted(grads.items())] for grads in scaled_grads])
            
            
            for i, s in zip(self.obj_idx, sol_i):
                sol[i] = .999 / len(self.obj_idx)
            
            print(sol, min_norm)
            
            self.model.zero_grad()
            for name, param in self.model.named_parameters():
                param.grad = sum(sol[o] * gradients[o][name] for o in range(len(self.objectives))).cuda()
        else:
            for name, param in self.model.named_parameters():
                param.grad = gradients[0][name].cuda()
