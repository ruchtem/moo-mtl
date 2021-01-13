import torch
import numpy as np
from copy import deepcopy

import time

from min_norm_solvers import MinNormSolver
from utils import calc_gradients


class ProposedSolver():

    def __init__(self, objectives, divisor, model, num_pareto_points):
        self.importances = [.5]#scheduler_values = np.linspace(.01, .99, num_pareto_points)
        self.objectives = objectives
        self.divisor = divisor
        self.model = model
        self.i = 0


    def new_point(self, train_loader, optimizer):
        self.importance = self.importances[self.i]
        self.i += 1


    def step(self, batch):

        gradients, obj_values = calc_gradients(batch, self.model, self.objectives)

        #print(obj_values)
        
        if len(self.objectives) > 1:
            # scale gradients
            #obj_values /= self.divisor
            alpha = (self.importance * (obj_values[1] + obj_values[2])) / (obj_values[0] - self.importance * obj_values[0])
            alpha = torch.Tensor([alpha]).cuda()

            scaled_grads = deepcopy(gradients)
            scaled_grads[0] = {k: v*alpha for k, v in scaled_grads[0].items()}

            # move towards pareto front
            tick = time.time_ns()
            # sol, min_norm = MinNormSolver.find_min_norm_element([[v for k, v in sorted(grads.items())] for grads in scaled_grads])
            sol, min_norm = MinNormSolver.scipy_impl([[v for k, v in sorted(grads.items())] for grads in scaled_grads])
            tock = time.time_ns()
            print(sol, min_norm, alpha, tock-tick)
            
            self.model.zero_grad()
            for name, param in self.model.named_parameters():
                param.grad = sum(sol[o] * gradients[o][name] for o in range(len(self.objectives))).cuda()
        else:
            for name, param in self.model.named_parameters():
                param.grad = gradients[0][name].cuda()
