import torch
from copy import deepcopy

from min_norm_solvers import MinNormSolver
from utils import calc_gradients


class ProposedSolver():

    def __init__(self, scheduled_importance, objectives, divisor, model):
        self.importances = scheduled_importance
        self.objectives = objectives
        self.divisor = divisor
        self.model = model
        self.i = 0


    def new_point(self, train_loader, optimizer):
        self.importance = self.importances[self.i]
        self.i += 1


    def step(self, data, labels, sensible_attribute):

        gradients, obj_values = calc_gradients(data, labels, sensible_attribute, self.model, self.objectives)

        #print(obj_values)
        
        if len(self.objectives) > 1:
            # scale gradients
            #obj_values /= self.divisor
            alpha = (self.importance * obj_values[1]) / (obj_values[0] - self.importance * obj_values[0])
            alpha = torch.Tensor([alpha]).cuda()

            scaled_grads = deepcopy(gradients)
            scaled_grads[0] = {k: v*alpha for k, v in scaled_grads[0].items()}

            # move towards pareto front
            sol, min_norm = MinNormSolver.find_min_norm_element_FW([[v for k, v in sorted(grads.items())] for grads in scaled_grads])
            
            for name, param in self.model.named_parameters():
                param.grad = sum(sol[o] * gradients[o][name] for o in range(len(self.objectives))).cuda()
        else:
            for name, param in self.model.named_parameters():
                param.grad = gradients[0][name].cuda()
