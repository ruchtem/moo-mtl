# code from https://github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py
# and adapted

import torch
from torch.autograd import Variable

from .base import BaseMethod
from multi_objective.min_norm_solvers import MinNormSolver, gradient_normalizers
from multi_objective.utils import calc_gradients


class MGDAMethod(BaseMethod):

    def __init__(self, objectives, model, cfg) -> None:
        super().__init__(objectives, model, cfg)
        self.approximate_norm_solution = cfg.approximate_mgda
        self.normalization_type = cfg.normalization_type
        self.loss_maxs = cfg.loss_maxs


    def step(self, batch):
        if self.approximate_norm_solution:
            # Approximate solution by Sener and Koltun 2019.
            self.model.zero_grad()

            # First compute representations (z)
            with torch.no_grad():
                rep = self.model.forward_feature_extraction(batch)
            
            # we require gradients wrt to (z)
            rep = Variable(rep, requires_grad=True)

            # Compute gradients of each loss function wrt z
            grads = {t: {} for t in self.task_ids}
            obj_values = {t: None for t in self.task_ids}
            for t, objective in self.objectives.items():
                # zero grad
                self.model.zero_grad()
                
                logits = self.model.forward_linear(rep, t)
                batch.update(logits)

                output = objective(**batch)
                output.backward()
                
                obj_values[t] = output.item()

                grads[t]['input'] = rep.grad.data.detach().clone()
                rep.grad.data.zero_()
        else:
            # This is plain MGDA
            grads, obj_values = calc_gradients(batch, self.model, self.objectives)

        if self.normalization_type == 'init_loss':
            gn = gradient_normalizers(grads, self.loss_maxs, 'loss')
        else:
            gn = gradient_normalizers(grads, obj_values, self.normalization_type)
        for t, task_grads in grads.items():
            for name, grad in task_grads.items():
                grads[t][name] = grad / gn[t]

        # Min norm solver by Sener and Koltun
        # They don't use their FW solver in their code either.
        # We can also use the scipy implementation by me, does not matter.
        grads = [[v for v  in d.values()] for d in grads.values()]
        sol, min_norm = MinNormSolver.find_min_norm_element(grads)

        # Scaled back-propagation
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        for i, (a, t) in enumerate(zip(sol, self.task_ids)):
            task_loss = self.objectives[t](**batch)
            if self.normalization_type == 'init_loss':
                task_loss /= self.loss_maxs[i]
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            
        loss_total.backward()
        return loss_total.item()


    @torch.no_grad()
    def eval_step(self, batch, preference_vector=None):
        self.model.eval()
        return self.model(batch)