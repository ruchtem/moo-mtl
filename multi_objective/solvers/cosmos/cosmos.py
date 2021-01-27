import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .epo_lp import EPO_LP
from utils import calc_gradients, num_parameters, model_from_dataset, circle_points
from solvers.base import BaseSolver
from .utils import alpha_from_epo, uniform_sample_alpha


class AlphaGenerator(nn.Module):
    def __init__(self, K, child_model, input_dim, hidden_dim=2):
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.main = nn.Sequential(
                nn.ConvTranspose2d(K, hidden_dim, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=6, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:])
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {dim}")

        self.child_model = child_model

    def forward(self, batch):
        b = batch['data'].shape[0]
        a = batch['alpha'].repeat(b, 1)
        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.main(a)
        x = torch.cat((batch['data'], a), dim=1)
        return self.child_model(dict(data=x))
    
    def private_params(self):
        if hasattr(self.child_model, 'private_params'):
            return self.child_model.private_params()
        else:
            return []



class COSMOSSolver(BaseSolver):

    def __init__(self, objectives, alpha_dir, dim, early_fusion, late_fusion, alpha_generator_dim, n_test_rays, internal_solver, **kwargs):
        self.objectives = objectives
        self.K = len(objectives)
        self.early_fusion = early_fusion
        self.late_fusion = late_fusion
        self.alpha_dir = alpha_dir
        self.n_test_rays = n_test_rays
        self.internal_solver = internal_solver

        dim = list(dim)
        dim[0] = dim[0] if not early_fusion else dim[0] + alpha_generator_dim

        model = model_from_dataset(method='cosmos', dim=dim, late_fusion=late_fusion, **kwargs)
        self.model = AlphaGenerator(self.K, model, dim, alpha_generator_dim).cuda()

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def step(self, batch):
        # step 1: sample alphas
        if self.alpha_dir > 0:
            batch['alpha']  = torch.from_numpy(
                np.random.dirichlet([self.alpha_dir for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            batch['alpha']  = uniform_sample_alpha(self.K)



        if self.internal_solver == 'epo':
            # calulate the gradient and update the parameters
            gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
            epo_lp = EPO_LP(m=self.K, n=self.n_params, r=batch['alpha'].cpu().numpy())
            batch['alpha'] = alpha_from_epo(epo_lp, gradients, obj_values, batch['alpha'].cpu().numpy())
        
            private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
            for name, param in self.model.named_parameters():
                not_private = all([p not in name for p in private_params])
                if not_private:
                    param.grad.data.zero_()
                    grad = None
                    for a, grads in zip(batch['alpha'], gradients):
                        if name in grads:
                            if grad is None:
                                grad = a * grads[name]
                            else:
                                grad += a * grads[name]
                    assert grad is not None
                    param.grad = grad
                    # param.grad = sum(a * grads[name] for a, grads in zip(batch['alpha'], gradients))
            return sum(obj_values)
        elif self.internal_solver == 'linear':

            self.model.zero_grad()
            logits = self.model(batch)
            batch.update(logits)
            loss_total = None
            for a, objective in zip(batch['alpha'], self.objectives):
                task_loss = objective(**batch)
                loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
                
            loss_total.backward()
            return loss_total.item()


    def eval_step(self, batch):
        self.model.eval()
        logits = []
        with torch.no_grad():
            test_rays = circle_points(self.n_test_rays, dim=self.K)

            for ray in test_rays:
                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                ray /= ray.sum()

                batch['alpha'] = ray
                logits.append(self.model(batch))
        return logits

