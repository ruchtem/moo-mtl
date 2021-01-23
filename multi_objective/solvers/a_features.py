import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import calc_gradients, flatten_grads, num_parameters, model_from_dataset
from solvers.base import BaseSolver

def uniform_sample_alpha(size):
    alpha = torch.rand(size)
    # unlikely but to be save:
    while sum(alpha) == 0.0:
        alpha = torch.rand(len(self.objectives))
    
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    
    return alpha / sum(alpha)


def dirichlet_sampling(size, alpha):
    result = np.random.dirichlet([alpha for _ in range(size)], 1).astype(np.float32).flatten()
    return torch.from_numpy(result).cuda()


def alpha_as_feature(batch, early_fusion=True, append=False, overwrite=False):
    if batch['data'].ndim == 2:
        # tabular data
        alpha_columnwise = batch['alpha'].repeat(len(batch['data']), 1)
        if early_fusion:
            if not overwrite:
                batch['data'] = torch.hstack((batch['data'], alpha_columnwise))
            else:
                batch['data'][:,-2:] = alpha_columnwise
        if append:
            batch['alpha_features'] = alpha_columnwise
    elif batch['data'].ndim == 4:
        # image data
        if early_fusion:
            b, c, w, h = batch['data'].shape
            alpha_channelwise = batch['alpha'].repeat(b, w, h, 1)
            alpha_channelwise = torch.movedim(alpha_channelwise, 3, 1)
            if not overwrite:
                batch['data'] = torch.cat((batch['data'], alpha_channelwise), dim=1)
            else:
                batch['data'][:, -2:, :, :] = alpha_channelwise
        if append:
            alpha_columnwise = batch['alpha'].repeat(len(batch['data']), 1)
            batch['alpha_features'] = alpha_columnwise
    return batch


class AlphaGenerator(nn.Module):
    def __init__(self, K, child_model, hidden_dim=2, target_size=(36, 36)):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(K, hidden_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(target_size)
        )
        self.child_model = child_model

    def forward(self, batch):
        b = batch['data'].shape[0]
        a = batch['alpha'].repeat(b, 1)
        a = a.reshape(b, len(batch['alpha']), 1, 1)

        a = self.main(a)

        x = torch.cat((batch['data'], a), dim=1)

        return self.child_model(dict(data=x))



class AFeaturesSolver(BaseSolver):

    def __init__(self, objectives, alpha_dir, dim, early_fusion, late_fusion, alpha_generator_dim, **kwargs):
        self.objectives = objectives
        self.K = len(objectives)
        self.early_fusion = early_fusion
        self.late_fusion = late_fusion
        self.alpha_dir = alpha_dir

        dim = list(dim)
        dim[0] = dim[0] if not early_fusion else dim[0] + alpha_generator_dim

        model = model_from_dataset(method='afeature', dim=dim, late_fusion=late_fusion, **kwargs)

        self.model = AlphaGenerator(self.K, model, alpha_generator_dim, target_size=dim[-2:]).cuda()

        print("Number of parameters: {}".format(num_parameters(self.model)))


    def step(self, batch):
        # step 1: sample alphas
        if self.alpha_dir:
            batch['alpha'] = dirichlet_sampling(self.K, self.alpha_dir)
        else:
            batch['alpha'] = uniform_sample_alpha(self.K)

        self.model.zero_grad()
        loss_total = None
        for a, objective in zip(batch['alpha'], self.objectives):
            logits = self.model(batch)
            batch.update(logits)
            task_loss = objective(**batch)

            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            
        loss_total.backward()
        return loss_total.item()


        # # calulate the gradient and update the parameters
        # gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
        
        # private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
        # for name, param in self.model.named_parameters():
        #     if name not in private_params:
        #         param.grad.data.zero_()
        #         grad = None
        #         for a, grads in zip(batch['alpha'], gradients):
        #             if name in grads:
        #                 if grad is None:
        #                     grad = a * grads[name]
        #                 else:
        #                     grad += a * grads[name]
        #         param.grad = grad
        #         # param.grad = sum(a * grads[name] for a, grads in zip(batch['alpha'], gradients))


    def eval_step(self, batch):
        assert self.K <= 2
        self.model.eval()
        logits = []
        with torch.no_grad():
            for i, a in enumerate(np.linspace(.001, .999, 20)):
                batch['alpha'] = torch.Tensor([a, 1-a]).cuda()
                logits.append(self.model(batch))
        return logits

