import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import calc_gradients, flatten_grads, num_parameters


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


class AFeaturesSolver():

    def __init__(self, objectives, model, early_fusion, late_fusion, alpha_dir, **kwargs):
        self.objectives = objectives
        self.model = model
        self.K = len(objectives)
        self.early_fusion = early_fusion
        self.late_fusion = late_fusion
        self.alpha_dir = alpha_dir

        print("Number of parameters: {}".format(num_parameters(model)))


    def step(self, batch):
        # step 1: sample alphas
        if self.alpha_dir:
            batch['alpha'] = dirichlet_sampling(self.K, self.alpha_dir)
        else:
            batch['alpha'] = uniform_sample_alpha(self.K)

        # append as features
        batch = alpha_as_feature(batch, self.early_fusion, self.late_fusion)

        # calulate the gradient and update the parameters
        gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
        
        private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
        for name, param in self.model.named_parameters():
            if name not in private_params:
                param.grad.data.zero_()
                grad = None
                for a, grads in zip(batch['alpha'], gradients):
                    if name in grads:
                        if grad is None:
                            grad = a * grads[name]
                        else:
                            grad += a * grads[name]
                param.grad = grad
                # param.grad = sum(a * grads[name] for a, grads in zip(batch['alpha'], gradients))


    def eval_step(self, batch):
        assert self.K <= 2
        logits = []
        with torch.no_grad():
            for i, a in enumerate(np.linspace(.001, .999, 20)):
                batch['alpha'] = torch.Tensor([a, 1-a]).cuda()
                # batch['alpha'] = torch.Tensor([1., 0.]).cuda()
                batch = alpha_as_feature(batch, self.early_fusion, self.late_fusion, overwrite=False if i==0 else True)
                logits.append(self.model(batch))
        return logits


    def new_point(self, *args):
        pass
