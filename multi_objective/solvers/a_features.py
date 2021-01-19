import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import calc_gradients, flatten_grads, angle


def uniform_sample_alpha(size):
    alpha = torch.rand(size)
    # unlikely but to be save:
    while sum(alpha) == 0.0:    
        alpha = torch.rand(len(self.objectives))
    
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    
    return alpha / sum(alpha)


def alpha_as_feature(batch, overwrite=False, append_to_data=True):
    if batch['data'].ndim == 2:
        # tabular data
        alpha_columnwise = batch['alpha'].repeat(len(batch['data']), 1)
        if append_to_data:
            if not overwrite:
                batch['data'] = torch.hstack((batch['data'], alpha_columnwise))
            else:
                batch['data'][:,-2:] = alpha_columnwise
        else:
            batch['alpha_features'] = alpha_columnwise
    elif batch['data'].ndim == 4:
        # image data
        if append_to_data:
            b, c, w, h = batch['data'].shape
            alpha_channelwise = batch['alpha'].repeat(b, w, h, 1)
            alpha_channelwise = torch.movedim(alpha_channelwise, 3, 1)
            if not overwrite:
                batch['data'] = torch.cat((batch['data'], alpha_channelwise), dim=1)
            else:
                batch['data'][:, -2:, :, :] = alpha_channelwise
        else:
            alpha_columnwise = batch['alpha'].repeat(len(batch['data']), 1)
            batch['alpha_features'] = alpha_columnwise
    return batch


class AFeaturesSolver():

    def __init__(self, objectives, model, early_fusion, **kwargs):
        self.objectives = objectives
        self.model = model
        self.K = len(objectives)
        self.alpha_extra = not early_fusion

        self.norms = []


    def step(self, batch):
        # step 1: sample alphas
        batch['alpha'] = uniform_sample_alpha(self.K)
        # batch['alpha'] = torch.Tensor([1., 0.]).cuda()

        # append as features
        batch = alpha_as_feature(batch, append_to_data=not self.alpha_extra)

        # calulate the gradient and update the parameters
        gradients, obj_values = calc_gradients(batch, self.model, self.objectives)

        # self.norms.append([torch.linalg.norm(g).cpu().numpy() for g in flatten_grads(gradients)])
        self.norms.append(angle(gradients))
        
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
                batch = alpha_as_feature(batch, overwrite=False if i==0 else True, append_to_data=not self.alpha_extra)
                logits.append(self.model(batch))
        return logits


    def new_point(self, *args):
        if len(self.norms) > 0:
            p = np.array(self.norms)
            plt.plot(p, '.')
            # plt.plot(p[:, 0])
            # plt.plot(p[:, 1])
            plt.savefig('u.png')
            plt.close()
            self.norms = []
