import torch
import numpy as np

from utils import calc_gradients


def uniform_sample_alpha(size):
    alpha = torch.rand(size)
    # unlikely but to be save:
    while sum(alpha) == 0.0:    
        alpha = torch.rand(len(self.objectives))
    
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    
    return alpha / sum(alpha)


def alpha_as_feature(batch, overwrite=False):
    if batch['data'].ndim == 2:
        # tabular data
        alpha_columnwise = batch['alpha'].repeat(len(batch['data']), 1)
        if not overwrite:
            batch['data'] = torch.hstack((batch['data'], alpha_columnwise))
        else:
            batch['data'][-2:] = alpha_columnwise
    elif batch['data'].ndim == 4:
        # image data
        b, c, w, h = batch['data'].shape
        alpha_channelwise = batch['alpha'].repeat(b, w, h, 1)
        alpha_channelwise = torch.movedim(alpha_channelwise, 3, 1)
        if not overwrite:
            batch['data'] = torch.cat((batch['data'], alpha_channelwise), dim=1)
        else:
            batch['data'][:, -2:, :, :] = alpha_channelwise
    return batch


class AFeaturesSolver():

    def __init__(self, objectives, model, **kwargs):
        self.objectives = objectives
        self.model = model
        self.K = len(objectives)


    def step(self, batch):
        # step 1: sample alphas
        batch['alpha'] = uniform_sample_alpha(self.K)

        # append as features
        batch = alpha_as_feature(batch)

        # calulate the gradient and update the parameters
        gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
        
        private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
        for name, param in self.model.named_parameters():
            if name not in private_params:
                param.grad.data.zero_()
                param.grad = sum(batch['alpha'][o] * gradients[o][name] for o in range(self.K))


    def eval_step(self, batch):
        assert self.K <= 2
        logits = []
        with torch.no_grad():
            for i, a in enumerate(np.linspace(.001, .999, 20)):
                batch['alpha'] = torch.Tensor([a, 1-a]).cuda()
                batch = alpha_as_feature(batch, overwrite=False if i==0 else True)
                logits.append(self.model(batch['data']))
        return logits


    def new_point(self, *args):
        pass
