import torch
import numpy as np

from utils import flatten_grads


def uniform_sample_alpha(size):
    alpha = torch.rand(size)
    # unlikely but to be save:
    while sum(alpha) == 0.0:
        alpha = torch.rand(size)
    
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    
    return alpha / sum(alpha)


def alpha_from_epo(epo_lp, grads, losses, preference):

    grads_list = flatten_grads(grads)
    G = torch.stack(grads_list)
    GG = G @ G.T
    losses = np.stack(losses)

    try:
        # Calculate the alphas from the LP solver
        alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
    except Exception as e:
        print(e)
        alpha = None
    if alpha is None:   # A patch for the issue in cvxpy
        alpha = preference / preference.sum()

    alpha = epo_lp.m * torch.from_numpy(alpha).cuda()

    return alpha


