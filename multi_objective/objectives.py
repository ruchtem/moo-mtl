import math
import torch
import torch.nn.functional as F
import numpy as np


def from_name(objectives, task_ids=None, **kwargs):
    map = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
        'L1Loss': L1Loss,
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        'ddp': DDPHyperbolicTangentRelaxation,
        'deo': DEOHyperbolicTangentRelaxation,
        'VAELoss': VAELoss,
        'WeightedVAELoss': WeightedVAELoss,
    }
    if len(task_ids) > 0:
        return {t: map[n]("labels_{}".format(t), "logits_{}".format(t), **kwargs) for n, t in zip(objectives, task_ids)}
    else:
        print("WARNING: No task ids specified, assuming all objectives use the same default output.")
        return {t: map[n](**kwargs) for t, n in enumerate(objectives)}


class CrossEntropyLoss():
    
    def __init__(self, label_name='labels', logits_name='logits', ignore_index=-100, **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
        self.ignore_index = ignore_index


    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        loss = F.cross_entropy(logits, labels, reduction=self.reduction, ignore_index=self.ignore_index)

        if self.reduction == 'none' and len(logits.shape) > 2:
            # for images we still need to average per image
            b = logits.shape[0]
            loss = loss.view(b, -1).mean(dim=1)

        return loss


class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, label_name='labels', logits_name='logits', **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)


class L1Loss():
    """
    Special loss for cityscapes
    """

    def __init__(self, label_name='labels', logits_name='logits', ignore_index=255, **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
        self.ignore_index = ignore_index
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        if 'inst' in self.label_name:
            mask = labels != self.ignore_index
            loss = F.l1_loss(logits[mask], labels[mask], reduction=self.reduction)

            if self.reduction == 'none':
                # average per image
                loss = torch.vstack(tuple(
                    l.mean()
                    for l
                    in torch.split(loss, split_size_or_sections=mask.sum(dim=[1, 2, 3]).tolist())
                ))

                # images which are ignored due to mask
                loss[torch.isnan(loss)] = 0
                
        else:
            loss = F.l1_loss(logits, labels, reduction=self.reduction)
        
        if self.reduction == 'none' and len(logits.shape) > 2:
            # for images we still need to average per image
            b = logits.shape[0]
            loss = loss.view(b, -1).mean(dim=1)
        
        return loss


class MSELoss():

    def __init__(self, label_name='labels', **kwargs):
        super().__init__()
        self.label_name = label_name
        self.reduction = 'mean'


    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        return F.mse_loss(logits, labels, reduction=self.reduction)


class L1Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


class L2Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)


class DDPHyperbolicTangentRelaxation():
    """See 
    
    Padh, K., Antognini, D., Glaude, E. L., Faltings, B., & Musat, C. (2020). 
    Addressing Fairness in Classification with a Model-Agnostic Multi-Objective 
    Algorithm. arXiv preprint arXiv:2009.04441.
    
    We could also use any other differentiable fairness metric.
    """

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1, **kwargs):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c
        self.reduction = 'mean'

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        s = kwargs[self.s_name].bool()

        if any(s) and not all(s):
            logits = torch.sigmoid(logits)
            tanh_convert = torch.tanh(self.c * torch.relu(logits))

            return torch.abs(tanh_convert[s].mean() - tanh_convert[~s].mean())
        else:
            # dpp is only defined if difference can be calculated
            # set gradients to zero
            return (logits * 0).mean()
        
        


class DEOHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1, normalize=False):
        super().__init__(label_name, logits_name, normalize)
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        logits = torch.sigmoid(logits)
        s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
        s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

        return torch.abs(torch.tanh(self.c * torch.relu(s_positive)).mean() - torch.tanh(self.c * torch.relu(s_negative)).mean())



class VAELoss():

    """taken from https://github.com/swisscom/ai-research-mamo-framework/blob/master/loss/vae_loss.py
    and adaped."""
    
    def __init__(self, label_name='labels', logits_name='logits', **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
        self.weighted_vector = None


    def __call__(self, **kwargs):
        y_pred = kwargs[self.logits_name]
        y_true = kwargs[self.label_name]

        anneal = kwargs['vae_beta']

        mean = kwargs['mean']
        log_variance = kwargs['log_variance']

        assert len(y_pred) == len(y_true)
        assert mean is not None
        assert log_variance is not None

        assert self.reduction == 'mean'
        
        # calculate the reconstruction loss
        if(self.weighted_vector is not None):
            # reconstruction loss if user provides a weighted vector
            BCE = -torch.mean(torch.sum(F.log_softmax(y_pred, 1)
                                        * y_true * self.weighted_vector, -1))
        else:
            # reconstruction loss without weigted vector
            BCE = -torch.mean(torch.sum(F.log_softmax(y_pred, 1) * y_true, -1))

        # calculate the 'regularization' loss based on Kullback-Leibler divergence.
        # here we compute the KLd between two multivariate normal distributions.
        # The first is a multivariate gaussian with mean 'mean' and log-variance 'log_variance'
        # The second is a multivariate standard normal distribution (mean 0 and unit variance)
        # The exact equation is given in: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(),
                                          dim=1))

        # return combined loss.
        return BCE + anneal * KLD
    
class WeightedVAELoss(VAELoss):

    def __init__(self, label_name='labels', logits_name='logits', loss_weights=None, **kwargs):
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
        self.weighted_vector = torch.from_numpy(np.load(loss_weights)).to(kwargs['device']).float()