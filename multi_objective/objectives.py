import math
import torch
import torch.nn.functional as F


def from_name(objectives, task_ids=None, **kwargs):
    map = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
        'L1Loss': L1Loss,
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        'ddp': DDPHyperbolicTangentRelaxation,
        'deo': DEOHyperbolicTangentRelaxation,
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

    def __init__(self, label_name='labels', logits_name='logits', **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.reduction = 'mean'
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        if 'inst' in self.label_name:
            mask = labels != 255  # ignore index for instances
            loss = F.l1_loss(logits[mask], labels[mask], reduction=self.reduction)
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

        logits = torch.sigmoid(logits)
        tanh_convert = torch.tanh(self.c * torch.relu(logits))

        reduced_loss = torch.abs(tanh_convert[s].mean() - tanh_convert[~s].mean())
        if self.reduction == 'mean':
            return reduced_loss
        elif self.reduction == 'none':
            # Our own adaption for instance-wise loss
            
            # hack for abs value
            test = tanh_convert.clone().detach()
            test[s] *= 1 / sum(s)
            test[~s] *= -1 / sum(~s)

            result = tanh_convert.clone()
            if test.sum() <= 0.:
                result[s] *= -1 / sum(s)
                result[~s] *= 1 / sum(~s)
            else:
                # change the order
                result[s] *= 1 / sum(s)
                result[~s] *= -1 / sum(~s)
            result = result * len(result)   # we reduce with mean instead of sum later

            assert math.isclose(result.mean().item(), reduced_loss.item(), abs_tol=1e-6)
            return result.squeeze()
        else:
            raise NotImplementedError()


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

