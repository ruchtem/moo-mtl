import torch

def from_name(names):
    objectives = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntroyLoss': BinaryCrossEntropyLoss,
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        'ddp': DDPHyperbolicTangentRelaxation,
        'deo': DEOHyperbolicTangentRelaxation,
    }
    return [objectives[n] for n in names]


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    
    def __init__(self):
        super().__init__(reduction='mean')
    

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        return super().__call__(logits, targets)


class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, label_name='labels', pos_weight=None):
        super().__init__(reduction='mean', pos_weight=torch.Tensor([pos_weight]).cuda() if pos_weight else None)
        self.label_name = label_name
    

    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return super().__call__(logits, labels)


class MSELoss(torch.nn.MSELoss):

    def __init__(self, label_name='labels'):
        super().__init__()
        self.label_name = label_name


    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        return super().__call__(logits, labels)


class L1Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


class L2Regularization():

    def __call__(self, **kwargs):
        model = kwargs['model']
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)


class DDPHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[sensible_attribute.bool()]
        s_positive = logits[~sensible_attribute.bool()]

        return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))


class DEOHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs['logits']
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[(sensible_attribute.bool()) & (targets == 1)]
        s_positive = logits[(~sensible_attribute.bool()) & (targets == 1)]

        return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))
