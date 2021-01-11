import torch



class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    
    def __init__(self):
        super().__init__(reduction='mean')
    

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        return super().__call__(logits, targets)


class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, pos_weight=None):
        super().__init__(reduction='mean', pos_weight=torch.Tensor([pos_weight]).cuda() if pos_weight else None)
    

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if targets.dtype != torch.float:
            targets = targets.float()
        return super().__call__(logits, targets)



class L1Regularization():

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


class L2Regularization():

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)


class DDPHyperbolicTangentRelaxation():
    # TODO: check if implementation is correct, default value of c

    def __init__(self, c=1):
        self.c = c

    def __call__(self, logits=None, targets=None, model=None, sensible_attribute=None):
        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[sensible_attribute.bool()]
        s_positive = logits[~sensible_attribute.bool()]

        return 1/n * (torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))) )
