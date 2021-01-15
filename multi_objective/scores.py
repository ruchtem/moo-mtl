import torch
import numpy as np

from abc import abstractmethod

import objectives as obj


def from_objectives(objectives):
    scores = {
        obj.CrossEntropyLoss: mcr,
        obj.BinaryCrossEntropyLoss: mcr,
        obj.DDPHyperbolicTangentRelaxation: DDP,
        obj.DEOHyperbolicTangentRelaxation: DEO,
        obj.MSELoss: L2Distance,
    }
    return [scores[o.__class__](o.label_name, o.logits_name) for o in objectives]

class BaseScore():

    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name


    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class L2Distance(BaseScore):

    def __call__(self, **kwargs):
        prediction = kwargs['logits']
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.linalg.norm(prediction - labels, ord=2)


class mcr(BaseScore):

    def __call__(self, **kwargs):
        # missclassification rate
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            if logits.shape[1] == 1:
                # binary case
                logits = torch.squeeze(logits)
                y_hat = torch.round(torch.sigmoid(logits))
            else:
                y_hat = torch.argmax(logits, dim=1)
            accuracy = sum(y_hat == labels) / len(y_hat)
        return 1 - accuracy.item()


class DDP(BaseScore):
    """Difference in Democratic Parity"""

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs['sensible_attribute']
    
        with torch.no_grad():
            n = logits.shape[0]
            logits_s_negative = logits[sensible_attribute.bool()]
            logits_s_positive = logits[~sensible_attribute.bool()]

            return torch.abs(1/n*sum(logits_s_negative > 0) - 1/n*sum(logits_s_positive > 0).item()).cpu().item()


class DEO(BaseScore):
    """Difference in Equality of Opportunity"""

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs['sensible_attribute']

        with torch.no_grad():
            n = logits.shape[0]
            logits_s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
            logits_s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

            return torch.abs(1/n*sum(logits_s_negative > 0) - 1/n*sum(logits_s_positive > 0).item()).cpu().item()        
