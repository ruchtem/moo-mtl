import torch
import numpy as np

from abc import abstractmethod
from pycocotools.mask import encode, iou

import objectives as obj


def from_objectives(obj_instances, metrics, task_ids, objectives, **kwargs):
    scores = {
        'CrossEntropyLoss': CrossEntropy,
        'BinaryCrossEntropyLoss': BinaryCrossEntropy,
        'DDPHyperbolicTangentRelaxation': DDP,
        'DEOHyperbolicTangentRelaxation': DEO,
        'MSELoss': L2Distance,
        'L1Loss': L1Loss,
        'mIoU': mIoU,
    }
    result = {
        'loss': {t: scores[o](obj_instances[t].label_name, obj_instances[t].logits_name) for t, o in zip(task_ids, objectives)},
    }
    if metrics is not None:
        result['metrics'] = {t: scores[o](obj_instances[t].label_name, obj_instances[t].logits_name) for t, o in zip(task_ids, metrics)}
    return result

class BaseScore():

    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name


    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class CrossEntropy(BaseScore):

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(logits, labels.long(), reduction='mean').item()


class BinaryCrossEntropy(BaseScore):
    
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        if len(logits.shape) > 1 and logits.shape[1] == 1:
            logits = torch.squeeze(logits)

        with torch.no_grad():
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean').item()


class L1Loss(BaseScore):
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.nn.functional.l1_loss(logits, labels.long(), reduction='mean').item()


class mIoU(BaseScore):

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        predictions = logits.max(dim=1)[1]
        ious = []
        for b in range(logits.shape[0]):
            score = 0
            j = 0
            for i in range(logits.shape[1]):
                mask_p = predictions[b] == i
                mask_l = labels[b] == i
                if mask_l.sum() > 0:
                    score += iou(
                        [encode(np.asfortranarray(mask_p.cpu().numpy()))],
                        [encode(np.asfortranarray(mask_l.cpu().numpy()))], 
                        [False]
                    ).squeeze().item()
                    j += 1
            ious.append(score / j)
        return sum(ious) / len(ious)



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
            if len(logits.shape) == 1:
                y_hat = torch.round(torch.sigmoid(logits))
            elif logits.shape[1] == 1:
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

            return (1/n * torch.abs(torch.sum(logits_s_negative > 0) - torch.sum(logits_s_positive > 0))).cpu().item()


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

            return (1/n * torch.abs(torch.sum(logits_s_negative > 0) - torch.sum(logits_s_positive > 0))).cpu().item()        
