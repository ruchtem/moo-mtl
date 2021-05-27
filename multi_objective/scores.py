import torch
import numpy as np

from abc import abstractmethod
from pycocotools.mask import encode, iou

import multi_objective.objectives as obj
from multi_objective import utils


def from_objectives(obj_instances, metrics, objectives, task_ids=None, **kwargs):
    scores = {
        'CrossEntropyLoss': CrossEntropy,
        'BinaryCrossEntropyLoss': BinaryCrossEntropy,
        'ddp': DDP,
        'DEOHyperbolicTangentRelaxation': DEO,
        'MSELoss': L2Distance,
        'L1Loss': L1Loss,
        'mIoU': mIoU,
        'VAELoss': VAELoss,
        'WeightedVAELoss': WeightedVAELoss,
        'RecallAtK': RecallAtK,
        'RevenueAtK': RevenueAtK,
    }
    if len(task_ids) == 0:
        task_ids = list(obj_instances.keys())
    result = {
        'loss': {t: scores[o](obj_instances[t].label_name, obj_instances[t].logits_name, **kwargs) for t, o in zip(task_ids, objectives)},
    }
    if metrics is not None:
        result['metrics'] = {t: scores[o](obj_instances[t].label_name, obj_instances[t].logits_name, **kwargs) for t, o in zip(task_ids, metrics)}
    return result

class BaseScore():

    def __init__(self, label_name='labels', logits_name='logits', ignore_index=-100, **kwargs):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.ignore_index = ignore_index
        self.kwargs = kwargs


    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class CrossEntropy(BaseScore):

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(logits, labels.long(), reduction='mean', ignore_index=self.ignore_index).item()


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
            if 'inst' in self.label_name:
                mask = labels != self.ignore_index
                return torch.nn.functional.l1_loss(logits[mask], labels[mask], reduction='mean').item()
            else:
                return torch.nn.functional.l1_loss(logits, labels, reduction='mean').item()


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


class VAELoss(obj.VAELoss):

    def __call__(self, **kwargs):
        kwargs['vae_beta'] = 1.
        with torch.no_grad():
            return super().__call__(**kwargs).item()


class WeightedVAELoss(obj.WeightedVAELoss):

    def __call__(self, **kwargs):
        kwargs['vae_beta'] = 1.
        with torch.no_grad():
            return super().__call__(**kwargs).item()


class RecallAtK(BaseScore):
    """
    Taken from https://github.com/swisscom/ai-research-mamo-framework/tree/master/metric
    and adapted
    """


    def __call__(self, **kwargs):
        y_pred = kwargs[self.logits_name]
        y_true = kwargs[self.label_name]
        x = kwargs['data']

        # Filtering out already chosen items:
        # In order to make sure we are not recommending already
        # chosen items, here we set the prediction of the already
        # chosen items to the minimum value. We know what items
        # have already been chosen since they are in our input X.
        y_pred[x == 1] = torch.min(y_pred)

        y_pred_binary = utils.find_top_k_binary(y_pred, self.kwargs['K'])
        y_true_binary = y_true > 0
        tmp = (y_true_binary & y_pred_binary).sum(dim=1).double()
        ones = torch.ones(y_true_binary.shape[0]).to(y_true.device).double()
        ks = torch.ones(y_true_binary.shape[0]).to(y_true.device).double()
        ks.fill_(self.kwargs['K'])
        d = torch.min(ks, torch.max(ones, y_true_binary.sum(dim=1).double()))
        recall = tmp / d
        result = round(recall.mean().item(), 6)
        if not (0 <= result <= 1):
            raise ValueError('The output of RecallAtK.evaluate ' + result
                             + ' must be in [0,1]')
        return result


class RevenueAtK(BaseScore):
    """
    Taken from https://github.com/swisscom/ai-research-mamo-framework/tree/master/metric
    and adapted
    """

    def __init__(self, label_name, logits_name, loss_weights=None, **kwargs):
        super().__init__(label_name, logits_name, **kwargs)
        self._revenue = np.load(loss_weights)

    def __call__(self, **kwargs):
        y_pred = kwargs[self.logits_name]
        y_true = kwargs[self.label_name]
        x = kwargs['data']

        # Filtering out already chosen items:
        # In order to make sure we are not recommending already
        # chosen items, here we set the prediction of the already
        # chosen items to the minimum value. We know what items
        # have already been chosen since they are in our input X.
        y_pred[x == 1] = torch.min(y_pred)

        y_true = y_true.cpu().numpy()

        if y_pred.shape[1] != len(self._revenue):
            raise ValueError('Arguments must have axis 1 of the same size as\
            the revenue.')

        y_pred_binary = utils.find_top_k_binary(y_pred, self.kwargs['K']).cpu().numpy()
        y_true_binary = (y_true > 0)
        tmp = np.logical_and(y_true_binary, y_pred_binary)
        revenue = 0
        for i in range(tmp.shape[0]):
            revenue += np.sum(self._revenue[tmp[i]])
        return revenue / float(tmp.shape[0])

