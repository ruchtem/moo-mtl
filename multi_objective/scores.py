import torch
import numpy as np

from abc import abstractmethod
from pycocotools.mask import encode, iou

import multi_objective.objectives as obj


def from_objectives(obj_instances, metrics, objectives, task_ids=None, **kwargs):
    scores = {
        'CrossEntropyLoss': CrossEntropy,
        'BinaryCrossEntropyLoss': BinaryCrossEntropy,
        'ddp': DDP,
        'DEOHyperbolicTangentRelaxation': DEO,
        'MSELoss': L2Distance,
        'L1Loss': L1Loss,
        'mIoU': mIoU,
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

        # metric = RunningMetric(metric_type = 'IOU', n_classes=19)

        # metric.update(logits, labels)
        
        # return metric.get_result()['mIOU']




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





class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0.0
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes**2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum()) 
            self.num_updates += predictions.shape[0]
    
        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti!=250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum( np.abs(gti[mask] - _pred.astype(np.int32)[mask]) ) 
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        
    def get_result(self):
        if self._metric_type == 'ACC':
            return {'acc': self.accuracy/self.num_updates}
        if self._metric_type == 'L1':
            return {'l1': self.l1/self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) 
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc':acc_cls, 'mIOU': mean_iou}