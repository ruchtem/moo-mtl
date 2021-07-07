import torch
from abc import abstractmethod


def from_objectives(obj_instances, metrics, objectives, task_ids=None, **kwargs):
    scores = {
        'CrossEntropyLoss': CrossEntropy,
        'mcr': mcr,
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
