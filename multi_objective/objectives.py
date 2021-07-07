import torch.nn.functional as F


def from_name(objectives, task_ids=None, **kwargs):
    map = {
        'CrossEntropyLoss': CrossEntropyLoss,
        # Add your custom loss here
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

