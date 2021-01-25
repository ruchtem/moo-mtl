# effitientnet code taken from here: https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
from models.efficient_net.model import EfficientNet

# small wrapper
class EfficientNetWrapper(EfficientNet):

    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result
    
    @classmethod
    def from_pretrained(cls, dim, task_ids, **override_params):
        cls.task_ids = task_ids
        return super().from_pretrained('efficientnet-b3', in_channels=dim[0], num_classes=len(task_ids))
    

    @classmethod
    def from_name(cls, dim, task_ids, **override_params):
        cls.task_ids = task_ids
        return super().from_name('efficientnet-b3', in_channels=dim[0], num_classes=len(task_ids))


