from .model import EfficientNet
from .utils import get_model_params, load_pretrained_weights

import torch

# small wrapper
class EfficientNetWrapper(EfficientNet):


    def __init__(self, blocks_args, global_params):
        global_params = global_params._replace(batch_norm_layer=torch.nn.Identity)
        super().__init__(blocks_args, global_params)

        self.task_layers = torch.nn.ModuleDict({
            f'task_fc_{t}': torch.nn.Linear(1000, 1) for t in self.task_ids
        })


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = super().forward(x)
        return x
    
        
    def forward_linear(self, x, i):
        # x = self._fc(x)
        result = {f'logits_{i}': self.task_layers[f'task_fc_{i}'](x)}
        return result
    
    def private_params(self):
        return [n for n, p in self.named_parameters() if "task_layers" not in n]


    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {f'logits_{t}': self.task_layers[f'task_fc_{t}'](x) for t in self.task_ids}
        return result


    def change_input_dim(self, dim):
        assert isinstance(dim, int)
        self._change_in_channels(dim)

    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **kwargs):
        cls.task_ids = task_ids
        return super().from_pretrained(model_name, in_channels=dim[0])
    
