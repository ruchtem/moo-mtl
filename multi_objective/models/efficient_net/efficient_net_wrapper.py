from .model import EfficientNet
from .utils import get_model_params, load_pretrained_weights

import torch

# small wrapper
class EfficientNetWrapper(EfficientNet):


    def __init__(self, blocks_args, global_params):
        global_params = global_params._replace(batch_norm_layer=torch.nn.Identity)
        super().__init__(blocks_args, global_params)

        self.task_layers = torch.nn.ModuleDict({
            f'task_fc_{t}': torch.nn.Linear(1792, 1) for t in self.task_ids
        })


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = super().forward(x)
        x = x.flatten(start_dim=1)
        return x
    
        
    def forward_linear(self, x, t):
        return {
            f'logits_{t}': self.task_layers[f'task_fc_{t}'](x)
        }


    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        x = x.flatten(start_dim=1)
        return {
            f'logits_{t}': self.task_layers[f'task_fc_{t}'](x) for t in self.task_ids
        }


    # this is required for cosmos
    def change_input_dim(self, dim):
        assert isinstance(dim, int)
        self._change_in_channels(dim)

    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **kwargs):
        cls.task_ids = task_ids
        return super().from_pretrained(model_name, in_channels=dim[0], include_top=False)
    
