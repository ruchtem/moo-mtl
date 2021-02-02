from models.efficient_net.model import EfficientNet
import torch

# small wrapper
class EfficientNetWrapper(EfficientNet):


    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

        self.task_layers = torch.nn.ModuleDict()
        for t in self.task_ids:
            self.task_layers[f"task_fc_{t}"] = torch.nn.Linear(1792 if not self.late_fusion else 1792 + len(self.task_ids), 1)


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
        
    def forward_linear(self, x, i):
        # x = self._dropout(x)
        # x = self._fc(x)
        result = {f'logits_{i}': self.task_layers[f'task_fc_{i}'](x)}
        return result
    
    def private_params(self):
        return [n for n, p in self.named_parameters() if "task_layers" not in n]


    def forward(self, batch):
        x = batch['data']
        b = x.shape[0]
        if self.late_fusion:
            a = batch['alpha'].repeat(b, 1)
            features = self.forward_feature_extraction(batch)
            x = torch.cat((features, a), dim=1)
        else:
            x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result
    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **override_params):
        cls.task_ids = task_ids
        return super().from_pretrained(model_name, in_channels=dim[0], num_classes=len(task_ids))
    

    @classmethod
    def from_name(cls, dim, task_ids, model_name, late_fusion, **override_params):
        cls.task_ids = task_ids
        cls.late_fusion=late_fusion
        
        # speeds up mgda
        
        #cls.task_linear = {f'fc_{t}': torch.nn.Linear(1792, 1).cuda() for t in task_ids}

        return super().from_name(model_name, in_channels=dim[0], num_classes=len(task_ids))


