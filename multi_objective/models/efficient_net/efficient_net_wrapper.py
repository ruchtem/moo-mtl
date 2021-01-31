from models.efficient_net.model import EfficientNet

# small wrapper
class EfficientNetWrapper(EfficientNet):


    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
        
    def forward_linear(self, x):
        x = self._dropout(x)
        x = self._fc(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result


    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result
    
    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, **override_params):
        cls.task_ids = task_ids
        return super().from_pretrained(model_name, in_channels=dim[0], num_classes=len(task_ids))
    

    @classmethod
    def from_name(cls, dim, task_ids, model_name, **override_params):
        cls.task_ids = task_ids
        return super().from_name(model_name, in_channels=dim[0], num_classes=len(task_ids))


