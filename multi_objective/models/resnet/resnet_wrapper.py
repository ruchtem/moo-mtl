from .resnet import ResNet, BasicBlock, Bottleneck, model_urls
from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class ResNetWrapper(ResNet):

    
    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        result = {'logits_{}'.format(t): x[:, i] for i, t in enumerate(self.task_ids)}
        return result


    @classmethod
    def from_name(cls, model_name, dim, task_ids, **override_params):
        cls.task_ids = task_ids
        return resnet18(
            pretrained=False, 
            progress=False,
            in_channels=dim[0],
            num_classes=len(task_ids))


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetWrapper:
    model = ResNetWrapper(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)