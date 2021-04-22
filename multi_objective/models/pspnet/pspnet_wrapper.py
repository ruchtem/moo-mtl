import torch

from .pspnet import get_segmentation_encoder, SegmentationDecoder


class PspNetWrapper(torch.nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()

        self.encoder = get_segmentation_encoder()
        self.segm_head = SegmentationDecoder(dim[-2:], num_class=19, task_type='C')
        self.inst_head = SegmentationDecoder(dim[-2:], num_class=2, task_type='R')
        self.depth_head = SegmentationDecoder(dim[-2:], num_class=1, task_type='R')
    

    def forward(self, data):
        x = data['data']
        x = self.encoder(x)

        return {
            'logits_segm': self.segm_head(x),
            'logits_inst': self.inst_head(x),
            'logits_depth': self.depth_head(x)
        }

    
    def forward_feature_extraction(self, batch):
        x = batch['data']
        return self.encoder(x)
    
    def forward_linear(self, x, t):
        return {
            f'logits_{t}': self.__getattr__(f'{t}_head')(x)
        }
    

    # this is required for cosmos
    def change_input_dim(self, dim):
        assert isinstance(dim, int)
        c = self.encoder.conv1
        self.encoder.conv1 = torch.nn.Conv2d(dim, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups)