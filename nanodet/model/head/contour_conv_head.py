import torch
import torch.nn as nn

from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.scale import Scale


class ContourConvHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channel,
        feat_channels=256,
        stacked_convs=4,
        strides=[8, 16, 32],
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        activation="LeakyReLU",
        reg_max=16,
        **kwargs
    ):
        super(ContourConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):
        outputs = []
        for x, scale in zip(feats, self.scales):
            reg_feat = x
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()
            outputs.append(bbox_pred.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
