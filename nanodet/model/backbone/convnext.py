from turtle import forward
import torch
import torch.nn as nn
import timm

class ConvNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_femto', features_only=True, pretrained=True)
    def forward(self,x):
        outs = self.model(x)[1:]
        return outs