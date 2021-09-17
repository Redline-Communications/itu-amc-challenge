#!/usr/bin/env python

"""ternary.py: Ternary VGG implementation for PyTorch"""

__author__      = "Armand Kamary"
__copyright__   = "Copyright 2021, Redline Communications"

from typing import Dict, Union, List, cast, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def Ternarize(v : torch.Tensor, nu = 0.7):
    '''
    Ternerize
    '''
    thres = nu * torch.mean(torch.abs(v))
    ge = torch.ge(v, thres).type(torch.FloatTensor)
    le = torch.le(v, -thres).type(torch.FloatTensor)
    unmasked = torch.multiply(ge + le, v)
    eta = torch.mean(unmasked)
    ret = torch.multiply(le, -eta)
    ret = ret + torch.multiply(ge, eta)
    return ret + (ret - v).detach()

def Quantize(v : torch.Tensor, k : int):
    '''
    Quantize
    '''
    factor = 2 ** k - 1
    return torch.round(factor * v) / factor

class SReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, k : float = 1.0, inplace: bool = False):
        super(SReLU, self).__init__()
        self.inplace = inplace
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        v = input.clone()
        cv = torch.clip(v, 0.0, 1.0)
        qv = Quantize(cv, self.k)
        result = cv + (qv - cv).detach()
        if self.inplace:
            input = result
        return result

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class TernaryLinear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(TernaryLinear,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.linear(input,self.weight,self.bias)
        return out

class TernaryConv1d(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        super(TernaryConv1d,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.conv1d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out

class TernaryConv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(TernaryConv2d,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out

class TernaryVGG(nn.Module):
    '''
     Ternary Visual Geometry Group (VGG)
    '''

    def __init__(self, features, num_classes=24, init_weights=True):
        super(TernaryVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            TernaryLinear(64 * 8, 128),
            nn.ReLU(True),
            TernaryLinear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, ternary : bool = False, srelu : bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 2
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if ternary:
                conv1d = TernaryConv1d(in_channels, v, kernel_size=3, padding=1, bias=False)
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1, bias=False)

            layers.append(conv1d)

            if batch_norm:
                layers.append(nn.BatchNorm1d(v))

            if srelu:
                layers.append(SReLU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))

            in_channels = v

    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'VGG10': [64, 'M', 64, 'M', 64, 'M', 64, 'M', 64, 'M', 64, 'M', 64, 'M'],
}

def _vgg(arch: str, cfg: str, batch_norm: bool, **kwargs: Any) -> models.VGG:

    if kwargs.pop('ternary', False):
        model = models.VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, srelu=kwargs.pop('srelu', False)), **kwargs)
    else:
        model = TernaryVGG(make_layers(cfgs[cfg], batch_norm=batch_norm, srelu=kwargs.pop('srelu', False)), **kwargs)

    return model


def VGG10(**kwargs: Any) -> models.VGG:
    '''
    VGG 10-layer model (configuration "VGG10") from
    '''
    return _vgg('VGG10', 'VGG10', batch_norm=True, **kwargs)

def TernaryVGG10(**kwargs: Any) -> models.VGG:
    '''
    TernaryVGG 10-layer model (configuration "VGG10") from
    '''
    return _vgg('TernaryVGG10', 'VGG10', batch_norm=True, ternary=True, **kwargs)