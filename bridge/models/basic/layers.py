import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class ConvEncoder(torch.nn.Module):
    
    def __init__(self, hidden_size=16, num_pixels=24, kernel_size=3, in_channels=3, out_channels=3, padding=0, stride=1):
        super().__init__()
        self.out_dim = ((num_pixels+2*padding-(kernel_size-1) - 1)//stride+1)
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.linear1= torch.nn.Linear(self.out_dim**2*out_channels, hidden_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        out = F.relu(x)
        return out

