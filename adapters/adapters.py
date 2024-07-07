
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

@dataclass
class LoRAConfig:
    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    enable: List[bool] = field(default_factory=lambda: [True, False, True])

class SimpleAdapterLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(SimpleAdapterLayer, self).__init__()
        if hidden_size == None:
            hidden_size = int(input_size / 2)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = nn.functional.relu(x1)
        x1 = self.fc2(x1)
        return x + x1

class HFPA(nn.Module):
    def __init__(self, input_size, adapter_heads, adapter_heads_size, dropout=0.1):
        super(HFPA, self).__init__()
        self.adapter_heads = adapter_heads
        self.adapter_heads_size = adapter_heads_size
        self.adapter_all_size = adapter_heads * adapter_heads_size
        self.d_model_heads_size = int(input_size / self.adapter_heads)
        assert input_size % self.adapter_heads == 0
        self.upsample = nn.ModuleList(
            [nn.Linear(self.d_model_heads_size, self.adapter_heads_size) for _ in range(self.adapter_heads)])
        self.unlinear = nn.ReLU()
        self.downsample = nn.ModuleList(
            [nn.Linear(self.adapter_heads_size, self.d_model_heads_size) for _ in range(self.adapter_heads)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, layernorm=None,resiual = True):
        residual = x
        x = x.view(x.size()[:-1] + (self.adapter_heads, self.d_model_heads_size))
        x = x.transpose(0,-2)
        x_out = []

        for i in range(self.adapter_heads):
            x_temp = self.upsample[i](x[i])
            x_temp = self.unlinear(x_temp)
            x_temp = self.downsample[i](x_temp)
            x_out.append(x_temp)
        x = torch.stack(x_out,dim=0).transpose(0,-2)
        x = x.reshape(residual.shape)
        x = self.drop(x)
        if layernorm is None:
            if resiual:
                x = x + residual
            else:
                return x
        else:
            if resiual:
                x = layernorm(x+residual)
            else:
                return layernorm(x)
        torch.cuda.empty_cache()
        return x

class HFPCA(HFPA):
    def __init__(self, input_size, adapter_heads, adapter_heads_size, dropout=0.1):
        super(HFPCA, self).__init__(input_size, adapter_heads, adapter_heads_size, dropout=dropout)
        #self.cross_linear = nn.Sequential(nn.Linear(self.adapter_all_size,self.adapter_all_size),nn.ReLU(),nn.Linear(self.adapter_all_size,self.adapter_all_size))
        self.cross_linear = nn.Linear(self.adapter_all_size,self.adapter_all_size)
    def forward(self, x, layernorm=None,resiual = True):
        residual = x
        x = x.view(x.size()[:-1] + (self.adapter_heads, self.d_model_heads_size))
        x = x.transpose(0, -2)
        x_out = []
        for i in range(self.adapter_heads):
            x_temp = self.upsample[i](x[i])
            #x_temp = self.unlinear(x_temp)
            x_out.append(x_temp)
        x = torch.stack(x_out,dim=0).transpose(0,-2)
        x_shape = x.shape
        x = x.reshape(-1,x_shape[-1]*x_shape[-2])
        x = x+self.cross_linear(x)
        x = self.unlinear(x)
        x = x.reshape(x_shape)
        x = x.transpose(0, -2)
        x_out = []
        for i in range(self.adapter_heads):
            x_out.append(self.downsample[i](x[i]))
       #x = x.transpose(0,-2)
        x = torch.stack(x_out,dim=0)
        x = x.transpose(0,-2)
        x = x.reshape(residual.shape)
        x = self.drop(x)
        if layernorm is None:
            if resiual:
                x = x + residual
            else:
                return x
        else:
            if resiual:
                x = layernorm(x+residual)
            else:
                return layernorm(x)
        torch.cuda.empty_cache()
        return x


