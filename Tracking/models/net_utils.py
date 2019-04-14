import os
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import pickle

from utils.log import logger


class ConcatAddTable(nn.Module):
    def __init__(self, *args):
        super(ConcatAddTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        x_out = None
        for module in self._modules.values():
            x = module(input)
            if x_out is None:
                x_out = x
            else:
                x_out = x_out + x
        return x_out


def load_net(path, net):
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        name = k[13:] # remove module.
        new_state_dict[name] = v
        
    #print(new_state_dict)
    state_dict = net.load_state_dict(new_state_dict)
    return epoch, state_dict


def is_cuda(model):
    p = next(model.parameters())
    return p.is_cuda


def get_device(model):
    if is_cuda(model):
        p = next(model.parameters())
        return p.get_device()
    else:
        return None