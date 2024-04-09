import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import sys
sys.path.append("/binf-isilon/renniegrp/vpx267/ucph_thesis/")
from model import ConfigurableModel, ConfigurableModelWoBatchNorm, TestMotifModel
import os
from wrapper import utils
from wrapper.utils import one_hot_to_sequence


def forward_to_RELU_1(model, x):
    """
    Assuming there is defined "self.CNN" layer in the model, It will get the output from the first layer after activation

    param: model: torch.nn.Module: model to be used
    param: x: torch.Tensor: input tensor
    return: torch.Tensor: output tensor after the first ReLU layer
    """
    for i, layer in enumerate(model.CNN): # Assuming there is defined "self.CNN" layer
        print(layer)
        x = layer(x)
        if isinstance(layer, nn.ReLU): # If it reaches ReLU, return the output 
            return x
