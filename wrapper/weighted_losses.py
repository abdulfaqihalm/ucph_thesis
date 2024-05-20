import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs:torch.Tensor, targets:torch.Tensor, weights:None|torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the weighted mean squared error loss between inputs and targets.

    param: inputs: tensor, model output
    param: targets: tensor, true value 

    return: weigthed loss
    """
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs:torch.Tensor, targets:torch.Tensor, weights:None|torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the weighted l1 loss between inputs and targets.

    param: inputs: tensor, model output
    param: targets: tensor, true value 

    return: weigthed loss
    """
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs:torch.Tensor, targets:torch.Tensor, weights:None|torch.Tensor = None, beta:float = 1.) -> torch.Tensor:
    """
    Calculate the weighted huber loss between inputs and targets. Beta is the threshold for the l1 loss.

    param: inputs: tensor, model output
    param: targets: tensor, true value 

    return: weigthed loss
    """
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss