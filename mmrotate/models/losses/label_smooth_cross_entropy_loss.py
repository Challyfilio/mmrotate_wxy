# Copyright (c) 2023 ✨Challyfilio✨. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from loguru import logger


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


@ROTATED_LOSSES.register_module()
class LabelSmoothCrossEntropyLoss(nn.Module):
    """
    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 avg_factor=None,
                 reduction_override=None,
                 epsilon=0.1,
                 reduction='mean'):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        # self.avg_factor = avg_factor

    def forward(self,
                preds,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        # logger.error(self.avg_factor)
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            
        Returns:
            torch.Tensor: The calculated loss
        """
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        loss_cls = linear_combination(loss / n, nll, self.epsilon)
        return loss_cls
