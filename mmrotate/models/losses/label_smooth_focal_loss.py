# Copyright (c) Challyfilio. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from ..builder import ROTATED_LOSSES


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y


@ROTATED_LOSSES.register_module()
class LabelSmoothFocalLoss(nn.Module):
    def __init__(self,
                 avg_factor=None,
                 reduction_override=None,
                 alpha=0.25,
                 gamma=2.0,
                 epsilon=0.1, 
                 reduction='mean',
                 ): 
        super(LabelSmoothFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.avg_factor = avg_factor
    
    
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
        log_preds = F.log_softmax(preds, dim=-1) # pt
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll_1 = F.nll_loss(log_preds, target, reduction=self.reduction) # -log(pt)
        # nll_2 = F.nll_loss(1-log_preds, target, reduction=self.reduction) # -log(1-pt)
        loss_cls = linear_combination(loss / n, self.alpha * (1 - log_preds).pow(self.gamma) * nll_1, self.epsilon)
        # loss_cls = self.alpha * (1 - log_preds).pow(self.gamma) * (1 - self.epsilon) * nll_1 + (1 - self.alpha) * log_preds.pow(self.gamma) * self.epsilon * nll_2
        return loss_cls