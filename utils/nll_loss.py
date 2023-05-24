import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from torch.nn.modules.loss import _WeightedLoss

EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce,
                                         reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        y_input = torch.log(y_input + EPSILON)
        y_input = y_input.view(-1)
        y_target = y_target.long().view(-1)
        return nll_loss(y_input,
                        y_target,
                        weight=self.weight,
                        ignore_index=self.ignore_index)
