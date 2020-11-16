from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class AD(nn.Module):
    def __init__(self,
                 in_feat,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(AD, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels,
                        kernel_size=1, padding=0,
                        bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(in_feat, num_target_channels),
            nn.ReLU(),
            conv1x1(num_target_channels, num_target_channels),
            nn.ReLU(),
            conv1x1(num_target_channels, num_target_channels)
        )

        self.log_scale = Parameter(np.log(np.exp(init_pred_var-eps)-1.0)* torch.ones(num_target_channels))
        self.eps = eps


    def forward(self, input, target):

        x = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = x.view(1, -1, 1, 1)

        pred_mean = self.regressor(input)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)

        return loss
