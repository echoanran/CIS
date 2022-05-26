from locale import normalize
import torch
from torch.cuda import device_count
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import os
import numpy as np
from scipy import stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CLFLoss(nn.Module):
    def __init__(self,
                 weights=None,
                 lambda_clf=1,
                 size_average=True,
                 **kwargs):
        super(CLFLoss, self).__init__()
        self.size_average = size_average
        self.weights = weights
        self.lambda_clf = lambda_clf

    def forward(self, outputs, targets, *args):
        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()
        loss_buff = 0
        for i in range(num_class):
            target = targets[:, i]
            output = outputs[:, i]
            if self.weights is None:
                loss_au = torch.sum(
                    -(target * torch.log((output + 0.05) / 1.05) +
                      (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            else:
                loss_au = torch.sum(-(
                    (1.0 - self.weights[i]) * target * torch.log(
                        (output + 0.05) / 1.05) + self.weights[i] *
                    (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return self.lambda_clf * loss_buff / (num_class * N)


if __name__ == '__main__':
    targets = [[1, 0, 0, 1, 1, 0, 1]]
    outputs = [[0.5, 0.4, 0.1, 0.6, 0.7, 0.2, 0.3]]

    targets = torch.tensor(targets, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)

    loss = CLFLoss()(outputs, targets)