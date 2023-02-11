
from __future__ import absolute_import
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchFewShot.models.resnet_drop import resnet12
from torchFewShot.models.wrn import wrn
from torchFewShot.models.conv4 import conv4

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, num_classes=64, backbone='resnet', support=False):
        super(Model, self).__init__()
   
        self.base = resnet12()

        self.nFeat = self.base.nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

        self.support = support

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def process_feature(self, f, ytrain, num_train, num_test, batch_size, return_test=False):
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.reshape(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.reshape(batch_size, -1, *f.size()[1:])

        ftrain_ = f[:batch_size * num_train].reshape(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:].reshape(batch_size, -1, *f.size()[1:])
        if not return_test:
            ftest = torch.cat([ftrain_, ftest], 1)
        ftrain, ftest = self.reshape(ftrain, ftest)

        # b, n2, n1, c, h, w
        ftrain = ftrain.transpose(1, 2)
        ftest = ftest.transpose(1, 2)
        return ftrain, ftest


    def get_score(self, ftrain, ftest):

        b, n2, n1, c, h, w = ftrain.shape
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        l_ftest = torch.norm(ftest, p=2, dim=3, keepdim=False)
        ftrain_norm = ftrain.clone()

        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        ftest_norm = ftest_norm.reshape(1, -1, *ftest_norm.size()[4:])
        conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)
        conv_weight = F.normalize(conv_weight, p=2, dim=1, eps=1e-12)

        base_cls_scores = F.conv2d(ftest_norm, conv_weight, groups=b*n1*n2)
        base_cls_scores = base_cls_scores.view(b* n2, n1, *base_cls_scores.size()[2:])
        base_cls_scores = base_cls_scores * l_ftest.view(-1, *l_ftest.size()[2:])
        return base_cls_scores


    def get_global_pred(self, ftest):
        h = ftest.shape[-1]
        global_pred = self.global_clasifier(ftest)
        return global_pred

    def get_test_score(self, score_list):
        if isinstance(score_list, list):
            return torch.stack(score_list, 0).mean(0).mean(-1).mean(-1)
        return score_list.mean(-1).mean(-1)

    def forward(self, xtrain, xtest, ytrain, ytest, global_labels=None, return_test=False):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        global_pred = self.get_global_pred(f)
        ftrain, ftest = self.process_feature(f, ytrain, num_train, 
                                                num_test, batch_size, return_test=return_test)

        cls_scores = [self.get_score(ftrain, ftest)]
        if return_test:
            return self.get_test_score(cls_scores)

        return global_pred, cls_scores
