
from __future__ import absolute_import
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchFewShot.models.resnet_drop import resnet12, BasicBlock
from torchFewShot.models.wrn import wrn
from torchFewShot.models.conv4 import conv4
#from torchFewShot.models.vit.t2t_vit import T2T_ViT, T2t_vit_24, T2t_vit_12, T2t_vit_14
#from torchFewShot.models.resnet12 import resnet12
#from cam import CAM

from torchFewShot.losses import CrossEntropyLoss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch import nn
from torch.nn import functional as F

class Upsample(nn.Module):
    def __init__(self, ndim, size=None):
        super(Upsample, self).__init__()
        downsample = nn.Sequential(
                nn.Conv2d(ndim, ndim//2,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(ndim//2),
            )

        self.conv = BasicBlock(ndim, ndim//2, downsample=downsample, pool=False)
        self.size = size

    def forward(self, x):
        x = self.conv(x)
        if self.size is None:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        else:
            x = F.interpolate(x, size=self.size)
        return x

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Decoder(nn.Module):
    def __init__(self, nin, ndim, input_size=11, output_size=84):
        super(Decoder, self).__init__()
        num_layers = int(math.log(math.ceil(output_size/input_size), 2))
        self.conv = nn.Conv2d(nin, ndim, kernel_size=3, stride=1, padding=1)

        self.blocks = []
        for i in range(num_layers):
            size = output_size if i == num_layers-1 else None
            self.blocks.append(Upsample(ndim, size=size))
            ndim = ndim // 2
        self.blocks = nn.ModuleList(self.blocks)

        self.out_norm = Normalize(ndim)
        self.out_conv = nn.Conv2d(ndim, 3, kernel_size=3, stride=1, padding=1)

    def corr_fuse(self, x, cond, mix_masks):
        '''
        x: (b, c, h, w)
        cond: (b, c, h, w)
        '''
        b, c, h, w = x.shape
        mix_masks = mix_masks.reshape(-1, h, w).reshape(-1, h*w).unsqueeze(1)
        x_ = F.normalize(x, p=2, dim=1).reshape(b, c, -1).permute(0, 2, 1)
        cond_ = F.normalize(cond, p=2, dim=1).reshape(b, c, -1)
        mask = (1 - torch.eye(h*w)).unsqueeze(0).repeat(b, 1, 1).cuda()

        x_sim = (torch.bmm(x_, x_.permute(0, 2, 1)) * mask * mix_masks).mean(-1).reshape(b, h, w)
        cond_sim = (torch.bmm(x_, cond_).permute(0, 2, 1) * mask * mix_masks).mean(-1).reshape(b, h, w)

        T = 0.1
        th = 0.7
        #softmax
        #x_sim = (x_sim/T).exp()
        #cond_sim = (cond_sim/T).exp()
        
        #sim_sum = x_sim + cond_sim
        #x_sim = x_sim / sim_sum
        #cond_sim = cond_sim / sim_sum

        # hard threshold
        #x_sim = (x_sim > th).float()
        #cond_sim = (cond_sim > th).float()

        # gumbel softmax sampling
        logits = torch.stack([x_sim/T, cond_sim/T], 1)
        #logits = torch.stack([x_sim, cond_sim], 1)
        action = F.gumbel_softmax(logits, hard=True, dim=1)
        x_sim, cond_sim = action[:, 0], action[:, 1]

        #feat = x_sim.unsqueeze(1).contiguous() * x + cond_sim.unsqueeze(1).contiguous() * cond
        feat = torch.cat([x_sim.unsqueeze(1) * x, cond_sim.unsqueeze(1) * cond], 1)
        cond_feat = torch.cat([(1-x_sim.unsqueeze(1)) * x, (1-cond_sim.unsqueeze(1)) * cond], 1)
        #feat = x + cond

        return torch.cat([feat, cond_feat], 0), logits
        #return feat, logits

    def forward(self, x, cond, masks):
        x, logits = self.corr_fuse(x, cond, masks)
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)
        return x, logits

class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64, dilations=[1], kernel=3, groups=1, mode=None, normalize=None, cascade=False, ode=False, backbone='resnet', base=True, depth=1, support=False):
        super(Model, self).__init__()
        self.scale_cls = scale_cls

   
        if backbone == 'resnet':
            self.base = resnet12()
        elif backbone == 'wrn':
            self.base = wrn()
        elif backbone == 'conv4':
            self.base = conv4()

        self.nFeat = self.base.nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

        self.baseline = base
        self.cascade = cascade
        self.support = support

        self.decoder = Decoder(self.nFeat*2, self.nFeat, 11)
        #self.decoder = Decoder(self.nFeat, self.nFeat)

        #self.graph_module = AdaGraph(ndim=self.nFeat)

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def process_feature(self, f, ytrain, num_train, num_test, batch_size, adapt=False):
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.reshape(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.reshape(batch_size, -1, *f.size()[1:])

        #add support samples into query samples
        ftrain_ = f[:batch_size * num_train].reshape(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:].reshape(batch_size, -1, *f.size()[1:])
        if self.training and self.support:
            ftest = torch.cat([ftrain_, ftest], 1)
        ftrain, ftest = self.reshape(ftrain, ftest)

        # b, n2, n1, c, h, w
        ftrain = ftrain.transpose(1, 2)
        ftest = ftest.transpose(1, 2)
        return ftrain, ftest


    def get_score(self, ftrain, ftest, adapt=False):

        b, n2, n1, c, h, w = ftrain.shape
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        l_ftest = torch.norm(ftest, p=2, dim=3, keepdim=False)
        #ftest_norm = ftest.clone()
        #ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain.clone()

        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        ftest_norm = ftest_norm.reshape(1, -1, *ftest_norm.size()[4:])
        if not adapt:
            conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)
        else:
            conv_weight = ftrain_norm.clone()
        conv_weight = F.normalize(conv_weight, p=2, dim=1, eps=1e-12)

        base_cls_scores = F.conv2d(ftest_norm, conv_weight, groups=b*n1*n2)
        base_cls_scores = base_cls_scores.view(b* n2, n1, *base_cls_scores.size()[2:])
        base_cls_scores = base_cls_scores * l_ftest.view(-1, *l_ftest.size()[2:])
        #base_cls_scores = base_cls_scores * 100
        return base_cls_scores


    def get_global_pred(self, ftest, num_test, batch_size, K):
        h = ftest.shape[-1]
        global_pred = self.global_clasifier(ftest)
        return global_pred

    def get_test_score(self, score_list):
        if isinstance(score_list, list):
            return torch.stack(score_list, 0).mean(0).mean(-1).mean(-1)
        return score_list.mean(-1).mean(-1)

    def recons(self, ftest, cond, masks):
        #pred = self.decoder(torch.cat([ftest, cond], 1))
        pred, logits = self.decoder(ftest, cond, masks)
        return pred, logits

    def forward(self, xtrain, xtest, ytrain, ytest, cond=None, global_labels=None, masks=None, recons=True):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        if self.training and cond is not None and recons:
            cond_feat = self.base(cond.reshape(-1, *cond.size()[2:]))
            recons, logits = self.recons(f[batch_size*num_train:], cond_feat, masks)
        else:
            recons, logits = None, None

        global_pred = self.get_global_pred(f, num_test, batch_size, K)

        ftrain, ftest = self.process_feature(f, ytrain, num_train, 
                                                num_test, batch_size)
        cls_scores = [self.get_score(ftrain, ftest)]

        if not self.training:
            return self.get_test_score(cls_scores)

        return global_pred, cls_scores, recons, logits
