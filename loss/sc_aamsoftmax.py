#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy
from basic.evaluate import equal_error_rate

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True   
        self.nClasses = nClasses
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses* 2,nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        nweight = F.normalize(self.weight)
        nembedding = F.normalize(x)
        sim_s3 = F.linear(nembedding, nweight)
        sim = sim_s3.view(-1,self.nClasses,2)
        sim = torch.max(sim,dim=2)[0]
        cosine = sim
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        output_tem = output.detach().cpu()
        label_tem = label.detach().cpu()
        loss = self.ce(output, label)
        eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
        prec1, _    = accuracy(output_tem,label_tem, topk=(1, 5))
        return loss, prec1, eer
