#! /usr/bin/python
# -*- encoding: utf-8 -*-

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
        self.sigma = 2
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        self.cur_m = self.m * (math.e ** (1 - cosine) ) / self.sigma
        self.cur_m = self.cur_m.cpu()
        self.cos_m = [t.numpy() for t in math.cos(self.cur_m)]
        self.sin_m = math.sin(self.cur_m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)
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