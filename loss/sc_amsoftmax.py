#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)
#https://github.com/cvqluu/GE2E-Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
from basic.evaluate import equal_error_rate

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.nClasses = nClasses
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses*2), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        # [B, E] [E, N*C] [B, N, C]
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        sim = costh.view(-1,self.nClasses,2)
        sim = torch.max(sim,dim=2)[0]
        costh = sim
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        output_tem = costh_m_s.detach().cpu()
        label_tem = label.detach().cpu()
        eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
        prec1, _    = accuracy(costh_m_s.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, prec1, eer

