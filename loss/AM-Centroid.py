#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Fast re-implementation of the GE2E loss (https://arxiv.org/abs/1710.10467) 
## Numerically checked against https://github.com/cvqluu/GE2E-Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
from basic.evaluate import equal_error_rate
import math

class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, scale=30, init_b=-5.0, easy_margin=False, margin=0.5, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True 
        self.m = margin
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.s = scale
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m) # -cos(m)
        self.mm = math.sin(math.pi - self.m) * self.m # sin(m) * m
        print('Initialised GE2E')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        x = F.normalize(x,p=2,dim=2)
        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]
        cos_sim_matrix = []
        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        for ii in range(0,gsize): 
            idx = [*range(0,gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:,idx,:], 1)
            cosine = F.cosine_similarity(x[:,ii,:],exc_centroids)
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m           
            phi = torch.where(cosine > 0, phi, cosine)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(0, label, 0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = self.s * output 
            cos_sim_diag = output
            cos_sim = self.s * F.cosine_similarity(x[:,ii,:].unsqueeze(-1),centroids.unsqueeze(-1).transpose(0,2))
            cos_sim[range(0,stepsize),range(0,stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim,1e-6))
        cos_sim_matrix = torch.stack(cos_sim_matrix,dim=1) 
        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        L4 = self.criterion(cos_sim_matrix.view(-1,stepsize), torch.repeat_interleave(label,repeats=gsize,dim=0).cuda())
        L5 = 0
        num = 0
        Min_dis = 100
        for ii in range(0,stepsize-1):
            tem_ii_center = centroids[ii].unsqueeze(0)
            dis = torch.cosine_similarity(tem_ii_center, centroids[ii+1:])
            L5 += dis.sum()
        L5 = 2.0 /  (stepsize *  (stepsize - 1 ) )   *  L5
        prec1, _  = accuracy(cos_sim_matrix.view(-1,stepsize).detach().cpu(), torch.repeat_interleave(label,repeats=gsize,dim=0).detach().cpu(), topk=(1, 5))
        output_tem = cos_sim_matrix.view(-1,stepsize).cpu()
        label_tem = torch.repeat_interleave(label,repeats=gsize,dim=0).cpu()
        eer = equal_error_rate(label_tem.detach().numpy(), output_tem.detach().numpy())
        nloss = L4 
        return nloss, prec1, eer
