# Implementation of Proxy-based deep Graph Metric Learning (ProxyGML) approach
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from utils import accuracy
from basic.evaluate import equal_error_rate


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses,  la=20, r=0.4, weight_lambda=0.3, gamma=0.1, tau=0.2, K=3,margin=0.01, scale=15, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        dim=nOut
        self.C = nClasses
        self.N = K
        self.r = r
        self.weight_lambda = weight_lambda
        self.Proxies = Parameter(torch.Tensor(dim, self.C*self.N), requires_grad=True)
        self.instance_label = torch.tensor(np.repeat(np.arange(self.C), self.N)).cuda()
        self.y_instance_onehot = self.to_one_hot(self.instance_label, n_dims=self.C).cuda()
        self.class_label = torch.tensor(np.repeat(np.arange(self.C), 1)).cuda()
        nn.init.xavier_normal_(self.Proxies, gain=1)
        print("#########")
        return

    def to_one_hot(self, y, n_dims=None):
        ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
        y_tensor = y.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

    def scale_mask_softmax(self,tensor,mask,softmax_dim,scale=1.0):
        #scale = 1.0 if self.opt.dataset != "online_products" else 20.0
        scale_mask_exp_tensor = torch.exp(tensor* scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)).unsqueeze(softmax_dim)
        return scale_mask_softmax_tensor

    def forward(self, input, target):
       
        centers = F.normalize(self.Proxies, p=2, dim=0)
        #constructing directed similarity graph
        similarity= input.matmul(centers)
        #relation-guided sub-graph construction
        positive_mask=torch.eq(target.view(-1,1).cuda()-self.instance_label.view(1,-1),0.0).float().cuda() #obtain positive mask
        topk = math.ceil(self.r * self.C * self.N)
        _, indices = torch.topk(similarity + 100 * positive_mask, topk, dim=1) # "1000*" aims to rank faster
        mask = torch.zeros_like(similarity)
        mask = mask.scatter(1, indices, 1)
        prob_a =mask*similarity
        #revere label propagation (including classification process)
        logits=torch.matmul(prob_a , self.y_instance_onehot)
        y_target_onehot = self.to_one_hot(target, n_dims=self.C).cuda()
        logits_mask=1-torch.eq(logits,0.0).float().cuda()
        predict=self.scale_mask_softmax(logits, logits_mask,1).cuda()
        # classification loss
        lossClassify=torch.mean(torch.sum(-y_target_onehot* torch.log(predict + 1e-20),dim=1))
        #regularization on proxies
        output_tem = (predict + 1e-20).detach().cpu()
        label_tem = target.detach().cpu()
        eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
        prec1, _    = accuracy(output_tem,label_tem, topk=(1, 5))
        if self.weight_lambda  > 0:
            simCenter = centers.t().matmul(centers)
            centers_logits = torch.matmul(simCenter , self.y_instance_onehot)
            reg=F.cross_entropy(centers_logits, self.instance_label)
            return lossClassify+self.weight_lambda*reg, prec1,eer
        else:
            return lossClassify,torch.tensor(0.0).to(self.opt.device)

