#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
from basic.evaluate import equal_error_rate

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
	    super(LossFunction, self).__init__()
	    self.test_normalize = True	  
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(nOut,nClasses)
	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):	
		half = x.shape[1] / 2
		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1, _ = accuracy(x.detach().cpu(), label.detach().cpu(), topk=(1, 5))
    	output_tem = x.detach().cpu()
		label_tem = label.detach().cpu()
		eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
		return nloss, prec1, eer