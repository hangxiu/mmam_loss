import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb
import numpy as np
from operator import itemgetter

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []
      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm
      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]
      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

## ，
def compute_eer(fnr, fpr):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])

def compute_acc(scores, labels):
    sorted_ndx = np.argsort(scores)
    labels = np.array(labels)[sorted_ndx]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def compute_pmiss_pfa(scores, labels):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial scores and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such
    partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = np.array(labels)[sorted_ndx]
    tgt = (labels == 1).astype('f8')
    imp = (labels == 0).astype('f8')
    fnr = np.cumsum(tgt) / np.sum(tgt)
    fpr = 1 - np.cumsum(imp) / np.sum(imp)
    return fnr, fpr


def compute_min_cost(scores, labels, p_target=0.01):
    fnr, fpr = compute_pmiss_pfa(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_c = compute_c_norm(fnr, fpr, p_target)
    return eer, min_c


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return c_det/c_def

def main():
    scores = np.random.rand(1000)
    labels = np.random.randint(2, size=1000)
    print(type(labels))
    eer, minc = compute_min_cost(scores, labels, p_target=0.05)
    tunedThreshold, eer1, fpr, fnr = tuneThresholdfromScore(scores, labels, target_fa=[1,0.001], target_fr = None)
    min_dcf, min_c_det_threshold = ComputeMinDcf(fnr, fpr, thresholds=tunedThreshold, p_target=0.05, c_miss=1, c_fa=1)
    print(eer,minc)
    print(eer1, min_dcf)
    
if __name__ == "__main__":
    main()