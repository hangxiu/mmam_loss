import scipy.io as io
import numpy as np
from tuneThreshold import *


def evaluateFromList(listfilename, norm_file_name):
    all_scores = [];
    all_labels = [];
    all_trials = [];
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline();
            if (not line):
                break;
            data = line.split();
            all_labels.append(int(float(data[-1])));
    with open(norm_file_name) as listfile:
        while True:
            line = listfile.readline();
            if (not line):
                break;
            data = line.split();
            all_scores.append(float(data[-1]));  
    print('\n')
    return (all_scores, all_labels, all_trials);

def main(listfilename, norm_file_name):
    sc, lab, _ = evaluateFromList(listfilename, norm_file_name)
    print(sc)
    print(lab)
    result = tuneThresholdfromScore(sc, lab, [1, 0.01]);
    print('EER %2.4f'%result[1])

    p_target = 0.5
    c_miss = 1
    c_fa = 1

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)
    eer, minc = compute_min_cost(sc, lab, p_target=0.5)
    print('EER %2.4f MinDCF %.5f'%(eer,minc))
    print('EER %2.4f MinDCF %.5f'%(result[1],mindcf / 2 ))
    quit();

if __name__ == "__main__":

    main("test_all.trials", "GML_N3_r04_test_all.score")
    
