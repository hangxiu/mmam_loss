import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm
import numpy as np
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

def draw_DET(peo, datum):
    up = np.max(datum)
    down = np.min(datum)
    pos_num = len(datum)//2

    x = []
    y = []
    dot_num = 1000  # DET
    step = (up - down) / (dot_num + 1)
    threshod = up
    size = len(datum)
    for i in range(dot_num):
        threshod -= step
        false_neg = 0
        false_pos = 0
        for d in range(size):
            if d < pos_num and datum[d] < threshod:
                false_pos += 1
            elif d > pos_num and datum[d] > threshod:
                false_neg += 1
        x.append(false_pos / size)
        y.append(false_neg / size)
    return x, y
def plot_DET_curve():
    # 
    pmiss_min = 0.01
    pmiss_max = 0.2
    pfa_min = 0.01

    pfa_max = 0.2
 
    # 
    pticks = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
            0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
            0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999]
 
    # 
    xlabels = [' 0.001', ' 0.002', ' 0.005', ' 0.01 ', ' 0.02 ', ' 0.05 ',
            '  0.1 ', '  0.2 ', ' 0.5  ', '  1   ', '  2   ', '  5   ',
            '  10  ', '  20  ', '  40  ', '  60  ', '  80  ', '  90  ',
            '  95  ', '  98  ', '  99  ', ' 99.5 ', ' 99.8 ', ' 99.9 ',
            ' 99.95', ' 99.98', ' 99.99', '99.995', '99.998', '99.999']
 
    ylabels = xlabels
 
    # 
    n = len(pticks)
    # 
    for k, v in enumerate(pticks[::-1]):
        if pmiss_min <= v:
            tmin_miss = n - k - 1   #
        if pfa_min <= v:
            tmin_fa = n - k - 1   # 
    # 
    for k, v in enumerate(pticks):
        if pmiss_max >= v:   
            tmax_miss = k+1         # 
        if pfa_max >= v:            
            tmax_fa = k+1            # 
 
    # FRR
    plt.figure()
    plt.xlim(norm.ppf(pfa_min), norm.ppf(pfa_max))
 
    plt.xticks(norm.ppf(pticks[tmin_fa:tmax_fa]), xlabels[tmin_fa:tmax_fa])
    plt.xlabel('False Alarm probability (in %)')
 
    # FAR
    plt.ylim(norm.ppf(pmiss_min), norm.ppf(pmiss_max))
    plt.yticks(norm.ppf(pticks[tmin_miss:tmax_miss]), ylabels[tmin_miss:tmax_miss])
    plt.ylabel('Miss probability (in %)')

    return plt
  
if __name__ == "__main__":
    
    sc_resnet_L, lab_resnet_L, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_xvector.score")
    sc_resnet_W, lab_resnet_W, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_resnet_W.score")
    sc_SEresnet_L, lab_SEresnet_L, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_nla_char_12.score")
    sc_SEresnet_W, lab_SEresnet_W, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_nla_char_23.score")
    # sc_amc, lab_amc, _ = evaluateFromList("ap17_test_all.trials", "ap17_amc_enroll_all_test_1s.score")

    sc_add, lab_tfc_add, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_tfc_all_add_v2.score")
    sc_avg, lab_tfc_avg, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_tfc_1234_max_v2.score")

    sc_max, lab_tfc_max, _ = evaluateFromList("ap17_test_1s.trials", "ap17_test_1s_mnla_tfc_1234_avg_v2.score")

    # sc_resnet_L, lab_resnet_L, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_xvector.score")
    # sc_resnet_W, lab_resnet_W, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_resnet_W.score")
    # sc_SEresnet_L, lab_SEresnet_L, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_resnet_w.score")
    # sc_SEresnet_W, lab_SEresnet_W, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_resnet_L.score")
    # # sc_amc, lab_amc, _ = evaluateFromList("ap17_test_all.trials", "ap17_amc_enroll_all_test_1s.score")

    # sc_add, lab_tfc_add, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_tfc_1234_add_v2.score")
    # sc_avg, lab_tfc_avg, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_tfc_1234_max_v2.score")

    # sc_max, lab_tfc_max, _ = evaluateFromList("ap17_test_3s.trials", "ap17_test_3s_tfc_1234_avg_v2_v2.score")

    # sc_resnet_L, lab_resnet_L, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_resnet_SE_L.score")
    # sc_resnet_W, lab_resnet_W, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_resnetse_W.score")
    # sc_SEresnet_L, lab_SEresnet_L, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_resnet_L.score")
    # sc_SEresnet_W, lab_SEresnet_W, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_resnet_W.score")
    # # sc_amc, lab_amc, _ = evaluateFromList("ap17_test_all.trials", "ap17_amc_enroll_all_test_1s.score")

    # sc_add, lab_tfc_add, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_tfc_1234_add_v2.score")
    # sc_avg, lab_tfc_avg, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_tfc_1234_avg_v2_v2.score")

    # sc_max, lab_tfc_max, _ = evaluateFromList("ap20_task3.trials", "ap20_task3_tfc_1234_max_v2_v2.score")

    #print(sc)
    #print(lab)
    # result_softmax = tuneThresholdfromScore(sc_softmax, lab_softmax, [1, 0.5]);
    # result_aam = tuneThresholdfromScore(sc_aam, lab_aam, [1, 0.5]);
    # result_Dam = tuneThresholdfromScore(sc_Dam, lab_Dam, [1, 0.5]);
    # result_ge2e = tuneThresholdfromScore(sc_ge2e, lab_ge2e, [1, 0.5]);
    # result_amc = tuneThresholdfromScore(sc_amc, lab_amc, [1, 0.5]);

    # result_sub = tuneThresholdfromScore(sc_sub, lab_sub, [1, 0.5]);
    # result_soft = tuneThresholdfromScore(sc_soft, lab_soft, [1, 0.5]);

    # result_mmam = tuneThresholdfromScore(sc_mmam, lab_mmam, [1, 0.5]);

    #print('EER %2.4f'%result[1])
   # print('EER %2.4f'%result1[1])
    p_target = 0.5
    c_miss = 1
    c_fa = 1

    fnrs_softmax, fprs_softmax, thresholds = ComputeErrorRates(sc_resnet_L, lab_resnet_L)
    fnrs_aam, fprs_aam, _ = ComputeErrorRates(sc_resnet_W, lab_resnet_W)
    fnrs_Dam, fprs_Dam, _ = ComputeErrorRates(sc_SEresnet_L, lab_SEresnet_L)
    fnrs_ge2e, fprs_ge2e, _ = ComputeErrorRates(sc_SEresnet_W, lab_SEresnet_W)
    # fnrs_amc, fprs_amc, _ = ComputeErrorRates(sc_amc, lab_amc)

    fnrs_sub, fprs_sub, _ = ComputeErrorRates(sc_add, lab_tfc_add)
    fnrs_soft, fprs_soft, _ = ComputeErrorRates(sc_avg, lab_tfc_avg)
    fnrs_mmam, fprs_mmam, _ = ComputeErrorRates(sc_max, lab_tfc_max)
    # 
    plt = plot_DET_curve()
    plt.grid(True)
    x_softmax, y_softmax = norm.ppf(fprs_softmax), norm.ppf(fnrs_softmax)
    x_aam, y_aam = norm.ppf(fprs_aam), norm.ppf(fnrs_aam)
    x_Dam, y_Dam = norm.ppf(fprs_Dam), norm.ppf(fnrs_Dam)
    x_ge2e, y_ge2e = norm.ppf(fprs_ge2e), norm.ppf(fnrs_ge2e)
    # x_amc, y_amc = norm.ppf(fprs_amc), norm.ppf(fnrs_amc)
    
    x_sub, y_sub = norm.ppf(fprs_sub), norm.ppf(fnrs_sub)
    x_soft, y_soft = norm.ppf(fprs_soft), norm.ppf(fnrs_soft)
    x_mmam, y_mmam = norm.ppf(fprs_mmam), norm.ppf(fnrs_mmam)
    xx, x11 = [1.5283,2,3], [1.4466]
    line_width = 1.5
    marker_size=3
    mark = ['.','o','x','8','s','v','D','+','*','p','^', '<','>','1','2','3','4','h','H']
    plt.plot(x_softmax, y_softmax, linewidth =line_width, markersize=marker_size,  color='r',linestyle='-', label='Resnet-34')
    plt.plot(x_aam, y_aam, linewidth =line_width, markersize=marker_size, color='b',linestyle='--', label='w-Resnet-34')
    plt.plot(x_Dam, y_Dam, linewidth =line_width, markersize=marker_size, color='c',linestyle='-', label='SE-Resnet-34')
    plt.plot(x_ge2e, y_ge2e, linewidth =line_width, markersize=marker_size, color='k',linestyle='--', label='w-SE-Resnet-34')
    # plt.plot(x_amc, y_amc, linewidth =line_width, markersize=marker_size, color='m',linestyle='-', label='AM-Centroid')
    plt.plot(x_sub, y_sub, linewidth =line_width, markersize=marker_size, color='y',linestyle='--', label='w-MDA-add')    
    plt.plot(x_soft, y_soft, linewidth =line_width, markersize=marker_size, color='gold',linestyle='-', label='w-MDA-avg')
    plt.plot(x_mmam, y_mmam, linewidth =line_width, markersize=marker_size, color='g',linestyle='--', label='w-MDA-max')
    plt.plot([-40, 1], [-40, 1])
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('ap17_test_1s_DET.png')
