import argparse
import numpy as np
from yaml import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from keras.utils.np_utils import to_categorical
from math import exp

def equal_error_rate(y_true, probabilities):
    y_one_hot = to_categorical(y_true, num_classes=probabilities.shape[1])
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), probabilities.ravel())
    eer = brentq(lambda x : 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer

def cal_eer(y_pred, y_true, threshold):
    s = y_pred > threshold
    tp = fp = fn = tn = 0
    for i in range(BATCH_SIZE):
        for j in range(langnum):
            if j == y_true[i]:
                if s[i][j]:
                    tp += 1
                else:
                    fn += 1
            else:
                if s[i][j]:
                    fp += 1
                else:
                    tn += 1
    return 1.*fp / (fp+tn), 1.*fn / (tp+fn)

def CountCavg(data, sn=20, lgn=4):
    Cavg = [0.0] * (sn + 1)
    precision = 1.0 / sn
    for section in range(sn + 1):
        threshold = section * precision
        target_Cavg = [0.0] * lgn
        for language in range(lgn):
            P_FA = [0.0] * lgn
            P_Miss = 0.0
            LTm = 0.0
            LTs = 0.0
            LNm = 0.0
            LNs = [0.0] * lgn
            for line in data:
                language_label = language + 1
                if line[0] == language_label:
                    if line[1] == language_label:
                        LTm += 1
                        if line[2] < threshold:
                            LTs += 1
                    for t in range(lgn):
                        if not t == language:
                            if line[1] == t + 1:
                                if line[2] > threshold:
                                    LNs[t] += 1
            LNm = LTm
            for Ln in range(lgn):
                P_FA[Ln] = LNs[Ln] / LNm
            P_Miss = LTs / LTm
            P_NonTarget = 0.5 / (lgn - 1)
            P_Target = 0.5
            target_Cavg[language] = P_Target * P_Miss + P_NonTarget * sum(P_FA)
        for language in range(lgn):
            Cavg[section] += target_Cavg[language] / lgn
    return Cavg, min(Cavg)

def cal_Cavg(probabilities, langnum):
    datas = []
    for i, item in enumerate(probabilities):
        label = np.argmax(item)
        for j in range(langnum + 1):
            datas.append([label + 1, j + 1 , float(item[j])])               
    for i in range(len(datas) // langnum):      
        s = 0                                   
        for j in range(langnum):                
            k = i * langnum + j                 
            s += exp(datas[k][2])               
        for j in range(langnum):                
            k = i * langnum + j                 
            datas[k][2] = exp(datas[k][2]) / s                             
    Cavg, minCavg = CountCavg(datas, 20, langnum)     
    print('Cavg:{}'.format(minCavg))           

def metrics_report(y_true, y_pred, probabilities, label_names=None):

    available_labels = range(0, len(label_names))
    print("Accuracy %s" % accuracy_score(y_true, y_pred))
    print("Equal Error Rate (avg) %s" % equal_error_rate(y_true, probabilities))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    print(confusion_matrix(y_true, y_pred, labels=available_labels))
    return 'Accuracy '+str(accuracy_score(y_true, y_pred))+'\n'+"EER "+ str(equal_error_rate(y_true, probabilities))+str(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))+'\n'+str(confusion_matrix(y_true, y_pred, labels=available_labels))+'\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--testset', dest='use_test_set', default=False, type=bool)
    cli_args = parser.parse_args()
    evaluate(cli_args)