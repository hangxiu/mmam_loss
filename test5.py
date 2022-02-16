import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm
import numpy as np
import scipy.io as io
import numpy as np
from tuneThreshold import *
from matplotlib.pyplot import MultipleLocator

if __name__ == "__main__":

    y_sub = [2.7346,2.4181,2.0201,2.6575,2.83,2.95]
    y_soft = [2.0343,2.3366,2.264,3.2289,3.0289,3.5289]
    y_mmam = [1.9762,1.4466,1.617,1.8049,1.823,1.819]
    x = ['2', '3', '4', '5', '6', '7']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.scatter(x,y, s=0.01)
    #marker='+'
    mark = ['.','o','x','8','s','v','D','+','*','p','^', '<','>','1','2','3','4','h','H']
    line_width = 1.5
    marker_size = 3
    ax.tick_params(axis='both',which='major',labelsize=14)
    ax.plot(x, y_sub, linewidth =line_width, markersize=marker_size, color='b',marker=mark[1], label='Sub-Center')
    ax.plot(x, y_soft, linewidth =line_width, markersize=marker_size, color='r', label='SoftTriple', marker=mark[2], alpha=0.5)
    ax.plot(x, y_soft, linewidth =line_width, markersize=marker_size, color='g', label='MMAM r=1.0', marker=mark[7], alpha=0.5)
    plt.xlim(xmin=1.0,xmax=4)
    plt.ylim([1.0,4.0])
    plt.xticks(x,rotation=40)
    ax.legend(loc='upper right')
    plt.savefig('test_all_eer_multi_center.png')
    plt.show()
    
    
