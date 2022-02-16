# coding='utf-8'
"""t-SNE"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as io

def get_data(path):
    data=np.load(path)
    X=data['vector']
    labels=data['utt']
    ID_list=[]
    for label in labels:
        ID_list.append(label.split('-')[0])
    x_dic={}
    for i in ID_list:
        x_dic[i]=[]
    for i in range(len(ID_list)):
        if ID_list[i] in x_dic:
           x_dic[ID_list[i]].append(X[i])
    x_choose={}
    for i in x_dic:
        if len(x_dic[i])>=250:
            x_choose[i]=x_dic[i]
    print('select speakers with utts large than 250')  
    x_vector=[]
    label_vector=[]
    n=1
    for i in x_choose:
        for j in x_choose[i]:
            x_vector.append(j)
            label_vector.append(n)
        n+=1
    n_samples, n_dim =len(x_vector), x_vector[0].shape
    n_labels=len(set(label_vector))
    print('n_samples={}, n_dim={}, n_labels={}'.format(n_samples,n_dim,n_labels))
    return x_vector, label_vector

def get_data2(path):
    print('Loading data')
    data = io.loadmat(path)
    ori_id = data['ori_id']
    dev_id = data['utt_id']
    dev_vector = data['embedding']
    dev_vector = np.squeeze(dev_vector)
    dev_id = np.squeeze(dev_id)
    ori_id = np.squeeze(ori_id)
    print(ori_id)
    ori_id = list(set(ori_id))
    ID_list= set(dev_id)
    print(ID_list)
    x_dic={}
    for i in ID_list:
        x_dic[i]=[]
    for i in range(len(dev_id)):
        if dev_id[i] in x_dic:
           x_dic[dev_id[i]].append(dev_vector[i])
    x_choose={}
    for i in x_dic:
        if len(x_dic[i])>=250:
            x_choose[i]=x_dic[i]
    print('select speakers with utts large than 250')  
    x_vector=[]
    label_vector=[]
    n=1
    for i in x_choose:
        tem_label = []
        for j in x_choose[i]:
            x_vector.append(j)
            tem_label.append(n)
        label_vector.append(tem_label)
        n+=1
        if n >= 21:
            break
    n_samples, n_dim =len(x_vector), x_vector[0].shape
    n_labels = len(label_vector)
    print('n_samples={}, n_dim={}, n_labels={}'.format(n_samples,n_dim,n_labels))
    return x_vector, label_vector, ori_id

def plot_embedding(data, label, ori_id):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    mark = ['.','o','x','8','s','v','D','+','*','p']
    list_legend = ['ja-jp','Uyghu','ru-ru', 'Tibet', 'vi-vn','ko-kr', 'ct-cn','zh-cn', 'id-id', 'Kazak']
    # list_legend = ['zh-cn', 'ct-cn', 'ja-jp', 'ru-ru', 'ko-kr']
    #list_legend = ori_id
    #list_legend = ['zh-cn','id-id','vi-vn','Tibet','ct-cn','Uyghu','ja-jp','ru-ru','Kazak','ko-kr']
    # 1s list_legend = ['zh-cn','id-id','vi-vn','Tibet','ct-cn','Uyghu','ja-jp','ru-ru','Kazak','ko-kr']
    color = ['b','c','g','k','m','r','y','pink','peru','gold']
    #plt.scatter(data[:, 0], data[:, 1], 10, c=label, marker=label, cmap=plt.cm.Spectral, alpha=0.5)
    i = 0 
    ax.scatter(data[i:i+len(label[0]), 0],data[i:i+len(label[0]), 1], s=1,c=color[0], marker=mark[0], alpha=0.5, label=list_legend[0])
    i += len(label[0])
    ax.scatter(data[i:i+len(label[1]), 0],data[i:i+len(label[1]), 1], s=1,c=color[1], marker=mark[1], alpha=0.5, label=list_legend[1])
    i += len(label[1])
    ax.scatter(data[i:i+len(label[2]), 0],data[i:i+len(label[2]), 1], s=1,c=color[2], marker=mark[2], alpha=0.5, label=list_legend[2])
    i += len(label[2])
    ax.scatter(data[i:i+len(label[3]), 0],data[i:i+len(label[3]), 1],s=1, c=color[3], marker=mark[3], alpha=0.5, label=list_legend[3])
    i += len(label[3])
    ax.scatter(data[i:i+len(label[4]), 0], data[i:i+len(label[4]), 1],s=1, c=color[4], marker=mark[4], alpha=0.5, label=list_legend[4])
    i += len(label[4])
    ax.scatter(data[i:i+len(label[5]), 0], data[i:i+len(label[5]), 1],s=1, c=color[5], marker=mark[5], alpha=0.5, label=list_legend[5])
    i += len(label[5])
    ax.scatter(data[i:i+len(label[6]), 0],data[i:i+len(label[6]), 1],s=1, c=color[6], marker=mark[6], alpha=0.5, label=list_legend[6])
    i += len(label[6])
    ax.scatter(data[i:i+len(label[7]), 0], data[i:i+len(label[7]), 1],s=1, c=color[7], marker=mark[7], alpha=0.5, label=list_legend[7])
    i += len(label[7])
    ax.scatter(data[i:i+len(label[8]), 0], data[i:i+len(label[8]), 1],s=1, c=color[8], marker=mark[8], alpha=0.5, label=list_legend[8])
    i += len(label[8])
    ax.scatter(data[i:, 0], data[i:, 1], c=color[9], s=1,marker=mark[9], alpha=0.5, label=list_legend[9])
    plt.legend(loc='upper right')
    return fig

def main(path0, epoch, m):
    data, labels_color, ori_id = get_data2(path0)
    print('Computing t-SNE embedding epoch')
    n_labels = len(labels_color)
    tsne = TSNE(n_components = 2, init='pca', random_state = 0)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, labels_color,ori_id)
    if not os.path.exists('./tsne'):
        os.mkdir('./tsne' );
    plt.savefig('./tsne/ap17_test_all_SEresnet_L_v4.mat.png')
    plt.close()

if __name__ == '__main__':
    main('./ap17_test_all_SEresnet_L_v4.mat', 1, 0.2)

  
