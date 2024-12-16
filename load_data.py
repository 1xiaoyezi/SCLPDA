import numpy as np
import scipy.sparse as sp
import sys
import torch as t

from features_proj import features_proj
from utils import Sizes, constructNet
import scipy.io as sio
import csv
def process(feature,hid_dim,out_dim):#hid_dim=
    model = features_proj(hid_dim,out_dim)
    return model(feature)

def normalize_features(feat):
    #求度矩阵，对所有行进行求和，设有n行，则得到n维度向量
    degree = np.asarray(feat.sum(1)).flatten() 

    # set zeros to inf to avoid dividing by zero
    #防止除0
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    #将度向量化成对角阵
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm
def preprocess_adj(adj):
    # adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    adj_normalized = adj +np.eye(adj.shape[0])

    return adj_normalized
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader] 
        return t.Tensor(md_data)
def load_data(args):
    attributes_list=[]
    for attribute in ['features','similarity']: #args.attributes
        if attribute =='features':
            F1 = read_csv('C:\\Users\\lg\\Desktop\\MSGCL-main\\p_p_f.csv')
            F2 = read_csv('C:\\Users\\lg\\Desktop\\MSGCL-main\\d_d_f.csv')
            feature = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                                  np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))

            attributes_list.append(feature) #1546*1546
        elif attribute =='similarity':
            F3 = read_csv('C:\\Users\\lg\\Desktop\\MSGCL-main\\p_p_s.csv')
            F4 = read_csv('C:\\Users\\lg\\Desktop\\MSGCL-main\\d_d_s.csv')
            similarity = np.vstack((np.hstack((F3, np.zeros(shape=(F3.shape[0], F4.shape[1]), dtype=int))),
                                     np.hstack((np.zeros(shape=(F4.shape[0], F3.shape[0]), dtype=int), F4))))
            attributes_list.append(similarity)
    '''miRNA-disease'''
    pir_dis_matrix=read_csv('C:\\Users\\lg\\Desktop\\MSGCL-main\\p_d.csv')#853*591
    adj_original = np.vstack((np.hstack((np.zeros(shape=(462,462),dtype=int), pir_dis_matrix)),np.hstack((pir_dis_matrix.T,np.zeros(shape=(102,102),dtype=int)))))
    pir_dis_matrix=t.FloatTensor(adj_original)

    features = np.hstack(attributes_list)

    features=features.astype(float) 
    features =t.FloatTensor(features)
    print(" features",features.shape)

    pir_sim = t.FloatTensor(F3)
    pir_fea = t.FloatTensor(F1)
    dis_sim = t.FloatTensor(F4)
    dis_fea = t.FloatTensor(F2)

    nfeats = features.shape[1]
    return features,nfeats,pir_dis_matrix,adj_original,pir_sim,pir_fea,dis_sim,dis_fea

