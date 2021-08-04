# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:50:03 2021

@author: 13412
"""

from HMM import *
from scipy.signal import medfilt
import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler

def inverse_transform_col(scaler,y,n_col):
    """将指定的特征反归一化
    
    参数
    ----------
    scaler:  MinMaxScaler实例
    y:用于反归一化的数据
    n_col：特征个数
    
    返回
    -------
    反归一化后的(n_col+1)列特征数据
    """

    y = y.copy()
    y -= scaler.min_[:n_col]
    y /= scaler.scale_[:n_col]
    return y

def transform_col(scaler,y,n_col):
    """将指定的特征归一化
    
    参数
    ----------
    scaler:  MinMaxScaler实例
    y:用于归一化的数据
    n_col：特征个数
    
    返回
    -------
    归一化后的(n_col+1)列特征数据
    """

    y = y.copy()
    y -= scaler.data_min_[:n_col]
    y *= scaler.scale_[:n_col]
    return y


def create_hmmlist(data_set,appliances_name,physical_quantity,cluster_list,train_min,train_max):
    """创建HMM模型列表及归一化参数列表
    
    参数
    ----------
    data_set: DataFrame格式数据集
    appliances_name:用电器名称列表
    [B1E,B2E,BME,CDE,CWE,DNE,DWE,EBE,EQE,FGE,FRE,GRE,HPE,HTE,OFE,OUE,TVE,UTE,WOE]
    physical_quantity: 用来训练HMM的电气物理量
    ['V','I','f','DPF','APF','P','Pt','Q','Qt','S','St']
    cluster_list:用电器的状态个数列表
    train_min：训练集的初始索引
    train_max：训练集的终止索引
    
    返回
    -------
    hmm_list:各用电器模型列表
    scaler_list：归一化参数列表
    """
    hmm_list=[]
    scaler_list=[]
    n_cluster=len(physical_quantity)
    for i in range(len(appliances_name)):
        name=[]
        name.append(appliances_name[i])
        appliances_quantity=[x for x in itertools.product(name,physical_quantity)]
        appliances_data=np.array(data_set[appliances_quantity][train_min:train_max])
        '''数值滤波'''
        for j in range(n_cluster):
            appliances_data[:,j]=medfilt(appliances_data[:,j],3)
        '''数据归一化'''
        scaler = MinMaxScaler(feature_range=(0, 1))
        appliances_trans_data=scaler.fit_transform(appliances_data)
        appliances_HMM=create_hmm(appliances_trans_data,cluster_list[i],n_cluster)
        hmm_list.append(appliances_HMM)
        scaler_list.append(scaler)
    return hmm_list,scaler_list

def fhmm_test_data(data_set,appliances_name,physical_quantity,scaler_list,train_min,train_max):
    data=[]
    original_data=[]
    n_cluster=len(physical_quantity)
    for i in range(len(appliances_name)):
        name=[]
        name.append(appliances_name[i])
        appliances_quantity=[x for x in itertools.product(name,physical_quantity)]
        appliances_original_data=np.array(data_set[appliances_quantity][train_min:train_max])
        appliances_data=np.zeros_like(appliances_original_data,float)
        '''数值滤波'''
        for j in range(n_cluster):
            appliances_data[:,j]=medfilt(appliances_original_data[:,j],3)
        '''数据归一化'''
        appliances_trans_data=transform_col(scaler_list[i],appliances_data,n_cluster)
        data.append(appliances_trans_data)
        original_data.append(appliances_original_data)
    return sum(data),original_data

def pre_inverse(pre_data,scaler_list):
    shape=np.shape(pre_data)
    pre_inverse_data=np.zeros_like(pre_data,float)
    for i in range(shape[1]):
        pre_inverse_data[:,i]=inverse_transform_col(scaler_list[i],pre_data[:,i],1)
    return pre_inverse_data