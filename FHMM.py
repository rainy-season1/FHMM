# -*- coding: utf-8 -*-
import numpy as np
from hmmlearn import hmm
from csv_handle import *
import pickle
import itertools
import math
import matplotlib.pyplot as plt

from HMM import *
import time

	
def con(o_value):
    cons = ({'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]-o_value})
    return cons
	
def compute_A_fhmm(list_A):
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result

def compute_pi_fhmm(list_pi):
	result = list_pi[0]
	for i in range(len(list_pi) - 1):
		result = np.kron(result, list_pi[i + 1])
	return result
	
def compute_means_fhmm(list_means):
	
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    #构造二维高斯分布
    means_stacked=means_stacked[:,:2]
    means = np.reshape(means_stacked, (num_combinations, 2))
    return means
 
def compute_covars_fhmm(list_covars):
    covars_copy=deepcopy(list_covars)
    for i in range(len(covars_copy)):
        shape1=covars_copy[i].shape[0]#状态数
        shape2=covars_copy[i].shape[1]#特征数
        covars_copy[i]=covars_copy[i].flatten()#转一维矩阵
        covars_copy[i]=covars_copy[i][covars_copy[i]!=0]#去0处理
        covars_copy[i]=covars_copy[i].reshape((shape1,shape2))
        covars_copy[i]=covars_copy[i].tolist()#转列表
    states_combination = list(itertools.product(*covars_copy))
    num_combinations = len(states_combination)
    covars_stacked = np.array([np.array(x[0])+np.array(x[1]) for x in states_combination])
    #构造二维高斯分布
    covars_stacked=covars_stacked[:,:2]
    return covars_stacked

def decode(length_sequence,appliances_lists):
	num=1
	decode_states=np.zeros((len(appliances_lists),len(length_sequence)),dtype=np.int)
	for j in range(len(length_sequence)):
		factor=length_sequence[j]
		for i in range(len(appliances_lists)):
			if i!=(len(appliances_lists)-1):
				for n in appliances_lists[i+1:]:
					num*=len(n.startprob_)
				temp=factor//num
				factor=factor%num
				num=1
			else:
				temp=factor
			decode_states[i][j]=int(temp)
	return decode_states



class FHMM():
    def __init__(self,HMM_lists):
        self.individual=HMM_lists
    def train(self):
        pi_lists=[self.individual[i].startprob_ for i in range(len(self.individual))]
        means_lists=[self.individual[i].means_ for i in range(len(self.individual))]
        covars_lists=[self.individual[i].covars_ for i in range(len(self.individual))]
        A_lists=[self.individual[i].transmat_ for i in range(len(self.individual))]
        FHMM_startprob=(compute_pi_fhmm(pi_lists))
        FHMM_transmat=(compute_A_fhmm(A_lists))
        FHMM_means=(compute_means_fhmm(means_lists))
        FHMM_covars=(compute_covars_fhmm(covars_lists))
        FHMM = hmm.GaussianHMM(n_components=len(FHMM_startprob), covariance_type='diag')
        FHMM.startprob_=FHMM_startprob
        FHMM.means_=FHMM_means
        #FHMM.n_features=2
        FHMM.covars_=FHMM_covars
        FHMM.transmat_=FHMM_transmat
        self.model=FHMM
        


