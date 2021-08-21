# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
from hmmlearn import hmm
from csv_handle import *
from scipy.signal import medfilt


def return_sorting_mapping(means):
	means_copy = deepcopy(means)
	means_copy=means_copy.T[0].reshape(-1,1)
	mean=deepcopy(means_copy)
	means_copy = np.sort(means_copy, axis=0)
	mapping = {}
	for i, val in enumerate(means_copy):
		mapping[i] = np.where(val == mean)[0][0]
	return mapping
    
def sort_startprob(mapping, startprob):

	num_elements = len(startprob)
	new_startprob = np.zeros(num_elements)
	for i in range(len(startprob)):
		new_startprob[i] = startprob[mapping[i]]
	return new_startprob

def sort_means(mapping, means):
    new_means = np.zeros_like(means)
    for i in range(len(means)):
        new_means[i] = means[mapping[i]]
    return new_means

def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
            new_covars[i] = covars[mapping[i]]
    shape1=new_covars.shape[0]
    shape2=new_covars.shape[1]
    new_covars=new_covars.flatten()
    new_covars=new_covars[new_covars!=0]
    new_covars=new_covars.reshape((shape1,shape2))
    return new_covars


def sort_transition_matrix(mapping, A):

    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new

def create_hmm(train_data,clusters,features):


	model = hmm.GaussianHMM(n_components=clusters,covariance_type='diag')
	model.fit(train_data)

	mapping=return_sorting_mapping(model.means_)
	new_startprob=sort_startprob(mapping,model.startprob_)
	new_means = sort_means(mapping,model.means_)
	new_covars=sort_covars(mapping,model.covars_)
	#new_covars=deepcopy(model.means_)
	new_transmat=sort_transition_matrix(mapping,model.transmat_)
	new_model = hmm.GaussianHMM(n_components=clusters,covariance_type='diag')
	new_model.startprob_=new_startprob
	new_model.means_=new_means
	new_model.n_features=features#特征数修改
	new_model.covars_=new_covars#参数不带0
	new_model.transmat_=new_transmat
	return new_model


if __name__=='__main__':
    data_set=pd.read_pickle('E:/NILM/pkl/WHOLE.pkl')#数据集读取
    train_data=np.array(data_set[[('WOE','P'),('WOE','Q')]][0:20000])
    for j in range(2):
        train_data[:,j]=medfilt(train_data[:,j],3)
    WOE=  create_hmm(train_data,3,2)
    print(WOE.means_)      
        
        






