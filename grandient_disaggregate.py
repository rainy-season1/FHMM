# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:53:36 2021

@author: 13412
"""
import numpy as np
from scipy.optimize import minimize

def gaussian(x,mu,sig):
	norm = 1/np.sqrt(2*np.pi*sig)
	return norm * np.exp(-np.power(x - mu, 2.) / (2*sig))


def fun(x,*arg):
    n=int(len(arg)/2)
    f=-1.0
    for i in range(n):
        f*=gaussian(x[i],arg[2*i],arg[2*i+1])
    return f


def con(o_value):
    cons = ({'type': 'eq', 'fun': lambda x: sum(x)-o_value})
    return cons

def power_decompose(mu_values,siga_values,hmm_sequence,all_values):
    shape=(np.shape(mu_values)[1],np.shape(mu_values)[0])#矩阵维度，功率观测值序列X用电器个数
    power_values=np.zeros(shape,dtype=float)#预测功率矩阵    
    #各时刻电器对应高斯分布均值及方差
    mu_states=mu_values.T
    siga_states=siga_values.T
    for i in range(shape[0]):
        o_value=all_values[i][0]
        cons=con(o_value)
        t_power=np.zeros(shape[1])
        #目标函数各高斯参数
        arg_mu=[]
        arg_siga=[]
        flag=[]#判断电器是否处于开启状态
        for j in range(shape[1]):
            if hmm_sequence[i][j]!=0:
                flag.append(True)
                arg_mu.append(mu_states[i][j])
                arg_siga.append(siga_states[i][j])
            else:
                flag.append(False)
        '''不全为关闭状态分解'''
        if sum(flag)==0:
            power_values[i]=t_power
        else:
            #目标函数初始值
            x0=np.array(arg_mu)
            #功率取值范围
            bnds=[(0,None) for j in range(sum(flag))]
            bnds=tuple(bnds)
            arg=tuple(arg_mu+arg_siga)
            #功率求解,res.x为返回功率值
            #限制条件，关闭状态不参与分解
            res = minimize(fun, x0,args=arg, method='SLSQP', constraints=cons,bounds=bnds)
            #print(np.shape(res.x))
            num=0
            for m in range(shape[1]):
                if flag[m]:
                    t_power[m]=res.x[num]
                    num+=1
                # else:
                #     t_power[m]=0
            power_values[i]=t_power
        #print(t_power)

    return power_values