# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:40:52 2021

@author: 13412
"""


from FHMM import *
from sklearn.metrics import mean_absolute_error
from nilmtk.losses import nde
from f1 import energy_f1score
import scipy.signal as signal
from data_process import *
from grandient_disaggregate import *

def draw(dic,appliances_name,pre_data,gt_data):
    n=np.shape(pre_data)[1]
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.plot(pre_data[:,i],label='Predict_data')
        plt.plot(gt_data[i][:,:1],label='True_data')
        plt.legend(frameon=False)
        print(dic[appliances_name[i]]+':')
        pre_temp=np.copy(pre_data[:,i]).reshape(-1,1)
        print('MAE:',round(mean_absolute_error(pre_temp,gt_data[i][:,:1]),3))
        print('F1_score',round(energy_f1score(gt_data[i][:,:1],pre_temp),3))
        print('NDE',round(nde(gt_data[i][:,:1],pre_temp),3))
        #print('F1_score',f1score(gt_data[i][:,:1],pre_data[:,i]))
    plt.show()
        
def fraction(appliances_name,pre_data,gt_data):
    labels=appliances_name
    
    
appliance_dic={'DWE':'Dishwasher','FGE':'Fridge','HPE':'Heat Pump',
     'CDE':'Dryer','CWE':'Clothes Washer','WOE':'Wall Oven'}

'''模型训练-参数设置'''
data_set=pd.read_pickle('E:/NILM/pkl/WHOLE.pkl')#数据集读取
appliances_name=['DWE','FGE','HPE','CDE','CWE','WOE']#电器名称
physical_quantity=['P','Q','S']#所用电器物理量
#physical_quantity=['P','Q','I','Pt','S']#所用电器物理量
cluster_list=[2,2,6,3,4,3]#各电器状态个数

'''模型训练代码'''
# train_min=0
# train_max=20000
# appliances_list,scaler_list=create_hmmlist(data_set,appliances_name,physical_quantity,cluster_list,train_min,train_max)
# fhmm=FHMM(appliances_list)
# fhmm.train()
# pickle.dump(fhmm,open('n_fhmm','wb'))
# pickle.dump(scaler_list,open('scaler_list','wb'))
# print('训练完毕')

'''模型测试'''
scaler_list=pickle.load(open('scaler_list','rb'))#归一化参数导入
fhmm=pickle.load(open('n_fhmm','rb'))#模型导入
#测试数据导入归一化
all_transform_data,gt_data=fhmm_test_data(data_set,appliances_name,['P','Q'],
                                  scaler_list,train_min='2012-04-15',train_max='2012-04-21')
#状态预测
sequence=fhmm.model.predict(all_transform_data)
hmm_sequence=decode(sequence,fhmm.individual)
#功率分解
mu_values=np.zeros_like(hmm_sequence,dtype=float)
siga_values=np.zeros_like(hmm_sequence,dtype=float)
print('ok')
for i in range(len(fhmm.individual)):
 	for j in range(len(sequence)):
         mu_values[i][j]=fhmm.individual[i].means_[hmm_sequence[i][j]][0]#取有功功率对应高斯均值
         siga_values[i][j]=fhmm.individual[i].covars_[hmm_sequence[i][j]][0][0]#取有功功率对应高斯方差
pre_power=power_decompose(mu_values,siga_values,hmm_sequence.T,all_transform_data)
#数值反归一化
#pre_data=pre_inverse(pre_power,scaler_list)
np.save('E:/NILM/npy/PQS.npy',pre_power)
draw(appliance_dic,appliances_name,pre_power,gt_data)


