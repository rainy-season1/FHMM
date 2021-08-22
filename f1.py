# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 09:27:39 2021

@author: 13412
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score

def energy_f1score(app_gt, app_pred):
    temp=np.hstack((app_gt,app_pred))
    molecule=temp.min(axis=1).sum()
    denominator=app_pred.sum()
    P=molecule/denominator
    denominator= app_gt.sum()
    R=molecule/denominator
    return (2*(P*R)/(P+R))
        
