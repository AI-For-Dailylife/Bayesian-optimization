# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:13:05 2020

@author: 81804
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_predict,GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn import metrics
#from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

ds = pd.read_csv("experimental_data.csv",index_col=0)
search_space_x = pd.read_csv("search_space_x.csv",index_col=0)
min_or_max = pd.read_csv("maximize_minimize.csv",index_col=0)

x = ds.iloc[:,1:]
y = ds.iloc[:,:1]

if min_or_max.values == 0:
    y = y*-1
else:
     pass
    
as_x = (x - x.mean())/x.std()
as_y = (y - y.mean())/y.std()
as_search_space_x = (search_space_x - x.mean(axis=0))/x.std(axis=0)


n_features = x.shape[1]

kernel = {1:ConstantKernel() * DotProduct() + WhiteKernel(),
          2:ConstantKernel() * RBF() + WhiteKernel(),
          3:ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
          4:ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel(),
          5:ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel() + ConstantKernel() * DotProduct(),
          6:ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
          7:ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
          8:ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
          9:ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
          10:ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
          11:ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
          12:RBF()
                     }
kernel_nums = len(kernel)


alphas = np.arange(0,1.1,0.1)
r2_list=[]

r2_best =-1000
for kernel_num in range(1,kernel_nums+1):
    for alpha_num in range(len(alphas)):
        gpr = GaussianProcessRegressor(kernel=kernel[kernel_num],\
                                       alpha=alphas[alpha_num])
        gpr.fit(as_x,as_y)
        as_cv_est_y = pd.DataFrame(cross_val_predict(gpr,as_x,as_y,cv=ds.shape[0])
                                   ,columns=y.columns)
        cv_est_y = (as_cv_est_y * y.std())+y.mean()
        r2_cv_y = metrics.r2_score(y,cv_est_y)
        r2_list.append(r2_cv_y)
        if r2_best<r2_cv_y:
            r2_best = r2_cv_y
            opt_kernel_key = kernel_num
            opt_alpha = alphas[alpha_num]

opt_gpr = GaussianProcessRegressor(kernel=kernel[opt_kernel_key]\
                                   ,alpha=opt_alpha)
opt_gpr.fit(as_x,as_y)

mu,sigma = opt_gpr.predict(as_search_space_x,return_std=True)

mu = (pd.DataFrame(mu,columns=y.columns)*y.std())+y.mean()
sigma = pd.DataFrame(sigma,columns=y.columns)*y.std()
        

epsilon =0.01
current_max_y = y.max()
pi = 1-norm.cdf(current_max_y.values+y.std()*epsilon,loc=mu,scale=sigma)
pi_max_index = pi.argmax()
pi_max_sample = search_space_x.iloc[pi_max_index,:]
print(pi_max_sample)


ei_term1 = pi*(mu-(current_max_y.values+y.std()*epsilon))
ei_term2 = (sigma**2)*norm.pdf(current_max_y.values+y.std()*epsilon
                               ,loc=mu
                               ,scale=sigma)
ei=ei_term1+ei_term2

ei_max_index = ei.idxmax()
ei_max_sample = search_space_x.iloc[ei_max_index,:]

ei_max_sample.to_csv("next_experiment_sample.csv")
print(ei_max_sample)

