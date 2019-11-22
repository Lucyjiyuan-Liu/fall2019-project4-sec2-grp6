#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 00:27:52 2019

@author: liujiyuan
"""

def ALSTD_CV (data_train,K,f,l,iteration):
    ## train set 
    ## K-fold
    ## iteration
    ## lamda
    ## iteration
    
    import pandas as pd
    import numpy as np
    from sklearn.utils import shuffle
    from ALS_TD import ALSTD
    
    #initialize train and test data and cv result 
    n_col=data_train.shape[1]
    
    #cv_result_mat=np.zeros((k,n_users,n_movies))
    n = data_train.shape[0] 
    n_fold=int(n/K)  
    
    data = np.zeros((K,n_fold,n_col))         
    data_train = shuffle(data_train) 
    
    train_rmse = np.zeros((K,iteration))
    test_rmse = np.zeros((K,iteration))
    
    for i in range(K):
        data[i] = data_train[i*n_fold:(i+1)*n_fold]
        vali = data[i]
        vali = pd.DataFrame(vali)
        train_new = data_train.drop(vali.index,axis=0)

        als_result =  ALSTD(f=f,l=l,iteration_max=iteration,train_new,vali)
        
        train_rmse[i,] = als_result.ALSTDfit()[6] 
        test_rmse[i,] = als_result.ALSTDfit()[7]
    
    ## result 
    mean_train_rmse = train_rmse.mean(axis=0)
    mean_test_rmse = test_rmse.mean(axis=0)
    sd_train_rmse = train_rmse.std(axis=0)
    sd_test_rmse = test_rmse.std(axis=0)
    
    return mean_train_rmse, mean_test_rmse, sd_train_rmse, sd_test_rmse
   