#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:42:57 2019

@author: xiyiyan
"""


########################
#####Step1: Run KRR#####
########################

def krr_postprocessing(mat_q,data,data_train):
        
    #rating_mat:610*9724 original data
    #mat_q:10*9724 predicted q matrix
    #data: rating_subset
    #data_train:data_train_subset
           
    import numpy as np
    import pandas as pd
    
    from sklearn import preprocessing
    from sklearn.kernel_ridge import KernelRidge  
    
        
    n_movies=np.unique(data.movieId).shape[0]
    n_users=np.unique(data.userId).shape[0]

    updated_rating_mat=np.zeros((n_users,n_movies))
    
    mat_q=mat_q.T
    
    #normalize q matrix
    q_normalize=preprocessing.normalize(mat_q) #mat_q:
    q_normalize.shape
    q_normalize=pd.DataFrame(q_normalize.T)
    q_normalize.columns=[np.unique(data.movieId)]
    
       
    for i in range(n_users):
        
        rating_i=data_train.loc[data_train['userId']==i+1,['movieId','rating']]
        movieId_i=rating_i.iloc[:,0]
        y_i=rating_i.iloc[:,1]#rating vector of user i
           
        #create X for user i
        X_i=q_normalize.loc[:,movieId_i]
        
        #predictions of krr
        krr = KernelRidge(alpha=0.5,kernel="rbf")
        krr.fit(X_i.T,y_i)

        pred_krr=krr.predict(q_normalize.T)            
        updated_rating_mat[i]=pred_krr          
        
    return(updated_rating_mat)

####################################################
######Step 2: Function for Calculating KRR_RMSE#####
####################################################
def rmse_krr(rating,est_rating):
    
    #rating: rating_subset
    #est_rating: 610*9724(predicted,updated_rating_mat)
    import numpy as np
    import math
    
    def sqr_err(obs):
        sqr_error=(obs[2]-est_rating.iloc[int(obs[0]-1),int(obs[4])])**2
        return(sqr_error)
        
    return(math.sqrt(np.mean(rating.apply(sqr_err,1))))
    
############################
#####Step 3: CV for KRR#####
############################
 
def cv_krr(data,data_train,data_test,k):   
    
    #data:rating_subset
    #data_train:data_train_subset
    #data_test:data_test_subset
    #k:k-fold cross validation
    
    import pandas as pd
    import numpy as np
    from sklearn.utils import shuffle
    
    #initialize train and test data and cv result 
    n_movies=np.unique(data.movieId).shape[0]
    n_users=np.unique(data.userId).shape[0]
    n_col=data_train.shape[1]
    
    cv_result_mat=np.zeros((k,n_users,n_movies))
    n=data_train.shape[0] 
    n_fold=int(n/k)  
    
    data=np.zeros((k,n_fold,n_col))         
    data_train=shuffle(data_train) 
    
    krr_rmse_train=np.zeros(k)
    krr_rmse_test=np.zeros(k)
  
   
    for i in range(k):
        
        data[i] = data_train[i*n_fold:(i+1)*n_fold]
        vali = data[i]
        vali=pd.DataFrame(vali)
        train_new = data_train.drop(vali.index,axis=0)
        krr_result=krr_postprocessing(mat_q2,ratings_subset,data_train_subset) 
        cv_result_mat[i]=krr_result
        
        #calculate rmse for train and test   
        krr_rmse_train[i]=rmse_krr(train_new,pd.DataFrame(krr_result))
        krr_rmse_test[i]=rmse_krr(vali,pd.DataFrame(krr_result))
        
    #get the predictions with smallest test error
    idx_min_rmse=krr_rmse_test.argmin()
        
    return(krr_rmse_test,cv_result_mat[idx_min_rmse])
 




