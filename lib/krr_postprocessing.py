#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:42:57 2019

@author: xiyiyan
"""




import pandas as pd
rating_time=pd.read_csv('rating_time.csv')

def krr_postprocessing(rating,mat_q):
    
    import numpy as np
    from sklearn import preprocessing
    from sklearn.kernel_ridge import KernelRidge  
    
    updated_rating=[]
    
    for i in range(610):
        rating_i=rating_time.loc[rating_time['userId']==i,['movieId','rating']]
        movieId_i=rating_i.iloc[:,0]
        y_i=rating_i.iloc[:,1]#rating vector of user i
           
        mat_q=pd.DataFrame(mat_q)
        mat_q=mat_q.T
        mat_q.columns=[np.unique(rating_time.movieId)]#change column names of q matrix
        
        X_i=mat_q.loc[:,movieId_i]
        X_i=preprocessing.normalize(X_i.T) #normalize latent vector for user i
        
        #predictions of krr
        krr = KernelRidge(alpha=0.5,kernel="rbf")
        krr.fit(X_i,y_i)
        pred_krr=krr.predict(X_i)

        updated_rating.append(pred_krr)
    
    updated_rating=np.array(updated_rating)    
    return(updated_rating)




