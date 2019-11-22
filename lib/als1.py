#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:10:01 2019

@author: liujiyuan
"""

I=9724
f=10
U=610
import numpy as np
import os
import pandas as pd
os.chdir('/Users/xiyiyan/Documents/GitHub/fall2019-project4-sec2-grp6/output')
rating_time=pd.read_csv('rating_time.csv')
rating_time.head()
rating_time.shape

###initialize
q=np.array(np.random.random((f,I)))
q=pd.DataFrame(q)
q.columns=[np.unique(rating_time.movieId)]

p=np.array(np.random.random((f,U)))
p=pd.DataFrame(p)
p.columns=[np.unique(rating_time.userId)]

bi=np.zeros((1,I))
bi=pd.DataFrame(bi)
bi.columns=[np.unique(rating_time.movieId)]

bu=np.zeros((1,U))
bu=pd.DataFrame(bu)
bu.columns=[np.unique(rating_time.userId)]

q.shape
p.shape

###global mean
#mu=np.mean(rating_time.loc[:,'rating'])

###sort movie id
movie_id=np.unique(rating_time.movieId)

###train_test

sample_rating.sort_values(['userId','movieId'],ascending=[True,True])
sample_rating=rating_time.sample(100)
sample_rating.head()
train=rating_time.head(10000)

#train and test rmse
train_rmse=[]
test_rmse=[]


mu=np.mean(train.loc[:,'rating'])

for l in range(max_iter):
    
    # step2: fix q, solve p
    # define new factors
    ones_movie=pd.DataFrame(np.repeat(1,I)).T
    ones_movie.columns=[np.unique(rating_time.movieId)]
    q.tilde=ones_movie.append(q,ignore_index=True)
    p.tilde=bu.append(p,ignore_index=True)
    
    for u in range(U):
        # find the moives rated by user u
        rating_i=train.loc[train['userId']==u+1,'movieId']
        #update p.tilde
        M1=np.linalg.inv(np.array(q.tilde.loc[:,rating_i].dot(q.tilde.loc[:,rating_i].T))+_lambda*np.eye(f+1))
        M2=np.array(q.tilde.loc[:,rating_i])
        M3=np.array(train.loc[train['userId']==u+1,'rating'])-mu-bi.loc[:,rating_i]
        p.tilde.iloc[:,u]=M1.dot(M2).dot(M3.T)
        
        #update bu and p
    bu=pd.DataFrame(p.tilde.iloc[0,:]).T
    p=p.tilde.drop([0])

    #step3:fix p, solve q
    
    ones_user=pd.DataFrame(np.repeat(1,U)).T
    ones_user.columns=[np.unique(rating_time.userId)]
    p.tilde=ones_user.append(p,ignore_index=True)
    q.tilde=bi.append(q,ignore_index=True)
    
    for i in range(I):
        rating_u=train.loc[train['index']==i,'userId']
        rating_u=rating_u.drop_duplicates()
            #update q.tilde
        M1=np.linalg.inv(np.array(p.tilde.loc[:,rating_u].dot(p.tilde.loc[:,rating_u].T))+_lambda*np.eye(f+1))
        M2=np.array(p.tilde.loc[:,rating_u])
        M3=np.array(train.rating[rating_u.index])-mu-bu.loc[:,rating_u]
        q.tilde.iloc[:,i]=M1.dot(M2).dot(M3.T)
   
        
    #update bi and q
    bi=q.tilde.iloc[0,:]
    q=q.tilde.drop([0])
    
    ##step4: fix p,q, slove bi,t
    #for t in range (1000)
        #for i in range(I):
            # find the moive rated by user u
            #rating_u=train.loc[train['index']==i,'userId']
            #rating_u=rating_u.drop_duplicates()
            #update q.tilde
            #M1=np.linalg.inv(np.array(p.tilde.loc[:,rating_u].dot(p.tilde.loc[:,rating_u].T))+_lambda*np.eye(f+1))
            #M2=np.array(p.tilde.loc[:,rating_u])
            #M3=np.array(train.rating[rating_u.index])-mu-bu.loc[:,rating_u]
            #q.tilde.iloc[:,i]=M1.dot(M2).dot(M3.T)
    
    
    #summarize
    p.T.dot(q)+bu
    
rating_u.unique()
train.rating[rating_u.index]
bu.shape