#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:12:55 2019

@author: liujiyuan
"""

import numpy as np
import pandas as pd
# import os

from rmse import rmse


class ALSTD(ModelBase):
    
    def __init__(self, f=10, l = 0.3, iteration_max, train, test):
        
        ## train test: must contain all the possible userID, movieID and 
        
        
        self.f = f # number of latent variables 
        self.l = l # lambda 
        self.iteration_max = iteration_max
        self.train = train
        self.test = test
        
        ## Get the dimension and global mean of the data set 
        self.n_user = np.unique(train.userId).shape[0]
        self.n_item = np.unique(train.movieId).shape[0]
        self.n_timebin = np.unique(train.bin).shape[0]
        self.mu = np.mean(train.loc[:,'rating'])
        self.min_rating = 0
        self.max_rating = train.rating.max()
        
        # data state
        self.q = np.array(np.random.random((f,n_item))) # Movie matrix
        self.p = np.array(np.random.random((f,n_user))) # User matrix
        self.bi = np.zeros((1,n_item)) # Movie bias
        self.bu = np.zeros((1,n_user)) # User bias
        self.bit = np.zeros((n_timebin*n_item,3)) # Movie time variate bias
    
    def RMSE(estimation, data):
    ###Root Mean Square Error
    
        def sqr_err(obs):
            sqr_error=(obs['rating']-estimation[int(obs['bin']-1),int(obs['userId']-1),int(obs['index'])])**2
            return(sqr_error)
            
        return(np.sqrt(np.mean(data.apply(sqr_err,1))))

     
    def ALSTDfit （ self )
    
    ##Alternating Least Squares with Temperal Dynamics
  
        ## Step 1: Initialize Movie matrix (q), User matrix (p), Movie bias(bi) and User bias(bu)
        #q = np.array(np.random.random((f,n_item)))
        q = pd.DataFrame(q)
        q.columns = [np.unique(train.movieId)]
        
        #p = np.array(np.random.random((f,n_user)))
        p = pd.DataFrame(p)
        p.columns = [np.unique(train.userId)]
        
        #bi = np.zeros((1,n_item))
        bi = pd.DataFrame(bi)
        bi.columns = [np.unique(train.movieId)]
        
        #bu = np.zeros((1,n_user))
        bu=pd.DataFrame(bu)
        bu.columns = [np.unique(train.userId)]
        
        #bit = np.zeros((n_timebin*n_item,3))
        bit = pd.DataFrame(bit)
        bit.columns = ['movieId','bin','bias']
        bit.movieId = np.unique(train.movieId).repeat(n_timebin)
        bit.bin = np.tile(np.unique(train.bin),n_item)
        bit.bias = np.zeros((n_timebin * n_item))
        
        
        # global mean 
        # mu=np.mean(train.loc[:,'rating'])
        
        # sort movie id
        movie_id=np.unique(train.movieId)
        
        # sort the data by userid then by movie id
        train = train.sort_values(['userId','movieId'],ascending=[True,True])
        
        #train and test rmse
        #train.insert(3,'estimate',0)
        #test.insert(3,'estimate',0)
        
        # RMSE
        test_RMSE = np.zeros(iteration)
        train_RMSE = np.zeros(iteration)
        
        for itera in range(iteration):
    
            # step2: fix q, solve p
            # define new factors
            ones_movie=pd.DataFrame(np.repeat(1,n_item)).T
            ones_movie.columns=[np.unique(train.movieId)]
            q.tilde=ones_movie.append(q,ignore_index=True)
            p.tilde=bu.append(p,ignore_index=True)
    
            for u in range(n_user):
                # find the moives rated by user u, and time_bin of the rating
                rating_i=train.loc[train['userId']==u+1,['movieId','bin']]
                bi_bin_u = np.zeros((1,rating_i.shape[0]))
                
                # get bibin(t)
                for s in range(rating_i.shape[0]):
                    bi_bin_u[0,s] = bit.loc[(bit['movieId'] == rating_i.movieId.iloc[s]) & (bit['bin'] == rating_i.bin.iloc[s]),'bias']
                
                #update p.tilde
                M1=np.linalg.inv(np.array(q.tilde.loc[:,rating_i.movieId].dot(q.tilde.loc[:,rating_i.movieId].T))+l*np.eye(f+1))
                M2=np.array(q.tilde.loc[:,rating_i.movieId])
                M3=np.array(train.loc[train['userId']==u+1,'rating'])- mu - bi.loc[:,rating_i.movieId] - bi_bin_u
                p.tilde.iloc[:,u]=M1.dot(M2).dot(M3.T)
        
            #update bu and p
            bu=pd.DataFrame(p.tilde.iloc[0,:]).T
            p=p.tilde.drop([0])

            
            #step3:fix p, solve q
            #update p.tilde
            ones_user=pd.DataFrame(np.repeat(1,n_user)).T
            ones_user.columns=[np.unique(train.userId)]
            p.tilde=ones_user.append(p,ignore_index=True)
            q.tilde=bi.append(q,ignore_index=True)
    
            for i in range(n_item):
                # find the users who rate movie i, and time_bin of the rating
                rating_u=train.loc[train['index']==i,['movieId','userId','bin']]
                bi_bin_i = np.zeros((1,rating_u.shape[0]))
                
                # get bibin(t)
                for s in range(rating_u.shape[0]):
                    bi_bin_i[0,s] = bit.loc[(bit['movieId'] == rating_u.movieId.iloc[s]) & (bit['bin'] == rating_u.bin.iloc[s]),'bias']
                
                #update q.tilde
                M1=np.linalg.inv(np.array(p.tilde.loc[:,rating_u.userId].dot(p.tilde.loc[:,rating_u.userId].T)) + l*np.eye(f+1))
                M2=np.array(p.tilde.loc[:,rating_u.userId])
                M3=np.array(train.loc[train['index']==i,'rating']) - mu - bu.loc[:,rating_u.userId] - bi_bin_i
                q.tilde.iloc[:,i]=M1.dot(M2).dot(M3.T)
   
        
            #update bi and q
            bi=pd.DataFrame(q.tilde.iloc[0,:]).T
            q=q.tilde.drop([0])
            
            
            #step4:fix others, solve bit
            for t in range(n_timebin):
                for i in range(n_item):
            
                    # find the number of users who rate movie i, in time_bin t
                    Nit = len(train[(train['index']==i) &(train['bin'] == t+1)])
                
                    # find total rating of movie i, in time_bin t
                    rit = np.array(train[(train['index']==i) & (train['bin'] == t+1)].rating) - mu - \
np.array(bu.loc[0,train[(train['index']==i) & (train['bin'] == t+1)].userId]) - \
float(bi.loc[0,movie_id[i]]) - \
np.array(p.loc[:,train[(train['index']==i) & (train['bin'] == t+1)].userId].T.dot(q.iloc[:,i]))
                    rit = rit.sum()
                
    
                    #update bit
                    bit.loc[(bit['movieId'] == movie_id[i]) & (bit['bin'] == t+1),'bias'] = rit/(Nit+l)


            
            # Summerize
            print("iter:", itera+1, "\t")
            
            
            # Estimate 
            est_rating = np.zeros((n_timebin, n_user, n_item))
            est_rating_1 = np.array(p.T.dot(q)) + mu + np.array(bu.T)
            est_rating_1 = np.add(est_rating_1, np.array(bu.T))
            est_rating_1 = np.add(est_rating_1, np.array(bi))
            
            for t in range(n_timebin):
                bt = bit[bit['bin'] == t + 1].bias
                est_rating_1 = np.add(est_rating_1, np.array(bt))
                est_rating[t,:,:] = est_rating_1
            
           
            
            # RMSE 
            train_RMSE[i] = RMSE(est_rating,train) ## first time: 1.150437954121489
            test_RMSE[i] = RMSE(est_rating,test) ## first time: 1.4897433028093507
            print("train RMSE:", train.RMSE[i], "\t")
            print("test RMSE:", test.RMSE[i], "\t")

        return mu, q, p, bi, bu, bit, train_RMSE, test_RMSE   
            
  
        

## Save output
q.to_csv('/Users/liujiyuan/Desktop/5243/Project4/fall2019-project4-sec2-grp6-master/output/q.csv', index = False)
p.to_csv('/Users/liujiyuan/Desktop/5243/Project4/fall2019-project4-sec2-grp6-master/output/p.csv', index = False)
bi.to_csv('/Users/liujiyuan/Desktop/5243/Project4/fall2019-project4-sec2-grp6-master/output/bi.csv', index = False)
bu.to_csv('/Users/liujiyuan/Desktop/5243/Project4/fall2019-project4-sec2-grp6-master/output/bu.csv', index = False)
bit.to_csv('/Users/liujiyuan/Desktop/5243/Project4/fall2019-project4-sec2-grp6-master/output/bit.csv', index = False)        
        
obs = train.iloc[57160,:]        
np.sum(train.loc[train['index']==1,'rating'])      
        