#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 01:28:12 2019

@author: xiyiyan
"""

def knn_postprocessing(rating_mat,mat_q,mat_p):
    
    #rating_mat:610*9724
    #mat_q:9724*k
    #mat_p:k*610
    
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity=cosine_similarity(mat_q,mat_q)    
    similarity=pd.DataFrame(similarity)
    
    rating_mat=pd.DataFrame(rating_mat)
    
    new_latent_factor=mat_q
    updated_q=np.zeros((610,9724,mat_q.shape[1]))
    
    for i in range(610):
        
            useri_rating=rating_mat.iloc[i,:] 
            num_norating_i=len(np.where(useri_rating==0)[0]) 
            
            for j in range(num_norating_i):
                
                ###update latent factor of each movie that was not rated by user_i
                
                cos_sim=similarity.iloc[np.where(useri_rating==0)[0][j],np.where(useri_rating!=0)[0]]
                most_similar=cos_sim.sort_values().tail(5)   
                ### get 5 most similar movies with movie_j      
                latent_factor5=mat_q[most_similar.index]
                new_latent_factor[j]=latent_factor5.mean(axis=0)
            updated_q[i]=new_latent_factor
    
            
    ###use the upadated q_matrix to predict the rating vector(9724*1) of user i      
    updated_rating_mat=np.zeros((610,9724))
    for i in range(610):
        updated_rating_mat[i]=np.dot(mat_q,mat_p.T[i]) 
        
    return(updated_rating_mat)
