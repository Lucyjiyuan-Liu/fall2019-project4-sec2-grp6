#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 01:28:12 2019

@author: xiyiyan
"""


#############################################
#####Run KNN for predicted movie matrix######
#############################################

def knn_postprocessing(data,mat_q):
    
    #data:trainning set
    #mat_q:10*9724 predicted q matrix
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    #n_users=np.unique(data.userId).shape[0]
    n_movies=np.unique(data.movieId).shape[0]
    n_users=np.unique(data.userId).shape[0]
    mat_q=mat_q.T
    updated_rating=np.zeros(n_movies)

    #get similarity matrix
    similarity=cosine_similarity(mat_q)   
    np.fill_diagonal(similarity,0)
    similarity.shape
    similarity=pd.DataFrame(similarity)   
     
    for i in range(n_movies):
                
        #get the most similar movie to i
        most_similar_idx=similarity[i].idxmax()            
        #number of ratings of movie_i
        num_rating= np.sum(data['index']== most_similar_idx)
                    
        #update rating of user i using average
        updated_rating[i]=np.sum(data.loc[data['index']==most_similar_idx,'rating'])/num_rating
    
    updated_rating=np.tile(updated_rating,(n_users,1)) 
    updated_rating=pd.DataFrame(updated_rating) 
    #updated_rating=updated_rating
    updated_rating.columns = [np.unique(data.movieId)]
    
    return(updated_rating)

   
    
