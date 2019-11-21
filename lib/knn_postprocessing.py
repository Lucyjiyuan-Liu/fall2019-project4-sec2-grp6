#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 01:28:12 2019

@author: xiyiyan
"""

import os
import pandas as pd
os.chdir('/Users/xiyiyan/Documents/GitHub/fall2019-project4-sec2-grp6/output')


data_train_subset=pd.read_csv('data_train_subset.csv')
mat_q1=pd.read_csv('q_1.csv')
mat_q2=pd.read_csv('q_2.csv')

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


updated_rating_1=knn_postprocessing(data_train_subset,mat_q1)
updated_rating_1=pd.DataFrame(updated_rating_1,index=[np.delete(a,0)])
updated_rating_1.to_csv('/Users/xiyiyan/Documents/GitHub/fall2019-project4-sec2-grp6/output/updated_rating_knn1.csv')


updated_rating_2=knn_postprocessing(data_train_subset,mat_q2)
updated_rating_2.to_csv('/Users/xiyiyan/Documents/GitHub/fall2019-project4-sec2-grp6/output/updated_rating_knn2.csv')
   
    