#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:08:22 2019

@author: xiyiyan
"""


def rmse(rating,est_rating):
    
    import numpy as np
    import math
    
    def sqr_err(obs):
        sqr_error=(obs[2]-est_rating.iloc[obs[0]-1,obs[4]])**2
        return(sqr_error)
        
    return(math.sqrt(np.mean(rating.apply(sqr_err,1))))
  
#######    rmse=0.401361160390804

