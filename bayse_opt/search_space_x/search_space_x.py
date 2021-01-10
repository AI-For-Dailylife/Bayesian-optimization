# -*- coding: utf-8 -*-



import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import itertools


ds = pd.read_csv("range_of_data.csv", index_col=0)


flag = 1

array = []

for col_num in range(ds.shape[1]):
    search_x = np.arange(ds.iloc[0,col_num],ds.iloc[1,col_num]+1\
                         ,ds.iloc[2,col_num])
    array.append(search_x)


search_space_x = list(itertools.product(*array))

search_space_x = pd.DataFrame(search_space_x,columns=ds.columns)

if flag==0:
    search_space_x.to_csv("search_space_x.csv")
elif flag==1:

    sum_search_space_x = sum(search_space_x.T.values).T

    under_100_x = pd.DataFrame(sum_search_space_x == 100)
    search_space_x = search_space_x.iloc[under_100_x.values[:,0],:]

    search_space_x = search_space_x.reset_index(drop=True)
    
    
    search_space_x.to_csv("search_space_x.csv")
