# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:47:11 2018

@author:
"""

import pandas as pd
import numpy as np
from pylab import pcolor, show, colorbar, xticks, yticks
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


########################## 1. Load the data ##################################
# Read the csv
df_sherlock = pd.read_csv(u"./Data/T2.csv")
df_sherlock['UUID'] = df_sherlock['UUID'].astype(np.int64)

df_moriarty = pd.read_csv(u"./Data/Moriarty.csv")
df_moriarty = df_moriarty.sort_values(by=['UUID'])
df_sherlock = df_sherlock.sort_values(by=['UUID'])
########################## Merge dataframes ##################################

df_sherlock['Malicious'] =0
            
i = 0
for m_index,m_row in df_moriarty.iterrows():
    for s_index in range(i, df_sherlock.shape[0]):
        if(df_moriarty.loc[m_index,'UUID'] >= df_sherlock.loc[s_index,'UUID'] and s_index < df_sherlock.shape[0]-1):
            if (df_moriarty.loc[m_index,'UUID'] < df_sherlock.loc[s_index+1,'UUID']):
                if df_moriarty.loc[m_index,'ActionType'] == 'malicious':
                    print("Maligno: "+ str(s_index) + " - " + str(m_index))
                    df_sherlock.loc[s_index,'Malicious']=2
                else:
                    print("Benigno: "+ str(s_index) + " - " + str(m_index))
                    df_sherlock.loc[s_index,'Malicious']=1
                i = s_index
                break
        else:
            i = s_index
            break