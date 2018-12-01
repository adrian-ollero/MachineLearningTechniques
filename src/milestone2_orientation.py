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
                    df_sherlock.loc[s_index,'Malicious']=1
                else:
                    print("Benigno: "+ str(s_index) + " - " + str(m_index))
                    df_sherlock.loc[s_index,'Malicious']=1
                i = s_index
                break
        else:
            i = s_index
            break
        
####################### 2. Features selection ################################

df_columns_filtered = df_sherlock[[c for c in df_sherlock if (c.startswith('Malicious') or (c.startswith('Orientation')and not c.endswith('FFT')))]]
df_columns_filtered = df_columns_filtered.dropna()
exclude = ["OrientationProbe_roll_MEDIAN", "OrientationProbe_roll_MIDDLE_SAMPLE", "OrientationProbe_pitch_MIDDLE_SAMPLE", "OrientationProbe_pitch_MEDIAN",
           "OrientationProbe_azimuth_MEDIAN", "OrientationProbe_azimuth_MIDDLE_SAMPLE"]
df_columns_filtered = df_columns_filtered.loc[:, df_columns_filtered.columns.difference(exclude)]


########################## 3. Build and tone a Model #########################
##### Data normalization

columna_malicious = df_columns_filtered['Malicious']

df_sin_malicious = df_columns_filtered.drop('Malicious',axis = 1)

from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df_sin_malicious)
################################ Naive-Bayes #################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_norm, columna_malicious, test_size=0.4)

## Gaussian
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

############################# METRIC RESULTS #################################
from sklearn.metrics import classification_report

print("Gaussian Naive-Bayes Results: \n" 
      +classification_report(y_true=y_test, y_pred=y_pred))

# Matriz de confusión

print("Confussion Matrix:\n")
matriz = pd.crosstab(y_test, y_pred, rownames=['actual'], colnames=['preds'])
print(matriz)

## Complement  #### NOT AVAILABLE UP TO VERSION 0.20.X OF SCIKIT-LEARN
from sklearn.naive_bayes import ComplementNB
model = ComplementNB()

model.fit(x_train,y_train)
y_predC = model.predict(x_test)

############################# METRIC RESULTS #################################
from sklearn.metrics import classification_report

print("Complement Naive-Bayes Results: \n" 
      +classification_report(y_true=y_test, y_pred=y_predC))

# Matriz de confusión

print("Confussion Matrix:\n")
matriz = pd.crosstab(y_test, y_predC, rownames=['actual'], colnames=['preds'])
print(matriz)

