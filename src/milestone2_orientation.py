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
exclude = ['OrientationProbe_azimuth_MEAN',"OrientationProbe_roll_MEDIAN", "OrientationProbe_roll_MIDDLE_SAMPLE", "OrientationProbe_pitch_MIDDLE_SAMPLE", "OrientationProbe_pitch_MEDIAN",
           "OrientationProbe_azimuth_MEDIAN", "OrientationProbe_azimuth_MIDDLE_SAMPLE"]
df_columns_filtered = df_columns_filtered.loc[:, df_columns_filtered.columns.difference(exclude)]


########################## 3. Build and tune a Model #########################
##### Data normalization

columna_malicious = df_columns_filtered['Malicious']

df_sin_malicious = df_columns_filtered.drop('Malicious',axis = 1)

from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df_sin_malicious)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_norm, columna_malicious, test_size=0.4)
################################ Naive-Bayes #################################

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

############################## CLUSTERING ####################################
df_columns_filtered.to_csv('./output/features_filtered.csv', index=False)
#First reduce dimensionality
df_reduced = df_columns_filtered.iloc[60000:df_columns_filtered.shape[0]]
#We drop azimuth as we considered en milestone 1 as not relevant
#df_reduced = df_reduced.drop('OrientationProbe_azimuth_MEAN',axis=1)

from sklearn.cluster import KMeans
from sklearn import metrics

# parameters
init = 'random' # initialization method 
# to run 10 times with different random centroids 
# to choose the final model as the one with the lowest SSE
iterations = 10
# maximum number of iterations for each single run
max_iter = 300 
# controls the tolerance with regard to the changes in the 
# within-cluster sum-squared-error to declare convergence
tol = 1e-04 
 # random seed
random_state = 10

distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(df_reduced)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(df_reduced, labels))
    
#Draw distorsion & silhoutte
plt.subplot(121)
plt.plot(range(2,11), distortions, marker='o', c='blue')
plt.xlabel('Number of clusters')
plt.ylabel('Distorsion')
plt.subplot(122)
plt.plot(range(2,11), silhouettes , marker='o', c='green')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()

k = 8

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(df_reduced)

from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_reduced, y_km))
      
print('Distortion: %.2f' % km.inertia_)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')   
ax.scatter(df_reduced.iloc[:,0], df_reduced.iloc[:,1], df_reduced.iloc[:,2], marker='o', c=km.labels_)
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], marker='o', c='green')
plt.show()

######################## GET CLUSTERS SEPARATED #############################
clusters_and_elements = []
for i in range(k):
    clusters_and_elements.append([]) #Initialize list

for x in range(0, len(df_reduced)): # Separate each element in a different list 
    clusters_and_elements[km.labels_[x]].append(df_reduced.iloc[x])

for x in range(len(clusters_and_elements)): # Transform to numpy array and store in cvs
    clusters_and_elements[x] = np.asarray(clusters_and_elements[x])

df_c3 = pd.DataFrame(data = clusters_and_elements[2],columns=["Malicious","Pitch","roll"])
df_c2 = pd.DataFrame(data = clusters_and_elements[1],columns=["Malicious","Pitch","roll"])
df_c7 = pd.DataFrame(data = clusters_and_elements[6],columns=["Malicious","Pitch","roll"])

df_union = df_c2.append(df_c3)

##### Data normalization

test_size = 0.4

columna_malicious = df_union['Malicious']

df_sin_malicious = df_union.drop('Malicious',axis = 1)

from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df_sin_malicious)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_norm, columna_malicious, test_size=test_size)

################################## KNN #######################################
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

X = df_norm
y = columna_malicious

h=.02

n_neighbors = 15
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(x_train, y_train)

#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


#y_predKNN = clf.predict(np.c_[xx.ravel(), yy.ravel()])
y_predKNN = clf.predict(x_test)

from sklearn.metrics import classification_report

print("KNN Results: \n" 
      +classification_report(y_true=y_test, y_pred=y_predKNN))

# Matriz de confusión

print("Confussion Matrix:\n")
matriz = pd.crosstab(y_test, y_predKNN, rownames=['actual'], colnames=['preds'])
print(matriz)

############################## Complement NB  ################################
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

#### Test with more data ####
df_columns_filtered = df_columns_filtered.drop('OrientationProbe_azimuth_MEAN',axis=1)
columna_malicious = df_columns_filtered['Malicious']

df_sin_malicious = df_columns_filtered.drop('Malicious',axis = 1)

from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df_sin_malicious)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_norm, columna_malicious, test_size=0.9)

y_predC = model.predict(x_test)

############################# METRIC RESULTS #################################
from sklearn.metrics import classification_report

print("Complement Naive-Bayes Results: \n" 
      +classification_report(y_true=y_test, y_pred=y_predC))

# Matriz de confusión

print("Confussion Matrix:\n")
matriz = pd.crosstab(y_test, y_predC, rownames=['actual'], colnames=['preds'])
print(matriz)
