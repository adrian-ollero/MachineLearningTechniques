# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:59 2018

@author: Adrian Ollero, Enrique Garrido, Fernando Vallejo
"""

import pandas as pd
import numpy as np
from pylab import pcolor, show, colorbar, xticks, yticks
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


########################## 1. Load the data ##################################
# Read the csv
df = pd.read_csv(u"../Data/T2.csv")

# 1. Filter days
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
df['date'] = [d.date() for d in df['TimeStemp']]
df['date'].unique()
df_2_days = df[(df['TimeStemp'] > '2016-05-03 00:00:00') & (df['TimeStemp'] <= '2016-05-04 23:59:59')]
print(df_2_days.shape)

############################### FILTER DATA ##################################
# 2. Filter columns
df_columns_filtered = df_2_days[[c for c in df if (c.startswith('Orientation')and not c.endswith('FFT'))]]
print(df_columns_filtered.shape)


# 3. Filter rows
# Merge 3 consecutive rows with the mean values
df_rows_n_columns_filtered = pd.DataFrame(columns = list(df_columns_filtered.columns.values))
for r in range(0, df_columns_filtered.shape[0], 3):
    serie = df_columns_filtered[r:r+3].mean()
    dataframe = pd.DataFrame(serie).transpose()
    df_rows_n_columns_filtered = df_rows_n_columns_filtered.append(dataframe)


############################## CORRELATION ###################################
# 4. Correlation
R = np.corrcoef(np.transpose(df_rows_n_columns_filtered))
pcolor(R)
colorbar()
yticks(np.arange(0,df_rows_n_columns_filtered.shape[1]),range(0,df_rows_n_columns_filtered.shape[1]))
xticks(np.arange(0,df_rows_n_columns_filtered.shape[1]),range(0,df_rows_n_columns_filtered.shape[1]))
show()

# The features listed in 'exclude' are those with high correlation, so are deleted (except one which is kept)
exclude = ["OrientationProbe_roll_MEDIAN", "OrientationProbe_roll_MIDDLE_SAMPLE", "OrientationProbe_pitch_MIDDLE_SAMPLE", "OrientationProbe_pitch_MEDIAN",
           "OrientationProbe_azimuth_MEDIAN", "OrientationProbe_azimuth_MIDDLE_SAMPLE"]
df_correlation = df_rows_n_columns_filtered.loc[:, df_rows_n_columns_filtered.columns.difference(exclude)]

# Calculate correlation with new dataframe
R = np.corrcoef(np.transpose(df_correlation))
pcolor(R)
colorbar()
yticks(np.arange(0,df_correlation.shape[1]),range(0,df_correlation.shape[1]))
xticks(np.arange(0,df_correlation.shape[1]),range(0,df_correlation.shape[1]))
show()

# Plot the heat map
sns.set(style="white")
mask = np.zeros_like(R, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8,
 square=True, xticklabels=2, yticklabels=2,
 linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


################################## HCA #######################################
# 5. HCA Feratures
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df_correlation)
df_norm_transposed = df_norm.transpose()

import sklearn.neighbors
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(df_norm_transposed)
avSim = np.average(matsim)
print ("%s\t%6.2f" % ('Distancia Media', avSim))

#Building the Dendrogram	
from scipy import cluster
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=15)
plt.show()

#####################  PLOT NORMALIZED DATA ##################################
# 6. Plot the data (Only Orientation as there are 3 features)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_norm[:,0], df_norm[:,1], df_norm[:,2], marker='o', c='red')
plt.show()

########################### CALCULATE PCA ####################################
# 7. PCA
from sklearn.decomposition import PCA
estimator = PCA (n_components = 3)
X_pca = estimator.fit_transform(df_norm.transpose())
print(estimator.explained_variance_ratio_) 

df_pca = pd.DataFrame(np.matrix.transpose(estimator.components_), columns=['PC-1', 'PC-2', 'PC-3'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
ax.scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], df_pca.iloc[:,2], marker='o', c='red')
plt.show()

##############################################################################
############################ KMEANS PCA VALUES ###############################
# 8. K-means 
## 8.1 Setting parameters
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

# 8.2 Calculate k
from sklearn.cluster import KMeans
from sklearn import metrics

distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(df_pca)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(df_pca, labels))
    
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

#8.3 Clustering PCA
k = 8

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(df_pca)

from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_pca, y_km))
      
print('Distortion: %.2f' % km.inertia_)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')   
ax.scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], df_pca.iloc[:,2], marker='o', c=km.labels_)
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], marker='o', c='green')
plt.show()

# 9. Outliers detection (LOF)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(df_pca)
X_scores = clf.negative_outlier_factor_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], df_pca.iloc[:,2], marker='o', c=km.labels_)
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
ax.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1],df_pca.iloc[:,2], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.show()

#############################################################################
##################### KMEANS WITH NORMALIZED DATA ###########################
distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(df_norm)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(df_norm, labels))
    
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

k = 8 #Selected value 

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(df_norm)

from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_norm, y_km))
      
print('Distortion: %.2f' % km.inertia_)

#plot clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')   
ax.scatter(df_norm[:,0], df_norm[:,1], df_norm[:,2], marker='o', c=km.labels_)
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], marker='+', c='green')
plt.show()

#Outlier detection
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(df_norm)
X_scores = clf.negative_outlier_factor_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_norm[:,0], df_norm[:,1], df_norm[:,2], marker='o')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
ax.scatter(df_norm[:, 0], df_norm[:, 1],df_norm[:,2], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.show()

# Outlier deleting (store in a csv)
rows_deleted = 0
outliers = []
np_correlation = df_correlation.values

for s in range(0, len(X_scores)):
    if X_scores[s-rows_deleted] < -100:
        outliers.append(df_correlation.iloc[s,:].tolist())
        df_norm = np.delete(df_norm, (s-rows_deleted), axis=0)
        X_scores = np.delete(X_scores, (s-rows_deleted), axis=0)
        np_correlation = np.delete(np_correlation, (s-rows_deleted), axis=0)
        rows_deleted+=1

outliers_to_csv = np.asarray(outliers)
np.savetxt(u"../output/outliers_orientation.csv",
           outliers_to_csv,
           fmt="%f",
           delimiter=",")
        
# Plot results without outliers
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_norm[:,0], df_norm[:,1], df_norm[:,2], marker='o')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
ax.scatter(df_norm[:, 0], df_norm[:, 1],df_norm[:,2], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.show()

# Calculate again value of k
distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(df_norm)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(df_norm, labels))
    
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

# Clustering without outliers
k = 8 #Selected value 

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(df_norm)

from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_norm, y_km))
      
print('Distortion: %.2f' % km.inertia_)

#plot clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')   
ax.scatter(df_norm[:,0], df_norm[:,1], df_norm[:,2], marker='o', c=km.labels_)
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], marker='+', c='green')
plt.show()

#############################################################################
######################## GET CLUSTERS SEPARATED #############################
clusters_and_elements = []
for i in range(k):
    clusters_and_elements.append([]) #Initialize list

for x in range(0, len(np_correlation)): # Separate each element in a different list 
    clusters_and_elements[km.labels_[x]].append(np_correlation[x])

for x in range(len(clusters_and_elements)): # Transform to numpy array and store in cvs
    clusters_and_elements[x] = np.asarray(clusters_and_elements[x])
    np.savetxt(u"../output/cluster"+str(x)+".csv",
           clusters_and_elements[x],
           fmt="%f",
           delimiter=",")
#############################################################################
