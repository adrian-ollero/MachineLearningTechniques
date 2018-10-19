# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:59 2018

@author: Adrian, Enrique, Fernando
"""

import pandas as pd
import numpy as np
from pylab import pcolor, show, colorbar, xticks, yticks
import seaborn as sns
import matplotlib.pyplot as plt


#0 . Load the data 
# read the csv
df = pd.read_csv(".\Data\T2.csv")


# 1. Filter days
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
# extract date from datetime
df['date'] = [d.date() for d in df['TimeStemp']]
# list the available days
df['date'].unique()
#filter data by date
df_2_days = df[(df['TimeStemp'] > '2016-05-03 00:00:00') & (df['TimeStemp'] <= '2016-05-04 23:59:59')]
print(df_2_days.shape)


# 2. Filter columns
# df28f = df28[[c for c in df if c.endswith('MEAN')]]
df_columns_filtered = df_2_days[[c for c in df if ((c.startswith('Orientation') or c.startswith('Gyroscope'))and not c.endswith('FFT'))]]
print(df_columns_filtered.shape)

# 3. Filter rows
# Merge 3 consecutive columns with the mean values
df_rows_n_columns_filtered = pd.DataFrame(columns = list(df_columns_filtered.columns.values))
for r in range(0, df_columns_filtered.shape[0], 3):
    serie = df_columns_filtered[r:r+3].mean()
    dataframe = pd.DataFrame(serie).transpose()
    df_rows_n_columns_filtered = df_rows_n_columns_filtered.append(dataframe)

# 4. Correlation
R = np.corrcoef(np.transpose(df_rows_n_columns_filtered))
pcolor(R)
colorbar()
yticks(np.arange(0,24),range(0,24))
xticks(np.arange(0,24),range(0,24))
show()

# The features listed in 'exclude' are those with high correlation, so are deleted (except one which is kept)
exclude = ["OrientationProbe_roll_MEDIAN", "OrientationProbe_roll_MIDDLE_SAMPLE", "OrientationProbe_pitch_MIDDLE_SAMPLE", "OrientationProbe_pitch_MEDIAN",
           "OrientationProbe_azimuth_MEDIAN", "OrientationProbe_azimuth_MIDDLE_SAMPLE"]
df_correlation = df_rows_n_columns_filtered.loc[:, df_rows_n_columns_filtered.columns.difference(exclude)]

# Calculate correlation with new dataframe wich looks much better
R = np.corrcoef(np.transpose(df_correlation))
pcolor(R)
colorbar()
yticks(np.arange(0,18),range(0,18))
xticks(np.arange(0,18),range(0,18))
show()

# Plot the 
sns.set(style="white")
mask = np.zeros_like(R, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8,
 square=True, xticklabels=2, yticklabels=2,
 linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# 5. HCA Feratures
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
# df_norm = min_max_scaler.fit_transform(df_correlation.transpose())  # PREGUNTAR!!
df_norm = min_max_scaler.fit_transform(df_correlation)

import sklearn.neighbors
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(df_norm)
avSim = np.average(matsim)
print ("%s\t%6.2f" % ('Distancia Media', avSim))

# 3.2. Building the Dendrogram	
from scipy import cluster
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
cluster.hierarchy.dendrogram(clusters, color_threshold=15)
plt.show()

# 6. PCA
from sklearn.decomposition import PCA
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(df_norm)
print(estimator.explained_variance_ratio_) 

pd.DataFrame(np.matrix.transpose(estimator.components_), columns=['PC-1', 'PC-2'], index=df_correlation.columns)

numbers = np.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1],'.') 
plt.xlim(-1, 2)
plt.ylim(-0.5, 1)
ax.grid(True)
fig.tight_layout()
plt.show()