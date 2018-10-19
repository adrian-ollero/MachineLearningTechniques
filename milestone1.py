# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:59 2018

@author: Adrian, Enrique, Fernando
"""

import pandas as pd


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
    