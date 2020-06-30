# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:32:38 2019

@author: 140524
"""

# Handling Missing Data in Data Frame from Youtube
#https://www.youtube.com/watch?v=EaGbS7eWSs0



import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/5_handling_missing_data_fillna_dropna_interpolate/weather_data.csv')
df.dtypes

# data comes as a string so changing the data fomat

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/pandas/5_handling_missing_data_fillna_dropna_interpolate/weather_data.csv', parse_dates=['day'])
df.dtypes
df

# set day column as index

df.set_index('day', inplace=True)
df
# filling Na values - first method with 0

new_df=df.fillna(0)
new_df

#sometimes 0 value always doesn make sense for example event column - 0 doesnt make sense
# now filling Na values of different columns with different values
# for example filling all temp Na values with 0, all windspeed with 2, all event Na values
# with no event

new_df=df.fillna({'temperature':0,'windspeed':2, 'event':'no event'})
new_df

# but filling always with 0 will drop down the mean which will drop down the mean
#so carry fwding the temperature,windspeed, event of previous day to the next day
# fflill carry fwds previous day's value

new_df=df.fillna(method='ffill')
new_df

#bfill means it will back fill next days value

new_df=df.fillna(method='bfill')
new_df

# specifying limit will set the limit of carryfwding only once
# meaning if there are 2 vacant cells consequetiveky in 1 column, it will fill the first one only
#leaving out the second one empty
#limit 1, limit 2 ...etc...

new_df=df.fillna(method='ffill',limit=1)
new_df

# using interpolate will come up with a better guess of finding the value in btn ..like
#A= 32 C = 30, B was mising 
#B using interporlate will be 30 
#B using ffill will 32
#B using fillna(0) will be 0
#this is called linear interpolation
new_df = df.interpolate()
new_df
#interpolation has many types - previously we used linera interpolation that is default
# we cud use polynomial, quatratic, cubic interpolation etc.
# below we will do time interpolation
# tempretaure of missing date 01/04 will be closer to 01/05(value already thr)
#rather than in btwn at 28 (avergae btwn temp at 01/01 and 01/05
new_df = df.interpolate(method='time')
new_df

# dropping all rows with Na values
new_df = df.dropna()
new_df
#drop all rows that has 3 Nan
new_df = df.dropna(how='all')
new_df

#if atleast 1 or 2 etc..Non Na value is there, keep the row - defining threshold parameter

new_df = df.dropna(thresh=1)
new_df

new_df = df.dropna(thresh=2)
new_df

#insert missing dates 
#To do that 1) define a date range 2)pass the date range to date time index
#3)reindex your dataframe

dt = pd.date_range('01-01-2017','01-11-2017')
idx = pd.DatetimeIndex(dt)
new_df=df.reindex(idx)
new_df