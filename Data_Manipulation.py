# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:45:47 2019

@author: 140524
"""

#Data Manipukation
# Basic functionality of NDIM
#NDIM returns the num of dimensions of a data structure or basically it returns 
#the number of columns of your data

import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
df
print(df.ndim)

#Passing a dict

import pandas as pd
import numpy as np
df=pd.DataFrame({'a':pd.Series(np.arange(1,50)),'b':pd.Series(np.arange(51,100))})
df
print(df.ndim)
print(df.axes)

# Basic functionality of AXES
# returns the axes of the row levels

import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
df
print(df.axes)
#Passing a dict
import pandas as pd
import numpy as np
df=pd.DataFrame({'a':pd.Series(np.arange(1,103)),'b':pd.Series(np.arange(51,100))})
df
print(df.axes)

#Basic functionality of AXES
#returns the actual data in the series of an array

import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
df
print(df.values)

import pandas as pd
import numpy as np
df=pd.DataFrame({'a':pd.Series(np.arange(1,103)),'b':pd.Series(np.arange(51,100))})
df
print(df.values)

#Basic functionality of Head
#returns the first N rows of the data structure
import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
df
print(df.head())

import pandas as pd
import numpy as np
df=pd.DataFrame({'a':pd.Series(np.arange(1,103)),'b':pd.Series(np.arange(51,100))})
df
print(df.head(10))
df.values

#Basic functionality of Tail
#returns the lst N rows of the data structure

import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
df
print(df.tail())

import pandas as pd
import numpy as np
df=pd.DataFrame({'a':pd.Series(np.arange(1,103)),'b':pd.Series(np.arange(51,100))})
df
print(df.tail(10))

#Basic functionality of Sum
#returns the sum of the values for the requested axis

import pandas as pd
import numpy as np
df = pd.DataFrame({'even':pd.Series(np.arange(0,100,2)),'odd':pd.Series(np.arange(1,100,2))})
df
print(df['odd'])
print(df['even'])
print(df.sum())

#Basic functionality of STD
#returns the STD of the values for the requested axis

import pandas as pd
import numpy as np
df = pd.DataFrame({'even':pd.Series(np.arange(0,100,2)),'odd':pd.Series(np.arange(1,100,2))})
print(df.std())
df.describe().head(3)

#Iterating through a data frame
#ITERITEMS iterates over each colunn as key,value pair

#The key,value pair iterated over - consistes of the column level as the key  and the
#series object of column values as values
#It takes a key at a item and iterates over the values associated with that key

# importing pandas as pd 
# importing pandas as pd 
import pandas as pd 
# Creating the DataFrame 
df = pd.DataFrame({'Weight':[45, 88, 56, 15, 71], 
                   'Name':['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'], 
                   'Age':[14, 25, 55, 8, 21]}) 
# Create the index 
index_ = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5'] 
df
# Set the index 
df.index = index_ 
# Print the DataFrame 
print(df) 
#loc and iloc function
print(df.loc[:,['Name','Age']])
print(df.loc['Row_2', ['Name','Age']]) 


import pandas as pd
import numpy as np
df=pd.DataFrame(np.random.rand(5,4), columns = ['col-1','col-2','col-3','col-4'])
df
for key, value in df.iteritems():
    print('KEY:',key)
    print('VALUE:',value)

# or
df = pd.DataFrame({'Weight':[45, 88, 56, 15, 71], 
                   'Name':['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'], 
                   'Age':[14, 25, 55, 8, 21]},index =['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5'] ) 
df
for key, value in df.iteritems():
    print('KEY:',key)
    print('----')
    print('VALUE:',value)
    
for i in df.iteritems():
    print(i)     

for i in df.iterrows():
    print(i)    

for key, value in df.iterrows():
    print('KEY:',key)
    print('---')
    print('VALUE:',value)        
    
for row in df.itertuples():
    print(row)
    
#ITERROWS iterates over each rows as key,value pair
    
import pandas as pd
import numpy as np
df=pd.DataFrame(np.random.rand(5,4), columns = ['col-1','col-2','col-3','col-4'])
for key, value in df.iterrows():
    print('KEY:',key)
    print('VALUE:',value)    
    
# trying to get the domernsion of each rows

import pandas as pd
import numpy as np
df=pd.DataFrame(np.random.rand(5,4), columns = ['col-1','col-2','col-3','col-4'])
for key, value in df.iterrows():
    print('KEY:',key)
    print('VALUE-DIMENSION:',value.ndim,type(value))        
    
#Iterating through a data frame
#IterTuples() returns a iterator yielding a named tuple for each row    

import pandas as pd
import numpy as np
df=pd.DataFrame(np.random.rand(5,4), columns = ['col-1','col-2','col-3','col-4'])
for row in df.itertuples():
    print(row)

#GROUP BY operation - grouping data by column

import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,1],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
df    
print(df.groupby('Team'))
print(df.groupby('Team').groups)

# Group by multiple column
import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,1],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
df    
print(df.groupby(['Team','Rank']).groups)

# Group by multiple column with duplicate values - changing one of AUS occurance rank=2
import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
df    
x = df.groupby('Team')
print(x.groups)

#--or

print(df.groupby('Team').groups)

print(df.groupby(['Team','Rank']).groups)

#Iterate over indivitual groups

import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
df
for name, group in df.groupby('Team'):
    print('Teamname:',name)
    print(group)
    

#lets find out the type of each group


import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
df
for name, group in df.groupby('Team'):
    print('Teamname:',name)
    print('---')
    print(group)
    print('---')
    print(type(group))

# or

x = df.groupby('Team')
print(x.groups)        
# printing only the group Rank    
import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)
for name, group in df.groupby('Team'):
    print('Teamname:',name)
    print('----')
    print(group['Rank'])
    print('----')
    print(group['Year'])

# a single group can be selected using get-group   
#Get-group gets the value of a single group - India    
import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)    
x = df.groupby('Team')
print(x.get_group('IND'))
    
#Get-group gets the value of a single group - Rank 1

import pandas as pd
import numpy as np
word_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
df = pd.DataFrame(word_cup)    
x = df.groupby('Rank')
print(x.get_group(1))

#Aggregations - get the single aggregated value of each group

import pandas as pd
import numpy as np
df = pd.DataFrame({'even':pd.Series(np.arange(0,100,2)),'odd':pd.Series(np.arange(1,100,2))})
df
print(df['odd'])
print(df['even'])
x = df.groupby('odd')
print(x.groups)



# concatenation concats 2 dataframes if their key and dim are same

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df1
df2 = pd.DataFrame(match_stats)
df2
print(pd.concat([df1,df2],sort=True))

# the above same as below which means concat method has default axis =0
#concateing acros the row with axis = 0
import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.concat([df1,df2],axis=0,sort=True))



#concateing acros the column with axis = 1
import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.concat([df1,df2],axis=1))

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.concat([df1,df2],sort=True))


#append - this works just like concat along row or axis=0

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(df1.append(df2,sort=True))

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(df2.append(df1,sort=True))

# Merging and Joining 
# it is simlar to concatenate but it joins on a specific column or axis =1

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)

#Merging on a key 
print(pd.merge(df1,df2,on='Team'))

#Left join merges 2 dataframes based on key from the left dataframe

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.merge(df1,df2,on='Team',how='left'))


#Right join merges 2 dataframes based on key from the right dataframe

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.merge(df1,df2,on='Team',how='right'))

# Outer join merges 2 data frames based on full union of the columns of both the dataframes more
# like the default

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.merge(df1,df2,on='Team',how='outer'))

# Outer join merges 2 data frames based on intersection of the columns of both the dataframes

import pandas as pd
import numpy as np
world_cup = {'Team':['WI','WI','IND','AUS','PAK','SRK','AUS','AUS','AUS','IND','AUS'],'Rank':[7,7,2,1,6,4,1,1,1,2,2],'Year':[1975,1979,1983,1987,1991,1995,1999,2003,2007,2011,2015]}
match_stats = {'Team':['WI','IND','PAK','SRK','AUS'],'WC_played':[11,11,9,8,10],'ODI':[733,988,712,678,700]}
df1 = pd.DataFrame(world_cup)  
df2 = pd.DataFrame(match_stats)
print(pd.merge(df1,df2,on='Team',how='inner'))


#----Use Case_Country_wise statistics

import pandas as pd
import numpy as np
table = pd.read_csv("C:\python\AllCountries.csv")
print(table.ndim)
print(table.shape)
print(table.dtypes)
print(table.head())
print(table.describe())


# find all the  ountries with sixe>2000*1000 sq km
#with open('C:\edureka\PRClass_3\Data_set\\minicountry.csv','w+') as fwrite:

# using iterrows
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',nrows=3)
ufo
ufo.columns
#print the rows
for c in ufo.iterrows():
    print(c)
# print the column names 
for c in ufo:
    print(c)    
#iterating through dataframes to print only the index, city and state

for index, row in ufo.iterrows():
        print(index,row.City,row.State)    
        
for key, value in ufo.iteritems():
        print(key,value)  
# how do DataFrames and Series work in regards to seldcting indivitual entires and iteration
# ietrating through series
        

selected_data = table.loc[:,['Country','LandArea']]
selected_data
for index,row in selected_data.iterrows():
     val = row['LandArea']
     if val>2000:
         print(row)
# or---using itertuples        
for row in selected_data.itertuples():
      if row[2]>2000:
          print(row)
          
#just using iloc

for i in selected_data.index:
    val = selected_data.loc[i,'LandArea']
    if val>2000:
         print(selected_data.loc[i,['Country','LandArea']])
   
# FINDING correlation btwn GDP and Birth rate - also using xticks - this is getting cluttered          

import matplotlib.pyplot as plt
import numpy as np
selected_data = table.loc[:,['Country','GDP','BirthRate']]
selected_data
x = np.array(selected_data['GDP'])
y = np.array(selected_data['BirthRate'])
plt.xlabel('GDP')
plt.ylabel('BirthRate')
plt.xticks(selected_data['GDP'],selected_data['Country'])  
plt.scatter(x,y)
plt.xlim(10)
plt.show()        

# FINDING correlation btwn GDP and Birth rate - this is not cluttered   

import matplotlib.pyplot as plt
import numpy as np
selected_data = table.loc[:,['Country','GDP','BirthRate']]
selected_data
x = np.array(selected_data['GDP'])
y = np.array(selected_data['BirthRate'])
plt.xlabel('GDP')
plt.ylabel('BirthRate')
#plt.xticks(selected_data['GDP'],selected_data['Country'])  
plt.scatter(x,y)
plt.xlim(10)
plt.show()       

# compare GDPs of the 10 richest countries
import matplotlib.pyplot as plt
import numpy as np
selected_data = table.loc[:,['Country','GDP']]
sorted__data = selected_data.sort_index(by='GDP',ascending = False)
Richest = sorted__data.head(10)
x = Richest['GDP']
labels = Richest['Country']
plt.pie(x,labels = labels)
plt.show()
        