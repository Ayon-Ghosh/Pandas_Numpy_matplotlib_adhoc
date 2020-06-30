# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:12:46 2019

@author: 140524
"""
# go through: https://jeffdelaney.me/blog/useful-snippets-in-pandas/

# Pandas from youtube
#https://www.youtube.com/watch?v=yzIMircGU5I&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y
# How do I read tabular data - table into Panda

# Video1

import pandas as pd
df = pd.read_csv('C:\python\ayontext')
df
#or directly reading from table

import pandas as pd
#df = pd.read_table('entire URL')
orders = pd.read_table('https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv')
# it will run perfectly because the source data is formatted well
# storing it in a variable and saving it means it saves the table as a dataframe object
orders

# first 5 rows
orders.head()

# passing URL where data is not formatted well by tabs and spaces

import pandas as pd
#df = pd.read_table('entire URL')
movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user')
# run this and see something is wrong
movieuser
# in read_table(URL, sep = tab) sep = tab is the default character. In chipotle there was tabs that is why
# the pandas could print the table perfectly. But in movie users the delimiter was pipe(|) so pandas could 
# separate or delimit. So we have to mention sep = '|'. when comma will be there sep = ',' etc.
movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user',sep='|')
movieuser

# Now the problem in the result is it takes 1st row as header which it should not be because this table doesnt
# not have any header
# so we can add another parameter in the read_table called header = none so the first row is not taken as header

movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user',sep='|', header=None)
movieuser

# now we are adding column name to the tables
# so we have to create a list of columns and pass the parameter to the read_table
import pandas as pd
user_cols = ['user_id','age','gender','occupation','zipcode']
#pass this into read_table
movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user',sep='|', header=None,names=user_cols)
movieuser


# bonus tip:
# skiprows=None parameter in read_table - u can use to skip a specific row or the top row which 
# may be some garbage txt not orderly formatted
#skipfooter=None parameter can be used to skip footer txt in a table if its not required

movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user',sep='|', header=None,names=user_cols,skiprows=range(1, 10))
movieuser.head(10)

# or - skip rows based on conditions

#video 2
# How do I select Panda series from a dataframe

# Pandas hold basically 2 types of object - 1) Dataframe which is a table of rows and column
# in that each column is called a 2)panda series. Though u can have panda series which are not
# part of dataframe but mostly it will be part of

import pandas as pd
uforeports = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',sep=',')
uforeports

#or read_cvs has sep=',' as default unlike sep=tab in read_table
uforeports = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
uforeports.head()
#type method tells the type of object - in this case the object is a dataframe
type(uforeports)

# first way to select a series (column-object) in a dataframe is the bracket method with the column name
uforeports['City']
# the below will tell that the type of object is a series
type(uforeports['City'])

# Second way to select a seriesSelecting using dot(.) notation
# this . method works because panda converts all column names into attributes - a cool trick
uforeports.City
#However if there is a space in the colum,n name for example col name = 'colours reported' thedot(.)
#method wont work. Then bracket only will work
# How to create a new column or series in the dataframe
# lets say we want to add state with the city, this works like string concatenation because 
# dataframe data is all strings. It also works with numbers as well

uforeports.City+uforeports.State
# putting comma space delimiter to make it more readable
uforeports.City+', '+uforeports.State

# Or u can completly create a new series - location using bracket method only
uforeports['Location'] = uforeports.City+', '+uforeports.State
uforeports

#Video 3
# why do some pandas command end with parenthesis and some command dont
# we are going to work with a csv - comma separated data file so we will use read_csv
import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()

# Dataframe is a object. A object has attributes (means descriptions) and method(actions or function)
# attributes doesnt have parenthesis at the end and the method has parenthesis so that
# u can add optional and required parameters within the parenthesis

#for example:
# attribute - shape - gives the # of rows and columns
movies.shape
# attribute datatypes - some are object somea re float and some are strings

movies.dtypes
# action - method - gives descriptive statictics of the table
movies.describe()
# passing optional parameter in describe to only give stats on the dtype objects
movies.describe(include=['object'])
# action - head to only print the first 5 columns or 10-15 whatever u define
movies.head()
# passing parameter in head
movies.head(20)


# video 5
# how do i rename columns in a pandas dataframe

import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',sep=',')
ufo.head()
# to know the column nam only use the column attribute of the dataframe
# prints the columns in form of a list but it is actually index object
ufo.columns
ufo['City']
ufo[['City','State','Time']]
# rename 'Color Reported' and 'shape Reported' to remove space

# option 1 to use a rename method and pass a dictionary as parameter
#The inplace = true parameter sets to affect the current dataframe
ufo.rename(columns = {'Color Reported':'Color_Reported','Shape Reported':'Shape_Reported'},inplace=True)
ufo.columns
ufo.head()

#option 2 - replacing all column names

ufo_cols = ['city','colors_reported','shape_reported','state','time']
ufo.columns = ufo_cols
ufo.columns
ufo.head()

#renaming columns while reading a file
#header = 0 means the existing 0 row already has a header but i am passing new headers (ufo_cols) to override them

ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',names = ufo_cols,header = 0)
ufo.head()

#Replacing spaces with '_" or doing some other replace for all columns at once
# in this case replace '_' with A
# call a string method on the column attribute 
# on that string method call the replace method and pass the replace parameter
type(ufo.columns)
#this single line code is very useful when there are 100s of columns
ufo.columns = ufo.columns.str.replace('_','A')
ufo.columns

#video 6
#how do i remove columns from a Pandas dataframe

import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',sep=',')
ufo.head()
#let delete colors_reported column
# option 1 - use drop method
ufo.columns
ufo.drop('Colors Reported',axis = 1,inplace = True)
ufo.head()

# dropping multiple columns

ufo.drop(['City','State'],axis = 1,inplace = True)
ufo.head()
# dropping rows instead of column menthod use axis 0
#drop row 5
ufo.drop(3,axis=0,inplace = True)
ufo.head()
#dropping multiple rows - here dropping 0 and 7-th row
ufo.drop([0,7],axis=0,inplace = True)
ufo.head(10)

#video 7
#how do i sort a dataframe or series

import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()
movies.columns
# sorting through a series - Title column within the movies
movies.title.sort_values()

#or
movies['title'].sort_values()

#  -with decending

movies['title'].sort_values(ascending = False)

# sorting a dataframe by a series

movies.sort_values('title')
movies.sort_values('duration',ascending = 'False')

# sorting by multiple columns 
movies.sort_values(['content_rating','duration'])

#syntax:

# =============================================================================
# data.sort_values(by=['col1','col2'],ascending=[True,False],inplace=False)
# =============================================================================


# Go through this example below from Hackarearth

data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'corned beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon'
}
data['food']=data['food'].str.lower()
data
data['animal']=data['food'].map(meat_to_animal)
data

# or
data['animal']=data['food'].map(str.lower).map(meat_to_animal)
data

# multiply ounces column *10

data['ounces'] = data['ounces']*10
data

#--------------
import numpy as np
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
#replace -999 with NaN values
data.replace(-999, np.nan,inplace=True)
data

#We can also replace multiple values at once.
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace([-999,-1000],np.nan,inplace=True)
data

#Renaming columns and index/rows all at once
import pandas as pd
import numpy as np
data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])
data
data.rename(index={'Ohio':'SanF'}, columns = {'one':'one_p','two':'two_p','three':'three_p'},inplace=True)
data
#video 8
#how to filter rows of a panda dataframe by column value

import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()

# filter this dataframe only for movies whose duration is >200 mins which means
#to only select rows whose duration is more than 200 min
# the below explanation is step by step process that will lead to a concise code

# Understanding Boolean - True, False: for example: type(False) will result in Bool
type(False)

# so now we want to create a list of Booleans of the same length as the Duration series
# within the Data frame that will show True for all>200 and False for all <=200
#like [False,True,False...........etc.]
#--------*******
booleans = []
for length in movies.duration:
    if length>=200:
        booleans.append(True)
    else:
        booleans.append(False)
booleans        
# see length of boolean list is same as the length of the dataframe or the series duration within the dataframe
len(booleans)
     
# now converting booleans list to a pandas series
is_long = pd.Series(booleans)
# see is_long now have indexing
is_long

# pass is_long to the DataFrame Movies using Bracket notation
# Normally braacket notation is used to select a specific column in dataframe
# panda is intelligent to print a dataframe of only those rows in which the durtaion is >200 mins
#this is panda specific feature that Python can't do
movies[is_long]
#----*****

# now the concise way to conde all that is there within the ---****
# below replaces the forlooop
# this is not present in python. Pandas return a whole series unlike python which returns a single result
is_long = movies.duration>=200
movies[is_long]

# even more concise
movies[movies.duration>=200]

# Now we want to just print genre of the movies that are more than >=200
#this prints out the selected dataframe based on the condiion - duration >200
#just like we select a column using [] or . we can do the same in selecting the duration only as below
movies[movies.duration>=200].genre
#or
movies[movies.duration>=200]['genre']

#sometimes if the above code doesnt run then use the loc method which is a better practise today
#the way .loc works is it enables use to select rows and columns by labels/indexwes

movies.loc[movies.duration>=200]

movies.loc[movies.duration>=200,'genre']

# how to apply multipe criteria to a pandas dataframne

import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()

# doing a previous step to select only movies that are >200 min
movies[movies.duration>=200]
# now adding one more filter - movies longer than 200 min which are of genre - drama
#use logical Or Operator
#True or True is True///True or False is True///Fale or True is True///False or False is False

# this is not going to work
#movies[movies.duration>=200 and movies.genre =='Drama']
# we will now tweak it to work by
# introducting the evaluation order - such as which worder the filter will be applied
# first the duration then the genre will be the order


movies[(movies.duration>=200) & (movies.genre =='Drama')]

# similarly use | -'pipe' character for Or

# now adding more OR condition such as movies whose genre is either crime or drame or action

movies[(movies.genre=='Crime') | (movies.genre =='Drama') |(movies.genre =='Action')]

# better way to do the above is to use a isin method and pass a list of the options

movies.genre.isin(['Crime','Drama','Action'])

#video 10

#Reading only first 2 columns from a csv file and ignore others

import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
# modify parameters of the read_csv FUNCTION add optional parameter usecols to select the required column 
#only

ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',usecols=['City','State'])
ufo.head()
# referencing coilumns by position - city is 0 column and Time is column 4

ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',usecols=[0,4])
ufo.head()

# pulling up only the first few rows when openning
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',nrows=3)
ufo
ufo.columns

# how do DataFrames and Series work in regards to seldcting indivitual entires and iteration
# ietrating through series

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
        
        
#whats the best way to drop every non-numeric column from a data frame

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink.dtypes     

# keep only the numercic columns

import numpy as np
# use a dataframe method select.dtypes
drink.select_dtypes(include=[np.number])
drink.select_dtypes(include=[np.number]).dtypes 

# now selecting object and float columns

drink.describe(include=['object','float64'])

#video 11
#how do i use axis parameter in pandas
#axis = 0 is for row - direction to go from top to down across row axis

# axis =1 is for column - direction to go from left to right across column axis

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
# drop a particular columan

drink.drop('continent',axis=1).head()

# drop a specific row

drink.drop(2,axis=0).head()

# using axis to find stats
# this gives us the mean of numeric columns. The default parameter of mean is axis = 0
#so it goes from left to right an in each column it reaches which is numeric it goes top to down to 
#give the mean for each columns
drink.mean()

# or for avergae of each  rows
drink.mean(axis=1).head()

# alias for axis numbers such as index for 0 and column for 1

# drink.mean(axis='columns') or drink.mean(axis='index')

# video 12
# how to use string method in pandas

#string methods such as upper()

import pandas as pd
orders = pd.read_table('https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv')
orders.head()
# making item name upper case

orders.item_name.str.upper().head()

# to check the presence of a substring - use contain method which returns a series of booleans
# true or false - true if substring present, false if not
orders.item_name.str.contains('chicken')
# now to publish the dataframe with rows containg chicken only
#put the above code in the dataframe
orders[orders.item_name.str.contains('Chicken')]

# chain together string methods
# remove all bracket characters in choice_description column

#------------Refer to API 

orders.choice_description.str.replace('[','')

# chain the above with another str replacement or any other str method

orders.choice_description.str.replace('[','').str.replace(']','')

#Many panda string method witl use regex
# using regex in the above code uing char class []
orders.choice_description.str.replace('[\[\]]','')

# video - 13

#how to change the data type of a pandas series

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink.head()
drink.dtypes
# gives the following input
#country                          object
#beer_servings                     int64
#spirit_servings                   int64
#wine_servings                     int64
#total_litres_of_pure_alcohol    float64
#continent                        object
#dtype: object

#lets now change the beer serving column to float64 rather than int 
# use series method to do so - astype()

drink.beer_servings.astype(float)
# now u can over ride the existing beer serving column by - 
drink['beer_servings'] = drink.beer_servings.astype(float)
drink.dtypes

# usefulness of this is when u have a datafile which has a column of numbers in
#string format on which u want to do mathematical calculation
#there u will have to convert that coulmn to int or float to do maths on it

#how to define the dtype of each column while reading the csv

# change the data type during read_csv

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv',dtype = {'beer_servings': float})
drink.dtypes

# another example
import pandas as pd
orders = pd.read_table('https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv')
orders.head()
# the item_price column has values such as $3.99 which is string / object format. You can find out it
# is object format if you do orders.dtypes. Now to do any math operation we will have to convert it into int or float
# in order to do that we will have to remove $ and the  cast the type to float
orders.item_price.str.replace('$','').astype(float)
# finding mean item price
orders.item_price.str.replace('$','').astype(float).mean()

#converting boolean array to 0 and 1 this is very imp in machine learning

import pandas as pd
orders = pd.read_table('https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv')
orders.head()
orders.item_name.str.contains('chicken').astype(int)


#video 14
# when should i use a groupby function

df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
df
grouped = df['data1'].groupby(df['key1'])
grouped.mean()

#----or


dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
df
df[:3]

#-------------or
import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv',dtype = {'beer_servings': float})
drink.head()
# find out the average beer serving across all countries
# in order to do that groupby - buy countries and then find mean with key as the group name

drink.beer_servings.mean()
drink.groupby('country').beer_servings.mean()

# now average of beer_serving of each continent
drink.groupby('continent').beer_servings.mean()

# how does the above work -step by step code - decluttering the chain method

# first let filter the data frame by Africa for a moment
import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv',dtype = {'beer_servings': float})
drink[drink.continent=='Africa']
# getting mean of Africa
drink[drink.continent=='Africa'].beer_servings.mean()
# the above groupby continent is repreating the filtering method for each continnent

# see the entire below piece of code is done in 1 single line code in groupby function

list_unique = []
for i in drink.country:
    if i not in list_unique:
        list_unique.append(i)
list_unique 
list_mean= []
d={}
for i in range(len(list_unique)):
    temp = drink[drink.country==list_unique[i]].beer_servings.mean()
    list_mean.append(temp)
list_mean    

# or finding unique coubtry list:

drink.country.unique()

# printing in a dict
for i in range(len(list_unique)):
    d[list_unique[i]] =list_mean[i]
d    
# or

import pandas as pd
country = pd.Series(list_unique)
mean = pd.Series(list_mean)
country_mean = pd.DataFrame({'country':country,'mean':mean})
country_mean

#when to use groupby in general?
# u use groupby when u want go analyse some panda series by some category
# in this case - panda series is beer_serving/// category is continent

# using Max function

drink.groupby('continent').beer_servings.max()

# using Min function

drink.groupby('continent').beer_servings.min()
drink.groupby('continent').groups

# .agg allows us to specifiy multiple aggregation functions at once

drink.groupby('continent').beer_servings.agg(['count','min','max','mean'])
import matplotlib.pyplot as plt
#%matplotlib inline
drink.groupby('continent').beer_servings.agg(['count','min','max','mean']).plot(kind='bar')

drink.groupby('continent').mean().plot(kind='bar')

# video = 15
#how do i explore a panda series
import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()
movies.columns
movies.dtypes
# gives all the decriptions of genre
movies.genre.describe()
# another use full method value_counts - shows the counts/ repitation of each unique value 
#in the series
movies.genre.value_counts()
# turning the counts into percentages
movies.genre.value_counts(normalize=True)

#the output of series and dataframe methods are series and dataframe objects
# to find a list of the unique values only

movies.genre.unique()

# to find the total number of unique values
movies.genre.nunique()

# cross tabulation 
pd.crosstab(movies.genre,movies.content_rating)

# working on a numeric columns
movies.duration.describe()
movies.duration.value_counts()

import matplotlib.pyplot as plt
movies.duration.plot(kind='hist')

movies.genre.value_counts().plot(kind='bar')

# video 16
# how to handle missing values
import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
ufo.tail()
ufo.columns
# the tail throws up a few missing values..Nan - Not a number , the missing value in csv
#is auto converted to Nan..

# working with missing value
# option 1 - work with isnull method - returns False is value present, true if missing
ufo.isnull().tail()

#notnull will return true if present false if not present
ufo.notnull().tail()
#number of missing values in each columns
ufo.isnull().sum()

# how did the above work

# shown step by step
#applying sum function on a boolean series works with panda assigning value of 1
#1 for true and 0 for false and then it sums it

pd.Series([True,False,True]).sum()

# Sum on a dataframe sum operates on axis =0 which means opreation across the rows for each columns
#down
# option 2 by boolean filters
#isnull turns out to series method as well
ufo[ufo.City.isnull()]
# there is no one answer to deal with missing values
# it is based on the data set and what type of analytics u want to do

#option 1 scenerio to drop missing values
# use dropna() method
# in that method how='any' is a default parameter - u dont have to mention it
# but it means that we should drop any row that has null values
ufo.shape
ufo.dropna(how='any')
# or
ufo.dropna()
# look at the shape now it drops all rows which has a null in any single column or multiple
#columsn
ufo.dropna().shape

# only drop a row if all of its values are missing - how='any' parameter will now have to be changed to
#how='all' 
ufo.dropna(how='all').shape

# i want to drop a row if there is Nan in any or just couple of specific columns
# examples - if there is Nan in City and Shape_Reported then drop it

ufo.dropna(subset=['City','Shape Reported'],how='any').shape

#drop a row if Nan is there in both city and share reported row

ufo.dropna(subset=['City','Shape Reported'],how='all').shape

# filling missing values

# first step count the number of missing values in a column
# see this does not give the count of Nan
ufo['Shape Reported'].value_counts()
ufo['Shape Reported'].value_counts().shape
# in order to get the value of Nan as well
ufo['Shape Reported'].value_counts(dropna=False).shape


# fill those Nan with various
#use fillna method
#in place = true will make changes in final dataframe
ufo['Shape Reported'].fillna(value='VARIOUS',inplace=True)
ufo
ufo['Shape Reported'].value_counts()

#Video 17
#Pandas index - part 1


import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink.head()


# attribute index - index also sometimes known as row levels
# index is object in panda that returns the column names or the row levels or the index in a series
drink.index
drink.columns
#column headers and the index/row level are not part of data frame
# default value of coilumns are also integers
# The index exist for !) selection 2) identification 3) alignment

# identification

# filtering dataframe - see even after filteriung the orginal dataframe the original row levels 
#are kept intact so u can see which rows u are working with

drink[drink.continent == 'South America']

#selection

#if we want a specific cell value use loc

drink.loc[23,'beer_servings']

# even more smarter method - we dont have to mention 23 
# in place = true doesnt require u to asign a new dataframe
# see the country series has now beccome the index

drink.set_index('country',inplace=True)

drink.head()
drink.columns
drink.shape
drink.loc['Brazil','beer_servings']

#moving the country back to a column

drink.reset_index(inplace=True)
drink.head()

# the output of drink.describe is a dataframe and it has a index

drink.describe()
drink.describe().index
drink.describe().columns
drink.describe().loc['25%','beer_servings']

# video 18
# pandas index part 2
# series index
import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink.head()
drink.continent.head()

# lets say we dont use the default index of the data frame

drink.set_index('country',inplace=True)
drink.head()
drink.continent.head()
# see the value_count also outputs a series which has index
drink.continent.value_counts()
#we can use that index to select values
drink.continent.value_counts().values
#or finding the index of africa
drink.continent.value_counts()['Africa']

# sorting

drink.continent.value_counts().sort_index()

# alignment
#lets create a small dataset of polupation of 2 countries and nultiply them with average 
#beerserving of those 2 countries

people = pd.Series([3000000,50000],index = ['Albania','Andorra'], name = 'population')
people

# see how panda aligns this small series with the beer serving series of the drinks dataframe
drink.beer_servings*people

#how to take people series and add it to the drink dataframe

# use concat method - the way to control that is with the axis parameter
# contact with axis = 0 contact rows while 1 put columns side by side

pd.concat([drink,people],axis=1).head()

# video 19
#How do i select multiple rows and columns from a panda datafram

#difference btwn loc, iloc, and ix - all dataframe methods for seldcting rows and columns


import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
# same things that head() does can be done with loc
# loc is used for filtering rows and selecting columns by label as in the rowlabel or column names
# selecting row 0 and all columns
# row selection
ufo.loc[0,:]
ufo.loc[[0,1,2,3],:]
# or
# see below the result - loc is inclusive of starting and ending number unlike range
ufo.loc[0:10,:]
# same as writing - 
# not mentioning row here as panda will assume all columns - not recommended
ufo.loc[0:2]

# column selection

ufo.loc[:,'City']
#or
ufo.loc[:,['City','State']]

# all rows city through state
ufo.loc[:,'City':'State']

# combining row and column selection

ufo.loc[0:10,'City':'State']

# same thing can be accomplished by

ufo.head(10).drop('Time',axis=1)

#using loc with boolean conditions

# previously using boolean to filter - lets see all sitings for Oakland

ufo[ufo.City=='Oakland']
# doing the same thing with loc
ufo.loc[ufo.City=='Oakland',:]

#or..selecting oakland rows and corresponding states column

ufo.loc[ufo.City=='Oakland','State']

#-----iloc----filtering rows and selecting columns by integer positions
# exlcusive of max number but inclusive of first number just like the range function

ufo.iloc[:,[0,3]]

# or a range

ufo.iloc[:,0:3]

ufo.loc[[0,1,2,3],:]

# ix....allows use u to mix labels and integers - kind of a blend btwn loc and iloc
# USE IX only when u have to mix row level and positional index

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv',index_col='country')  
drink.head()

# using ix - label of row and position of column


drink.ix['Albania',0]

# or

drink.ix[1,'beer_servings']
drink.ix['Albania':'Andorra',0:2]

# if u have a string index as in this case - it will print columns of position 0 and 1 (exclusive of 
#last number) but if row levels are numbers then it takes the labels itself not positional as below example
#prints 3 rows -0-1-2 row levels but 2 columns - string column - position 0-1
ufo.ix[0:2,0:2]

# VIDEO 20
# when should I use in place parameter in pandas

import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
ufo.shape
ufo.drop('City',axis=1)
ufo.head()
# check even after dropping the city column is not removed. The reason is because the drop
#method has this inplace=False by default which means that this operation won't affect the 
#main dataframe by default. In order to make a permanent drop we will have to set in place = True
#so it affects the dataframe. The reason it is set False by default is for u to experiment with data
# before making a parmanent change
ufo.drop('City',axis=1,inplace=True)
ufo.head()
#similarly the dropna for dropping missing values also have a option parameter inplace
#set to default as Fault. Nothing has been lost from dataframe
#u can check the shape to see how many rows will be deleted and then decide if
#u want to delete them
ufo.dropna(how='any').shape

ufo.set_index('Time',inplace=True)

# using assignment statements also u can by pass using inplace = false

ufo = ufo.set_index('Time',inplace=True)
ufo.tail()

# both the above will give the same result

# Fill mising values taking advantage of the inplace = false parameter
# so the main dataframe is not changed
#always check method parameters
ufo.fillna(method = 'bfill').tail()

#check if method =ffill is suitable
ufo.fillna(method='ffill').tail()

#it seems ffill is suitable more in this situation
#u could experiment in this way becaue inplace = false and u r not doing any change
#to the original dataframe

#Video 21
#how do i make my dataframe smaller and faster
import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink.head()
#info dataframe method tells us more abt dataframe - check it
drink.info()
#look at the dtypes and memory usage
#In dtypes there are objects. objects are usually strings but can also be python list, dicts etc..
#whatever is used to built the dataframe. Python user object reference for all of them
# now since info method is run very fast, python really doesnt go to those obecjt location and cal
#memory size. In stead python just gives the memory of the reference to the object as in the memory of
#the object address. That is why memory usage is said to ATLEAST 9.1+ kb - + because orginal memory for storing
#object is more than memory used for storing their address

#forcing panda to cal the true memory usage - using memory_usage='deep' parameter
#see original memory size is 30.4 Kb

drink.info(memory_usage='deep')

# how much space is each column taking

drink.memory_usage(deep='true')

#memory_usage method gives a series so we can sum i to get the total mem which will be close
#or equal to info(deep=true)
drink.memory_usage(deep='true').sum()

# Bottomline - object column takes a lot of space. how can i be more efficient with object column
#int columns takes up a lot less memory. take the drink dataframe. the continnent column
#has many repeatations. if we assign a int to each continent and replace the object continent column
#with int then  it will be more space efficient
#we still have to store a look up table. but u store in once and can reference it 

#u can do that using CATEGORY dtype
sorted(drink.continent.unique())

drink['continent'] = drink.continent.astype('category')
# check the dtype now
drink.dtypes
drink.continent.head()
#see continent series has been changed to int
drink.continent.cat.codes.head()
# see now memory usage has reduced
drink.memory_usage(deep=True)
# repeatoing this for country
drink['country'] = drink.continent.astype('category')
drink.dtypes
drink.memory_usage(deep=True)
# the memory usage hasnot reduced a lot. because countries are all different
# so for 193 diff countries u need 193 int categories and then u need a look up table for 193 objects-cat 
#mapping. so ultimately the memory usage increases. so only use categories when a string/object
#column has many duplications

#--

df=pd.DataFrame({'ID':[100,101,102,103],'quality':['good','v good','good','excellent']})
df
#sort this dataframe by quality
df.sort_values('quality')
# but u see there is a logical ordering to these qualities/ excellenent, V good, nd then good
#not like sorted above
# so how to tell pandas the logical ordering
# u can tell the logical ordering of the qualities by using categories
df['quality'] = df.quality.astype('category',categories=['good','v good','excellent'],ordered= True)
df.quality
# now it will sort them in logical order
df.sort_values('quality')

# best thing is u can also use boolean conditions once u assign categories
#lets say we want all rows where the quality is better than good
# assigning categories helps u to use comparision operators
df.loc[df.quality>'good',:]

# Video 24 - How do I create dummy - (also known as) indicator variable in pandas

#encode unordered categorical features 

import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')
train.head()
train.columns
train.shape
train.Sex

# objective is to create a dummy variable for the sex column - 0 for Female and 1 for Male
#one way to do it will be to use series - map method

train['Sex_Male'] = train.Sex.map({'female':0,'male':1})
train['Sex_Male']
train.head()

# the more flexible way to do it is by using pd.to_dummy

pd.get_dummies(train.Sex)

# if u have N possible values of categorical variable, we need N-1 dummy variables
#Having one less will staill capture all the information.
# dropping the first column


pd.get_dummies(train.Sex,prefix='Sex').iloc[:,1:].head()
train.head()
# -----we achieved the same result in both the process

#but the pd.get_dummies gets very useful when there are many categorical variable such as below

train.Embarked.value_counts()
pd.get_dummies(train.Embarked, prefix = 'Embarked').head()
pd.get_dummies(train.Embarked, prefix = 'Embarked').head().iloc[:,1:]
# see the out put of the below head(). The new dummy var columns are not there
# so we will have to concat
train.head()
embarked_dummies = pd.get_dummies(train.Embarked, prefix = 'Embarked').head().iloc[:,1:]
train_new = pd.concat([train, embarked_dummies])
train_new.columns


# how to pass a dataframe to pd.get_dummies - before we passed a series

import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')
train.head()
train.columns
# we pass the entire dataframe and pass the columns on which we should convert to dummies
pd.get_dummies(train, columns = ['Sex','Embarked'])
# see the original sex and embarked columns are dropped
pd.get_dummies(train, columns = ['Sex','Embarked']).columns

# Now the problem we are dealing with is- how to drop the N-1 dummy columns such as 
#dummy female and dummy Embarked_c....WE can use iloc method but if we have 40 different dummies
# then very difficult to use iloc. we will then use the following attribute in 
#get_dummies - drop first

pd.get_dummies(train, columns = ['Sex','Embarked'], drop_first = True)
pd.get_dummies(train, columns = ['Sex','Embarked'], drop_first = True).columns

# see above it dropped the ppropriater dummy columns..This is done without using iloc
# and without using concat....super smart

#video - 25: How do I work with date and times in Panda

#there is a lot of pwowerfull time series functions. we will use a few.

import pandas as pd
ufo = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv',sep=',')
ufo.head()
# what if I wanted to analyse the sitings by year or by time of day

ufo.dtypes

#see above the time column is a object - which means its stored as strings
# we can do string formatting/ slicing to just select the year or the time but thats not 
#smart

ufo.Time.str.slice(-5,-3).astype(int).head()
# but this apprach will easily break - lets use a better methodology

# solution here is to convert the time column into pandas date time format

ufo['Time']=pd.to_datetime(ufo.Time)
ufo.head()
ufo.dtypes
# see the dtype of the Time column has changed and the format of the time column 
# values has changed
# this conversion to date time format exposes some really great attributes such as below
# the below pulls out the hour
ufo.Time.dt.hour
# or lets find the name of the day of the week when each ufo was sited

ufo.Time.dt.weekday_name

# or a number version of he weekday

ufo.Time.dt.weekday

# or what day of the year

ufo.Time.dt.dayofyear

# now lets pass a string to the datatime function and it outputs what is known 
# as a timestamp

pd.to_datetime('1/1/1999')
ts = pd.to_datetime('1/1/1999')
# now use the time stamp to compare and select only those rows whose time is greater
# or equal to 1/1/1999
ufo.loc[ufo.Time>=ts]

#or

ufo.loc[ufo.Time>=ts,:]

# doing mathematical operations with time stamp
#this will tell me the latest time stamp in the time series
ufo.Time.max()
#to find the range of period of siting, I can even do

ufo.Time.max() - ufo.Time.min()

# time delta object also has attributes such as days

(ufo.Time.max() - ufo.Time.min()).days

#plotting number of ufo reportings by year

import matplotlib.pyplot as plt

ufo.Time.dt.year.unique()
ufo.Time.dt.year.value_counts()
# or if we want to sort it by index
ufo.Time.dt.year.value_counts().sort_index()


plt.xlabel('Year')
plt.ylabel('Number of sigthings')
plt.plot(ufo.Time.dt.year.unique(),ufo.Time.dt.year.value_counts())

# or

plt.plot(ufo.Time.dt.year.unique(),ufo.Time.dt.year.value_counts().sort_index())


# do the whole things as
import matplotlib.pyplot as plt
ufo['year'] = ufo.Time.dt.year
ufo.head()
ufo.year.value_counts().sort_index()
ufo.year.value_counts().sort_index().plot

# video 26: how to remove duplicate rows in pandas

# how to count the number of duplicates in a particular column in a data frame

import pandas as pd
user_cols = ['user_id','age','gender','occupation','zipcode']
#pass this into read_table
movieuser = pd.read_table('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/u.user',sep='|', header=None,names=user_cols,index_col='user_id')
movieuser.head()
movieuser.shape

# we want toi identify duplicate zipcodes - use the duplicated method 
# it returns a bool series of true and false. It returns a true only when the value is duplicated 
# before
# but we still dont know which previous one is duplicated
movieuser.zipcode.duplicated()

# we can count the number of duplicates
# panda converts the the trues to 1s and false to 0 and then sums it up
movieuser.zipcode.duplicated().sum()

# now lets look at the duplication in the data frame as a whole instead of 1 column or series
# this will return a True only if the entire row is identical to it - previously - and now duplicated
movieuser.duplicated()
movieuser.duplicated().sum()

# to see the rows which are duplicated

movieuser.loc[movieuser.duplicated()]

#actually the default paramneter in duplicate function is duplicated (keep = 'first')
movieuser.duplicated(keep='first')
#the logic or default logic even if u dont mention keep= first is
#mafk duplicate as true except for the first one
# u can change the logic to keep='last' to mark the first occurance as duplcate and keep the last ones

movieuser.loc[movieuser.duplicated(keep='last')]

# or we can use the logic as keep=false which is it will mark all the duplicates as true and show the false
#thus it shows both the orginal and duplicates
movieuser.loc[movieuser.duplicated(keep=False)]

# we want to drop the duplicate from the dataframe
#keep =first is default - we can use keep = last or false
# we can add the inplace = true as well
movieuser.drop_duplicates()
movieuser.drop_duplicates().shape
# see it dropped the 7 duplicate rows

# what if we only wanted to consider certain columns when identifying duplicates

# example below - we are assuming age and zipcode is a unique identifier and hence we wantb to
# find duplicates in those

movieuser.duplicated(subset=['age','zipcode']).sum()

# the above output means there are 16 rows where age and zipcode was together duplicated

#similarly dropping them

movieuser.drop_duplicates(subset=['age','zipcode']).shape


#Video 27: How do i avoid a setting with copy warning

import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()
#lets look for missing values in content_rating

movies.content_rating.isnull().sum()

#lets do the value counts of each rating

movies.content_rating.value_counts()

# Not rated is of no use - lets convert those to NaN, so now NaN should be 65+3=68
#----
#selecting the dataframe whose content_rating column is Unrated
movies[movies.content_rating =='UNRATED']
#selecting the column only with unrated values
movies[movies.content_rating =='UNRATED'].content_rating

#converting unrated to NaN
import numpy as np
movies[movies.content_rating =='UNRATED'].content_rating = np.nan

# see we recive a future copy warning. The reason is -
#movies[movies.content_rating =='UNRATED']-------------------is called Get in Pandas
#content_rating = np.nan-----------------is called set in pandas
# pandas doesnt know that Get creates a view or a copy in order to set a value to it
#hence the warning.
#Use .loc method to avopud that


movies.loc[movies.content_rating=='NOT RATED','content_rating'] = np.nan
movies.content_rating.isnull().sum()

# now it worked because Loc solved it by turning it from 2 operations - Get and set to just 1 operation
#set

#we want all movies with star_rating >9

topmovies = movies.loc[movies.star_rating>=9, :]
topmovies

#correcting the duration of shwashank redemption
 
movies.loc[0, 'duration'] = 150
topmovies

