# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:27:36 2019

@author: 140524
"""

#Top 25 Panda Tricks
#https://www.youtube.com/watch?v=RlIiVeig3hc

#https://www.dataschool.io/python-pandas-tips-and-tricks/

#Trick1: Show installed versions

import pandas as pd
import numpy as np
pd.__version__
pd.show_versions()

#Trick2: Create an example DataFrame
#pass a dictionary to the DataFrame constructor, in which the dictionary keys 
#are the column names and the dictionary values are lists of column values:
df = pd.DataFrame({'col1':[100,200],'col2':[300,400]})
df
#Now if you need a much larger DataFrame, the above method will require way too 
#much typing. In that case, you can use NumPy's random.rand() function, tell it 
#the number of rows and columns, and pass that to the DataFrame constructor:
df= pd.DataFrame(np.random.randn(4,8))
df
np.random.randn(4,8)
#That's pretty good, but if you also want non-numeric column names, you can coerce 
#a string of letters to a list and then pass that list to the columns parameter:
df= pd.DataFrame(np.random.randn(4,8),columns=list('abcdefgh'))
df

#Trick3: Rename columns
df = pd.DataFrame({'col1':[100,200],'col2':[300,400]})
df
df = df.rename({'col1':'col_1','col2':'col_2'},axis='columns')
df
#The best thing about this method is that you can use it to rename any number of columns, 
#whether it be just one column or all columns.
#Now if you're going to rename all of the columns at once, a simpler method is just 
#to overwrite the columns attribute of the DataFrame:

df.columns = ['colA','colB']
df

#Now if the only thing you're doing is replacing spaces with underscores, an even 
#better method is to use the str.replace() method, since you don't have to type out 
#all of the column names:

df.columns = df.columns.str.replace('','_')
df

#Finally, if you just need to add a prefix or suffix to all of your column names, 
#you can use the add_prefix() method.

df.add_prefix('X_')
df

#Reversing row order

import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
drinks.columns
drinks.loc[::-1].head()

#What if you also wanted to reset the index so that it starts at zero?
#You would use the reset_index() method and tell it to drop the old index entirely:

drinks.loc[::-1].reset_index(drop=True).head()

#Reverse column order

#Similar to the previous trick, you can also use loc to reverse the left-to-right 
#order of your columns:

drinks.loc[:, ::-1].head()

#Trick 6: Select columns by data type

drinks.dtypes
#Let's say you need to select only the numeric columns. You can use the select_dtypes() method:

drinks.select_dtypes(include='number').head()

#This includes both int and float columns.
#You could also use this method to select just the object columns:

drinks.select_dtypes(include='object').head()

#ou can tell it to include multiple data types by passing a list:
drinks.select_dtypes(include=['number','object','category','datetime']).head()

#You can also tell it to exclude certain data types:
drinks.select_dtypes(exclude='number').head()

#Trick 7: convert strings to numbers

df=pd.DataFrame({'col_1':['1.1','2.3','4.6'],'col_2':['4.4','5.6','8.9'],'col_3':['1','2','--']})
df
df.dtypes
df = df.astype({'col_1':'float', 'col_2':'float'})
df
df.dtypes
#However, this would have resulted in an error if you tried to use it on the 
#third column, because that column contains a dash to represent zero and pandas 
#doesn't understand how to handle it.

#nstead, you can use the to_numeric() function on the third column and tell 
#it to convert any invalid input into NaN values:

pd.to_numeric(df.col_3,errors='coerce')

#If you know that the NaN values actually represent zeros, you can fill them with 
#zeros using the fillna() method:
pd.to_numeric(df.col_3,errors='coerce').fillna(0)

#Finally, you can apply this function to the entire DataFrame all at once by 
#using the apply() method:

df=df.apply(pd.to_numeric,errors='coerce').fillna(0)
df.dtypes
df
# or just the short cut
df1 = pd.DataFrame({'colA':['1.1','2.3','4.6'],'colB':['4.4','5.6','8.9'],'colC':['1','2','--']})
df1=df1.apply(pd.to_numeric,errors='coerce').fillna(0)
df1
df1.dtypes
#Trick8: Reduce DataFrame size

#pandas DataFrames are designed to fit into memory, and so sometimes you need 
#to reduce the DataFrame size in order to work with it on your system.
#Here's the size of the drinks DataFrame:

drinks.info(memory_usage='deep')

#You can see that it currently uses 30.4 KB. If you're having performance problems 
#with your DataFrame, or you can't even read it into memory, there are two easy steps 
#you can take during the file reading process to reduce the DataFrame size.
#The first step is to only read in the columns that you actually need, which we 
#specify with the "usecols" parameter:

cols = ['beer_servings', 'continent']
small_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols)
small_drinks.info(memory_usage='deep')

#The second step is to convert any object columns containing categorical data to the 
#category data type, which we specify with the "dtype" parameter:
dtypes = {'continent':'category'}
smaller_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols, dtype=dtypes)
smaller_drinks.info(memory_usage='deep')
#By reading in the continent column as the category data type, we've further 
#reduced the DataFrame size to 2.3 KB.Keep in mind that the category data type 
#will only reduce memory usage if you have a small number of categories relative 
#to the number of rows.


#Trick 9: Build a DataFrame from multiple files (row-wise)
from glob import glob
#You can pass a pattern to glob(), including wildcard characters, and it will return a list of all files that match that pattern.
#In this case, glob is looking in the "data" subdirectory for all CSV files that 
#start with the word "stock":

stock_files = sorted(glob('C:/python/stock*'))
stock_files

# =============================================================================
# glob returns filenames in an arbitrary order, which is why we sorted the list 
# using Python's built-in sorted() function.
# We can then use a generator expression to read each of the files using read_csv() 
# and pass the results to the concat() function, which will concatenate the rows 
# into a single DataFrame:
# 
# =============================================================================
pd.concat((pd.read_csv(file) for file in stock_files))

#Unfortunately, there are now duplicate values in the index. To avoid that, we 
#can tell the concat() function to ignore the index and instead use the default integer index:

pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)


# #Trick 10: Build a DataFrame from multiple files (column-wise)
# same process as rows just add = axis = 'columns'
pd.concat((pd.read_csv(file) for file in stock_files), axis = 'columns')

#Trick 11: Split a dataframe into 2 random subsets

#Let's say that you want to split a DataFrame into two parts, randomly assigning 
#75% of the rows to one DataFrame and the other 25% to a second DataFrame.
#For example, we have a DataFrame of movie ratings with 979 rows:
import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()
len(movies)
#We can use the sample() method to randomly select 75% of the rows and assign them to the "movies_1" DataFrame:

movies_1= movies.sample(frac=0.75,random_state=1234)
#Then we can use the drop() method to drop all rows that are in "movies_1" and 
#assign the remaining rows to "movies_2":

movies_2 = movies.drop(movies_1.index)

len(movies_1)+len(movies_2)

#And you can see from the index that every movie is in either "movies_1":

movies_1.index.sort_values()
movies_2.index.sort_values()

#Keep in mind that this approach will not work if your index values are not unique.

#Filter a data frame by multiple categories

import pandas as pd
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
movies.head()
movies.columns

#selecting genre 

movies.genre

#unique list of genre

movies.genre.unique()

#If we wanted to filter the DataFrame to only show movies with the genre Action 
#or Drama or Western, we could use multiple conditions separated by the "or" operator:
movies[(movies.genre =='Drama')|(movies.genre =='Action')|(movies.genre =='Western')].head(50)
# OR better with isin
movies[movies.genre.isin(['Action','Drama','Western'])].head(50)

#And if you want to reverse this filter, so that you are excluding (rather than including) 
#those three genres, you can put a tilde in front of the condition:

movies[~movies.genre.isin(['Action','Drama','Western'])].head(50)
movies[~movies.genre.isin(['Action','Drama','Western'])].head(50).genre.unique()

#This works because ~tilde is the "not" operator in Python.

#Trick 13: Filter a dataframe by largest categories

#Let's say that you needed to filter the movies DataFrame by genre, but only include the 3 largest genres.

movies.columns
movies.groupby('genre').title.agg('count').sort_values(ascending=False)
# or


#The Series method nlargest() makes it easy to select the 3 largest values in this Series:
movies.genre.value_counts().nlargest(3)
#And all we actually need from this Series is the index:

movies.genre.value_counts().nlargest(3).index
# or
genres = list(movies.genre.value_counts().nlargest(3).index)
genres

#Finally, we can pass the index object to isin(), and it will be treated like a list of genres:

movies[movies.genre.isin(movies.genre.value_counts().nlargest(3).index)]

# or

movies[movies.genre.isin(genres)]

#Trick 15: Handling missing values

import pandas as pd
ufo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv')
ufo.head()

#You'll notice that some of the values are missing.
#To find out how many values are missing in each column. you can use the isna() method and then take the sum():
ufo.isna().sum()

#isna() generated a DataFrame of True and False values, and sum() converted all 
#of the True values to 1 and added them up. Similarly, you can find out the percentage 
#of values that are missing by taking the mean() of isna():

ufo.isna().mean()
ufo.columns
#If you want to drop the columns that have any missing values, you can use the dropna() method:
ufo.dropna(axis='columns').head()
#drop rows where all values are nan
ufo.dropna(how = 'all').head()

#drop rows where any of the values is nan
ufo.dropna(how = 'any').head()

#Trick 16: Split a string into multiple columns

df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],
                   'location':['Los Angeles, CA', 'Washington, DC']})
df
df[['First_Name','Middle Name', 'Last Name']] = df['name'].str.split(' ',expand=True)
df.drop(columns = 'name',inplace=True)
df

df['city'] = df.location.str.split(',',expand=True)[0]
df.drop(columns='location',inplace=True)
df

#Trick 17: Expand a series of list into Dataframe
#There are two columns, and the second column contains regular Python lists of integers.
#If we wanted to expand the second column into its own DataFrame, we can use the apply() 
#method on that column and pass it the Series constructor:

df = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10, 40], [20, 50], [30, 60]]})
df_new = df.col_two.apply(pd.Series)
df_new
df = pd.concat([df,df_new], axis='columns')
df.rename({'0':'col_A','1':'col_B'},axis='columns')
df.columns

df.columns = ['colA','colB','colC','colD']
df.drop(columns='colB',inplace=True)
df

# or- the simple way
df1 = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10, 40], [20, 50], [30, 60]]})


df1['col~A'] = pd.Series(map(lambda x:x[0],pd.Series(df1.col_two)))
df1['col~A']
df1['col~B'] = pd.Series(map(lambda x:x[1],pd.Series(df1.col_two)))
df1['col~B']
df1.drop(columns=['col_two','colA','colB'],inplace=True)
df1

# Trick 18:Aggrigate by multiple functions

import pandas as pd
orders = pd.read_table('https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv')
orders.head()
orders.info()
#Each order has an order_id and consists of one or more rows. To figure out the 
#total price of an order, you sum the item_price for that order_id. For example, 
#here's the total price of order number 1:
orders.item_price = orders.item_price.str.replace('$','').astype(float)
orders.item_price
orders.groupby('order_id').item_price.sum()
orders[orders.order_id==1].item_price.sum()

orders.groupby('order_id').item_price.agg(['sum', 'count']).head()


#Trick 19: Combine the output of an aggregation with a dataframe

#if we want to add a new column total_price which is the sum of each order then
#there is a problem. see the problem that occurs

len(orders.groupby('order_id').item_price.sum())
# length of this is 1834

len(orders.item_price)
#length of this is 4622

#so if we want to add total_price - which is the sum of the prices of each order
#we have a length mis match
#To solve this - we use the transform method - transform() method, which performs 
#the same calculation but returns output data that is the same shape as the input data:

total_price = orders.groupby('order_id').item_price.transform('sum')
total_price
len(total_price)
orders['total_price'] = total_price
orders.head()

#As you can see, the total price of each order is now listed on every single line.
#That makes it easy to calculate the percentage of the total order price that each 
#line represents:
orders['percent of total'] = orders.item_price/orders.total_price
orders.head() 

#Trick 20: Select a slice of rows and columns
import pandas as pd
import numpy as np
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
titanic.head()
#This is the famous Titanic dataset, which shows information about passengers 
#on the Titanic and whether or not they survived.
#If you wanted a numerical summary of the dataset, you would use the describe() method:

titanic.describe()

#However, the resulting DataFrame might be displaying more information than you need.
#If you wanted to filter it to only show the "five-number summary", you can use the 
#loc accessor and pass it a slice of the "min" through the "max" row labels:

titanic.describe().loc['min':'max']
#And if you're not interested in all of the columns, you can also pass it a 
#slice of column labels:

titanic.describe().loc['min':'max', 'Pclass':'Parch']

#Trick 21: Reshape a multi indexed series

titanic.Survived.mean()

#If you wanted to calculate the survival rate by a single category such as "Sex",
# you would use a groupby():

titanic.groupby('Sex').Survived.mean()

#And if you wanted to calculate the survival rate across two different categories at once, 
#you would groupby() both of those categories:

titanic.groupby(['Sex','Pclass']).Survived.mean()
#This shows the survival rate for every combination of Sex and Passenger Class.
# It's stored as a MultiIndexed Series, meaning that it has multiple index levels
# to the left of the actual data.
#It can be hard to read and interact with data in this format, 
#so it's often more convenient to reshape a MultiIndexed Series into a DataFrame 
#by using the unstack() method:

titanic.groupby(['Sex','Pclass']).Survived.mean().unstack()
type(titanic.groupby(['Sex','Pclass']).Survived.mean().unstack())

#Trick 22: Create a pivot table

#If you often create DataFrames like the one above, you might find it more convenient 
#to use the pivot_table() method instead:
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
#With a pivot table, you directly specify the index, the columns, the values, 
#and the aggregation function.
#An added benefit of a pivot table is that you can easily add row and column 
#totals by setting margins=True
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean',margins=True)
#This shows the overall survival rate as well as the survival rate by Sex and Passenger Class.
#Finally, you can create a cross-tabulation just by changing the aggregation function from "mean" to "count":
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='count',
                    margins=True)

#Trick 23: Convert continuous data intop categorical data

#Let's take a look at the Age column from the Titanic dataset:
titanic.Age.head(10)
#It's currently continuous data, but what if you wanted to convert it into categorical data?
#One solution would be to label the age ranges, such as "child", "young adult", and "adult". 
#The best way to do this is by using the cut() function:    

pd.cut(titanic.Age, bins=[0, 18, 25, 99], labels=['child', 'young adult', 'adult']).head(10)

#This assigned each value to a bin with a label. Ages 0 to 18 were assigned the label "child",
# ages 18 to 25 were assigned the label "young adult", and ages 25 to 99 were assigned the 
#label "adult".
#Notice that the data type is now "category", and the categories are automatically ordered.


#Format a dataframe separately for each column

stocks =pd.DataFrame({'Date':pd.date_range(start='2017-01-02',end='2017-01-10'),'Close':[31.5,
112.52,
57.42,
113,
57.24,
31.35,
57.64,
31.59,
113.05
], 'Volume':[14070500,
21701800,
19189500,
29736800,
20085900,
18460400,
16726400,
11808600,
21453100
],'Symbol':['CSCO',
'AAPL',
'MSFT',
'AAPL',
'MSFT',
'CSCO',
'MSFT',
'CSCO',
'AAPL'
]})
stocks

#We can create a dictionary of format strings that specifies how each column should be formatted:

format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}
#And then we can pass it to the DataFrame's style.format() method
stocks.style.format(format_dict)

# ---------- New not covered by Kevin ---------------
# Using Counter()
import pandas as pd
from collections import Counter
data = pd.read_csv('https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/02-BarCharts/data.csv')
data.head()
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

language_counter = Counter()

for response in lang_responses:
    language_counter.update(response.split(';'))
language_counter 
language_counter.most_common(15)

languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])
languages
popularity
languages.reverse()
popularity.reverse()

# Finding number of null values in each column in the data frame

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/titanic.csv'
titanic = pd.read_csv(url)
titanic.head()
titanic.isnull().sum()


# Trick 19: Create a empty dataframe with a list of columns ,without index and create a 
# 1000s rows to forma complete dataframe

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
#finding the length of each row
l = len(digits.data[0])
#finding the # of rows in the new dataframe
digits.data.shape
#creating the dataframe by appedning rows
import pandas as pd
import numpy as np
df = pd.DataFrame(columns = [str(x) for x in range(0,64)])
df

for i in range(1797):
   df.loc[i] = digits.data[i]
df.head(5)
type(df)
df.shape

# Trick 20

#define an array and map it to all the cagetories in a column in the dataframe
# result will be an array 
# very useful to plot

import pandas as pd
beer = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/beer.txt', sep = ' ')
type(beer)
beer
# The name column wll not be useful
X = beer.drop('name',axis=1)
X
# trying the first clustering model

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X)

# finding the clusters

km.labels_

# or it can be found by

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
ypred = km.fit_predict(X)
ypred

beer['cluster'] = km.labels_
beer

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
colors

c = colors[beer.cluster]
c

# Trick 21
#combine multiple rows into a single column 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##Step 1: Read CSV File
df = pd.read_csv('https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv')
df.head()
df.columns
df.isnull().sum()

##Step 2: Select Features

features = ['keywords','cast','genres','director']
df[features].head()
for feature in features:
    df[feature] = df[feature].fillna('')
    
## combine features into a single column

def combine_row(row):
     return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
 

df["combined_features"] = df[features].apply(combine_row, axis=1)   
df["combined_features"].head() 
