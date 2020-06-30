# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:40:02 2019

@author: 140524
"""
#--Numpy Array - NdArray
# An NdArray is a multi dimensional array that has 2 parts - the actual data,
# and metadata which describes the stored data. They are indexed just like sequences
# in python starting from 0
# single dimensional numpy array

import numpy as np
a =np.array([1,2,3])
print(a)
print(type(a))

# multi dimensional numpy array: the lenght of a sublist determines the number
# of colums. The number of list determined the number of rows.
# all the sublist have to be same lenght in order to be a multi dimensional array

import numpy as np
a = np.array([[1,2,3],[3,4,5]])
print(a)

#creating a array using ARANGE function

import numpy as np
a = np.arange(0,100,dtype=np.int32)
print(a)
a.itemsize
import numpy as np
a = np.arange(0,100,dtype=np.int8)
a.itemsize
# creating a array of zeroes, the zero function takes in a tuple whose first element is
# is the number of rows and second element i the number of columsn

import numpy as np
a = np.zeros((5,5))
print(a)

import numpy as np
a = np.zeros((4,8))
print(a)

import numpy as np
a = np.zeros((5,3))
print(a)

a = np.zeros((5,3,2))
print(a)

# creating array using linspace. It takes the fist element, last element, and steps or
# lenear gaps - also called lenearly space vectors


import numpy as np
a = np.linspace(0,20,5)
print(a)

import numpy as np
a = np.linspace(0,20,10)
print(a)

# Asarray - used for converting python sequences into nparrays

import numpy as np
a = [1,2,3,4]
x = np.asarray(a)
print(x)

# restructuring a numpy array meaning converting a flat/lenear array of 8
#elements into 2x2x2 3D array

import numpy as np
a = np.linspace(0,20,8)
print(a)
a = a.reshape((2,2,2))
print(a)

# OR

import numpy as np
a = np.zeros(8)
print(a)
a = a.reshape((2,2,2))
print(a)
a = a.reshape((4,2))
print(a)

# Ravel - flattens out the dimensional array

import numpy as np
a = np.zeros(8)
print(a)
a = a.reshape((2,2,2))
print(a)
print(a.ravel())

# Indexing and slicing of Numpy array - very similar to python

import numpy as np
a = np.arange(2,20)
print(a)
x = a[7]
print(x)

import numpy as np
a = np.arange(2,20)
print(a)
x = slice(2,10,4)
print(x)
print(a[10:15])
print(a[:15])

# indexing and slicing a multidimensional array - Basic indexing

import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(a[0:2,0:2])
print(a[0:3,0:1])
print(a[0:2,0:3])
print(a[1:,:3])


# Advanced indexing
# =============================================================================
# 
# difference btw basic and advanced indexing
# 
# 1) basic slicing operates on views of the data - while
# advanced copying operates on copy of the data
# 2) basic indexing is done with integers - start:stop:step-while advanced indexing
# can be done by ndarray, list of indexes, and boolean masking
# 3) basic indexing can perform basic operations while advanced indexing performs
# complex operation satisfying some conditions
# 
# see advanced indexing examples below
# =============================================================================

import numpy as np
a=np.arange(1,10)
a

#lets get the # 2,5,6 from the above array

#step1: create a array of index of 2,5,6 in a

index = np.array([1,4,5])
#step2: pass the array of index into a
a[index]


b=np.array([[1,2,3],[4,5,6],[7,8,9]])
b

# lets print 3 and 7

index = [[0,2],[2,0]]
b[index]

# or

index=((0,2),(2,0))
b[index]

# or
b[[0,2],[2,0]]
# lets print 4,6,8
b[[1,1,2],[0,2,1]]
# or 

index =((1,1,2),(0,2,1))
b[index]

# printing repeatedly the same number

import numpy as np
a=np.arange(1,10)
a
index=np.array([1,4,1,4,1,4,1,4])
a[index]

# or

a[[1,4,1,4,1,4,1,4]]

#boolean indexing
import numpy as np
a=np.array([[1,2,-3],[4,-2,3]])
a

#print only the negative #s

a[a<0]

# print only postive numbers
a[a>0]

# multiply 10 to all negative #

a[a<0]*10

# printing thr whole array with the multiplies #s

a[a<0] = a[a<0]*10
a

# or

n=a<0
a[n]=a[n]*10
a

# =============================================================================
# Fancy indexing
# 
# We will index an array C in the following example by using a Boolean mask. 
# It is called fancy indexing, if arrays are indexed by using boolean or integer arrays (masks). 
# The result will be a copy and not a view.
# 
# In our next example, we will use the Boolean mask of one array to select the 
# corresponding elements of another array. The new array R contains all the elements of C 
# where the corresponding value of (A<=5) is True.
# C = np.array([123,188,190,99,77,88,100])
# =============================================================================
A = np.array([4,7,2,8,6,9,5])
C[A<=5]
R = C[A<=5]
R


# arithmetic operations on arrays using scalar

import numpy as np
a=np.arange(1,5)
a
#when we use scalars like +2,*2 etc, then it createsa array of same size and adds
#or multiplies 2 to each element of the original array
a*2
a+2
a%2

a=np.arange(1,5)
b=np.arange(6,10)
a+b
b/a

# we can ALSO use:

# =============================================================================
# np.add(a,b)
# np.subtract(a,b)
# np.multiply(a,b)
# np.divide(a,b)
# np.mod(a,b)
# np.power(a,b)
# 
# =============================================================================


# Braodcasting - arithmatic operation on arrays of different shape and size
#braodcasting will stretch the value or the shape to the required dimension and 
# then perfom the arithmetic operation

#Refer: https://www.youtube.com/watch?v=0u9OzBSRZec&list=PLzgPDYo_3xunaO-noMnBzc3KzzToOcxVY&index=18
# =============================================================================
# 
# brodcasting rules:
#     size of each dimension will be same
#     size of one of the dimension should be one
#     if the 2 arrays differ in their number of dimensions, the shape of the one:
#         with the fewer dimension is padded with 1 on its leading side(left side):
#     if the shape of the 2 arrays does not match in any dimension, the array with the
#         shape equal to 1 in that dimension is stretched to match the other shape
#     if in any dimension the sizes disagree and neither 1, then error is raised    
# =============================================================================

        

#Numpy array attributes

import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
#returns a tuple consisting of array dimension
print(a.shape)
#returns the number of array dimensions
print(a.ndim)
#returns the length of each element in byte size of an array
print(a.itemsize)
print(a.size)
# Numpy array creation routines

import numpy as np
a = np.zeros(5)
print(a)


#Reshape and resize the array

#https://www.youtube.com/watch?v=KehyltXMrZE&list=PLzgPDYo_3xunaO-noMnBzc3KzzToOcxVY&index=19
# =============================================================================
# 
# array.reshape(array,shape,order)
# order = C,A,F
# C=row wise operation---------------default
# A=column wise operation
# F=fortrain order
# =============================================================================
# even after reshaping - the size (which is the number of elements) must be same
a=np.arange(10)
a.size
np.reshape(a,(5,2))
np.reshape(a,(5,2),order='F')
a.size

# or 

a.reshape((2,5))
#resize will chnge the total # of elements of the array - the data is repreated in the 
#order they r stored..see outpiut of example below

np.resize(a,(6,3))
#numpy.empty - uninitialized array (doesnt have ny value - just been created)
# of specific shape and dtype
# it picks up any random value

# constuctor syntax: numpy.empty(shape,dtype=float//int etc.)

import numpy as np
x = np.empty([2,2],dtype=int)
print(x)


##set a random seed
np.random.seed(0)

# Reading and writing from txt files

# Numpy provides the option to import data from files directly into ndarray
#using the loadtxt function

#The savetxt function can be used to write data from ndarray to txt file

import numpy as np
a = np.arange(0,20)
print(a)
np.savetxt('ayon.txt',a)


import numpy as np
a = np.array([[1,2],[2,3]])
print(a)
np.savetxt('test.txt',a)

new_arr = np.loadtxt('test.txt')
print(new_arr)



# Reading and writing from CSV file
#numpy arrays can be dumped into CSV file using savetxt and delimited which is a comma
#The genfromtxt can be used to read data from a CSV file into a numpy array
import numpy as np
a = np.array([[1,2,3],[2,3,4]])
print(a)
np.savetxt('C:\\python\\files\\ayonbuli.csv',a,delimiter=',')
import numpy as np
a = np.genfromtxt('ayonbuli.csv',delimiter=',')
print(a)


#---PANDA ---- PANEL DATA
#--data structure in pandas
# Series - dimension 1 - label homogenous array with immutable size but value mutable
# dataframe --- dimension 2 -- label heterogenously typed, size - mutable  and value mutable
#tabular data structure
#Panels - dimension 3 -- labeled size mutable array
# all values in panda are mutable

# series ---------

# a series is a one dimesional array that contain homogenous data (same type data)
# such as ndarray, series, dic, list, constant
# all elements are value mutable but size immutable
# indexes can be unique, hashable, and have the same length as data  -defaults to 
#np.arange(n) if no index is passed
# data type of each column if none is mentioned will be inferred automatically
# deep copies data, set to false as default meaning even copies the reference and 
# properties, memory location of the data

# create a series

import pandas
series = pandas.Series()
print(series)


import numpy as np
import pandas
a = np.array([10,20,30,40])
type(a)
series = pandas.Series(a)
series.dtypes
type(series)
print(series)


import pandas
my_dict = {'a':10, 'b':20,'c':30}
series = pandas.Series(my_dict)
print(series)

# Retrieving a part of the series using slicing

import pandas 
a = pandas.Series([10,20,30,40])
print(a[1:4])

import pandas
series = pandas.Series({'a':10, 'b':20,'c':30, 'd':40,'e':50,'f':60})
series
print(series[2:4])
print(series[:'e'])


# Data Frame

# a dataframe is a 2-D data structure in which data is aligned in a tabular 
# fashion cosnsiting of rows and columes
# constructor of dataframe: pandas.DataFrame(data,index,data type, copy)
# can be of multiple data type such as ndarray, list, constant, series, dic
#row and colume level of a data frame defaults to np.arange(n) if no index is passed
#data type of each colume
# deep copies data, set to false as default

# create a data frame

import pandas
a = pandas.DataFrame([10,20,30,40])
print(a)

import pandas
a = pandas.DataFrame([{'a':10,'b':20,'c':30},{'age':20,'name':'Ram'}])
print(a)

import pandas
listx = [{'a':10,'b':20,'c':30},{'age':20,'name':'Ram','city':'coral'}]
a = pandas.DataFrame(listx, index = ['row-1','row-2'])
print(a)

#converting a dic of series into dataframe

import pandas
a = pandas.DataFrame([{'one':pandas.Series([1,2,3,4], index =['row-1','row-2','row-3','row-4']),'two':pandas.Series(['a','b','c','d'], index =['row-1','row-2','row-3','row-4'])}])
print(a)

import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90], index =['eng','eco','math'])
table = pandas.DataFrame({'jim':series1,'joe':series2})
print(table)

# Addition and deletion of columes in DataFrame

#A new colume can be added to a dataframe when it is passed as a series


import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90,100], index =['eng','eco','math','CSE'])
table = pandas.DataFrame({'jim':series1,'joe':series2})
table['Ayon'] = pandas.Series([10,20,30,5,15],index =['eng','eco','math','CSE','beng'])
print(table)


import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90,100], index =['eng','eco','math','CSE'])
table = pandas.DataFrame({'jim':series1,'joe':series2,'Ayon':pandas.Series([10,20,30,5,15],index =['eng','eco','math','CSE','beng'])})
print(table)

#del(table['jim'])
#print(table)

new_table = table.pop('jim')
print(new_table)

# Addition and Deletion of ROW in DataFrame

#DataFrame rows can be selected by passing the selected row level to loc function
# table.loc()

import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90,100], index =['eng','eco','math','CSE'])
table = pandas.DataFrame({'jim':series1,'joe':series2,'Ayon':pandas.Series([10,20,30,5,15],index =['eng','eco','math','CSE','beng'])})
table
print(table.loc['eng'])
print(table.jim)
#or
print(table['jim'])

# now using iloc - mathematical level of the row

import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90,100], index =['eng','eco','math','CSE'])
table = pandas.DataFrame({'jim':series1,'joe':series2,'Ayon':pandas.Series([10,20,30,5,15],index =['eng','eco','math','CSE','beng'])})
table
print(table.iloc[2])

# Appending a row

import pandas
data = {'one':pandas.Series([10,20,30], index = ['a','b','c']), 'two':pandas.Series([11,21,31], index = ['a','b','c'])}
table = pandas.DataFrame(data)
table
# addting colume
table['three'] = pandas.Series([10,20,30], index = ['a','b','c'])
print(table)
row = pandas.DataFrame([[40,50],[41,51]], columns = ['one','two'], index = ['d','e'])
table = table.append(row, sort = False)
print(table)

# Appending a row - another example

import pandas
series1 = pandas.Series([45,56,87], index =['eng','eco','math'])
series2 = pandas.Series([23,65,90,100], index =['eng','eco','math','CSE'])
# addting colume
table = pandas.DataFrame({'jim':series1,'joe':series2,'Ayon':pandas.Series([10,20,30,5,15],index =['eng','eco','math','CSE','beng'])})
print(table)
row = pandas.DataFrame([[100,100],[99,98]], columns = ['Ayon','jim'], index =['DS','PM'])
print(table.append(row, sort = False))

# Imporitng and Exporting Data Using pandas

# Loading CSV into Data Frames

# Data can be loaded into DataFrames from input data stored in CSV file using the read_CSV()
#function in which u have to give path of the file

import pandas
table = pandas.read_csv("C:\python\DataSet_Numpy_Panda_Matplotlib\CityTemps.csv")
print(table)



# MATPLOTLIB

# Plotting using Matplotlib

# below example gives only Y axis values
# X axis values are taken by default
# plot(x,y)

import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show()

# below example we are computing X axis using list comprehension

import matplotlib.pyplot as plt
y_values = [1,2,3,4,5,10]
x_values = [i**2 for i in y_values]
plt.plot(x_values,y_values)
plt.show()

# below example using arange of numpy

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,11)
y = [i**2 for i in x]
plt.plot(x,y)
plt.show()

# or

import matplotlib.pyplot as plt
import numpy as np
#x = np.arange(2,11)
#y = [i**2 for i in x]
plt.plot(np.arange(2,11),[i**2 for i in np.arange(2,11)])
#plt.plot(x,y)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20,3)
print(x)
y = [i**2 for i in x]
plt.plot(x,y)
plt.show()

# MultiLine plot

#Multiple functions can be used to draw the same plot

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20,3)
print(x)    
plt.plot(x,x)
plt.plot(x,[i**2 for i in x])
plt.plot(x,[i**3 for i in x])
plt.show()

# or

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20,3)
print(x)
plt.plot(x,x,x,[i**2 for i in x],x,[i**3 for i in x])
plt.show()


# Adding a GRID

# Grid functions add a grid to the plot
# Grid takes the Boolean value

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20,3)
print(x)
plt.plot(x,x,x,[i**2 for i in x],x,[i**3 for i in x])
plt.grid(True)
plt.show()

# limiting the Axes

# if it is a very large graph and u want to see a certain portion of it
# you will have to limit the axes

# The scale of the plot can be set using axis function - thereby limiting axes

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20,3)
print(x)
plt.plot(x,x,x,[i**2 for i in x],x,[i**10 for i in x])
plt.grid(True)
plt.axis([5,15,1,200])
plt.show()

# Using xlim and ylim instead of axis

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20)
print(x)
plt.plot(x,x,x,[i**2 for i in x],x,[i**10 for i in x])
plt.grid(True)
plt.xlim(3,6)
plt.ylim(3,100)
plt.show()

# Adding labels and title to a plot

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20)
print(x)
plt.plot(x,x,x,[i**2 for i in x],x,[i**10 for i in x])
plt.grid(True)
plt.xlim(3,6)
plt.ylim(3,100)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Learning Matplotlib')
plt.show()

# Adding Legends

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20)
print(x)
plt.plot(x,x, label = 'linear')
plt.plot(x,[i**2 for i in x], label = 'square')
plt.plot(x,[i**3 for i in x], label = 'cube')
plt.legend()
plt.grid(True)
#plt.xlim(3,6)
#plt.ylim(3,100)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Learning Matplotlib')
plt.show()

# Saviung plot

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(2,20)
print(x)
plt.plot(x,x, label = 'linear')
plt.plot(x,[i**2 for i in x], label = 'square')
plt.plot(x,[i**3 for i in x], label = 'three')
plt.legend()
plt.grid(True)
#plt.xlim(3,6)
#plt.ylim(3,100)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Learning Matplotlib')
#plt.show()
plt.savefig('testplot.png')

# HISTOGRAM
# histogram is a distribution of a variable
# here dont provide the x and y VALUES
# Use random.randn() from numpy 
#returns a sample from the standard normal distribution
#study the details from:
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html?highlight=randn#numpy.random.randn

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(400,80)
plt.hist(x)
plt.show()

# passing the Bin parameter to space it out and make non overlapping

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(400,80)
plt.hist(x,20)
plt.show()


# BAR CHART

# FOR BAR graph we need to provide 2 array.
# one array that defines the midpoint of each bars
# another array that defines the height of each bars

import matplotlib.pyplot as plt
plt.bar([2,4,6,8,10],[10,20,50,30,5])
plt.show()

# example showing to plot a dictionary using a bar chart

import matplotlib.pyplot as plt
my_dict = {'a':20,'b':40,'c':78,'d':65}
for i, key in enumerate(my_dict):
    print(i,key)
    plt.bar(i,my_dict[key])
plt.show()    


#plotting a dictionary into Bar chart using xticks
#xtkicks add the labels as keys instead of the enumerate loop i
# xticks((height of bars in bar chart - Y axis), labels in X axis)
# example: xticks([1,2,3],['A-bar','B-bar','C-bar'])

import matplotlib.pyplot as plt
import numpy as np
my_dict = {'A':20,'B':40,'C':78,'D':65}
for i, key in enumerate(my_dict):
    plt.bar(i,my_dict[key])
plt.xticks(np.arange(len(my_dict)),my_dict.keys())    
plt.show()   

# PIE CHART

# Give the size of the square plot in inches within which the piechart circle fits in
#Using figsize(x,y)

# Give the proportions of the sectors

import matplotlib.pyplot as plt
plt.figure(figsize=(3,3))
x = [20,10,40]
labels = ['bikes', 'cars', 'cycles']
plt.pie(x,labels = labels)
plt.show()

# SCATTER PLOT
# use 2 randns one for x and another for y
# 2 gaussian distributions
# X and Y must be of same sie

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x,y)
plt.show()


s = '100 BROAD ROAD'
s[:-4]
s[-4:]