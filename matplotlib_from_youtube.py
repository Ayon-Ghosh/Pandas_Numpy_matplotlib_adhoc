# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:09:05 2019

@author: 140524
"""

# very useful: COUNTPLOT AND CATPLOT

#function of a plot

#https://www.programcreek.com/python/example/96210/seaborn.pairplot

#https://towardsdatascience.com/hands-on-python-data-visualization-seaborn-count-plot-90e823599012
#https://seaborn.pydata.org/generated/seaborn.countplot.html

# MORE into HISTOGRAM AND DISTPLOTS

#https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

# SUBPLOT
#https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib

#Great tutorial from Matplotlib

#https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py


#Matplotlib tutorial - MaxNlocator, subplots, and figsize
#https://www.youtube.com/watch?v=qErBw-R2Ybk

# pyplot is a submodule within matplotlib which allows us to built charts

import matplotlib.pyplot as plt
apl_price = [93.5,101.7,87.3,76.9,97.2]
ms_price = [39.01,50.29,57.05,69.98,95.39]
year = [2014,2015,2016,2017,2018]
plt.plot(year,apl_price)
plt.scatter(year,ms_price)
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.show()

# or both line charts
plt.plot(year,apl_price,year,ms_price)

# changing line colors

plt.plot(year,apl_price,'k',year,ms_price,'g')

#changing line pattern as well

plt.plot(year,apl_price,':k',year,ms_price,'--g')

#limits for X and Y axis

#plt.axis([2013,2019,30,100])
plt.show()

#figures - subplots and charts

# figure is a container - the sie of which u mention in figsize
#subplots are spaces within the figure - container assigned based on how many subplots
#u want to have
#charts are graohs that u print or assign in each subplots

fig_1=plt.figure(1,figsize = (15,7))
# u assign any random # to the figure - just like a var name to hold the fig
chart_1=fig_1.add_subplot(121)
chart_2=fig_1.add_subplot(122)
chart_1
chart_2

# 121 ---1 row 2 columns chart name 1
# 122 - 1 row 2 column chart name 2

# 221---2 rows 2 columns ctart name 1
#----
#224---2 rows 2 columns ctart name 4
# 4 charts can go in 2 rows and 2 columns

# and so on
import matplotlib.pyplot as plt
apl_price = [93.5,101.7,87.3,76.9,97.2]
ms_price = [39.01,50.29,57.05,69.98,95.39]
year = [2014,2015,2016,2017,2018]
chart_1.plot(year,apl_price)
chart_2.scatter(year,ms_price)
plt.xlabel('Year')
plt.ylabel('Stock Price')
plot.show()

#lets fix the ticks of the X axis to int years not float years

#refer: https://matplotlib.org/api/ticker_api.html
import matplotlib.pyplot as plt
apl_price = [93.5,101.7,87.3,76.9,97.2]
ms_price = [39.01,50.29,57.05,69.98,95.39]
year = [2014,2015,2016,2017,2018]
from matplotlib.ticker import MaxNLocator
chart_1.plot(year,apl_price)
chart_1.xaxis.set_major_locator(MaxNLocator(integer = True))
chart_2.scatter(year,ms_price)
chart_2.xaxis.set_major_locator(MaxNLocator(integer = True))
plt.xlabel('Year')
plt.ylabel('Stock Price')
plot.show()







#Matplotlib tutorial 
#https://www.youtube.com/watch?v=UO98lJQ3QGI&list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_

#video 1

from matplotlib import pyplot as plt
# the below is a antigravity comit style that python has as a method - mimicing the 
#xkcd comics
#plt.xkcd()

ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666,
            84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708, 108423, 101407, 112542, 122870, 120000]
plt.plot(ages_x, py_dev_y, color = 'b',marker = 'o', linewidth = 1,label='Python')

js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000,
            78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000, 91660, 99240, 108000, 105000, 104000]
plt.plot(ages_x, js_dev_y, linewidth = 1.5, label='JavaScript')


dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752, 77232,
         78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988, 100000, 108923, 105000, 103117]

# format string consist of color,marker and line
#format string URL: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
#using hex values for color - first two # are red, second 2 #s are green, last 2 blue

plt.plot(ages_x, dev_y, color='#444444', linestyle='--', marker = '.', linewidth = 2,label='All Devs')

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')

plt.legend()
#showing grid
plt.grid(True)
# to auto adjust padding issues if any
plt.tight_layout()

#saving the graph image in your local drive 

#plt.savefig('C:\python\plot.png')

plt.show()
#-----------------------------
#--using style - u can remove linewidth, color rtc..in the same above example

from matplotlib import pyplot as plt

#to see the available styles in plotting

print(plt.style.available)
# ost used are ggplot, fivethrityeigth, seaborn etc.
# below using fivethirtyeight style

plt.style.use('fivethirtyeight')



#plt.xkcd()

ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666,
            84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708, 108423, 101407, 112542, 122870, 120000]
plt.plot(ages_x, py_dev_y, label='Python')

js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000,
            78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000, 91660, 99240, 108000, 105000, 104000]
plt.plot(ages_x, js_dev_y, label='JavaScript')


dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752, 77232,
         78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988, 100000, 108923, 105000, 103117]

# format string consist of color,marker and line
#format string URL: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
#using hex values for color - first two # are red, second 2 #s are green, last 2 blue

plt.plot(ages_x, dev_y, label='All Devs')

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')

plt.legend()
# to auto adjust padding issues if any
plt.tight_layout()

#plt.savefig('plot.png')

plt.show()


# video 2

plt.bar(ages_x, dev_y, color='#444444', label='All Devs')

#if you want to over your line plot with bar chart

plt.plot(ages_x, dev_y, label='All Devs')       
plt.bar(ages_x, dev_y, color='#444444', label='All Devs') 
plt.show()

#plotting bar charts stacking top of each other
plt.bar(ages_x, dev_y, color='#444444', label='All Devs') 
plt.bar(ages_x, py_dev_y,color='#008fd5', label='Python')     
plt.bar(ages_x, js_dev_y, color='#e5ae38',label='JavaScript')
plt.show()


#the above is incovinient to view, so stacking them side by side
# =============================================================================
# step1:
# to do that we have to covert the list of age_x to a index of len of the same array
# and use it for every bars
# step2:
# define a width of each bar otherwise again it will stack on each othet
# step3:
# add the width in one, and subtract from another so all three stackes beside each other    
# =============================================================================
    
    



# =============================================================================
# 
# We can now try to change back the x labels from index_x to ages_x
# step: Use xticks
# =============================================================================

plt.xticks(ticks = index_x, labels = ages_x)
# to auto adjust padding issues if any
plt.tight_layout()

#plt.savefig('plot.png')

plt.show()       
        
#Horizontal bar

# addition concept of Counter used - which we will add on in Numpy_Panda_Matplotlib

import csv
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

data = pd.read_csv('https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/02-BarCharts/data.csv')
data.head()
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

language_counter = Counter()

for response in lang_responses:
    language_counter.update(response.split(';'))
language_counter 
len(language_counter)   

#finding the most common items in the counter

language_counter.most_common(15)

languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

languages.reverse()
popularity.reverse()

plt.barh(languages, popularity)

plt.title("Most Popular Languages")
# plt.ylabel("Programming Languages")
plt.xlabel("Number of People Who Use")

plt.tight_layout()

plt.show()

# video 3 - pie charts:
#piecharts are hard to compare and more so it doesnt have percentages.
#Max 5 items can be represented in pie charts - else - for more - best use 
#bar charts for comparisions

from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

slices = [120,80,30,20]
labels = ['extra1','extra2','extra3','extra4']
colors = ['blue','red','yellow','green']
plt.pie(slices,labels = labels,colors = colors)
#to add borders across each wedge in the pie chart

plt.pie(slices,labels = labels,colors = colors,wedgeprops = {'edgecolor':'black'})

plt.title("My Awesome Pie Chart")
plt.tight_layout()
plt.show()


slices = popularity
slices
labels = languages 
labels
plt.pie(slices,labels = labels,wedgeprops = {'edgecolor':'black'})

# see this looks so crowded and unreadable
#so taking the 5 most popular language

slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
#explode = [0, 0, 0, 0.1, 0]

plt.pie(slices, labels=labels,wedgeprops={'edgecolor': 'black'})

plt.title("My Awesome Pie Chart")
plt.tight_layout()
plt.show()

#Now the above is much easier to read. Lets learn how to emphasise one section of
#the chart

#to emphasis the Python wedge in this chart:
#step: we will have to pass in an explode argument. The explode argument will be a list of 
#values that will offset the slice. This list will be a list of floats that will
#represents how much we want to emphasise the slice
# a 0 will keep the value as is but any # besides 0 will indicate how much we want to emphasie that
#wedge/fraction
#0.1 is 10% of the radias that the wedge exploded out
#0.5 is 50% of the radius that the empahsised wedge exploded out
#and so

#to give a 3D effect we can pass - shadow =True

# to play around with starting angle or refernce we can pass in the argument startangle = value
#play around with the value

# to also print the percentages of each wedge - use the following argument
#autopct='%1.1f%%'

slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
explode = [0, 0, 0, 0.1, 0]

plt.pie(slices, labels=labels,explode = explode,shadow = True, startangle = 90,
        autopct='%1.1f%%',wedgeprops={'edgecolor': 'black'})

plt.title("My Awesome Pie Chart")
plt.tight_layout()
plt.show()

#video 4 - stack plots

# =============================================================================
# stack plots are essential in uses cases such as one variable is spread across a
# duration and you have several other variables (each of which) contributing something
# in that duration
# for example: you have a duration of first 9 mins in a game. You have 3 players scoring >=0
# points every minute of that game
# You want a chart of map each players contribution each minute and also the total game
# Pie chart will give u that break up for a specific minute but not throughout that duration
# 
# 
# =============================================================================
from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")


minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

player1 = [8, 6, 5, 5, 4, 2, 1, 1, 0]
player2 = [0, 1, 2, 2, 2, 4, 4, 4, 4]
player3 = [0, 1, 1, 1, 2, 2, 3, 3, 4]

labels = ['player1', 'player2', 'player3']
colors = ['#6d904f', '#fc4f30', '#008fd5']

plt.stackplot(minutes, player1, player2, player3, labels=labels, colors=colors)

plt.legend()
# if you want to change the position of the legend box then u mention 
#plt.legend(loc='upper left')
# or for example: plt.legend(loc=(0.07, 0.05))

plt.title("My Awesome Stack Plot")
plt.tight_layout()
plt.show()
# example 2

from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")


minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

player1 = [1, 2, 3, 3, 4, 4, 4, 4, 5]
player2 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
player3 = [1, 1, 1, 2, 2, 2, 3, 3, 3]
labels = ['player1', 'player2', 'player3']
plt.stackplot(minutes,player1,player2,player3,labels = labels)
plt.legend(loc='upper left')
plt.title("My Awesome Stack Plot")
plt.tight_layout()
plt.show()


# Colors:
# Blue = #008fd5
# Red = #fc4f30
# Yellow = #e5ae37
# Green = #6d904f 

#video 5: filling area of line plots
#this fillng up the line plotrs gives a lot of insights into the data

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/05-Fill_Betweens/data.csv')
data
ages = data['Age']
dev_salaries = data['All_Devs']
py_salaries = data['Python']
js_salaries = data['JavaScript']

plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')

plt.plot(ages, py_salaries, label='Python')

#filling up from python to the x axis

plt.fill_between(ages, py_salaries)

# increasing the transparancy a bit more using the alpha arguent
plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')

plt.plot(ages, py_salaries, label='Python')
plt.fill_between(ages, py_salaries, alpha = 0.25)

# only filling up python salaries to the overall median instead of x axis
overall_median = 57287
plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')
plt.plot(ages, py_salaries, label='Python')
plt.fill_between(ages, py_salaries, overall_median,alpha=0.25)


# now filling up one area greater than median

plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')
plt.plot(ages, py_salaries, label='Python')
#to fill the crevive where it didnt fill color use interpolate: you can customize colors
#
plt.fill_between(ages, py_salaries, overall_median,
                 where=(py_salaries > overall_median),
                 alpha=0.25, interpolate = True, label='Above Avg')
# now if we want to fill above aveage sal in one color and below avg salary in another color
plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')
plt.plot(ages, py_salaries, label='Python')
plt.fill_between(ages, py_salaries, overall_median,
                 where=(py_salaries > overall_median),
                 alpha=0.25, interpolate = True, label='Above Avg')
plt.fill_between(ages, py_salaries, overall_median,
                 where=(py_salaries <= overall_median),
                 alpha=0.25, interpolate = True, label='below Avg')

#now filling where python salary is greater than dev salary - which filling in the area btwn
# 2 different plots
plt.plot(ages, dev_salaries, color='#444444',
         linestyle='--', label='All Devs')
plt.plot(ages, py_salaries, label='Python')
plt.fill_between(ages, py_salaries, dev_salaries,
                 where=(py_salaries > dev_salaries),
                 interpolate=True, color='blue', alpha=0.25, label='Above Avg')
plt.fill_between(ages, py_salaries, dev_salaries,
                 where=(py_salaries <= dev_salaries),
                 interpolate=True, color='red', alpha=0.25, label='Below Avg')

plt.legend()

plt.title('Median Salary (USD) by Age')
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')

plt.tight_layout()

plt.show()

#video 6: Histograms - great of visualizing distribution of data divided by some boundaries
# =============================================================================
# 
# Use cases: lets say you conduct a survey and you track age of each of the person who responded to your
# survey. With that you want to measure the age of your sample of the survey. 
# You can do it with a bar chart but if you have 50 different ages you will have 50 bars which will
# be difficult to see
# this is where the histogram comes in use when u can use bins for our data and plot how many
#values falls in those bins
# =============================================================================

import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

ages = [18, 19, 21, 25, 26, 26, 30, 32, 38, 45, 55]

# Bins can be a single integer or a list of values
#if Bin is a single integer for example 5 - it divides the 55-18/5 = 7.4 so each
#bars will be at 7.4 approx
# u can put some edge color to divide btwn each bins

plt.hist(ages,bins=5,edgecolor='black')

#However if we want more control over our slots of bins then we can provide a list of Bins

bins=[10,20,30,40,50,60]

plt.hist(ages,bins=bins,edgecolor='black')

# if we dont want to plot ages 10 to 20 and exclude ages -18,19 etc in the list we can start 
# from 20 - for example
bins=[20,30,40,50,60]

plt.hist(ages,bins=bins,edgecolor='black')

#example2
data = pd.read_csv('https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/06-Histograms/data.csv')
data.head()
ids = data['Responder_id']
ages = data['Age']
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(ages, bins=bins, edgecolor='black')

# see we have data from 70 to 100 but its so small that its dwarfed in compariosn to the
#number from other bins. In that case we use the log argument to bring that out


plt.hist(ages, bins=bins, edgecolor='black', log=True)

#lets add on some some information to the histogram. Lets add a vertical line on 
#median age by using the axis vertical line - axvline method

median_age = 29
color = '#fc4f30'
plt.hist(ages, bins=bins, edgecolor='black', log=True)
plt.axvline(median_age, color=color, label='Age Median', linewidth=2)

plt.legend()

plt.title('Ages of Respondents')
plt.xlabel('Ages')
plt.ylabel('Total Respondents')

plt.tight_layout()

plt.show()

for i, binwidth in enumerate([1, 5, 10, 15]):
    print(i)
    print(binwidth)

#video 7: scatter plot - great if you want to show the relation ship btwn 2 sets of
#values and see if they r correlated - they r good to see trends, outliers etc.

import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('seaborn')

x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]
y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]

plt.scatter(x,y)

#increasing the size of the dots by using s

plt.scatter(x,y,s=100)

#changing the color of the dots

plt.scatter(x,y,s=100,c='green')

#changing the marker style of the dots

plt.scatter(x,y,s=100,c='green',marker = 'X')

#adding edge to the circle markers and making them a bit transparent

plt.scatter(x,y,s=100,c='green',edgecolor='black',linewidth=1,alpha=0.75)


#adding different color for each set of markers - gives u an idea of how each 
#points are plotted...lets say to add a dimension we add another list called color
#that represnts the satisfaction level . Each points in the color list corresponds to a 
#a point in x and y list, for example 7 corresponds to x-5, and y-7

colors = [7, 5, 9, 7, 5, 7, 2, 5, 3, 7, 1, 2, 8, 1, 9, 2, 5, 6, 7, 5]
plt.scatter(x,y,s=100,c=colors,edgecolor='black',linewidth=1,alpha=0.75)

#I dont like the color so i want to change the shade to green - use cmap
plt.scatter(x,y,s=100,c=colors,edgecolor='black',cmap='Greens',linewidth=1,alpha=0.75)

# now I want to add a color bar and add a level to it - usecbar

cbar = plt.colorbar()
cbar.set_label('Satisfaction')

#we can change in the sizes of each data points as well to add one more dimension to
#the data - just like the colors, these sizes correspond to the data points in x and y

sizes = [209, 486, 381, 255, 191, 315, 185, 228, 174, 538, 239, 394, 399, 153, 273, 293, 436, 501, 397, 539]
plt.scatter(x,y,s=sizes,c=colors,edgecolor='black',cmap='Greens',linewidth=1,alpha=0.75)
cbar = plt.colorbar()
cbar.set_label('Satisfaction')

#example 2 - youtube data on top videos - views, likes, and counts
#want to see if there is a correlation btwn the view counts and likes
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('seaborn')

data = pd.read_csv('https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/07-ScatterPlots/2019-05-31-data.csv')
view_count = data['view_count']
likes = data['likes']
ratio = data['ratio']

plt.scatter(view_count, likes, edgecolor='black', linewidth=1, alpha=0.75)

#see we have a outlier on the top right corner. We can minimise the impact of the outlier
#using the log scale
plt.scatter(view_count, likes, edgecolor='black', linewidth=1, alpha=0.75)
plt.xscale('log')
plt.yscale('log')

#lets use the ratio of the likes/dislikes for the colors
plt.scatter(view_count, likes, edgecolor='black', c=ratio, cmap='summer',linewidth=1, alpha=0.75)
plt.xscale('log')
plt.yscale('log')
cbar = plt.colorbar()
cbar.set_label('Like/Dislike Ratio')
plt.title('Trending YouTube Videos')
plt.xlabel('View Count')
plt.ylabel('Total Likes')
plt.tight_layout()
plt.show()

# getting data from API
# here API is the yahoo finance - stock data
# we want to create a function and use to plat for every company

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def graph_data(stock):
    
    stock_price_URL = 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('')
    plt.legend()
    plt.show()

# histogram, bins, and xticks
import matplotlib.pyplot as plt    
x = [10,6,7,8,6,5,4,2,12,9,1,18,13,14,15,18,12,13,19,11]
bins=[list(range(0,11))]    
plt.xticks(bins)
plt.hist(x,bins=20)
plt.show()
    
# xticks
# not below the set_ticks doesnt plot 101,102,103, as given i+100
import pylab as plt

x = range(1, 7)
y = (220, 300, 300, 290, 320, 315)

def test(axes):
    axes.bar(x,y)
    axes.set_xticks(x, [i+100 for i in x])

a = plt.subplot(1,2,1)
test(a)
b = plt.subplot(1,2,2)
test(b)

# set_ticks defines the position of the x or y scale
#set_ticklabels sticks the labels on those position
# for example: Now see the labels on the scale

import matplotlib.pyplot as plt

x = range(10)
y = range(10)

fig, ax = plt.subplots(nrows=2, ncols=2)

for row in ax:
    for col in row:
        col.plot(x, y)

plt.show()

#----------------
        

def test(axes):
    axes.bar(x,y)
    axes.set_xticks(x)
    axes.set_xticklabels([i+100 for i in x])

a = plt.subplot(1,2,1)
test(a)
b = plt.subplot(1,2,2)
test(b)

#bar values ---are values of each bar
import os
import numpy as np
import matplotlib.pyplot as plt

x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME', u'DISH', u'NEIGHBOURHOOD']
y = [160, 167, 137, 18, 120, 36, 155, 130]

fig,ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(x))  # the x locations for the groups
#text function sticks the text in the picture
#parameters are - figurename.text(x-postion,y-position, txt to stick in the graph etc)
#u can adjustment the x position and y position to make it as neat possible 
# as shown below: x position - v+3 (example: 163,170 etc.), y position i+.25
#(example: 1.25,2.25 etc.)
for i, v in enumerate(y):
    print(i,v)
    ax.text(v+3, i + .25, str(v), color='blue', fontweight='bold')
# see X is a list of string/words so cannot be plotted on X axis like below
#ax.barh(x,y,width, color="blue")
#therefore the way around to plot X values is to take the length of x
#use set_yticks to set the positions of the stickers
#use_set_ytick label to stick the labels on the axis   
ax.barh(ind,y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('title')
plt.xlabel('x')
plt.ylabel('y')      
plt.show()
