#!/usr/bin/env python
# coding: utf-8

# In[142]:


#Question 1 : Occupations:
#Step-1


# In[2]:


import pandas as pd
import numpy as np


# In[ ]:


#step-2


# In[3]:


#step-3


# In[4]:


users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep = '|')


# In[5]:


users


# In[6]:


#step-4


# In[134]:


print('Mean age of per occupation :')
users.groupby('occupation').age.mean()


# In[9]:


#step-5


# In[133]:


print("The male ratio by ocupation in descending order is : ")

gender_count_per_occupation= users.groupby(['occupation', 'gender']).gender.count()
total_number_people = users.groupby('gender').gender.count()
print((gender_count_per_occupation /total_number_people).sort_values(ascending=False).xs('M',level=1))


# In[11]:


#step-6


# In[12]:


users.groupby('occupation').age.agg(['min', 'max'])


# In[13]:


#step-7


# In[136]:


print('According to occupation and sex, the mean age was computed as follows.:')
users.groupby(['occupation', 'gender']).age.mean()


# In[15]:


#step-8


# In[18]:


gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})

occup_count = users.groupby(['occupation']).count()

occup_gender = gender_ocup.div(occup_count*0.01, level = "occupation")
occup_gender.loc[:, 'gender']


# In[19]:


#Question-2: Euro Teams.
#Step-1


# In[20]:


import numpy as np
import pandas as pd


# In[ ]:


#step-2


# In[ ]:


#Step-3


# In[21]:


euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')


# In[22]:


print('Imported data & assign as euro12 as below:')
euro12


# In[ ]:


#step-4


# In[24]:


euro12.Goals


# In[ ]:


#Step-5


# In[131]:


print('The number of teams who competed in Euro 2012 is:')
no_of_teams_participate = len(euro12.groupby('Team').groups)
print(no_of_teams_participate)


# In[26]:


#Step-6


# In[130]:


print('The datasets total number of columns is:')
total_columns_count = euro12.shape[1]
print(total_columns_count)


# In[ ]:


#step-7


# In[138]:


print('Please locate the discipline dataframe.:')
df_discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
print(df_discipline)


# In[ ]:


#step-8


# In[128]:


print('Below is a list of each teams calculated mean yellow cards.:')
mean_yellow_cards_team = euro12.groupby('Team').agg({'Yellow Cards': 'mean'})
print(mean_yellow_cards_team)


# In[ ]:


#step-9


# In[126]:


print('Below is a list of each teams calculated mean yellow cards.:')
mean_yellow_cards_team = euro12.groupby('Team').agg({'Yellow Cards': 'mean'})
print(mean_yellow_cards_team)


# In[ ]:


#step-10


# In[124]:


print('Teams that score over six goals are:')
goals_gt_6 = euro12[euro12.Goals>6]
print(goals_gt_6)


# In[ ]:


#step-11


# In[31]:


euro12[euro12.Team.str.startswith('G')]


# In[ ]:


#step-12


# In[32]:


print('the first seven listed columns:')
euro12.head(7)


# In[ ]:


#step-13


# In[123]:


print('all columns except the final three are:')
euro12.iloc[:, :-3]


# In[ ]:


#step-14


# In[122]:


print('Russia, Italy, and Englands Shooting Accuracy:')
shooting_accuracy = euro12[['Team', 'Shooting Accuracy']]
SHA_ENG_ITY_RUS = shooting_accuracy[shooting_accuracy.Team.isin(["England","Italy", "Russia"])]
print(SHA_ENG_ITY_RUS)


# In[ ]:


#Question-3 : Housing.
#Step-1


# In[110]:


import pandas as pd
import numpy as np


# In[ ]:


#step-2


# In[111]:


A = pd.Series(np.random.randint(2,6,200))
B = pd.Series(np.random.randint(2,5,300))
C = pd.Series(np.random.randint(20000,40000,200))


# In[40]:


#step-3


# In[119]:



SC = pd.concat([A,B,C],axis=1)
SC.head()


# In[ ]:


#step-4


# In[113]:


SC.columns = ['bedrs','bathrs','price_sqr_meter']
SC.head()


# In[44]:


#step-5


# In[121]:


bigcolumn = pd.concat([A,B,C],axis=1)
bigcolumn


# In[46]:


#step-6


# In[115]:


len(bigcolumn)


# In[ ]:


#step-7


# In[116]:


bigcolumn.reset_index(drop=True, inplace=True)
bigcolumn


# In[ ]:


Question-5:


# In[ ]:


#step-1: Import necessary libraries;


# In[49]:


import pandas as pd
import numpy as np


# In[ ]:


#step-2: import Data set from the address given;


# In[50]:


print('Imported dataset:')
chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv',sep='\t')
print(chipo)


# In[ ]:


#step-4 : See the first 10 entries;


# In[106]:


print('The First ten entries are:')
first_10_entries = chipo.head(10)
print(first_10_entries)


# In[ ]:


#step-5 : what is the no of observations;


# In[105]:


print('the total number of observations in the dataset are:')
no_of_obs = chipo.shape
print(no_of_obs)


# In[ ]:


#step-6 : number of Columns;


# In[104]:


print('The total number of columns in dataset are:')
no_of_columns = chipo.shape[1]
print(no_of_columns)


# In[ ]:


#step-7 : print the names of all the columns;


# In[52]:


print('Column names were:')
column_names = chipo.columns
print(column_names)


# In[ ]:


#step-8: how is dataset indexed ;


# In[54]:


print(chipo.index)


# In[55]:


#step-9 Most ordered item ?


# In[140]:


print('Most ordered item:')
most_ordered_item = chipo[chipo.quantity==chipo.quantity.max()]
print(most_ordered_item)


# In[ ]:


#step-10


# In[102]:


print('The number of orders on the most popular items is:')
most_ordered_item = chipo[chipo.quantity==chipo.quantity.max()]
print(most_ordered_item)


# In[ ]:


#step-11


# In[101]:


print('The choice_description columns most popular item is:')

moi_cd = chipo.groupby('choice_description').agg({'quantity':'sum'}).sort_values(by='quantity', ascending=False).head(1)
print(moi_cd)


# In[ ]:


#step-12


# In[99]:


print('The complete quantity of ordered items was:')
total_items = chipo.quantity.sum()
print(total_items)


# In[ ]:


#step-14


# In[100]:


print('The datasets overall revenue amount over the entire period:')
revenue = print(chipo['item_price'].sum())
print(revenue)


# In[ ]:


#step-15


# In[95]:


print('The total number of orders were in the period of:')
total_orders = print(chipo['order_id'].sum())
print(total_orders)


# In[62]:


#step-16


# In[94]:


print('The number of the different items were sold are:')
diff_items = chipo.item_name.nunique()
print(diff_items)


# In[ ]:


#Question-6


# In[93]:


import matplotlib.pyplot as plt
import pandas as pd
import os
import csv



marriage = pd.read_csv('us-marriages-divorces-1867-2014(1).csv')

columns = marriage.columns.drop(['Marriages','Divorces','Population','Year'])

fig,ax= plt.subplots()

for column in columns:

    ax.plot(marriage['Year'],marriage[column])

ax.set_title('US Marriages and Divorces')

ax.set_xlabel('Year')

ax.set_ylabel('Numbers')

plt.show()


# In[68]:


#Question-7


# In[90]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=[12, 6])
marriage = [708000, 1668000, 2316000]
divorce = [57000, 386000, 945000]

X = np.arange(len(marriage))
plt.bar(X, marriage, color = 'pink', width = 0.25)
plt.bar(X + 0.25, divorce, color = 'yellow', width = 0.25)
plt.legend(['Marriage', 'Divorce'])
plt.xticks([i + 0.25 for i in range(3)], ['1900', '1950', '2000'])
plt.title("Vertical Bar Chart")
plt.xlabel('Marriages & Divorces per capita of US between 1900, 1950, and 2000')
plt.ylabel('Total')
plt.show()


# In[ ]:


#Question-8


# In[87]:


import pandas as pd
import matplotlib.pyplot as plt

dataframe_Actor = pd.read_csv('actor_kill_counts.csv')


dataframe_Actor.plot.barh(x='Actor', y='Count')

plt.title("horizontal Bar Chart")
plt.ylabel('Actor')
plt.xlabel('Kill Counts per (numbers)')
plt.grid(axis='x', linestyle = '--')
plt.show()


# In[ ]:


#Question-9


# In[83]:


import matplotlib.pyplot as plt
import os
import pandas as pd

roman_emperors= pd.read_csv('roman-emperor-reigns.csv')
assassination_death = roman_emperors[roman_emperors['Cause_of_Death'].apply(lambda x: 'assassinated' in x.lower())]
emperor = assassination_death["Emperor"]
cause_of_death = assassination_death["Cause_of_Death"]
plt.pie(range(len(cause_of_death)), labels=emperor,autopct='%1.2f%%', startangle=50, radius=0.055 * 100,rotatelabels = 300)
fig = plt.figure(figsize=[12, 6])
plt.title("The Roman Emperor who was assassinated")
plt.show()


# In[ ]:


#Question-10


# In[141]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
arcade_revenuew = pd.read_csv('arcade-revenue-vs-cs-doctorates.csv')
sns.scatterplot(x='Total Arcade Revenue (billions)', y='Computer Science Doctorates Awarded (US)', hue='Year', palette ='deep',data=arcade_revenuew)


# In[ ]:


#Theend.

