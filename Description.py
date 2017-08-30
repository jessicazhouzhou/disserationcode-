
# coding: utf-8

# In[1]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:

dataframe=pd.read_csv('cleandata.csv')
# Using panda to read the csv file


# In[3]:

dataframe=dataframe.drop('srch_id',axis=1)


# In[4]:

dataframe=dataframe.drop('site_id',axis=1)


# In[5]:

dataframe=dataframe.drop('prop_country_id',axis=1)


# In[6]:

dataframe=dataframe.drop('prop_id',axis=1)


# In[7]:

dataframe=dataframe.drop('prop_log_historical_price',axis=1)


# In[9]:

dataframe=dataframe.drop('visitor_location_country_id',axis=1)


# In[11]:

dataframe = dataframe.ix[dataframe.ix[:,'click_bool'] == 1,:]


# In[12]:

dataframe


# In[13]:

dataframe.describe().transpose()

