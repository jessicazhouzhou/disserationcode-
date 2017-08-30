
# coding: utf-8

# ## Data Wrangling
# 
# ### Importing Data with Pandas

# In[34]:

# Magic to show the plots within the notebook:
get_ipython().magic(u'matplotlib inline')

# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[35]:

# Read the csv file (see Lecture notes for tips)
df = pd.read_csv("cleandata.csv")


# Show an overview of our data: 

# In[36]:

# Get more information about the dataframe
df.info()


# Above is a summary of our data contained in a Pandas dataframe.
# 
# We can get more information about the dataframe using `.info()`

# We can describe the feature in our dataframe and get some basic statistics using `.describe()`

# In[37]:

# Get descriptive statistics about the dataframe
df


# In[38]:

df.describe()


# In[39]:

names=df.columns.values
names


# #### Cleaning Data
# 
# The features `ticket` and `cabin` have many missing values and so can’t add much value to our analysis. To handle this we will drop them from the dataframe to preserve the integrity of our dataset.
# 
# To do that we'll use this line of code to drop the features entirely:
# 
#     df = df.drop(['feature_1','feature_2'], axis=1) 

# In[40]:

df = df.drop(['srch_id'], axis = 1)


# In[41]:

df = df.drop(['site_id'], axis = 1)


# In[42]:

df = df.drop(['visitor_location_country_id'],axis = 1)


# In[43]:

df = df.drop (['prop_country_id'],axis = 1)


# In[44]:

df = df.drop (['prop_id'],axis = 1)


# In[45]:

df=  df.drop (['click_bool'],axis = 1)


# In[46]:

df=  df.drop (['random_bool'],axis = 1)


# In[47]:

df = df.drop(['srch_destination_id'],axis=1)


# In[48]:

df = df.drop(['position'],axis=1)


# In[49]:

import numpy as np


# In[50]:

import pylab as pl


# In[51]:

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Random Forest
# 
# **From Wikipedia:**
# >Random forests are an ensemble learning method for classification (and regression) that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes output by individual trees.
# 
# Random Forest are a form of non-parametric modeling.
# 
# A random forest algorithm randomly generates many simple decision tree models to explain the variance observed in random subsections of our data.  These models are may perform poorly individually but once they are averaged, they can be powerful predictive tools. The averaging step is important. While the vast majority of those models were extremely poor; they were all as bad as each other on average. So when their predictions are averaged together, the bad ones average their effect on our model out to zero. The thing that remains, *if anything*, is one or a handful of those models have stumbled upon the true structure of the data.
# 
# Below we show the process of instantiating and fitting a random forest, scoring the results and generating feature importances.

# ### Training a Basic Random Forest

# Firstly we need to import the `RandonForestClassifier` and `preprocessing` modules from the Scikit-learn library

# In[89]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# We need to seed the random seed and initialize the label encoder in order to preprocess some of our features

# In[90]:

# Set the random seed
np.random.seed(11)

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()


# In[91]:

# Initialize the Random Forest model
randomforest = RandomForestClassifier(max_features=2,n_estimators=500,oob_score=True )



# In[92]:

# Define our features
features = ["price_usd","prop_location_score2","prop_review_score"
            ,"promotion_flag","prop_brand_bool"]
# Train the model
randomforest.fit(X=df[features],y=df["booking_bool"])


# Finally we print the out-of-bag accuracy using `rf_model.oob_score_` followed by the feature importances.

# In[93]:

# Print OOB accuracy
print(randomforest.oob_score)


# In[94]:

feature_importance = randomforest.feature_importances_
feature_importance


# In[95]:

# Plot the feature importances of the factors   
objects = ("price_usd","prop_location_score2","prop_review_score",
            "promotion_flag","prop_brand_bool")
# Sort variable importance from high to low
def hilo_sort(my_list):
    sorted_list = sorted(my_list)
    idx=[];
    for s in sorted_list:
        idx.append(my_list.index(s))
    return(idx, sorted_list)

(idx_1, feature_importance_1)= hilo_sort(list(feature_importance)) 
sorted_predictors_1 = [objects[i] for i in idx_1]


# In[96]:

#Plot a nice bar chart to show variable importance for starrating
y_pos = np.arange(len(sorted_predictors_1))
plt.barh(y_pos,feature_importance_1, align='center', alpha=0.5)
plt.xticks(np.arange(0, 1, 0.1))
plt.yticks(y_pos,sorted_predictors_1)
plt.xlabel('varianle importance')
plt.title('Varible importance for brand_bool')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



