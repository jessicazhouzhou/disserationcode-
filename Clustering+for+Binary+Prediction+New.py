
# coding: utf-8

# # Clustering

# In[1]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[37]:

dataframe= pd.read_csv('Clusetring Data.csv')
# Using panda to read the csv file
dataframe


# In[ ]:




# In[38]:

from sklearn.cluster import KMeans
clustering_model = KMeans(n_clusters=2)


# In[39]:

get_ipython().magic(u'time clusters = clustering_model.fit_predict(train)')


# In[40]:

y=clustering_model.labels_
train['Cluster Number']=pd.Series(y,index=train.index)


# In[41]:

Y = train['Cluster Number'] 
X=train.ix[:,0:4]
X_norm=(X-X.min())/(X.max()-X.min())
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
transformed


# In[42]:

plt.scatter(transformed[Y==0][0], transformed[Y==0][1], label='Cluster0', c='red')
plt.scatter(transformed[Y==1][0], transformed[Y==1][1], label='Cluster1', c='blue')
plt.legend()
plt.show()


# In[43]:

print(clustering_model.inertia_)


# In[44]:

from sklearn.metrics import silhouette_score
silhouette = silhouette_score(train.values, clusters, metric='euclidean', sample_size=2000)
print ("Silhouette score :", silhouette)


# In[45]:

final = X_norm.join(pd.Series(clusters, index=train.index, name='cluster'))
final['cluster'] = final['cluster'].map(lambda cluster_id: 'cluster' + str(cluster_id))


# In[46]:

size = pd.DataFrame({'size': final['cluster'].value_counts()})
size.head()


# In[47]:

axis_x=train.columns[1]
axis_y = train.columns[2]  
from ggplot import ggplot, aes, geom_point
ggplot(aes(axis_x, axis_y, colour='cluster'), final) + geom_point()


# In[48]:

#Initializes plotting library and functions for 3D scatter plots 
from pyspark.ml.feature import VectorAssembler
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import six
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

def rename_columns(train, prefix='x'):
    """
    Rename the columns of a dataframe to have X in front of them

    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    df = train.copy()
    return df


# In[49]:

# create an artificial dataset with 2 clusters
Y=clustering_model.labels_
X=X_norm.values
df = pd.DataFrame(X)
# ensure all values are positive (this is needed for our customer 360 use-case)
df = df.abs()
# and add the Y
df['y'] = Y
# split df into cluster groups
grouped = df.groupby(['y'], sort=True)
# compute sums for every column in every group
mean = grouped.mean()


# In[51]:

data = [go.Heatmap(z=mean.values.tolist(), 
                   y=['Cluster 0', 'Cluster 1', 'Outliers'],
                   x=[u'price_usd', u'prop_location_score1',  u'prop_review_score', 
                         u'promotion'],
                   colorscale='Viridis')]
plotly.offline.iplot(data, filename='pandas-heatmap')


# In[63]:

dataframe['Cluster']=y
dataframe


# In[68]:

dataframe.to_csv('Clustering New.csv')


# In[56]:

clusters=()
   cluster = 0
   for item in labels:
    if item in clusters:
        clusters[item].append(row_dict[n])
    else:
        clusters[item] = [row_dict[n]]
    cluster +=1


# In[57]:

for item in clusters:
    print " Cluster", item
    for i in clusters[item]:
        print i


# In[ ]:




# In[ ]:



