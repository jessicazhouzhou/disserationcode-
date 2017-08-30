
# coding: utf-8

# # Predicting booking_bool in Cluster0

# ### Notebook automatically generated from your model

# Model Logistic Regression, trained on 2017-08-23 12:02:26.

# #### Generated on 2017-08-30 21:14:54.468377

# prediction
# This notebook will reproduce the steps for a BINARY_CLASSIFICATION on  Cluster0.
# The main objective is to predict the variable booking_bool

# Let's start with importing the required libs :

# In[ ]:

import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter


# And tune pandas display options:

# In[ ]:

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# #### Importing base data

# The first step is to get our machine learning dataset:

# In[ ]:

# We apply the preparation that you defined. You should not modify this.
preparation_steps = [{u'metaType': u'PROCESSOR', u'alwaysShowComment': False, u'disabled': False, u'params': {u'normalizationMode': u'EXACT', u'matchingMode': u'FULL_STRING', u'appliesTo': u'SINGLE_COLUMN', u'values': [u'1'], u'columns': [u'click_bool'], u'action': u'KEEP_ROW', u'booleanMode': u'AND'}, u'preview': True, u'type': u'FilterOnValue'}]
preparation_output_schema = {u'userModified': False, u'columns': [{u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'string', u'name': u'date_time', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'site_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'visitor_location_country_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'prop_country_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'prop_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'prop_starrating', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'prop_review_score', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'prop_brand_bool', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'prop_location_score1', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'prop_location_score2', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'prop_log_historical_price', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'position', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'price_usd', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'promotion_flag', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_destination_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_length_of_stay', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_booking_window', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_adults_count', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_children_count', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_room_count', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'srch_saturday_night_bool', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'orig_destination_distance', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'random_bool', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'click_bool', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'booking_bool', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'Cluster', u'maxLength': -1}]}

ml_dataset_handle = dataiku.Dataset('Cluster0')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
get_ipython().magic(u'time ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)')

print 'Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1])
# Five first records",
ml_dataset.head(5)


# #### Initial data management

# The preprocessing aims at making the dataset compatible with modeling.
# At the end of this step, we will have a matrix of float numbers, with no missing values.
# We'll use the features and the preprocessing steps defined in Models.
# 
# Let's only keep selected features

# In[ ]:

ml_dataset = ml_dataset[[u'srch_saturday_night_bool', u'prop_location_score1', u'srch_room_count', u'srch_booking_window', u'promotion_flag', u'prop_starrating', u'booking_bool', u'prop_brand_bool', u'prop_review_score', u'srch_length_of_stay', u'price_usd', u'srch_children_count', u'position', u'srch_adults_count']]


# Let's first coerce categorical columns into unicode, numerical features into floats.

# In[ ]:

# astype('unicode') does not work as expected
def coerce_to_unicode(x):
    if isinstance(x, str):
        return unicode(x,'utf-8')
    else:
        return unicode(x)

categorical_features = [u'promotion_flag', u'prop_brand_bool']
numerical_features = [u'srch_saturday_night_bool', u'prop_location_score1', u'srch_room_count', u'srch_booking_window', u'prop_starrating', u'prop_review_score', u'srch_length_of_stay', u'price_usd', u'srch_children_count', u'position', u'srch_adults_count']
text_features = []
from dataiku.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]'):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')


# We are now going to handle the target variable and store it in a new variable:

# In[ ]:

target_map = {u'1': 1, u'0': 0}
ml_dataset['__target__'] = ml_dataset['booking_bool'].map(str).map(target_map)
del ml_dataset['booking_bool']


# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]


# #### Cross-validation strategy

# The dataset needs to be split into 2 new sets, one that will be used for training the model (train set)
# and another that will be used to test its generalization capability (test set)

# This is a simple cross-validation strategy.

# In[ ]:

train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
print 'Train data has %i rows and %i columns' % (train.shape[0], train.shape[1])
print 'Test data has %i rows and %i columns' % (test.shape[0], test.shape[1])


# #### Features preprocessing

# The first thing to do at the features level is to handle the missing values.
# Let's reuse the settings defined in the model

# In[ ]:

drop_rows_when_missing = []
impute_when_missing = [{'impute_with': u'MEAN', 'feature': u'srch_saturday_night_bool'}, {'impute_with': u'MEAN', 'feature': u'prop_location_score1'}, {'impute_with': u'MEAN', 'feature': u'srch_room_count'}, {'impute_with': u'MEAN', 'feature': u'srch_booking_window'}, {'impute_with': u'MODE', 'feature': u'promotion_flag'}, {'impute_with': u'MEAN', 'feature': u'prop_starrating'}, {'impute_with': u'MODE', 'feature': u'prop_brand_bool'}, {'impute_with': u'MEAN', 'feature': u'prop_review_score'}, {'impute_with': u'MEAN', 'feature': u'srch_length_of_stay'}, {'impute_with': u'MEAN', 'feature': u'price_usd'}, {'impute_with': u'MEAN', 'feature': u'srch_children_count'}, {'impute_with': u'MEAN', 'feature': u'position'}, {'impute_with': u'MEAN', 'feature': u'srch_adults_count'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print 'Dropped missing records in %s' % feature

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print 'Imputed missing values in feature %s with value %s' % (feature['feature'], unicode(str(v), 'utf8'))


# We can now handle the categorical features (still using the settings defined in Models):

# Let's dummy-encode the following features.
# A binary column is created for each of the 100 most frequent values.

# In[ ]:

LIMIT_DUMMIES = 100

categorical_to_dummy_encode = [u'promotion_flag', u'prop_brand_bool']

# Only keep the top 100 values
def select_dummy_values(train, features):
    dummy_values = {}
    for feature in categorical_to_dummy_encode:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values

DUMMY_VALUES = select_dummy_values(train, categorical_to_dummy_encode)

def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print 'Dummy-encoded feature %s' % feature

dummy_encode_dataframe(train)

dummy_encode_dataframe(test)


# Let's rescale numerical features

# In[ ]:

rescale_features = {u'srch_booking_window': u'AVGSTD', u'price_usd': u'AVGSTD', u'prop_starrating': u'AVGSTD', u'srch_saturday_night_bool': u'AVGSTD', u'prop_location_score1': u'AVGSTD', u'srch_children_count': u'AVGSTD', u'srch_room_count': u'AVGSTD', u'prop_review_score': u'AVGSTD', u'position': u'AVGSTD', u'srch_adults_count': u'AVGSTD', u'srch_length_of_stay': u'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print 'Feature %s was dropped because it has no variance' % feature_name
    else:
        print 'Rescaled %s' % feature_name
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


# #### Modeling

# Before actually creating our model, we need to split the datasets into their features and labels parts:

# In[ ]:

train_X = train.drop('__target__', axis=1)
test_X = test.drop('__target__', axis=1)

train_Y = np.array(train['__target__'])
test_Y = np.array(test['__target__'])


# Now we can finally create our model !

# In[ ]:

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty="l2",random_state=1337)


# ... And train it

# In[ ]:

get_ipython().magic(u'time clf.fit(train_X, train_Y)')


# Build up our result dataset

# The model is now being trained, we can apply it to our test set:

# In[ ]:

get_ipython().magic(u'time _predictions = clf.predict(test_X)')
get_ipython().magic(u'time _probas = clf.predict_proba(test_X)')
predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')
cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(label_id), label) for (label, label_id) in target_map.iteritems()])
]
probabilities = pd.DataFrame(data=_probas, index=test_X.index, columns=cols)

# Build scored dataset
results_test = test_X.join(predictions, how='left')
results_test = results_test.join(probabilities, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'booking_bool'})


# #### Results

# You can measure the model's accuracy:

# In[ ]:

from dataiku.doctor.utils.metrics import mroc_auc_score
test_Y_ser = pd.Series(test_Y)
print 'AUC value:', mroc_auc_score(test_Y_ser, _probas)


# We can also view the predictions directly.
# Since scikit-learn only predicts numericals, the labels have been mapped to 0,1,2 ...
# We need to 'reverse' the mapping to display the initial labels.

# In[ ]:

inv_map = { label_id : label for (label, label_id) in target_map.iteritems()}
predictions.map(inv_map)


# That's it. It's now up to you to tune your preprocessing, your algo, and your analysis !
# 
