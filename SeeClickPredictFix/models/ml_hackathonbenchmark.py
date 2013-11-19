''' 
DATA DICTIONARY
id - a randomly assigned id
latitude - the latitude of the issue
longitude - the longitude of the issue
summary - a short text title
description - a longer text explanation
num_votes - the number of user-generated votes
num_comments - the number of user-generated comments
num_views - the number of views
source - a categorical variable indicating where the issue was created
created_time - the time the issue originated
tag_type - a categorical variable (assigned automatically) of the type of issue
'''

''' Libraries and globals '''
import numpy as np
import csv
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn import cross_validation

''' Functions and classes '''

''' Load data '''
# load training data into dict
data = []
file_name = "../data/train.csv"
reader = csv.DictReader(open(file_name, 'rb'), delimiter=',', quotechar='"')
for row in reader:
    data.append(row)

# convert appropriate keys from string to appropriate type
for sub in data:
    for key in sub:
        if key == 'id' or key == 'num_votes' or key == 'num_comments' or key == 'num_views':
            sub[key] = int(sub[key])
        elif key == 'latitude' or key =='longitude':
            sub[key] = float(sub[key])
        elif key == 'created_time':
            sub[key] = time.mktime(time.strptime(sub[key], "%Y-%m-%d %H:%M:%S"))    # make time into datetime (float)

print "Training data loaded successfully!"

''' Pre-process training data '''
# select which variables to fit the model on
X_vars = ['latitude', 'longitude', 'source', 'tag_type']
y_vars = ['num_votes', 'num_comments', 'num_views']
X = []
y = []
for d in data:
    subdict_X = dict([(i, d[i]) for i in X_vars if i in d])
    subdict_y = dict([(i, d[i]) for i in y_vars if i in d])
    X.append(subdict_X)
    y.append(subdict_y)
    
# encode categorical variables in X
vec_X = DictVectorizer()
vec_y = DictVectorizer()
X_encoded = vec_X.fit_transform(X).toarray()
# convert y into proper numpy array
y_encoded = vec_y.fit_transform(y).toarray()
# transform y to log(y_i + 1)
y_encoded = np.log(y_encoded + 1)


''' Build the model '''
# segment into geographic clusters using KMeans
km = KMeans(n_clusters = 4)
km.fit(X_encoded[:, 1:3])
cluster_centers_indices = km.cluster_centers_
labels = km.labels_

print X_encoded
print labels.T
raw_input()

# loop through each geo cluster (city) and build model
for cluster in range(0, 4):
    X_subset = X_encoded[labels == cluster]
    y_subset = y_encoded[labels == cluster]
    X_subset = X_subset[:, 3::]
    # fit linear model (column 1 is time, 2 and 3 are lat long, rest are indictator vars)
    clf_0 = linear_model.LinearRegression()
    clf_1 = linear_model.LinearRegression()
    clf_2 = linear_model.LinearRegression()
    
    clf_0.fit(X_subset, y_subset[:, 0])
    clf_1.fit(X_subset, y_subset[:, 1])
    clf_2.fit(X_subset, y_subset[:, 2])
    
    cv_scores_0 = cross_validation.cross_val_score(clf_0, X_subset, y_subset[:, 0], cv=5)
    cv_scores_1 = cross_validation.cross_val_score(clf_0, X_subset, y_subset[:, 1], cv=5)
    cv_scores_2 = cross_validation.cross_val_score(clf_0, X_subset, y_subset[:, 2], cv=5)
    
    print cv_scores_0
    print cv_scores_1
    print cv_scores_2
    raw_input()
    



