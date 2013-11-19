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
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn import ensemble

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
X_vars = ['latitude', 'longitude', 'source', 'tag_type', 'created_time']
X_text_vars = ['summary', 'description']
y_vars = ['num_votes', 'num_comments', 'num_views']
X = []
X_text = []
y = []
for d in data:
    # extract numeric/categorical elements
    subdict_X = dict([(i, d[i]) for i in X_vars if i in d])
    subdict_y = dict([(i, d[i]) for i in y_vars if i in d])
    X.append(subdict_X)
    y.append(subdict_y)
    # extract text elements
    subdict_X = dict([(i, d[i]) for i in X_text_vars if i in d])
    X_text.append(subdict_X)

#===============================================================================

# 
# summary_arr = np.array(summary_lst)
# print summary_arr
# raw_input()
# description_arr = np.array(description_lst)
# print description_arr
# raw_input()
#===============================================================================

# encode categorical variables in X
vec_X = DictVectorizer()
vec_y = DictVectorizer()
X_encoded = vec_X.fit_transform(X).toarray()
# convert y into proper numpy array
y_encoded = vec_y.fit_transform(y).toarray()
# transform y to log(y_i + 1)
y_encoded = np.log(y_encoded + 1)
# use tf-idf on 'summary' and 'description' fields
summary_lst = []
description_lst = []
for x in X_text:
    summary_lst.append(x['summary'])
    description_lst.append(x['description'])
   
vec_summary = text.CountVectorizer()
vec_description = text.CountVectorizer()
raw_input("initialized")
summary_encoded = vec_summary.fit_transform(summary_lst)#.toarray()
description_encoded = vec_description.fit_transform(summary_lst)#.toarray()
#X_text_encoded = vec_X_text.fit_transform(X_text)
raw_input("fit")
print summary_encoded
raw_input()
print description_encoded
raw_input()




''' Build the model '''
# segment into geographic clusters using KMeans
km = KMeans(n_clusters = 4)
km.fit(X_encoded[:, 1:3])
cluster_centers_indices = km.cluster_centers_
labels = km.labels_

# loop through each geo cluster (city) and build model
resid_square_sum = 0
test_size = 0
clf_0_lst = []
clf_1_lst = []
clf_2_lst = []
for cluster in range(0, 4):
    X_subset = X_encoded[labels == cluster]
    y_subset = y_encoded[labels == cluster]
    
    median_time = np.median(X_subset[:, 0])
    #X_subset = X_subset[:, 3::]
    
    # split into train (1st 50% time) and test (2nd 50% time)
    X_train_subset = X_subset[X_subset[:,0] <= median_time]
    y_train_subset = y_subset[X_subset[:,0] <= median_time]
    X_test_subset = X_subset[X_subset[:,0] < median_time]
    y_test_subset = y_subset[X_subset[:,0] < median_time]
    
    # remove time and lat long
    X_train_subset = X_train_subset[:, 3::]
    X_test_subset = X_test_subset[:, 3::]    
    
    # fit linear model (column 1 is time, 2 and 3 are lat long, rest are indictator vars)
    clf_0 = linear_model.Ridge()
    clf_1 = linear_model.Ridge()
    clf_2 = linear_model.Ridge()
    ##clf_0 = ensemble.GradientBoostingRegressor()
    ##clf_1 = ensemble.GradientBoostingRegressor()
    ##clf_2 = ensemble.GradientBoostingRegressor()
    
    clf_0.fit(X_train_subset, y_train_subset[:, 0])
    clf_1.fit(X_train_subset, y_train_subset[:, 1])
    clf_2.fit(X_train_subset, y_train_subset[:, 2])
    
    clf_0_lst.append(clf_0)
    clf_1_lst.append(clf_1)
    clf_2_lst.append(clf_2)
    
    rmse_0 = np.sqrt(np.sum(np.square(clf_0.predict(X_test_subset) - y_test_subset[:, 0])) / y_test_subset[:, 0].size)
    rmse_1 = np.sqrt(np.sum(np.square(clf_1.predict(X_test_subset) - y_test_subset[:, 1])) / y_test_subset[:, 1].size)
    rmse_2 = np.sqrt(np.sum(np.square(clf_2.predict(X_test_subset) - y_test_subset[:, 2])) / y_test_subset[:, 2].size)
    rmse_total = np.sum(np.square(clf_0.predict(X_test_subset) - y_test_subset[:, 0])) + np.sum(np.square(clf_1.predict(X_test_subset) - y_test_subset[:, 1])) +np.sum(np.square(clf_2.predict(X_test_subset) - y_test_subset[:, 2]))
    resid_square_sum += rmse_total
    test_size += y_test_subset[:, 0].size + y_test_subset[:, 1].size + y_test_subset[:, 2].size
    rmse_total /= (y_test_subset[:, 0].size + y_test_subset[:, 1].size + y_test_subset[:, 2].size)
    rmse_total = np.sqrt(rmse_total)

    print rmse_0, rmse_1, rmse_2
    print rmse_total
    
print "RMSE =", np.sqrt(resid_square_sum / test_size)


''' Load test data '''
# load test data into dict
data = []
file_name = "../data/test.csv"
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
            sub[key] = time.mktime(time.strptime(sub[key], "%Y-%m-%d %H:%M:%S"))

print "Testing data loaded successfully!"

''' Pre-process test data '''
# select which variables to fit the model on
X = []
for d in data:
    subdict_X = dict([(i, d[i]) for i in X_vars if i in d])
    X.append(subdict_X)
    
# encode categorical variables in X
vec = DictVectorizer()
X_encoded = vec_X.transform(X).toarray()

''' Fit model to test data '''
# segment into geographic clusters
labels = km.predict(X_encoded[:, 1:3])

# loop through each geo cluster
y_predicted = np.zeros((X_encoded.shape[0], 3))
for cluster in range(0, 4):
    X_subset = X_encoded[labels == cluster]
    X_subset = X_subset[:, 3::]
    
    clf_0 = clf_0_lst[cluster]
    clf_1 = clf_1_lst[cluster]
    clf_2 = clf_2_lst[cluster]
    
    y0_predicted = clf_0.predict(X_subset)
    y1_predicted = clf_1.predict(X_subset)
    y2_predicted = clf_2.predict(X_subset)
    
    y_subset_predicted = np.vstack((y0_predicted, y1_predicted, y2_predicted)).T
    y_predicted[labels == cluster] = y_subset_predicted

# transform fit data
y_predicted = np.exp(y_predicted) + 1
print "Testing model fit!"

''' Write submission file for test data '''
raw_input("Press enter to write submission file")
id_list = [d['id'] for d in data]
id_list = np.array(id_list)[:, np.newaxis]
header_text = "id,num_views,num_votes,num_comments"
submission_fname = "../submissions/submission.csv"
output_array = np.hstack((id_list, y_predicted))
np.savetxt(submission_fname, output_array, delimiter=",", header=header_text, fmt="%f")
print "Submission file written to '", submission_fname, "'"


