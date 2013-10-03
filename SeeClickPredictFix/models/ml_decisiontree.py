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
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation

''' Functions and classes '''
def loss(p_num_votes, p_num_comments, p_num_views, a_num_votes, a_num_comments, a_num_views):
    # check if shape is the same
    n = p_num_votes.size + p_num_comments.size + p_num_views.size
    log_p_num_votes = np.log(p_num_votes + 1)
    log_p_num_comments = np.log(p_num_comments + 1)
    log_p_num_views = np.log(p_num_views + 1)
    log_a_num_votes = np.log(a_num_votes + 1)
    log_a_num_comments = np.log(a_num_comments + 1)
    log_a_num_views = np.log(a_num_views + 1)
    print (log_p_num_votes).shape
    print (log_a_num_votes).shape
    print log_p_num_votes
    print log_a_num_votes
    raw_input()
    sum_num_votes = np.sum(np.square(log_p_num_votes - log_a_num_votes))
    sum_num_comments = np.sum(np.square(log_p_num_comments - log_a_num_comments))
    sum_num_views = np.sum(np.square(log_p_num_views - log_a_num_views))
    rmsle = np.sqrt((sum_num_votes + sum_num_comments + sum_num_views) / n)
    return rmsle

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
            sub[key] = time.strptime(sub[key], "%Y-%m-%d %H:%M:%S")

print "Training data loaded successfully!"

''' Pre-process training data '''
# select which variables to fit the model on
X_vars = ['latitude', 'longitude', 'source']
y_vars = ['num_votes']
X = []
y = []
for d in data:
    subdict_X = dict([(i, d[i]) for i in X_vars if i in d])
    subdict_y = dict([(i, d[i]) for i in y_vars if i in d])
    X.append(subdict_X)
    y.append(subdict_y)
    
# encode categorical variables in X
vec = DictVectorizer()
X_encoded = vec.fit_transform(X).toarray()
# convert y into proper numpy array
y_encoded = vec.fit_transform(y).toarray()
# separate training/testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_encoded, y_encoded, test_size=0.3)
print y_train.shape
print y_test.shape

''' Begin building the model on training data'''
# decision tree model
clf = DecisionTreeRegressor(max_depth=2)
clf.fit(X_train, y_train)
y_train_predicted = clf.predict(X_train)
y_test_predicted = clf.predict(X_test)[0:]
print y_train_predicted.shape
print y_test_predicted.shape
raw_input()
train_loss = loss(y_train_predicted, np.zeros(y_train.shape),  np.zeros(y_train.shape),  y_train, np.zeros(y_train.shape),  np.zeros(y_train.shape))
test_loss = loss(y_test_predicted, np.zeros(y_test.shape), np.zeros(y_test.shape), y_test, np.zeros(y_test.shape), np.zeros(y_test.shape))
print "Train loss =", train_loss
print "Test loss =", test_loss