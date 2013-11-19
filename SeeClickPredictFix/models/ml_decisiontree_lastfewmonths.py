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
from sklearn.metrics import make_scorer

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
    sum_num_votes = np.sum(np.square(log_p_num_votes - log_a_num_votes))
    sum_num_comments = np.sum(np.square(log_p_num_comments - log_a_num_comments))
    sum_num_views = np.sum(np.square(log_p_num_views - log_a_num_views))
    rmsle = np.sqrt((sum_num_votes + sum_num_comments + sum_num_views) / n)
    return rmsle

def scorer_loss(y, y_predicted):
    rmsle = loss(y_predicted[:, 0], y_predicted[:, 1], y_predicted[:, 2], y[:, 0], y[:, 1], y[:, 2])
    return rmsle

##def scorer_rmsle(clf, X, y):
    ##clf.fit(X, y)
    ##y_predicted = clf.predict(X)
    ##rmsle = loss(y_predicted[:, 0], y_predicted[:, 1], y_predicted[:, 2], y[:, 0], y[:, 1], y[:, 2])
    ##return rmsle
    

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

print "month cutoff, depth, test score mean, test score stdev"
# remove all data before time_cutoff
for month in range(3, 4+1):
    #month = 4
    time_cutoff = time.mktime(time.strptime("2013-" + str(month) + "-1 0:0:0", "%Y-%m-%d %H:%M:%S"))
    created_time_index = vec_X.get_feature_names().index('created_time')
    X_encoded = X_encoded[X_encoded[:, created_time_index] > time_cutoff]
    y_encoded = y_encoded[X_encoded[:, created_time_index] > time_cutoff]
    
    # separate training/testing data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_encoded, y_encoded, test_size=0.3)
    # remove unneeded dimension
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    
    ''' Begin building the model on training data'''
    # decision tree model
    for depth in range(1, 25+1):
        #depth = 15 # 12 is optimal for num_votes
        #print "max_depth =", depth
        clf = DecisionTreeRegressor(max_depth=depth)
        clf.fit(X_train, y_train)
        y_train_predicted = clf.predict(X_train)
        y_test_predicted = clf.predict(X_test)
        train_loss = loss(y_train_predicted[:, 0], y_train_predicted[:, 1],  y_train_predicted[:, 2],  y_train[:, 0], y_train[:, 1],  y_train[:, 2])
        test_loss = loss(y_test_predicted[:, 0], y_test_predicted[:, 1], y_test_predicted[:, 2], y_test[:, 0], y_test[:, 1], y_test[:, 2])
        #print "month cutoff =", month
        #print "Train loss =", train_loss
        #print "Test loss =", test_loss
        
        # cross validation
        clf_new = DecisionTreeRegressor(max_depth=depth)
        rmsle_scorer = make_scorer(scorer_loss, greater_is_better=False)
        n_samples = X_encoded.shape[0]
        cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=0.3)
        scores = cross_validation.cross_val_score(clf_new, X_encoded, y=y_encoded, scoring=rmsle_scorer, cv=cv)
        scores = -scores
        #print scores
        #print "test score mean =", scores.mean()
        #print "test score stdev =", scores.std()
        print month, depth, scores.mean(), scores.std()
        clf_new.fit(X_encoded, y_encoded)

raw_input("!!!!")

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

# remove all data before time_cutoff
X_encoded = X_encoded[X_encoded[:, created_time_index] > time_cutoff]

''' Fit model to test data '''
y_predicted = clf.predict(X_encoded)
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
