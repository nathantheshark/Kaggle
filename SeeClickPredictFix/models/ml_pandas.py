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
import pandas as pd
import csv
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


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
# create data frame
df = pd.DataFrame(data)

## add new columns
# add city index column, based on geographic (lat/long) clusters
km = KMeans(n_clusters = 4)
geo_coords = df[['latitude','longitude']]
km.fit(geo_coords)
cluster_centers_indices = km.cluster_centers_
labels = km.labels_
df['city'] = labels

# add binary feature based on if description exists or not (1 exists, 0 not)
has_descr = df['description'] != ""
df["has_descr"] = has_descr

# add text feature extractor to summary
vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(df['summary'])
df["summary_tfidf"] = tfidf

# transpose y vars (num_comments, num_views, num_votes) to ln(var + 1)
df["log_num_comments"] = np.log(df["num_comments"] + 1)
df["log_num_views"] = np.log(df["num_views"] + 1)
df["log_num_votes"] = np.log(df["num_votes"] + 1)

# vectorize source field to binary indicator variables
vectorizer = CountVectorizer()
source_vec = vectorizer.fit_transform(df['source'])
df['source_vec'] = source_vec

# vectorize tag_type field to binary indicator variables
vectorizer = CountVectorizer()
tag_type_vec = vectorizer.fit_transform(df['tag_type'])
df['tag_type_vec'] = tag_type_vec


''' Build the model '''
features = ['latitude','longitude','source_vec','tag_type_vec','has_descr','summary_tfidf','city']
targets = ['log_num_comments','log_num_views','log_num_votes']
X= df[features]
clf_lst = [linear_model.LinearRegression()] * len(targets)

for t in targets:
    y = df[t]
    i = targets.index(t)
    clf_lst[i].fit(X, y)
    
