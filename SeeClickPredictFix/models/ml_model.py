''' Libraries and globals '''
import numpy as np
import csv

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

''' Load data '''
# create empty lists to add data to
unique_id = []
latitude = []
longitude = []
summary = []
description = []
num_votes = []
num_comments = []
num_views = []
source = []
created_time = []
tag_type = []

# read training data
file_name = "../data/train.csv"
reader = csv.reader(open(file_name, 'rb'), delimiter=',', quotechar='"')
reader.next()       # skip header row
for row in reader:
    unique_id.append(int(row[0]))
    latitude.append(float(row[1]))
    longitude.append(float(row[2]))
    summary.append(row[3])
    description.append(row[4])
    num_votes.append(int(row[5]))
    num_comments.append(int(row[6]))
    num_views.append(int(row[7]))
    source.append(row[8])
    created_time.append(row[9])
    tag_type.append(row[10])

#  convert to numpy arrays
unique_id = np.array(unique_id)
latitude = np.array(latitude)
longitude = np.array(longitude)
summary = np.array(summary)
#description = np.array(description)
num_votes = np.array(num_votes)
num_comments = np.array(num_comments)
num_views = np.array(num_views)
source = np.array(source)
created_time = np.array(created_time)
tag_type = np.array(tag_type)

print "Training data loaded successfully!"

''' Pre-process training data '''
X = np.vstack((latitude, longitude, summary, source, created_time, tag_type)).T
y = np.vstack((num_votes, num_comments, num_views)).T
print X
print y

''' Begin building the model on training data'''

