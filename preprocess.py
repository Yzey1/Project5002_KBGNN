import pandas as pd
import numpy as np
import random
import pickle as pkl
from tqdm import tqdm
import os
from math import cos, asin, sqrt, pi


def distance(lat1, lon1, lat2, lon2):
    ''' Calculate the distance between two points on the earth using Haversine formula

    Args:
        lat1, lon1: latitude and longitude of the first point
        lat2, lon2: latitude and longitude of the second point

    Returns:
        The distance between two points in kilometers
    '''
    # radius of the earth in km
    r = 6371
    # convert factor from degree to radian
    p = pi / 180
    # calculate the haversine theta according to the formula
    hav_theta = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * \
        cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(hav_theta))


# src_path = './dataset_tsmc2014/dataset_TSMC2014_TKY.txt'
src_path = './dataset_tsmc2014/dataset_TSMC2014_NYC.txt'
# dst_path = './processed_data/tky/raw/'
dst_path = './processed_data/nyc/raw/'

# create the destination directory if not exist
os.makedirs(dst_path, exist_ok=True)

# set random seed
random.seed(42)

# the columns of the dataset
col_names = ['uid', 'poi', 'cat_id', 'cat_name',
             'latitude', 'longitude', 'offset', 'time']

print('Reading data...')

# read the data
data = pd.read_csv(src_path, sep='\t', header=None,
                   names=col_names, encoding='unicode_escape')

# remove the columns that are not needed
data.drop(['cat_id', 'cat_name', 'offset', 'time'], axis=1, inplace=True)

# count the number of users and POIs
num_user = pd.unique(data['uid']).shape[0]
num_poi = pd.unique(data['poi']).shape[0]
print("#Users: {}".format(num_user))
print("#POIs: {}".format(num_poi))

# map the user and POI to a continuous index
uid_map = dict(zip(pd.unique(data['uid']), range(num_user)))
poi_map = dict(zip(pd.unique(data['poi']), range(num_poi)))
data['uid'] = data['uid'].map(uid_map)
data['poi'] = data['poi'].map(poi_map)

print('Finish reading data.')

print('Generating dataset...')

sum_seqlen=0

# the coordinates of each POI
coords = {poi: None for poi in range(num_poi)}
for poi, item in data.groupby('poi'):
    lat, lon = item['latitude'].iloc[0], item['longitude'].iloc[0]
    coords[poi] = (lat, lon)

# generate training, testing and validation set
train_set, eval_set = [], []
for uid in range(num_user):
    # the sequence of POIs that the user visits
    true_seq = data[data['uid'] == uid]['poi'].tolist()
    true_seq_set = set(true_seq)

    # calculate the sequence length 
    seqlen=len(true_seq)
    # calculate the sum of sequence length, this is also the interactions in this sequence
    sum_seqlen+=seqlen    

    # take only the POIs that the user never visits
    false_seq = []
    while len(false_seq) < len(true_seq):
        poi = random.randint(0, num_poi - 1)
        if poi not in true_seq_set:
            false_seq.append(poi)

    for i in range(1, len(true_seq) - 1):
        train_set.append(
            (uid, true_seq[i], true_seq[:i], coords[true_seq[i]], 1))
        train_set.append(
            (uid, false_seq[i], true_seq[:i], coords[false_seq[i]], 0))

    # we use the last POI of a user as the evaluation set
    eval_set.append(
        (uid, true_seq[-1], true_seq[:-1], coords[true_seq[-1]], 1))
    eval_set.append(
        (uid, false_seq[-1], true_seq[:-1], coords[false_seq[-1]], 0))

print(f'avgSeqLen = {sum_seqlen/num_user}')
print(f'interactions = {sum_seqlen}')       
    
    
# random shuffle the training and evaluation set
random.shuffle(train_set)
random.shuffle(eval_set)

# split the evaluation set into validation and testing set
sep = len(eval_set) // 2
val_set = eval_set[:sep]
test_set = eval_set[sep:]

print(f'#Train: {len(train_set)}')
print(f'#Validation: {len(val_set)}')
print(f'#Test: {len(test_set)}')

# save the datasets
with open(dst_path+'train.pkl', 'wb') as f:
    pkl.dump(train_set, f, pkl.HIGHEST_PROTOCOL)
with open(dst_path+'test.pkl', 'wb') as f:
    pkl.dump(test_set, f, pkl.HIGHEST_PROTOCOL)
with open(dst_path+'val.pkl', 'wb') as f:
    pkl.dump(val_set, f, pkl.HIGHEST_PROTOCOL)
with open(dst_path+'info.pkl', 'wb') as f:
    pkl.dump((num_user, num_poi), f, pkl.HIGHEST_PROTOCOL)

print('Finish generating dataset.')

print('Generating neighborhood graph...')

# only regard the POIs with distance less or equal than 0.5km as neighbors
threshold = 0.5
edges = [[], []]

for i in tqdm(range(num_poi)):
    for j in range(i + 1, num_poi):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        if distance(lat1, lon1, lat2, lon2) <= threshold:
            edges[0].append(i)
            edges[1].append(j)

# convert the list to numpy array, size: (2, num_edges)
edges = np.array(edges)

# save the neighborhood graph
with open(dst_path+'dist_graph.pkl', 'wb') as f:
    pkl.dump(edges, f, pkl.HIGHEST_PROTOCOL)

# the distance of each edge, size: (num_edges,)
dist_on_graph = np.array([distance(coords[edges[0, i]][0], coords[edges[0, i]][1],
                         coords[edges[1, i]][0], coords[edges[1, i]][1]) for i in range(edges.shape[1])])

np.save(dst_path + 'dist_on_graph.npy', dist_on_graph)

print('Finish generating neighborhood graph.')
