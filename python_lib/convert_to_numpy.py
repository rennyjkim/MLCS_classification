# numpy

import os, glob
import numpy as np
import csv

path_to_train_2d_datasets = os.path.join('..', '001 csv', 'outputs', 'train', '*.csv')
path_to_train2_2d_datasets = os.path.join('..', '001 csv', 'outputs', 'train_2', '*.csv')
path_to_test_2d_datasets = os.path.join('..', '001 csv', 'outputs', 'test', '*.csv')

train_2d_files = glob.glob(path_to_train_2d_datasets)
train2_2d_files = glob.glob(path_to_train2_2d_datasets)
test_2d_files = glob.glob(path_to_test_2d_datasets)

train_data = np.empty((len(train_2d_files), 28, 28))
train2_data = np.empty((len(train2_2d_files), 28, 28))
test_data = np.empty((len(test_2d_files), 28, 28))

train_label = np.empty((len(train_2d_files)), dtype=np.uint8)
train2_label = np.empty((len(train2_2d_files)), dtype=np.uint8)
test_label = np.empty((len(test_2d_files)), dtype=np.uint8)

for data_idx, data_path in enumerate(train_2d_files):
    replaced_path = data_path.replace('../001 csv/outputs/train/#','')
    label = replaced_path[0]
    train_label[data_idx] = label
    csv_data = csv.reader(open(data_path))
    tmp = []
    for row in csv_data:
        tmp.append(row)
    for i in range(28):
        for j in range(28):
            train_data[data_idx, i, j] = tmp[i][j]

for data_idx, data_path in enumerate(train2_2d_files):
    replaced_path = data_path.replace('../001 csv/outputs/train_2/#','')
    label = replaced_path[0]
    train2_label[data_idx] = label
    csv_data = csv.reader(open(data_path))
    tmp = []
    for row in csv_data:
        tmp.append(row)
    for i in range(28):
        for j in range(28):
            train2_data[data_idx, i, j] = tmp[i][j]


for data_idx, data_path in enumerate(test_2d_files):
    replaced_path = data_path.replace('../001 csv/outputs/test/#','')
    label = replaced_path[0]
    test_label[data_idx] = label
    csv_data = csv.reader(open(data_path))
    tmp = []
    for row in csv_data:
        tmp.append(row)
    for i in range(28):
        for j in range(28):
            test_data[data_idx, i, j] = tmp[i][j]

data = np.concatenate((train_data, test_data), axis=0)
label = np.concatenate((train_label, test_label), axis=0)

shuffle = np.arange(data.shape[0])
np.random.shuffle(shuffle)
shuffled_data = data[shuffle]
shuffled_label = label[shuffle]

[train, valid, test] = np.split(data, [int(data.shape[0]*0.7), int(data.shape[0]*0.9)])
[train_l, valid_l, test_l] = np.split(label, [int(label.shape[0]*0.7), int(label.shape[0]*0.9)])

np.savez('train_x', train_x = train)
np.savez('train_y', train_y = train_l)

np.savez('valid_x', valid_x = valid)
np.savez('valid_y', valid_y = valid_l)

np.savez('test_x', test_x = test)
np.savez('test_y', test_y = test_l)

print("fin")