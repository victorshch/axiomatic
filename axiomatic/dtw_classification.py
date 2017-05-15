# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean

from axiomatic.utils import transform_dataset


def dtw_distance_from_matrix(D):
    dtw_dist = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
    dtw_dist[0, :] = np.inf
    dtw_dist[:, 0] = np.inf
    dtw_dist[0][0] = 0.0
    R = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
    R[0][0] = -1

    for i in xrange(1, dtw_dist.shape[0]):
        for j in xrange(1, dtw_dist.shape[1]):
            possible_dists = [dtw_dist[i-1, j], dtw_dist[i, j-1], dtw_dist[i-1, j-1]]
            min_idx = np.argmin(possible_dists)

            dtw_dist[i][j] = D[i-1][j-1] + possible_dists[min_idx]

            if min_idx == 0:
                R[i][j] = R[i-1][j] + 1
            elif min_idx == 1:
                R[i][j] = R[i][j-1] + 1
            elif min_idx == 2:
                R[i][j] = R[i-1][j-1] + 1

    return dtw_dist[D.shape[0]][D.shape[1]] / R[D.shape[0]][D.shape[1]]


def dtw_distance(ts1, ts2):
    """
    Calculate DTW distance
    @param ts1: pd.DataFrame representing time series
    @param ts2: pd.DataFrame representing time series
    @return: dtw distance between time series
    """
    ts1_val = ts1.values
    ts2_val = ts2.values

    D = np.zeros((len(ts1_val), len(ts2_val)))

    for i in xrange(len(ts1_val)):
        for j in xrange(len(ts2_val)):
            D[i][j] = euclidean(ts1_val[i], ts2_val[j])

    return dtw_distance_from_matrix(D)


class TimeSeriesDTW1NNClassifier(object):
    def __init__(self, dataset):
        train_ts_list, train_labels = transform_dataset(dataset['train'])
        val_ts_list, val_labels = transform_dataset(dataset['validate'])

        train_ts_list.extend(val_ts_list)
        train_labels.extend(val_labels)

        self.ts_list = train_ts_list
        self.labels = train_labels

    def predict(self, ts_list):
        labels = []
        min_dist = np.zeros(len(ts_list))
        min_dist[:] = np.inf

        for i, ts in enumerate(ts_list):
            print i+1, 'out of', len(ts_list)

            cur_label = 0
            for j, train_ts in enumerate(self.ts_list):
                dist = dtw_distance(ts, train_ts)
                if dist < min_dist[i]:
                    min_dist[i] = dist
                    cur_label = j
            labels.append(self.labels[cur_label])

        return np.array(labels)
