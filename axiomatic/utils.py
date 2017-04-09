# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def dtw_distances_from_matrix(D):
    N = D.shape[0]
    Nmax = D.shape[1]
    S = np.zeros(D.shape)
    R = np.zeros(D.shape)
    S[0, 0] = D[0, 0]
    R[0, 0] = 1
    for a in xrange(1, N):
        S[a, 0] = D[a, 0] + S[a - 1, 0]
        R[a, 0] = 1 + R[a - 1, 0]
    for b in xrange(1, Nmax):
        # S[N - 1, b] = D[N - 1, b] + S[N - 1, b + 1]
        # R[N - 1, b] = R[N - 1, b + 1] + 1
        S[0, b] = D[0, b]
        R[0, b] = 1

    for a in xrange(1, N):
        for b in xrange(1, Nmax):
            Dab = D[a, b]
            Sa1b1 = S[a - 1, b - 1]
            Ra1b1 = R[a - 1, b - 1]
            Sab1 = S[a, b - 1]
            Rab1 = R[a, b - 1]
            Sa1b = S[a - 1, b]
            Ra1b = R[a - 1, b]
            diag = (Dab + Sa1b1) * (Ra1b1 + 1)
            right = (Dab + Sab1) * (Rab1 - 1)
            down = (Dab + Sa1b) * (Ra1b + 1)
            if down < diag and down < right:
                S[a, b] = Dab + Sa1b
                R[a, b] = Ra1b + 1
            elif diag <= down and diag <= right:
                S[a, b] = Dab + Sa1b1
                R[a, b] = Ra1b1 + 1
            elif right < diag and right <= down:
                S[a, b] = Dab + Sab1
                R[a, b] = Rab1 + 1
            else:
                print "Warning: dtw_from_matrix() strange behavior"

    dist = S[N - 1, :] / R[N - 1, :]
    return dist


def dtw_distances(model, observed_marking, metric):
    distancesM = pairwise_distances(
        np.array(model).reshape(-1, 1),
        np.array(observed_marking).reshape(-1, 1),
        metric=metric,
    )
    return dtw_distances_from_matrix(distancesM)


def time_series_embedding(series, left_neighbourhood, right_neighbourhood):
    """
    @param series a single-dimension TS as a pandas.Series object
    @param left_neighbourhood the size of left neighborhood (not counting the central point)
    @param right_neighbourhood the size of right neighborhood (not counting the central point)
    @returns a pandas.DataFrame containing copies of series with shifts so that each row corresponds to 
    a neighborhood and the row position corresponds to central point of the neighborhood in the original series.
    Rows where the neighborhood exceeds series limits contain NaN values. The part where neighborhood doesn't
    exceed TS limits corresponds to res.iloc[left_neighbourhood:-right_neighbourhood]
    
    Example:
    
    In [1]: series
    Out[1]:
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64

    In [2]: time_series_embedding(series, 1, 2)
    Out[2]:
        -1    0    1    2
    0  NaN  1.0  2.0  3.0
    1  1.0  2.0  3.0  4.0
    2  2.0  3.0  4.0  5.0
    3  3.0  4.0  5.0  NaN
    4  4.0  5.0  NaN  NaN
    """
    ts_embedding_list = [(-shift_value, series.shift(shift_value)) for shift_value in
                         xrange(left_neighbourhood,  -right_neighbourhood - 1, -1)]
    
    return pd.DataFrame(dict(ts_embedding_list))
