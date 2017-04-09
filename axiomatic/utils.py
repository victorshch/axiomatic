# -*- coding: utf-8 -*-

import pandas as pd


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
