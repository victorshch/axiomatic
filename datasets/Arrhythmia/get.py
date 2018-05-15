import os, sys
import scipy.io
import pandas as pd
import scipy.signal
import numpy as np

def load(path, event_types, true_or_false, resample, ecg_only = False):
    res = []

    for filename in os.listdir(path):
        filename = path + '/' + filename

        if os.path.isfile(filename):
            name, ext = os.path.splitext(filename)

            if ext == '.hea':
                desc = open(filename, 'r')
                num_dim = int(desc.readline().split()[1])
                row_types = []

                for i in range(num_dim):
                    row_types.append(desc.readline().split()[-1])
                curr_event_type = desc.readline()[1 :].rstrip()
                curr_true_or_false = desc.readline()[1 : ].split()[0].lower()

                if (event_types == 'all' or curr_event_type in event_types) and curr_true_or_false == true_or_false:
                    matrix = scipy.io.loadmat(name + '.mat')['val']

                    if len(matrix[0]) != 75000:
                      continue
                    now = dict()
                    ecg_num, pulse_num = 1, 1
                    not_append = False

                    for row_type in range(len(row_types)):
                        if row_types[row_type] in ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V', 'MCL']:
                            if resample:
                              now['ECG' + str(ecg_num)] = scipy.signal.resample(matrix[row_type], 300)
                            else:
                              now['ECG' + str(ecg_num)] = matrix[row_type]

                            if min(now['ECG' + str(ecg_num)]) == max(now['ECG' + str(ecg_num)]):
                                not_append = True
                            ecg_num += 1
                        if not ecg_only and row_types[row_type] in ['PLETH', 'ABP']:
                            if pulse_num == 1:
                                if resample:
                                  now['PULSE' + str(pulse_num)] = scipy.signal.resample(matrix[row_type], 300)
                                else:
                                  now['PULSE' + str(pulse_num)] = matrix[row_type]
                                pulse_num += 1
                    
                    if not not_append:
                        res.append(pd.DataFrame(data=now))
    return res
