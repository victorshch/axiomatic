import os, sys
import scipy.io
import pandas as pd
import scipy.signal
import numpy as np

def load(path, event_type, true_or_false):
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

                if curr_event_type == event_type and curr_true_or_false == true_or_false:
                    matrix = scipy.io.loadmat(name + '.mat')['val']

                    if len(matrix[0]) != 75000:
                      continue
                    now = dict()
                    ecg_num, pulse_num = 1, 1
                    not_append = False

                    for row_type in range(len(row_types)):
                        if row_types[row_type] in ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V', 'MCL']:
                            now['ECG' + str(ecg_num)] = scipy.signal.resample(matrix[row_type], 300)

                            if min(now['ECG' + str(ecg_num)]) == 0 and max(now['ECG' + str(ecg_num)]) == 0:
                                not_append = True
                            ecg_num += 1
                        if row_types[row_type] in ['PLETH', 'ABP']:
                            if pulse_num == 1:
                                now['PULSE' + str(pulse_num)] = scipy.signal.resample(matrix[row_type], 300)
                                pulse_num += 1
                    
                    if not not_append:
                        res.append(pd.DataFrame(data=now))
    return res
    
#res = load(sys.argv[1], sys.argv[2], sys.argv[3])
#print(res)
#print()
#print(len(res))
