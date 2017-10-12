import os, sys
import scipy.io
import pandas as pd

event_type = sys.argv[2]
true_or_false = sys.argv[3]
res = []

for filename in os.listdir(sys.argv[1]):
    if os.path.isfile(filename):
        name, ext = os.path.splitext(os.path.basename(filename))

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
                now = dict()

                for row_type in range(len(row_types)):
                    now[row_types[row_type]] = matrix[row_type]
                res.append(pd.DataFrame(data=now))

print(res)
print()
print(len(res))
