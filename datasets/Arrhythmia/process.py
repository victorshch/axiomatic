import os

strange_types = dict()
abnormal_events = {'Tachycardia' : {}, 'Ventricular_Tachycardia' : {}, 'Asystole' : {}, 'Ventricular_Flutter_Fib' : {}, 'Bradycardia' : {}}

for event in abnormal_events:
  basic_set = {'abp_pleth' : 0, 'abp' : 0, 'pleth' : 0}
  abnormal_events[event] = {'false' : [0, basic_set], 'true' : [0, basic_set]}

for filename in os.listdir('.'):
    if os.path.isfile(filename):
        name, ext = os.path.splitext(os.path.basename(filename))
        
        if ext == '.hea':
            desc = open(filename, 'r')
            num_dim = int(desc.readline().split()[1])
            types = []

            for i in range(num_dim):
                row_type = desc.readline().split()[-1]

                if row_type == 'PLETH' or row_type == 'ABP':
                    types.append(row_type.lower())
                elif row_type not in ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V']:
                    if row_type in strange_types:
                        strange_types[row_type] += 1
                    else:
                        strange_types[row_type] = 1
            event_type = desc.readline()[1 :].rstrip()
            true_or_false = desc.readline()[1 : ].split()[0].lower()
            row_types = '_'.join(sorted(types))
            abnormal_events[event_type][true_or_false][1][row_types] += 1
            abnormal_events[event_type][true_or_false][0] += 1

print(abnormal_events)
print()
print(strange_types)
