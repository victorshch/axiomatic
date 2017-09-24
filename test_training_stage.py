# coding=UTF-8
import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem, MinMaxAxiom, IntegralAxiom
from axiomatic.axiom_training_stage import FrequencyECTrainingStage, FrequencyAxiomTrainingStage

stage = FrequencyECTrainingStage({'num_part': 5, 'left_window': 5, 'right_window': 5, 'num_axioms': 10, 'axiom_list': [MinMaxAxiom, IntegralAxiom], 'enable_cache': True})

ts = []

for i in range(10):
    ts.append(pd.DataFrame(np.random.random((10, 1))))

for i in range(6):
    if i == 0:
        print('normal:')
    elif i == 3:
        print()
        print('class1:')
    print(' '.join(list(map(str, ts[i][0]))))
print()
print("axioms:")

artifacts = stage.train({"train": {"normal": [ts[0], ts[1], ts[2]], "class1": [ts[3], ts[4], ts[5]]}}, {"axioms": []})

stage = FrequencyAxiomTrainingStage({'num_axioms': 10, 'max_depth': 5, 'num_step_axioms': 10})
artifacts = stage.train({"train": {"normal": [ts[0], ts[1], ts[2]], "class1": [ts[3], ts[4], ts[5]]}}, artifacts)

print(artifacts["axioms"])
