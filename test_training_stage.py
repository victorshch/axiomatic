import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem, MinMaxAxiom
from axiomatic.axiom_training_stage import FrequencyECTrainingStage

class SampleAxiom(object):
    def __init__(self, params):
        pass
    
    # ts -- объект pandas.DataFrame
    # возвращает булевский pandas.Series, в котором true соотв. точкам, где аксиома выполняется
    def run(self, ts):
        return ts[0].shift(1) - ts[0] > 0

stage = FrequencyECTrainingStage(None)

ts = []

for i in range(10):
    ts.append(pd.DataFrame(np.random.random((10, 2))))

stage.train({"normal": [ts[0], ts[1], ts[2]], "class1": [ts[3], ts[4], ts[5]]}, None)
