import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem

class SampleAxiom(object):
    def __init__(self, params):
        pass
    
    # ts -- объект pandas.DataFrame
    # возвращает булевский pandas.Series, в котором true соотв. точкам, где аксиома выполняется
    def run(self, ts):
        return ts[0].shift(1) - ts[0] > 0

axiom_list = [SampleAxiom(1), SampleAxiom(1)]
ts = pd.DataFrame(np.random.random((10, 2)))
print(ts)
print(SampleAxiom(1).run(ts))

now = AxiomSystem(axiom_list)
print(now.perform_marking(ts))
