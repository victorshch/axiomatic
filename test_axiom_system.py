import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem, MinMaxAxiom

class SampleAxiom(object):
    def __init__(self, params):
        pass
    
    # ts -- объект pandas.DataFrame
    # возвращает булевский pandas.Series, в котором true соотв. точкам, где аксиома выполняется
    def run(self, ts):
        return ts[0].shift(1) - ts[0] > 0

params = {"l": 1, "r": 1, "pmin": -0.8, "pmax": 0.8}
axiom_list = [MinMaxAxiom(params)]
ts = pd.DataFrame(np.random.random((10, 2)))
print(ts)
print(MinMaxAxiom(params).run(ts))

now = AxiomSystem(axiom_list)
print(now.perform_marking(ts))
