import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

class SampleAxiom(object):
    def __init__(self, params):
        pass
    
    # ts -- объект pandas.DataFrame
    # возвращает булевский pandas.Series, в котором true соотв. точкам, где аксиома выполняется
    def run(self, ts):
        return ts[0].shift(1) - ts[0] > 0

axiom_list = [SampleAxiom(1), SampleAxiom(1)]
ts = pd.DataFrame(np.random.random((10, 2)))

now = AxiomSystem(axiom_list)
print(now.perform_marking(ts))

class1 = pd.DataFrame(np.random.random((4, 2)))
class2 = pd.DataFrame(np.random.random((4, 2)))

print(now.perform_marking(class1))
print(now.perform_marking(class2))

rec = AbnormalBehaviorRecognizer(now, {"class1": class1, "class2": class2}, {"maxdelta": 2, "bound": 0})
res = rec.recognize(ts)

for a, b in res:
    print(a, b)
