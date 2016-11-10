from axiomatic.objective_function import ObjectiveFunction
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

c1 = now.perform_marking(class1)
c2 = now.perform_marking(class2)

print(c1)
print(c2)

rec = AbnormalBehaviorRecognizer(now, {"class1": c1, "class2": c2}, {"maxdelta": 0.1, "bound": 0})
res = rec.recognize(ts)

for a, b in res:
    print(a, b)

now = ObjectiveFunction(1, 0)
print(now.calculate_one(rec, ts, "class1"))

now = ObjectiveFunction(0, 1)
print(now.calculate_one(rec, ts, "class1"))

now = ObjectiveFunction(1, 2)
print(now.calculate(rec, {"class1" : [ts, ts, ts]}))
