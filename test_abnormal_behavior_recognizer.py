# coding=UTF-8
import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem, DummyAxiom
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

axiom_list = [DummyAxiom(), DummyAxiom()]
ts = pd.DataFrame(np.random.random((10, 2)))

now = AxiomSystem(axiom_list)
print(now.perform_marking(ts))

class1 = pd.DataFrame(np.random.random((4, 2)))
class2 = pd.DataFrame(np.random.random((4, 2)))

c1 = now.perform_marking(class1)
c2 = now.perform_marking(class2)

print(c1)
print(c2)

rec = AbnormalBehaviorRecognizer(now, {"class1": c1, "class2": c2}, {"maxdelta": 0.25, "bound": 0})
res = rec.recognize(ts)

for a, b in res:
    print(a, b)
