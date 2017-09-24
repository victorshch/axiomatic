# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from axiomatic.base import AxiomSystem
from axiomatic.elementary_conditions import MinMaxAxiom

# l, r, pmin, pmax
params = [1,  1, -0.8,  0.8]
axiom_list = [MinMaxAxiom(params)]
ts = pd.DataFrame(np.random.random((10, 2)))
print(ts)
print(MinMaxAxiom(params).run(ts, dict()))

now = AxiomSystem(axiom_list)
print(now.perform_marking(ts))
