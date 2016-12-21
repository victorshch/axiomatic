from scipy import optimize
from axiomatic.base import MinMaxAxiom
from scipy.ndimage import maximum_filter
import numpy as np

class FrequencyECTrainingStage(object):
    def __init__(self, config):
        pass

    def train(self, data_set, artifacts):
        normal = data_set["normal"]

        for name in data_set:
            if name != "normal":
                rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
                resbrute = optimize.brute(MinMaxAxiom.static_run, rranges, args=(data_set[name], normal), full_output=True, finish=optimize.fmin)
                res = (resbrute[2] == maximum_filter(resbrute[2], size=3))

                res = res.ravel()
                pmin = resbrute[3][0].ravel()
                pmax = resbrute[3][1].ravel()
                params = np.dstack((pmin, pmax))

                print(res)

                print(pmin)
                print(pmax)
        return artifacts
