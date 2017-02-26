from scipy import optimize
from axiomatic.base import AbstractAxiom
from scipy.ndimage import maximum_filter
import numpy as np

class FrequencyECTrainingStage(object):
    def __init__(self, config):
        self.num_part = config['num_part']
        self.max_window = config['max_window']
        self.num_axioms = config['num_axioms']
        self.axiom_list = config['axiom_list']

    def train(self, data_set, artifacts):
        normal = data_set["normal"]
        maxdim = data_set["normal"][0].shape[1]

        for name in data_set:
            if name != "normal":
                for axiom in self.axiom_list:
                    for dim in range(maxdim):
                        for sign in [-1, 1]:
                            abstract_axiom = AbstractAxiom(sign, dim, axiom)
                            rranges = (slice(0, self.max_window, 1), slice(0, self.max_window, 1))
                            rranges = rranges + abstract_axiom.bounds(data_set[name] + normal, self.num_part)

                            resbrute = optimize.brute(abstract_axiom.static_run, rranges,
                                args=(data_set[name], normal), full_output=True, finish=optimize.fmin)

                            res = (resbrute[3] == maximum_filter(resbrute[3], size=3)).ravel()
                            
                            axioms = list(zip(res, resbrute[3].ravel(), *[param.ravel() for param in resbrute[2]]))
                            axioms.sort(reverse=True)

                            high = sorted(res, reverse=True).index(False) if False in res else len(res)

                            axioms = axioms[0: min(self.num_axioms, high)]
                            axioms = [AbstractAxiom(sign, dim, axiom, params[2:]) for params in axioms]
                            artifacts["axioms"] += axioms
        
        return artifacts
