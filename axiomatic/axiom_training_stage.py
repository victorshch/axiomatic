from scipy import optimize
from axiomatic.base import AbstractAxiom, Axiom
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
        artifacts = dict()
        artifacts["axioms"] = dict()

        for name in data_set:
            if name != "normal":
                class_axioms = []

                for dim in range(maxdim):
                    dim_axioms = []

                    for axiom in self.axiom_list:
                        for sign in [-1, 1]:
                            abstract_axiom = AbstractAxiom(sign, dim, axiom)
                            rranges = (slice(0, len(normal[0]), 1), slice(0, self.max_window, 1))
                            rranges = rranges + abstract_axiom.bounds(data_set[name] + normal, self.num_part)

                            resbrute = optimize.brute(abstract_axiom.static_run, rranges,
                                args=(data_set[name], normal), full_output=True, finish=optimize.fmin)

                            res = (resbrute[3] == maximum_filter(resbrute[3], size=3)).ravel()
                            
                            axioms = list(zip(res, resbrute[3].ravel(), *[param.ravel() for param in resbrute[2]]))
                            axioms = [(param[0], param[1], int(param[2]), int(param[3]), *param[4:]) for param in axioms]
                            axioms.sort(reverse=True)

                            high = sorted(res, reverse=True).index(False) if False in res else len(res)

                            axioms = axioms[0: min(self.num_axioms, high)]

                            axioms = [(params, sign, dim) for params in axioms]
                            dim_axioms += axioms
                    
                    dim_axioms.sort(reverse=True)
                    dim_axioms = dim_axioms[0: self.num_axioms]
                    print(dim_axioms)
                    dim_axioms = [AbstractAxiom(sign, dim, axiom, params[2:]) for params, sign, dim in dim_axioms]
                    
                    class_axioms += dim_axioms
                artifacts["axioms"][name] = class_axioms
        return artifacts


class FrequencyAxiomTrainingStage:
    def __init__(self, config):
        self.num_axioms = config['num_axioms']
        self.num_step_axioms = config['num_step_axioms']
        self.max_depth = config['max_depth']

    def train(self, data_set, artifacts):
        axioms = artifacts["axioms"]
        artifacts["full_axioms"] = dict()
        normal = data_set["normal"]

        for name in axioms:
            now = [Axiom(x) for x in axioms[name]]
            nums = [x.run_all(data_set[name], normal) for x in now]
            
            for i in range(self.max_depth):
                add = []

                for pos1 in range(len(now)):
                    for pos2 in range(pos1 + 1, len(now)):
                        axiom_or = now[pos1].logical_or(now[pos2])
                        axiom_and = now[pos1].logical_and(now[pos2])
                        num_or = axiom_or.run_all(data_set[name], normal)
                        num_and = axiom_and.run_all(data_set[name], normal)

                        if max(num_or, num_and) > min(nums[pos1], nums[pos2]):
                            add.append(axiom_or if num_or > num_and else axiom_and)
                now += add
                nums = [x.run_all(data_set[name], normal) for x in now]
                res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_step_axioms]

                now = [x[1] for x in res]
                nums = [x[0] for x in res]

            res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_axioms]
            res = [x[1] for x in res]
            artifacts["full_axioms"][name] = res
        return artifacts
