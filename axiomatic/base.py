import numpy as np
from random import randint

class AxiomSystem(object):
    def __init__(self, axiom_list):
        self.axiom_list = axiom_list
    
    # возвращает разметку временного ряда ts как array-like (напр., np.array) с номерами аксиом, выполняющихся в соотв. точках ts
    def perform_marking(self, ts):
        axiom_result = np.empty(shape=[len(ts[0]), 0])

        for x in self.axiom_list:
            axiom_result = np.column_stack((axiom_result, x.run(ts)))
        result = np.zeros(len(ts))

        for i in range(len(ts)):
            good = False

            for j in range(len(self.axiom_list)):
                if axiom_result[(i, j)]:
                    result[i] = j
                    good = True
                    break

            if not good:
                result[i] = -1
        return result

class MinMaxAxiom(object):
    def __init__(self, params):
        self.l = params["l"]
        self.r = params["r"]
        self.pmin = params["pmin"]
        self.delta = params["delta"]

    def bounds(self, data):
        best = 0

        for ts in data:
            best = max(best, max(ts[0]))
    
    def run_one(self, ts, ind):
        for i in range(max(0, ind - self.l), min(len(ts), ind + self.r + 1)):
            if ts[0][i] > self.pmin + self.delta or ts[0][i] < self.pmin:
                return False
        return True
   #     now = ts[0][max(0, ind - self.l): min(len(ts), ind + self.r + 1)] 
   #     return min(min(self.pmin <= now, now <= self.pmin + self.delta))

    def run(self, ts):
        res = np.zeros(len(ts))

        for i in range(len(ts)):
            res[i] = self.run_one(ts, i)
        return res
    
    def static_run_one(params, data):
        freq = 0

        for ts in data:
            now = MinMaxAxiom({"pmin": params[0], "delta": params[1], "l": 0, "r": len(ts)})
            freq += now.run(ts)[0]
        return freq

    def static_run(params, *data):
        data_abnorm, data_norm = data
        return MinMaxAxiom.static_run_one(params, data_abnorm) / (1 + MinMaxAxiom.static_run_one(params, data_norm))
