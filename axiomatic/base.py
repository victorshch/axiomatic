import numpy as np

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
        self.pmax = params["pmax"]

    def run(self, ts):
        res = np.zeros(len(ts))
        print(len(ts))

        for i in range(len(ts)):
            res[i] = 1

            for j in range(max(0, i - self.l), min(len(ts), i + self.r + 1)):
                for k in range(len(ts.columns)):
                    if self.pmin > ts[k][j] or self.pmax < ts[k][j]:
                        res[i] = 0
                        break
        return res
