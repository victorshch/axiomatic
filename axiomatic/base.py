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


class Axiom(object):
    def __init__(self, axiom = []):
        self.dnf = [] if axiom == [] else [[axiom]]

    def run(self, ts):
        res = np.zeros(len(ts))

        for kf in self.dnf:
            now = np.ones(len(ts))

            for axiom in kf:
                now = np.logical_and(now, axiom.run(ts))
            res = np.logical_or(res, now)
        return res

    def run_all(self, data_abnorm, data_norm):
        res = 0

        for ts in data_abnorm:
            res += sum(self.run(ts))
        res /= len(data_abnorm)

        freq = 0

        for ts in data_norm:
            freq += sum(self.run(ts))
        freq /= len(data_norm)
        return res / (1 + freq)

    def logical_or(self, another):
        res = Axiom()
        res.dnf = self.dnf + another.dnf
        return res

    def logical_and(self, another):
        res = Axiom()

        for x1 in self.dnf:
            for x2 in another.dnf:
                res.dnf.append(x1 + x2)
        return res


class AbstractAxiom(object):
    def __init__(self, sign, dim, axiom, params = []):
        self.sign = sign
        self.dim = dim
        self.axiom = axiom

        if len(params) == axiom.num_params:
            self.concrete_axiom = axiom(params)

    def bounds(self, data, num_part):
        bnd = self.axiom.bounds([ts[self.dim].values for ts in data])
        res = tuple()

        for now in bnd:
            left, right = now
            step = (right - left) / num_part
            res = res + (slice(left, right, step), )
        return res

    def run(self, ts):
        res = np.zeros(len(ts))

        for i in range(len(ts)):
            res[i] = self.concrete_axiom.run_one(ts[self.dim].values, i)
        
        if self.sign == 1:
            return res
        return np.logical_not(res)
    
    def static_run_one(self, params, data):
        freq = 0

        for ts in data:
            now = AbstractAxiom(self.sign, self.dim, self.axiom, params)
            freq += sum(now.run(ts))
        return freq / len(data)

    def static_run(self, params, *data):
        params = list(params)
        params[0] = int(params[0])
        params[1] = int(params[1])

        data_abnorm, data_norm = data
        return self.static_run_one(params, data_abnorm) / (1 + self.static_run_one(params, data_norm))
    

class MinMaxAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.r += self.l
        self.pmax += self.pmin

    def bounds(data):
        best, worst = max(data[0]), min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst, best), (0, best - worst))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        return self.pmin <= min(seg) and max(seg) <= self.pmax


class IntegralAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.r += self.l
        self.pmax += self.pmin

    def bounds(data):
        bestsum = sum(data[0])

        for ts in data:
            bestsum = max(bestsum, sum(ts))
        return ((0, bestsum), (0, bestsum))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        now = sum(seg) - seg[0] / 2 - seg[len(seg) - 1] / 2
        return self.pmin <= now <= self.pmax
