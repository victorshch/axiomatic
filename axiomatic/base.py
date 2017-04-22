# -*- coding: utf-8 -*-

from axiomatic.elementary_conditions import *

QuestionMarkSymbol = -2

class AxiomSystem(object):
    def __init__(self, axiom_list):
        self.axiom_list = axiom_list

    def perform_marking(self, ts):
        """
        Возвращает разметку временного ряда ts как array-like (напр., np.array) с номерами аксиом,
        выполняющихся в соотв. точках ts
        @ts: временной ряд, который необходимо разметить
        """
        cache = {}
        
        result = np.full(ts.shape[0], -1, dtype=int)
        
        if len(self.axiom_list) == 0:
            print "Warning: perform_marking for an empty axiom system"
            return result
        
        axiom_result = np.hstack([np.array(x.run(ts).reshape(-1, 1)) for x in self.axiom_list])
        
        any_axiom_fulfilled = np.any(axiom_result, axis=1)
        min_axiom_no = np.argmax(axiom_result, axis=1)
        
        result[any_axiom_fulfilled] = min_axiom_no[any_axiom_fulfilled]

        return result
    
    def __repr__(self):
        return "AxiomSystem(" + ", ".join(repr(a) for a in self.axiom_list) + ")"


class DummyAxiom(object):
    """
    Dummy axiom is satisfied on every position in all dimensions
    """
    def run(self, ts):
        """
        Check whether axiom is satisfied.
        @param ts: pd.DataFrame containing time series
        @return: bool numpy.array, containing True values on positions where axiom is satisfied
        """
        return np.ones(ts.shape[0])
    
    def __str__(self):
        return "Dummy(" + str(id(self) % 1000) + ")"
    
    def __repr__(self):
        return "Dummy(" + str(id(self) % 1000) + ")"


class Axiom(object):
    def __init__(self, axiom=[]):
        self.dnf = [] if axiom == [] else [[axiom]]

    def run(self, ts, cache=[]):
        res = np.zeros(len(ts))

        for kf in self.dnf:
            now = np.ones(len(ts))

            for axiom in kf:
                now = np.logical_and(now, axiom.run(ts, cache))
            res = np.logical_or(res, now)
        return res

    def run_all(self, data_abnorm, data_norm, cache_abnorm, cache_norm):
        res = 0

        for i in range(len(data_abnorm)):
            res += sum(self.run(data_abnorm[i], cache_abnorm[i]))
        res /= len(data_abnorm)

        freq = 0

        for ts in range(len(data_norm)):
            freq += sum(self.run(data_norm[i], cache_norm[i]))
        freq /= len(data_norm)
        return res / (0.000005 + freq)

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
    def __init__(self, sign, dim, axiom, params=[]):
        self.sign = sign
        self.dim = dim
        self.axiom = axiom

        if len(params) == axiom.num_params:
            self.concrete_axiom = axiom(params)

    def bounds(self, data, num_part):
        bnd = self.axiom.bounds([ts[ts.columns[self.dim]].values for ts in data])
        res = tuple()

        for now in bnd:
            left, right = now
            step = (right - left) / num_part
            res += (slice(left, right, step), )
        return res

    def run(self, ts, cache=[]):
        res = self.concrete_axiom.run(ts[ts.columns[self.dim]].values, [] if len(cache) == 0 else cache[self.dim])
        
        if self.sign == 1:
            return res
        return np.logical_not(res)
    
    def static_run_one(self, params, data, cache):
        freq = 0

        for i in range(len(data)):
            now = AbstractAxiom(self.sign, self.dim, self.axiom, params)
            freq += sum(now.run(data[i], cache[i]))
        return freq / len(data)

    def static_run(self, params, *data):
        params = list(params)
        data_abnorm, data_norm, left_window, right_window, cache_abnorm, cache_norm = data
        params = [left_window, right_window] + params
        return self.static_run_one(params, data_abnorm, cache_abnorm) / (1 + self.static_run_one(params, data_norm, cache_norm))


class TrainingPipeline(object):
    """
    This class allows to run training stages consecutively
    """
    def __init__(self, stage_list):
        self.stage_list = stage_list

    def train(self, data_set):
        """
        Run stages from self.stage_list consecutively on same artifacts dict
        """
        artifacts = dict()
        for stage in self.stage_list:
            artifacts = stage.train(data_set, artifacts)
        return artifacts
