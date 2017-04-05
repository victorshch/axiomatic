import numpy as np

def form_matrix(ts, left, right):
    n = len(ts)
    res = np.empty(shape=(left + right + 1, n))
    roll = np.append(ts, np.full(right + left, np.nan), axis=0)

    for i in range(-left, right + 1):
        res[i + left] = np.roll(roll, -i)[:n]
    return res


class MinMaxAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        best, worst = max(data[0]), min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst, best), (0, best - worst))
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        return np.logical_and(np.less_equal(np.nanmax(matrix, axis=0), self.pmax), np.greater_equal(np.nanmin(matrix, axis=0), self.pmin))


class MaxAxiom(object):
    num_params = 2

    def __init__(self, params):
        self.l, self.r = params

    @classmethod
    def bounds(self, data):
        return tuple()
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        return np.nanmax(matrix, axis=0) <= ts


class MinAxiom(object):
    num_params = 2

    def __init__(self, params):
        self.l, self.r = params
    
    @classmethod
    def bounds(self, data):
        return tuple()
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        return np.nanmin(matrix, axis=0) >= ts


class ChangeAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        best = max(abs(data[0]))

        for ts in data:
            best = max(best, max(abs(ts)))
        best *= 2
        return ((0, best), (0, best))
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        diff = np.concatenate((np.full((1, len(ts)), self.pmin), abs(matrix[1:] - matrix[:-1])), axis=0)
        return np.logical_and(np.nanmax(diff, axis=0) <= self.pmax, np.nanmin(diff, axis=0) >= self.pmin)


class IntegralAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        bestsum = sum(data[0])
        worstsum = sum(data[0])

        for ts in data:
            bestsum = max(bestsum, sum(ts))
            worstsum = min(worstsum, sum(ts))
        return ((worstsum, bestsum), (0, bestsum - worstsum))
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        summ = np.nansum(np.array([np.nansum(matrix, axis=0), -matrix[0] / 2, -matrix[-1] / 2]), axis=0)
        return np.logical_and(summ >= self.pmin, summ <= self.pmax)


class RelativeChangeAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst - best, best - worst), (0, (best - worst) * 2))

    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        diff = np.repeat(ts[np.newaxis, :], self.r + self.l + 1, axis=0)
        diff[self.l:] *= -1
        matrix[self.l:] *= -1
        diff -= matrix
        rrange = abs(np.repeat(np.arange(-self.l, self.r + 1)[:, np.newaxis], len(ts), axis=1))
        return np.logical_and(np.nanmin(diff - rrange * self.pmin, axis=0) >= 0, np.nanmin(rrange * self.pmax - diff, axis=0) >= 0)

class FirstDiffAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst - best, best - worst), (0, (best - worst) * 2))
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        diff = np.concatenate((np.full((1, len(ts)), self.pmin), matrix[1:] - matrix[:-1]), axis=0)
        return np.logical_and(np.nanmin(diff, axis=0) >= self.pmin, np.nanmax(diff, axis=0) <= self.pmax)


class SecondDiffAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    @classmethod
    def bounds(self, data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst * 2 - best * 2, best * 2 - worst * 2), (0, (best - worst) * 4))
    
    def run(self, ts, cache):
        matrix = cache if len(cache) > 0 else form_matrix(ts, self.l, self.r)
        diff = np.concatenate((np.full((1, len(ts)), self.pmin), matrix[2:] + matrix[:-2] - 2 * matrix[1:-1]), axis=0)
        return np.logical_and(np.nanmin(diff, axis=0) >= self.pmin, np.nanmax(diff, axis=0) <= self.pmax)
