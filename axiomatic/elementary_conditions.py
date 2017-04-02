import numpy as np

def form_matrix(ts, left=2, right=2, default_elem=0):
    n = len(ts)
    res = np.empty(shape=(left + right + 1, n))
    roll = np.append(ts, np.full(right + left, default_elem), axis=0)

    for i in range(-left, right + 1):
        res[i + left] = np.roll(roll, -i)[:n]
#   res = np.array([np.append(np.append(np.full(max(-i, 0), default_elem), ts[max(i, 0): min(n, n + i)]),
#          np.full(max(i, 0), default_elem)) for i in range(-left, right + 1)])
    return res

class MinMaxAxiom(object):
    num_params = 4
    cnt = 0

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        best, worst = max(data[0]), min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst, best), (0, best - worst))
    
    def run(self, ts, cache):
        matrix1 = form_matrix(ts, self.l, self.r, min(ts)) #cache
        matrix2 = form_matrix(ts, self.l, self.r, max(ts)) #cache
        res = np.logical_and(np.less_equal(matrix1.max(0), self.pmax), np.greater_equal(matrix2.min(0), self.pmin))
#        n = len(ts)
#        res = np.zeros(n)

#        for i in range(n):
#            seg = ts[max(0, i - self.l): min(n, i + self.r + 1)]
#            res[i] = self.pmin <= min(seg) and max(seg) <= self.pmax
        return res


class MaxAxiom(object):
    num_params = 2

    def __init__(self, params):
        self.l, self.r = params

    def bounds(data):
        return tuple()
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        return max(seg) <= ts[ind]


class MinAxiom(object):
    num_params = 2

    def __init__(self, params):
        self.l, self.r = params

    def bounds(data):
        return tuple()
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        return min(seg) >= ts[ind]


class ChangeAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        best = max(abs(data[0]))

        for ts in data:
            best = max(best, max(abs(ts)))
        best *= 2
        return ((0, best), (0, best))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        diff = abs(seg[1:] - seg[:-1])
        return len(diff) == 0 or self.pmin <= min(diff) and max(diff) <= self.pmax


class IntegralAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        bestsum = sum(data[0])
        worstsum = sum(data[0])

        for ts in data:
            bestsum = max(bestsum, sum(ts))
            worstsum = min(worstsum, sum(ts))
        return ((worstsum, bestsum), (0, bestsum - worstsum))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        now = sum(seg) - seg[0] / 2 - seg[len(seg) - 1] / 2
        return self.pmin <= now <= self.pmax


class RelativeChangeAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst - best, best - worst), (0, (best - worst) * 2))

    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        diff = np.full(len(seg), ts[ind]) - seg
        left = np.full(len(seg), self.pmin) * (np.full(len(seg), ind) - np.arange(max(0, ind - self.l), min(len(ts), ind + self.r + 1)))
        right = np.full(len(seg), self.pmax) * (np.full(len(seg), ind) - np.arange(max(0, ind - self.l), min(len(ts), ind + self.r + 1)))
        return min(abs(diff) - abs(left)) >= 0 and min(abs(right) - abs(diff)) >= 0


class FirstDiffAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst - best, best - worst), (0, (best - worst) * 2))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        diff = seg[1:] - seg[:-1]
        return len(diff) == 0 or self.pmin <= min(diff) and max(diff) <= self.pmax


class SecondDiffAxiom(object):
    num_params = 4

    def __init__(self, params):
        self.l, self.r, self.pmin, self.pmax = params
        self.pmax += self.pmin

    def bounds(data):
        best = max(data[0])
        worst = min(data[0])

        for ts in data:
            best = max(best, max(ts))
            worst = min(worst, min(ts))
        return ((worst * 2 - best * 2, best * 2 - worst * 2), (0, (best - worst) * 4))
    
    def run_one(self, ts, ind):
        seg = ts[max(0, ind - self.l): min(len(ts), ind + self.r + 1)]
        diff = seg[2:] + seg[:-2] - 2 * seg[1:-1]
        return len(diff) == 0 or self.pmin <= min(diff) and max(diff) <= self.pmax
