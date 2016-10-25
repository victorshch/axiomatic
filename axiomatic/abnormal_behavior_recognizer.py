import numpy as np

class AbnormalBehaviorRecognizer(object):
    # axiom_system -- система аксиом
    # model_dict -- набор эталонов по одному для каждого класса нештатного поведения (эталон = [abnormal behavior] model)
    # params -- параметры распознавателя
    def __init__(self, axiom_system, model_dict, params):
        self.axiom_system = axiom_system
        self.model_dict = model_dict
        self.maxdelta = params["maxdelta"]
        self.bound = params["bound"]
    
    # возвращаем маркировку участков нештатного поведения -- т. е. список пар (конец участка, класс)

    def DTW(self, a, b):
        d = np.zeros((len(a) + 1, len(b) + 1))
        INF = 1e100

        for i in range(1, len(a) + 1):
            d[i][0] = INF

        for j in range(1, len(b) + 1):
            d[0][j] = INF

        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i][j] = abs(a[i - 1] - b[j - 1]) + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[len(a)][len(b)]

    def recognize(self, ts):
        mark = self.axiom_system.perform_marking(ts)
        res = []

        for s in self.model_dict:
            now = self.axiom_system.perform_marking(self.model_dict[s])
            
            for i in range(len(mark)):
                for j in range(i + len(now), i + len(now) + self.maxdelta):
                    if j <= len(mark) and self.DTW(mark[i: j], now) <= self.bound:
                        res.append((i, s))
                        break
        return res
