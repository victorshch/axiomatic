# -*- coding: utf-8 -*-
import numpy as np
from fastdtw import fastdtw
from sklearn.metrics.pairwise import pairwise_distances

QuestionMarkSymbol = -2

def dtw_distances_from_matrix(D):
    N = D.shape[0]
    Nmax = D.shape[1]
    S = np.zeros(D.shape)
    R = np.zeros(D.shape)
    S[0, 0] = D[0, 0]
    R[0, 0] = 1
    for a in xrange(1, N):
        S[a, 0] = D[a, 0] + S[a - 1, 0]
        R[a, 0] = 1 + R[a - 1, 0]
    for b in xrange(1, Nmax):
        #S[N - 1, b] = D[N - 1, b] + S[N - 1, b + 1]
        #R[N - 1, b] = R[N - 1, b + 1] + 1
        S[0, b] = D[0, b] 
        R[0, b] = 1
    
    for a in xrange(1, N):
        for b in xrange(1, Nmax):
            Dab = D[a, b]
            Sa1b1 = S[a - 1, b - 1]
            Ra1b1 = R[a - 1, b - 1]
            Sab1 = S[a, b - 1]
            Rab1 = R[a, b - 1]
            Sa1b = S[a - 1, b]
            Ra1b = R[a - 1, b]
            diag = (Dab + Sa1b1) * (Ra1b1 + 1)
            right = (Dab + Sab1) * (Rab1 - 1)
            down = (Dab + Sa1b) * (Ra1b + 1)
            if down < diag and down < right:
                S[a, b] = Dab + Sa1b
                R[a, b] = Ra1b + 1
            elif diag <= down and diag <= right:
                S[a, b] = Dab + Sa1b1
                R[a, b] = Ra1b1 + 1
            elif right < diag and right <= down:
                S[a, b] = Dab + Sab1
                R[a, b] = Rab1 + 1
            else:
                print "Warning: dtw_from_matrix() strange behavior"
    
    dist = S[N - 1, :] / R[N - 1, :]
    return dist

def dtw_distances(model, observed_marking, metric):
    distancesM = pairwise_distances(np.array(model).reshape(-1, 1), np.array(observed_marking).reshape(-1, 1), \
        metric=metric)
    return dtw_distances_from_matrix(distancesM)

class AbnormalBehaviorRecognizer(object):
    def __init__(self, axiom_system, model_dict, params):
        """Инициализируем структуру распознавателя
        Параметры:
        axiom_system -- система аксиом
        model_dict -- набор эталонов по одному для каждого класса нештатного поведения (эталон = [abnormal behavior] model)
        params -- параметры распознавателя (maxdelta - разброс "окна" поиска, bound - граница отсечения между штатным и нештатным результатом)
        """

        self.axiom_system = axiom_system
        self.model_dict = model_dict
        self.maxdelta = params["maxdelta"]
        self.bound = params["bound"]
    

    def recognize(self, ts):
        """Возвращаем маркировку участков нештатного поведения -- т. е. список пар (конец участка, класс)
        Параметры:
        ts - текущий путь
        """

        mark = self.axiom_system.perform_marking(ts)
        res = []

        for s in self.model_dict:
            now = self.model_dict[s]
            
            distances = dtw_distances(now, mark, lambda a, b: a != b and a != QuestionMarkSymbol)
            
            for i, dist in enumerate(distances):
                if dist <= self.bound:
                    res.append((i, s))
                    break
        return res
