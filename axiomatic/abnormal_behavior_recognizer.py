# coding=UTF-8
import numpy as np
from fastdtw import fastdtw

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
            
            for i in range(len(mark)):
                bound1 = max(i + 1, int(i + len(now) * (1 - self.maxdelta)))
                bound2 = min(len(mark) + 1, int(i + len(now) * (1 + self.maxdelta) + 1))

                for j in range(bound1, bound2):
                    dist, path = fastdtw(self.model_dict[s], mark[i: j], dist = lambda a, b: a == b)

                    if dist <= self.bound * len(path):
                        res.append((i, s))
                        break
        return res
