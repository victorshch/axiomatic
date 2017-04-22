# -*- coding: utf-8 -*-
#from axiomatic.utils import dtw_distances
from axiomatic.dtw_on_steroids import dtw_distances

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
            
            distances = dtw_distances(now, mark)
            
            for i, dist in enumerate(distances):
                if dist <= self.bound:
                    res.append((i, s))
                    break
        return res
