import pandas as pd
import numpy as np

class ObjectiveFunction(object):
    def __init__(self, k_e1, k_e2):
        """Иницализация
        Параметры:
        k_e1 - стоимость ошибки первого рода
        k_e2 - стоимость ошибки второго рода
        """

        self.k_e1 = k_e1
        self.k_e2 = k_e2
    
    def calculate_one(self, recognizer, ts, true_class):
        """Тестируем данный распознаватель - вычислим стоимость ошибок распознавателя на данном участке, если известен истинный класс нештатного
        поведения на данном участке
        Параметры:
        recognizer - данный распознаватель (AbnormalBehaviorRecognizer)
        ts - траектория
        true_class - класс нештатного поведения (пара позиция - класс) на данном участке
        """

        res = recognizer.recognize(ts)
        first_error = 0
        second_error = 1
        
        for x in res:
            if x != true_class:
                first_error += 1
            if x == true_class:
                second_error = 0
        return first_error * self.k_e1 + second_error * self.k_e2
    
    def calculate(self, recognizer, data_set):
        """Вычислим стоимость ошибок распознавателя на данном множестве учатков, если для них известен истинный класс
        Параметры:
        recognizer - данный распознаватель
        data_set - список пар (траектория - true_class)
        """

        res = 0

        for ts, true_class in data_set:
            res += self.calculate_one(recognizer, ts, true_class)
        return res
