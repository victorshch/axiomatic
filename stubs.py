import pandas as pd
import numpy as np

class SampleAxiom(object):
    def __init__(self, params):
        pass
    
    # ts -- объект pandas.DataFrame
    # возвращает булевский pandas.Series, в котором true соотв. точкам, где аксиома выполняется
    def run(ts):
        return ts[0].shift(1) - ts[0] > 0

# TODO implement -- preskes
class AxiomSystem:
    def __init__(self, axiom_list):
        pass
    
    # возвращает разметку временного ряда ts как array-like (напр., np.array) с номерами аксиом, выполняющихся в соотв. точках ts
    def perform_marking(ts):
        return np.random.randint(0, 5, len(ts))

# training stage concept
class TrainingStage(object):
    #config -- dict с параметрами
    def __init__(self, config):
        pass
    
    def train(self, data_set, artifacts):
        # artifacts -- dict, содержащий промежуточные результаты (например, наборы аксиом, системы аксиом, эталоны)
        return artifacts

# TODO implement -- natalia
class ClusteringAxiomsStage(TrainingStage):
    pass

# TODO implement -- preskes
class GeneticRecognizerStage(TrainingStage):
    pass

class TrainingPipeline(object):
    def __init__(self, stage_list):
        self.stage_list = stage_list
        
    def train(self, data_set):
        artifacts = dict()
        for stage in stage_list:
            artifacts = stage.train(data_set, artifacts)
        
        # для случая распознавания нештатного поведения в artifacts должны быть система аксиом и набор эталонов
        return artifacts

# TODO implement -- alexworld
class AbnormalBehaviorRecognizer(object):
    # axiom_system -- система аксиом
    # model_dict -- набор эталонов по одному для каждого класса нештатного поведения (эталон = [abnormal behavior] model)
    # params -- параметры распознавателя
    def __init__(self, axiom_system, model_dict, params):
        pass
    
    # возвращаем маркировку участков нештатного поведения -- т. е. список пар (конец участка, класс)
    def recognize(self, ts):
        return [(0, "class1")]

# TODO implement -- preskes
class ObjectiveFunction(object):
    def __init__(self, k_e1, k_e2):
        self.k_e1 = k_e1
        self.k_e2 = k_e2
        
    def calculate(recognizer, ts, true_class):
        pass
    
    def calculate(recognizer, data_set):
        pass

def stubs_test():
    clustering_axioms_params = dict()
    genetic_recognizer_params = dict()
    
    training_pipeline = TrainingPipeline([
        ClusteringAxioms(clustering_axioms_params),
        GeneticRecognizerStage(genetic_recognizer_params)
        ])
    
    # формат датасета для распознавания нештатного поведения -- dict, в котором normal -- список траекторий нормального поведения,
    # остальные ключи -- списки предаварийных траекторий классов нештатного поведения
    data_set = dict(normal = [pd.DataFrame(np.sin(np.arange(0, 2*np.pi, 0.1)))], class1 = [pd.DataFrame(np.cos(np.arange(0, 2*np.pi, 0.1)))])
    
    artifacts = training_pipeline.train(data_set)
    
    recognizer_params = dict()
    
    recognizer = AbnormalBehaviorRecognizer(artifacts["axiom_system"], artifacts["model_list"], recognizer_params)
    
    ts = pd.DataFrame(np.sin(np.arange(0, 2*np.pi, 0.1)))
    
    labeling = recognizer.recognize(ts)
    
    for pos, klass in labeling:
        print "found abnormal behavior of class ", klass, "at position", pos
