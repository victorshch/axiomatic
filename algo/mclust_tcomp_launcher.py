from datasets.Arrhythmia.get import load
from datetime import datetime
import sys
import numpy as np

import pandas as pd
import pickle

from axiomatic.base import AxiomSystem, TrainingPipeline
from axiomatic.axiom_training_stage import *
from axiomatic.genetic_recognizer_training_stage import GeneticRecognizerTrainingStage
from axiomatic.elementary_conditions import *
from axiomatic.objective_function import ScoreObjectiveFunction as ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer, UnitedAbnormalBehaviorRecognizer

from axiomatic.features import *
import json


class MclustTCompLauncher:        
  def run(self, config):
    print(datetime.now())

    FEATURES = {'Leakage' : Leakage(), 'BinaryCovariance' : BinaryCovariance(), 'BinaryFrequency' : BinaryFrequency(), 'AreaBinary' : AreaBinary(), 'Complexity' : Complexity(), 'Kurtosis' : Kurtosis(), 'Count2' : Count2(20), 'LinearRegressionCoef' : LinearRegressionCoef(), 'Skewness' : Skewness(), 'Count1' : Count1(20), 'Count3' : Count3(20), 'IntegralRatioHigh' : IntegralRatioHigh(), 'IntegralRatioLow' : IntegralRatioLow()}

    false = load('./datasets/Arrhythmia/training', 'all', 'false', resample = True, ecg_only = True)

    for table in false:
      for row in table.columns:
        table[row] = np.nan_to_num(table[row])

    fin_dataset = {'train' : {'normal' : false[: len(false) // 2], 'abnormal' : []}, 'test' : {'normal' : false[len(false) // 2 :], 'abnormal' : []}}
    
    def calc_one(abn_type):
      print 'Calculating', abn_type
      true = load('./datasets/Arrhythmia/training', [abn_type], 'true', resample = True, ecg_only = True)
      
      for table in true:
        for row in table.columns:
          table[row] = np.nan_to_num(table[row])

      dataset = {'train' : {'normal' : false[: len(false) // 2], 'abnormal' : true[: len(true) // 2]},
            'test' : {'normal' : false[len(false) // 2 : ], 'abnormal' : true[len(true) // 2 : ]}}
      fin_dataset['train']['abnormal'] += dataset['train']['abnormal']
      fin_dataset['test']['abnormal'] += dataset['test']['abnormal']

      clustering_axiom_stage = [KMeansClusteringAxiomStage({'clustering_params': {'n_clusters': 5,
            'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2, 'features' :
            [FEATURES[feature] for feature in config[abn_type]]}})]

      genetic_recognizer_stage = GeneticRecognizerTrainingStage(dict(n_jobs=1, population_size=100, iteration_count=10, use_question_mark=False, num_axioms_weight=0.0))

      axiom_union_stage = AxiomUnionStage(clustering_axiom_stage)
      training_pipeline = TrainingPipeline([axiom_union_stage, genetic_recognizer_stage])

      artifacts = training_pipeline.train(dataset, dict())

      print "Artifacts after training: ", artifacts

      recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                            dict(bound=0.1,maxdelta=0.5))

      obj_fn = ObjectiveFunction()
      test_result = obj_fn.calculate(recognizer, dataset['test'])

      print test_result
      return recognizer
    
    recognizers = []
    
    for key in config.keys():
      recognizers.append(calc_one(key))

    unite_recognizer = UnitedAbnormalBehaviorRecognizer(recognizers)

    dataset = fin_dataset
    obj_fn = ObjectiveFunction()
    test_result = obj_fn.calculate(unite_recognizer, dataset['test'])

    print test_result
    return unite_recognizer
