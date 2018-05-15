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
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

from axiomatic.features import *

class MclustT1Launcher:
  def run(self, features):
    true = load('./datasets/Arrhythmia/training', 'all', 'true', resample = True, ecg_only = True)
    false = load('./datasets/Arrhythmia/training', 'all', 'false', resample = True, ecg_only = True)

    for pack in [true, false]:
      for table in pack:
        for row in table.columns:
          table[row] = np.nan_to_num(table[row])

    dataset = {'train' : {'normal' : false[: len(false) // 2], 'abnormal' : true[: len(true) // 2]},
            'test' : {'normal' : false[len(false) // 2 : ], 'abnormal' : true[len(true) // 2 : ]}}

    FEATURES = {'Leakage' : Leakage(), 'BinaryCovariance' : BinaryCovariance(), 'BinaryFrequency' : BinaryFrequency(), 'AreaBinary' : AreaBinary(), 'Complexity' : Complexity(), 'Kurtosis' : Kurtosis(), 'Count2' : Count2(20), 'LinearRegressionCoef' : LinearRegressionCoef(), 'Skewness' : Skewness(), 'Count1' : Count1(20), 'Count3' : Count3(20), 'IntegralRatioHigh' : IntegralRatioHigh(), 'IntegralRatioLow' : IntegralRatioLow()}

    clustering_axiom_stage = [KMeansClusteringAxiomStage({'clustering_params': {'n_clusters': 5,
          'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2, 'features' :
          [FEATURES[feature] for feature in features]}})]

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
